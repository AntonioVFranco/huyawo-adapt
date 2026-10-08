"""Bounded reproducibility evaluation of scripted local workspace action tapes.

This module measures deterministic in-process replay, not task success, model
performance, independent state attestation, or eligibility for training rewards.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_adapt.agentic.environment.interface import (
    AgentEnvironmentAction,
    AgentEnvironmentResetRequest,
    AgentEnvironmentResetResult,
    AgentEnvironmentStepRequest,
    AgentEnvironmentStepResult,
    validate_reset_exchange,
    validate_step_exchange,
)
from huyawo_adapt.agentic.environment.workspace import (
    BoundedWorkspaceEnvironment,
    WorkspaceFixture,
)
from huyawo_adapt.contracts.base import ContractModel, NonEmptyString, Sha256Digest

ReplayStatus = Literal["reproducible", "incomplete", "blocked"]
TerminalStatus = Literal["continue", "terminated", "truncated", "error", "not_available"]
ReplayReason = Literal[
    "terminal_replay_match",
    "action_tape_incomplete",
    "action_rejected",
    "step_after_terminal",
    "environment_replay_conflict",
    "evidence_conflict",
]
ActionTape = Annotated[tuple[AgentEnvironmentAction, ...], Field(min_length=1, max_length=16)]


def _evidence(data: object) -> str:
    canonical = json.dumps(
        {"domain": "scripted-agent-baseline-v1", "data": data},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AgentScriptedEvaluationCase(ContractModel):
    """Caller-authored action tape; never interpreted as model-generated actions."""

    contract_type: Literal["agent_scripted_evaluation_case"] = "agent_scripted_evaluation_case"
    fixture: WorkspaceFixture
    episode_id: NonEmptyString
    actions: ActionTape
    declared_source: Literal["scripted_local_fixture"] = "scripted_local_fixture"
    case_provenance_sha256: Sha256Digest


class AgentEvaluationStepEvidence(ContractModel):
    """Public, actual M7 transition references with no private state payload."""

    contract_type: Literal["agent_evaluation_step_evidence"] = "agent_evaluation_step_evidence"
    step_index: int = Field(ge=0)
    action_fingerprint: Sha256Digest
    before_state_sha256: Sha256Digest
    after_state_sha256: Sha256Digest
    public_observation_sha256: Sha256Digest
    transition_evidence_sha256: Sha256Digest
    terminal_status: Literal["continue", "terminated", "truncated", "error"]


class AgentScriptedEvaluationResult(ContractModel):
    """Reproducibility finding; explicitly not a success or reward verdict."""

    contract_type: Literal["agent_scripted_evaluation_result"] = "agent_scripted_evaluation_result"
    authority_scope: Literal["scripted_local_replay_only"] = "scripted_local_replay_only"
    case_fingerprint: Sha256Digest
    environment_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    episode_id: NonEmptyString
    reset_result_fingerprint: Sha256Digest
    steps: tuple[AgentEvaluationStepEvidence, ...]
    attempted_action_count: int = Field(ge=0, le=16)
    executed_action_count: int = Field(ge=0, le=16)
    final_state_sha256: Sha256Digest | None
    final_public_observation_sha256: Sha256Digest | None
    terminal_status: TerminalStatus
    replay_status: ReplayStatus
    reason: ReplayReason
    replay_evidence_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (
            self.executed_action_count != len(self.steps)
            or self.attempted_action_count < self.executed_action_count
        ):
            raise ValueError("inconsistent action counters")
        if (self.final_state_sha256 is None) != (self.final_public_observation_sha256 is None):
            raise ValueError("partial final observation commitment")
        expected = {
            "reproducible": {"terminal_replay_match"},
            "incomplete": {"action_tape_incomplete"},
            "blocked": {
                "action_rejected",
                "step_after_terminal",
                "environment_replay_conflict",
                "evidence_conflict",
            },
        }
        if self.reason not in expected[self.replay_status]:
            raise ValueError("replay status and reason are inconsistent")
        if self.replay_status == "reproducible" and self.terminal_status not in (
            "terminated",
            "truncated",
        ):
            raise ValueError("reproducible requires an environment terminal")
        if self.replay_status == "incomplete" and self.terminal_status != "continue":
            raise ValueError("incomplete requires a continuing environment")
        return self


class ScriptedAgentBaselineEvaluator:
    """Execute the same local tape on two new deterministic workspaces."""

    def evaluate(self, case: AgentScriptedEvaluationCase) -> AgentScriptedEvaluationResult:
        if type(case) is not AgentScriptedEvaluationCase:
            raise TypeError("expected AgentScriptedEvaluationCase")
        checked = AgentScriptedEvaluationCase.model_validate(case.model_dump(mode="python"))
        fixture = checked.fixture
        left = BoundedWorkspaceEnvironment(fixture)
        right = BoundedWorkspaceEnvironment(fixture)
        if left is right:
            raise RuntimeError("replay instances must be independent")
        reset_request = AgentEnvironmentResetRequest(
            environment_identity=fixture.environment_identity,
            task_identity=fixture.task_identity,
            episode_id=checked.episode_id,
            seed=fixture.seed,
        )
        # An invalid reset has no admissible evidence record, so it fails closed.
        first = left.reset(reset_request)
        second = right.reset(reset_request)
        validate_reset_exchange(reset_request, first)
        validate_reset_exchange(reset_request, second)

        prior_left: AgentEnvironmentResetResult | AgentEnvironmentStepResult = first
        prior_right: AgentEnvironmentResetResult | AgentEnvironmentStepResult = second
        records: list[AgentEvaluationStepEvidence] = []
        attempted = 0
        reason: ReplayReason = "action_tape_incomplete"
        status: ReplayStatus = "incomplete"
        terminal: TerminalStatus = "continue"

        if first.canonical_bytes() != second.canonical_bytes():
            reason = "environment_replay_conflict"
            status = "blocked"
        else:
            for action in checked.actions:
                attempted += 1
                if terminal != "continue":
                    status = "blocked"
                    reason = "step_after_terminal"
                    break

                index = prior_left.observation.observation_index
                request_left = AgentEnvironmentStepRequest(
                    environment_identity=fixture.environment_identity,
                    task_identity=fixture.task_identity,
                    episode_id=checked.episode_id,
                    step_index=index,
                    before_state_sha256=prior_left.observation.state_sha256,
                    action=action,
                )
                request_right = AgentEnvironmentStepRequest(
                    environment_identity=fixture.environment_identity,
                    task_identity=fixture.task_identity,
                    episode_id=checked.episode_id,
                    step_index=prior_right.observation.observation_index,
                    before_state_sha256=prior_right.observation.state_sha256,
                    action=action,
                )
                rejected = 0
                step_left: AgentEnvironmentStepResult | None = None
                step_right: AgentEnvironmentStepResult | None = None
                try:
                    step_left = left.step(request_left)
                    validate_step_exchange(prior_left, request_left, step_left)
                except (TypeError, ValueError):
                    rejected += 1
                try:
                    step_right = right.step(request_right)
                    validate_step_exchange(prior_right, request_right, step_right)
                except (TypeError, ValueError):
                    rejected += 1

                if rejected:
                    status = "blocked"
                    reason = "action_rejected" if rejected == 2 else "evidence_conflict"
                    break
                if step_left is None or step_right is None:
                    raise RuntimeError("missing validated step without rejection")
                if step_left.canonical_bytes() != step_right.canonical_bytes():
                    status = "blocked"
                    reason = "environment_replay_conflict"
                    break
                prior_left, prior_right = step_left, step_right
                terminal = step_left.terminal_status
                records.append(
                    AgentEvaluationStepEvidence(
                        step_index=step_left.step_index,
                        action_fingerprint=step_left.action_fingerprint,
                        before_state_sha256=step_left.transition.before_state_sha256,
                        after_state_sha256=step_left.transition.after_state_sha256,
                        public_observation_sha256=step_left.observation.public_observation.sha256,
                        transition_evidence_sha256=step_left.transition.transition_evidence_sha256,
                        terminal_status=step_left.terminal_status,
                    )
                )
                if terminal != "continue":
                    status = "reproducible"
                    reason = "terminal_replay_match"

        final_state = prior_left.observation.state_sha256
        final_public = prior_left.observation.public_observation.sha256
        fingerprint = checked.fingerprint()
        digest = _evidence(
            {
                "case_fingerprint": fingerprint,
                "left_reset_fingerprint": first.fingerprint(),
                "right_reset_fingerprint": second.fingerprint(),
                "steps": [entry.canonical_data() for entry in records],
                "attempted_action_count": attempted,
                "executed_action_count": len(records),
                "terminal_status": terminal,
                "replay_status": status,
                "reason": reason,
            }
        )
        return AgentScriptedEvaluationResult(
            case_fingerprint=fingerprint,
            environment_fingerprint=fixture.environment_identity.fingerprint(),
            task_fingerprint=fixture.task_identity.fingerprint(),
            episode_id=checked.episode_id,
            reset_result_fingerprint=first.fingerprint(),
            steps=tuple(records),
            attempted_action_count=attempted,
            executed_action_count=len(records),
            final_state_sha256=final_state,
            final_public_observation_sha256=final_public,
            terminal_status=terminal,
            replay_status=status,
            reason=reason,
            replay_evidence_sha256=digest,
        )
