from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from huyawo_adapt.contracts.base import (
    ContractModel,
    NonEmptyString,
    Sha256Digest,
    StrictFrozenModel,
)

ExactString = Annotated[str, StringConstraints(strip_whitespace=False)]
NonEmptyTokenIds = Annotated[
    tuple[Annotated[int, Field(ge=0)], ...],
    Field(min_length=1),
]


class AgentTrajectoryPayload(StrictFrozenModel):
    """Exact UTF-8 payload with content-addressed evidence."""

    media_type: NonEmptyString
    content: ExactString
    sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_content_digest(self) -> Self:
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("sha256 must identify the exact UTF-8 content")
        return self


class AgentToolCallRecord(StrictFrozenModel):
    """One agent tool invocation and exact arguments."""

    call_id: NonEmptyString
    tool_fingerprint: Sha256Digest
    arguments: AgentTrajectoryPayload


class AgentToolResultRecord(StrictFrozenModel):
    """One tool response associated with an invocation."""

    call_id: NonEmptyString
    tool_fingerprint: Sha256Digest
    result: AgentTrajectoryPayload
    is_error: bool


class AgentStateTransitionRecord(StrictFrozenModel):
    """Identities of a state transition and its proof."""

    before_state_sha256: Sha256Digest
    after_state_sha256: Sha256Digest
    transition_evidence_sha256: Sha256Digest


class AgentRewardComponent(StrictFrozenModel):
    """Reward component and the source that produced it."""

    component_id: NonEmptyString
    value: float
    source_fingerprint: Sha256Digest
    details_sha256: Sha256Digest | None = None


class AgentFailureRecord(StrictFrozenModel):
    """Terminal failure evidence without retry semantics."""

    failure_type: NonEmptyString
    message: ExactString | None = None
    details_sha256: Sha256Digest | None = None
    retryable: bool


class AgentTrajectoryStep(StrictFrozenModel):
    """Canonical observation, generation, and action evidence for one step."""

    step_index: int = Field(ge=0)
    observation: AgentTrajectoryPayload
    model_input: AgentTrajectoryPayload
    model_input_token_ids: NonEmptyTokenIds | None = None
    generated_output: AgentTrajectoryPayload
    generated_token_ids: NonEmptyTokenIds | None = None
    tool_call: AgentToolCallRecord | None = None
    tool_result: AgentToolResultRecord | None = None
    state_transition: AgentStateTransitionRecord | None = None
    reward_components: tuple[AgentRewardComponent, ...] = ()
    terminal_status: Literal["continue", "success", "failure", "truncated", "error"]
    failure: AgentFailureRecord | None = None
    provenance_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_step_evidence(self) -> Self:
        if self.tool_result is not None:
            if self.tool_call is None:
                raise ValueError("tool_result requires a tool_call in the same step")
            if self.tool_result.call_id != self.tool_call.call_id:
                raise ValueError("tool call/result call_id mismatch")
            if self.tool_result.tool_fingerprint != self.tool_call.tool_fingerprint:
                raise ValueError("tool call/result tool_fingerprint mismatch")

        if self.terminal_status in ("continue", "success") and self.failure is not None:
            raise ValueError("continue and success steps cannot declare failure evidence")
        if self.terminal_status in ("failure", "error") and self.failure is None:
            raise ValueError("failure and error steps require failure evidence")
        return self


NonEmptySteps = Annotated[tuple[AgentTrajectoryStep, ...], Field(min_length=1)]


class AgentTrajectory(ContractModel):
    """Backend-independent canonical record for one agent trajectory."""

    contract_type: Literal["agent_trajectory"] = "agent_trajectory"
    trajectory_id: NonEmptyString
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    harness_fingerprint: Sha256Digest
    tool_set_fingerprint: Sha256Digest
    environment_fingerprint: Sha256Digest
    verifier_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    rollout_policy_fingerprint: Sha256Digest
    steps: NonEmptySteps
    capture_provenance_sha256: Sha256Digest
    parent_trajectory_fingerprint: Sha256Digest | None = None
    transformation_fingerprint: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_trajectory(self) -> Self:
        if (self.parent_trajectory_fingerprint is None) != (
            self.transformation_fingerprint is None
        ):
            raise ValueError("parent and transformation fingerprints must appear together")

        for index, step in enumerate(self.steps):
            if step.step_index != index:
                raise ValueError("step indexes must be contiguous from zero")
            if index < len(self.steps) - 1 and step.terminal_status != "continue":
                raise ValueError("non-final steps must have terminal_status continue")
            if index == len(self.steps) - 1 and step.terminal_status == "continue":
                raise ValueError("final step must declare a terminal status")
        return self
