"""Immutable, fixture-scoped state verification without environment execution."""

from __future__ import annotations

from typing import Literal, Protocol, Self

from pydantic import model_validator

from huyawo_adapt.agentic.contracts import (
    AgentEnvironmentIdentity,
    AgentTaskIdentity,
    AgentVerifierIdentity,
)
from huyawo_adapt.agentic.trajectory import AgentTrajectory, AgentTrajectoryPayload
from huyawo_adapt.contracts.base import ContractModel, Sha256Digest

VerifierVerdict = Literal["pass", "fail", "blocked"]
VerifierReason = Literal[
    "exact_state_match",
    "exact_state_mismatch",
    "missing_observation",
    "missing_final_transition",
    "transition_evidence_conflict",
    "terminal_evidence_conflict",
]


class ExpectedStateReference(ContractModel):
    """Exact expected UTF-8 final state for one task and success criterion."""

    contract_type: Literal["agent_expected_state_reference"] = "agent_expected_state_reference"
    task_fingerprint: Sha256Digest
    success_criteria_sha256: Sha256Digest
    expected_final_state: AgentTrajectoryPayload
    reference_provenance_sha256: Sha256Digest


class AgentVerifierRequest(ContractModel):
    """Frozen cross-identified verifier inputs and observed state evidence."""

    contract_type: Literal["agent_verifier_request"] = "agent_verifier_request"
    verifier_identity: AgentVerifierIdentity
    task_identity: AgentTaskIdentity
    environment_identity: AgentEnvironmentIdentity
    trajectory: AgentTrajectory
    reference: ExpectedStateReference
    observed_final_state: AgentTrajectoryPayload | None
    observation_provenance_sha256: Sha256Digest | None

    @model_validator(mode="after")
    def validate_identity_links(self) -> Self:
        if self.verifier_identity.verifier_kind != "deterministic":
            raise ValueError("reference verifier requires deterministic verifier identity")

        environment_fingerprint = self.environment_identity.fingerprint()
        verifier_fingerprint = self.verifier_identity.fingerprint()
        task_fingerprint = self.task_identity.fingerprint()

        if self.task_identity.environment_fingerprint != environment_fingerprint:
            raise ValueError("task environment fingerprint does not match environment identity")
        if self.task_identity.verifier_fingerprint != verifier_fingerprint:
            raise ValueError("task verifier fingerprint does not match verifier identity")
        if self.trajectory.environment_fingerprint != environment_fingerprint:
            raise ValueError(
                "trajectory environment fingerprint does not match environment identity"
            )
        if self.trajectory.verifier_fingerprint != verifier_fingerprint:
            raise ValueError("trajectory verifier fingerprint does not match verifier identity")
        if self.trajectory.task_fingerprint != task_fingerprint:
            raise ValueError("trajectory task fingerprint does not match task identity")
        if self.reference.task_fingerprint != task_fingerprint:
            raise ValueError("reference task fingerprint does not match task identity")
        if self.reference.success_criteria_sha256 != self.task_identity.success_criteria_sha256:
            raise ValueError("reference success criteria do not match task identity")
        if (self.observed_final_state is None) != (self.observation_provenance_sha256 is None):
            raise ValueError("observed final state and provenance must appear together")

        return self


class AgentVerifierResult(ContractModel):
    """Canonical local-only verdict, not a training or benchmark reward."""

    contract_type: Literal["agent_verifier_result"] = "agent_verifier_result"
    verifier_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    environment_fingerprint: Sha256Digest
    trajectory_fingerprint: Sha256Digest
    reference_fingerprint: Sha256Digest
    observed_final_state_sha256: Sha256Digest | None
    verdict: VerifierVerdict
    reason: VerifierReason
    authority_scope: Literal["local_fixture_comparison_only"] = "local_fixture_comparison_only"
    observation_provenance_sha256: Sha256Digest | None

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        if (self.observed_final_state_sha256 is None) != (
            self.observation_provenance_sha256 is None
        ):
            raise ValueError("observed state and provenance must appear together")

        if self.verdict == "pass":
            if self.reason != "exact_state_match":
                raise ValueError("pass requires exact_state_match")
            if self.observed_final_state_sha256 is None:
                raise ValueError("pass requires observed state evidence")
        elif self.verdict == "fail":
            if self.reason != "exact_state_mismatch":
                raise ValueError("fail requires exact_state_mismatch")
            if self.observed_final_state_sha256 is None:
                raise ValueError("fail requires observed state evidence")
        elif self.reason not in (
            "missing_observation",
            "missing_final_transition",
            "transition_evidence_conflict",
            "terminal_evidence_conflict",
        ):
            raise ValueError("blocked requires an evidence insufficiency or conflict reason")

        if self.reason == "missing_observation" and self.observed_final_state_sha256 is not None:
            raise ValueError("missing_observation requires absent observed state")
        if (
            self.reason
            in (
                "transition_evidence_conflict",
                "terminal_evidence_conflict",
            )
            and self.observed_final_state_sha256 is None
        ):
            raise ValueError("evidence conflict requires observed state")
        return self


class AgentVerifier(Protocol):
    """Pure verification boundary for a strict canonical request."""

    def verify(self, request: AgentVerifierRequest) -> AgentVerifierResult: ...


class ExactStateReferenceVerifier:
    """Exact fixture-state comparator; never executes tools or environments."""

    def verify(self, request: AgentVerifierRequest) -> AgentVerifierResult:
        if type(request) is not AgentVerifierRequest:
            raise TypeError("verify requires an AgentVerifierRequest")

        validated = AgentVerifierRequest.model_validate(request.model_dump(mode="python"))
        final = validated.trajectory.steps[-1]
        observed = validated.observed_final_state
        transition = final.state_transition

        verdict: VerifierVerdict
        reason: VerifierReason

        if transition is None:
            verdict, reason = "blocked", "missing_final_transition"
        elif observed is None:
            verdict, reason = "blocked", "missing_observation"
        elif transition.after_state_sha256 != observed.sha256:
            verdict, reason = "blocked", "transition_evidence_conflict"
        elif (
            final.terminal_status in ("failure", "error", "truncated")
            and observed.content == validated.reference.expected_final_state.content
        ):
            verdict, reason = "blocked", "terminal_evidence_conflict"
        elif observed.content == validated.reference.expected_final_state.content:
            verdict, reason = "pass", "exact_state_match"
        else:
            verdict, reason = "fail", "exact_state_mismatch"

        return AgentVerifierResult(
            verifier_fingerprint=validated.verifier_identity.fingerprint(),
            task_fingerprint=validated.task_identity.fingerprint(),
            environment_fingerprint=validated.environment_identity.fingerprint(),
            trajectory_fingerprint=validated.trajectory.fingerprint(),
            reference_fingerprint=validated.reference.fingerprint(),
            observed_final_state_sha256=observed.sha256 if observed is not None else None,
            verdict=verdict,
            reason=reason,
            observation_provenance_sha256=validated.observation_provenance_sha256,
        )
