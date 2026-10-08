"""Pure immutable exchange contracts for executable agent environments."""

from __future__ import annotations

from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from huyawo_adapt.agentic.contracts import AgentEnvironmentIdentity, AgentTaskIdentity
from huyawo_adapt.agentic.trajectory import AgentStateTransitionRecord, AgentTrajectoryPayload
from huyawo_adapt.contracts.base import ContractModel, NonEmptyString, Sha256Digest


class AgentEnvironmentAction(ContractModel):
    """One canonical agent action with exact arguments and provenance."""

    contract_type: Literal["agent_environment_action"] = "agent_environment_action"
    action_id: NonEmptyString
    arguments: AgentTrajectoryPayload
    action_provenance_sha256: Sha256Digest
    tool_fingerprint: Sha256Digest | None = None


class AgentEnvironmentObservation(ContractModel):
    """Public observation and separate opaque internal-state commitment."""

    contract_type: Literal["agent_environment_observation"] = "agent_environment_observation"
    episode_id: NonEmptyString
    observation_index: int = Field(ge=0)
    public_observation: AgentTrajectoryPayload
    state_sha256: Sha256Digest
    state_provenance_sha256: Sha256Digest
    visibility_policy_sha256: Sha256Digest


class AgentEnvironmentResetRequest(ContractModel):
    """Immutable reset intent bound to an environment, task and seed."""

    contract_type: Literal["agent_environment_reset_request"] = "agent_environment_reset_request"
    environment_identity: AgentEnvironmentIdentity
    task_identity: AgentTaskIdentity
    episode_id: NonEmptyString
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_task_environment_link(self) -> Self:
        if self.task_identity.environment_fingerprint != self.environment_identity.fingerprint():
            raise ValueError("task environment fingerprint mismatch")
        return self


class AgentEnvironmentResetResult(ContractModel):
    """Reported initial observation; not proof of an executed reset."""

    contract_type: Literal["agent_environment_reset_result"] = "agent_environment_reset_result"
    environment_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    episode_id: NonEmptyString
    seed: int = Field(ge=0)
    observation: AgentEnvironmentObservation
    reset_evidence_sha256: Sha256Digest


class AgentEnvironmentStepRequest(ContractModel):
    """A step request consuming one explicitly identified prior state."""

    contract_type: Literal["agent_environment_step_request"] = "agent_environment_step_request"
    environment_identity: AgentEnvironmentIdentity
    task_identity: AgentTaskIdentity
    episode_id: NonEmptyString
    step_index: int = Field(ge=0)
    before_state_sha256: Sha256Digest
    action: AgentEnvironmentAction

    @model_validator(mode="after")
    def validate_task_environment_link(self) -> Self:
        if self.task_identity.environment_fingerprint != self.environment_identity.fingerprint():
            raise ValueError("task environment fingerprint mismatch")
        return self


class AgentEnvironmentStepResult(ContractModel):
    """One reported transition and terminality without success or reward authority."""

    contract_type: Literal["agent_environment_step_result"] = "agent_environment_step_result"
    environment_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    episode_id: NonEmptyString
    step_index: int = Field(ge=0)
    action_fingerprint: Sha256Digest
    observation: AgentEnvironmentObservation
    transition: AgentStateTransitionRecord
    terminal_status: Literal["continue", "terminated", "truncated", "error"]
    terminal_evidence_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_terminal_evidence(self) -> Self:
        if self.terminal_status == "continue":
            if self.terminal_evidence_sha256 is not None:
                raise ValueError("continue must not carry terminal evidence")
        elif self.terminal_evidence_sha256 is None:
            raise ValueError("terminal status requires terminal evidence")
        return self


class ExecutableAgentEnvironment(Protocol):
    """Synchronous environment boundary; implementation is out of M6 scope."""

    def reset(self, request: AgentEnvironmentResetRequest) -> AgentEnvironmentResetResult: ...

    def step(self, request: AgentEnvironmentStepRequest) -> AgentEnvironmentStepResult: ...


def _checked_reset_result(value: AgentEnvironmentResetResult) -> AgentEnvironmentResetResult:
    if type(value) is not AgentEnvironmentResetResult:
        raise TypeError("expected AgentEnvironmentResetResult")
    return AgentEnvironmentResetResult.model_validate(value.model_dump(mode="python"))


def _checked_step_result(value: AgentEnvironmentStepResult) -> AgentEnvironmentStepResult:
    if type(value) is not AgentEnvironmentStepResult:
        raise TypeError("expected AgentEnvironmentStepResult")
    return AgentEnvironmentStepResult.model_validate(value.model_dump(mode="python"))


def validate_reset_exchange(
    request: AgentEnvironmentResetRequest,
    result: AgentEnvironmentResetResult,
) -> None:
    """Reject contradictions in a reported reset without executing an environment."""
    if type(request) is not AgentEnvironmentResetRequest:
        raise TypeError("expected AgentEnvironmentResetRequest")
    request = AgentEnvironmentResetRequest.model_validate(request.model_dump(mode="python"))
    result = _checked_reset_result(result)
    if result.environment_fingerprint != request.environment_identity.fingerprint():
        raise ValueError("reset environment fingerprint mismatch")
    if result.task_fingerprint != request.task_identity.fingerprint():
        raise ValueError("reset task fingerprint mismatch")
    if (
        result.episode_id != request.episode_id
        or result.observation.episode_id != request.episode_id
    ):
        raise ValueError("reset episode mismatch")
    if result.seed != request.seed:
        raise ValueError("reset seed mismatch")
    if result.observation.observation_index != 0:
        raise ValueError("reset observation index must be zero")
    if result.observation.state_sha256 != request.task_identity.initial_state_sha256:
        raise ValueError("reset initial state commitment mismatch")


def validate_step_exchange(
    previous: AgentEnvironmentResetResult | AgentEnvironmentStepResult,
    request: AgentEnvironmentStepRequest,
    result: AgentEnvironmentStepResult,
) -> None:
    """Reject inconsistent consecutive exchanges, including steps after terminality."""
    checked_previous: AgentEnvironmentResetResult | AgentEnvironmentStepResult
    if isinstance(previous, AgentEnvironmentResetResult):
        checked_previous = _checked_reset_result(previous)
    elif isinstance(previous, AgentEnvironmentStepResult):
        checked_previous = _checked_step_result(previous)
        if checked_previous.terminal_status != "continue":
            raise ValueError("cannot step after a terminal environment result")
    else:
        raise TypeError("expected previous reset or step result")
    if type(request) is not AgentEnvironmentStepRequest:
        raise TypeError("expected AgentEnvironmentStepRequest")
    request = AgentEnvironmentStepRequest.model_validate(request.model_dump(mode="python"))
    result = _checked_step_result(result)

    if isinstance(checked_previous, AgentEnvironmentResetResult):
        if checked_previous.observation.observation_index != 0:
            raise ValueError("previous reset observation index must be zero")
        if checked_previous.observation.state_sha256 != request.task_identity.initial_state_sha256:
            raise ValueError("previous reset initial-state mismatch")
    else:
        if checked_previous.observation.observation_index != checked_previous.step_index + 1:
            raise ValueError("previous step observation index mismatch")
        if (
            checked_previous.transition.after_state_sha256
            != checked_previous.observation.state_sha256
        ):
            raise ValueError("previous step state transition mismatch")

    environment_fingerprint = request.environment_identity.fingerprint()
    task_fingerprint = request.task_identity.fingerprint()
    if checked_previous.environment_fingerprint != environment_fingerprint:
        raise ValueError("previous environment fingerprint mismatch")
    if checked_previous.task_fingerprint != task_fingerprint:
        raise ValueError("previous task fingerprint mismatch")
    if result.environment_fingerprint != environment_fingerprint:
        raise ValueError("step environment fingerprint mismatch")
    if result.task_fingerprint != task_fingerprint:
        raise ValueError("step task fingerprint mismatch")
    if (
        checked_previous.episode_id != request.episode_id
        or checked_previous.observation.episode_id != request.episode_id
        or result.episode_id != request.episode_id
        or result.observation.episode_id != request.episode_id
    ):
        raise ValueError("step episode mismatch")
    if request.step_index != checked_previous.observation.observation_index:
        raise ValueError("step index does not follow previous observation")
    if result.step_index != request.step_index:
        raise ValueError("step result index mismatch")
    if result.observation.observation_index != request.step_index + 1:
        raise ValueError("step observation index mismatch")
    if request.before_state_sha256 != checked_previous.observation.state_sha256:
        raise ValueError("step previous state commitment mismatch")
    if result.action_fingerprint != request.action.fingerprint():
        raise ValueError("step action fingerprint mismatch")
    if result.transition.before_state_sha256 != request.before_state_sha256:
        raise ValueError("step transition before-state mismatch")
    if result.transition.after_state_sha256 != result.observation.state_sha256:
        raise ValueError("step transition after-state mismatch")
