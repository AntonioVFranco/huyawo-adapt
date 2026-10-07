from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_adapt.contracts.base import (
    ContractModel,
    NonEmptyString,
    Sha256Digest,
)

NonEmptyFingerprintTuple = Annotated[
    tuple[Sha256Digest, ...],
    Field(min_length=1),
]

VerifierKind = Literal[
    "deterministic",
    "llm_judge",
    "hybrid",
]


class AgentHarnessIdentity(ContractModel):
    """Immutable identity for agent harness execution semantics."""

    contract_type: Literal["agent_harness_identity"] = "agent_harness_identity"
    harness_id: NonEmptyString
    source_revision: NonEmptyString
    implementation_sha256: Sha256Digest
    configuration_sha256: Sha256Digest
    chat_template_sha256: Sha256Digest | None = None
    observation_format_sha256: Sha256Digest
    control_flow_sha256: Sha256Digest
    stopping_behavior_sha256: Sha256Digest


class AgentToolIdentity(ContractModel):
    """Immutable identity for one callable agent tool."""

    contract_type: Literal["agent_tool_identity"] = "agent_tool_identity"
    tool_id: NonEmptyString
    tool_schema_sha256: Sha256Digest
    implementation_sha256: Sha256Digest
    result_schema_sha256: Sha256Digest | None = None
    side_effect_policy_sha256: Sha256Digest | None = None


class AgentToolSetIdentity(ContractModel):
    """Immutable identity for the exact set of tools exposed to an agent."""

    contract_type: Literal["agent_tool_set_identity"] = "agent_tool_set_identity"
    tool_fingerprints: NonEmptyFingerprintTuple

    @field_validator("tool_fingerprints")
    @classmethod
    def normalize_tool_fingerprints(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize tool membership into deterministic fingerprint order."""
        if len(values) != len(set(values)):
            raise ValueError("tool_fingerprints must not contain duplicates")

        return tuple(sorted(values))


class AgentEnvironmentIdentity(ContractModel):
    """Immutable identity for reusable executable environment semantics."""

    contract_type: Literal["agent_environment_identity"] = "agent_environment_identity"
    environment_id: NonEmptyString
    definition_sha256: Sha256Digest
    state_schema_sha256: Sha256Digest
    reset_semantics_sha256: Sha256Digest
    transition_semantics_sha256: Sha256Digest
    terminal_conditions_sha256: Sha256Digest


class AgentVerifierIdentity(ContractModel):
    """Immutable identity for deterministic or judge-backed verification."""

    contract_type: Literal["agent_verifier_identity"] = "agent_verifier_identity"
    verifier_id: NonEmptyString
    verifier_kind: VerifierKind
    implementation_sha256: Sha256Digest
    configuration_sha256: Sha256Digest
    output_schema_sha256: Sha256Digest
    judge_model_fingerprint: Sha256Digest | None = None
    judge_prompt_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_verifier_kind(self) -> Self:
        """Validate judge identity requirements for each verifier kind."""
        judge_values = (
            self.judge_model_fingerprint,
            self.judge_prompt_sha256,
        )

        if self.verifier_kind == "deterministic":
            if any(value is not None for value in judge_values):
                raise ValueError(
                    "deterministic verifiers must not declare judge model "
                    "or judge prompt fingerprints"
                )
            return self

        if any(value is None for value in judge_values):
            raise ValueError(
                f"{self.verifier_kind} verifiers require judge model and judge prompt fingerprints"
            )

        return self


class AgentTaskIdentity(ContractModel):
    """Immutable identity for one executable agent task instance."""

    contract_type: Literal["agent_task_identity"] = "agent_task_identity"
    task_id: NonEmptyString
    environment_fingerprint: Sha256Digest
    verifier_fingerprint: Sha256Digest
    task_definition_sha256: Sha256Digest
    initial_state_sha256: Sha256Digest
    allowed_actions_sha256: Sha256Digest
    success_criteria_sha256: Sha256Digest


class AgentRolloutPolicyIdentity(ContractModel):
    """Immutable identity for generation and agent-loop rollout semantics."""

    contract_type: Literal["agent_rollout_policy_identity"] = "agent_rollout_policy_identity"
    policy_id: NonEmptyString
    generation_config_sha256: Sha256Digest
    agent_loop_config_sha256: Sha256Digest
    sampling_policy_sha256: Sha256Digest
    seed_policy_sha256: Sha256Digest
