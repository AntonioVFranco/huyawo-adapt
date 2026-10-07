from __future__ import annotations

import pytest
from pydantic import ValidationError

import huyawo_adapt
from huyawo_adapt.agentic import (
    AgentEnvironmentIdentity,
    AgentHarnessIdentity,
    AgentRolloutPolicyIdentity,
    AgentTaskIdentity,
    AgentToolIdentity,
    AgentToolSetIdentity,
    AgentVerifierIdentity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def make_harness_identity() -> AgentHarnessIdentity:
    return AgentHarnessIdentity(
        harness_id="reference-harness",
        source_revision="0123456789abcdef",
        implementation_sha256=SHA_A,
        configuration_sha256=SHA_B,
        chat_template_sha256=SHA_C,
        observation_format_sha256=SHA_D,
        control_flow_sha256=SHA_E,
        stopping_behavior_sha256=SHA_F,
    )


def make_environment_identity() -> AgentEnvironmentIdentity:
    return AgentEnvironmentIdentity(
        environment_id="database-environment",
        definition_sha256=SHA_A,
        state_schema_sha256=SHA_B,
        reset_semantics_sha256=SHA_C,
        transition_semantics_sha256=SHA_D,
        terminal_conditions_sha256=SHA_E,
    )


def make_deterministic_verifier() -> AgentVerifierIdentity:
    return AgentVerifierIdentity(
        verifier_id="database-verifier",
        verifier_kind="deterministic",
        implementation_sha256=SHA_A,
        configuration_sha256=SHA_B,
        output_schema_sha256=SHA_C,
    )


def test_harness_identity_round_trip_and_fingerprint() -> None:
    identity = make_harness_identity()
    restored = AgentHarnessIdentity.from_canonical_json(identity.canonical_bytes())

    assert restored == identity
    assert restored.fingerprint() == identity.fingerprint()
    assert identity.contract_type == "agent_harness_identity"


def test_agentic_contract_is_frozen() -> None:
    identity = make_harness_identity()

    with pytest.raises(ValidationError):
        identity.harness_id = "mutated-harness"


def test_agentic_contract_rejects_unknown_fields() -> None:
    values = make_harness_identity().model_dump()
    values["unexpected"] = True

    with pytest.raises(ValidationError):
        AgentHarnessIdentity.model_validate(values)


def test_agentic_contract_rejects_malformed_sha256() -> None:
    values = make_harness_identity().model_dump()
    values["implementation_sha256"] = "not-a-sha256"

    with pytest.raises(ValidationError):
        AgentHarnessIdentity.model_validate(values)


def test_tool_identity_supports_explicit_optional_semantics() -> None:
    identity = AgentToolIdentity(
        tool_id="sql_query",
        tool_schema_sha256=SHA_A,
        implementation_sha256=SHA_B,
        result_schema_sha256=SHA_C,
        side_effect_policy_sha256=SHA_D,
    )

    assert identity.contract_type == "agent_tool_identity"
    assert identity.result_schema_sha256 == SHA_C
    assert identity.side_effect_policy_sha256 == SHA_D


def test_tool_set_normalizes_fingerprints() -> None:
    identity = AgentToolSetIdentity(tool_fingerprints=(SHA_C, SHA_A, SHA_B))

    assert identity.tool_fingerprints == (SHA_A, SHA_B, SHA_C)


def test_tool_set_fingerprint_is_independent_of_input_order() -> None:
    first = AgentToolSetIdentity(tool_fingerprints=(SHA_A, SHA_B, SHA_C))
    second = AgentToolSetIdentity(tool_fingerprints=(SHA_C, SHA_A, SHA_B))

    assert first.fingerprint() == second.fingerprint()


def test_tool_set_requires_at_least_one_tool() -> None:
    with pytest.raises(ValidationError):
        AgentToolSetIdentity(tool_fingerprints=())


def test_tool_set_rejects_duplicate_fingerprints() -> None:
    with pytest.raises(ValidationError):
        AgentToolSetIdentity(tool_fingerprints=(SHA_A, SHA_A))


def test_deterministic_verifier_rejects_judge_identity() -> None:
    with pytest.raises(ValidationError):
        AgentVerifierIdentity(
            verifier_id="deterministic-verifier",
            verifier_kind="deterministic",
            implementation_sha256=SHA_A,
            configuration_sha256=SHA_B,
            output_schema_sha256=SHA_C,
            judge_model_fingerprint=SHA_D,
            judge_prompt_sha256=SHA_E,
        )


@pytest.mark.parametrize(
    "verifier_kind",
    ["llm_judge", "hybrid"],
)
def test_judge_backed_verifier_requires_complete_judge_identity(
    verifier_kind: str,
) -> None:
    with pytest.raises(ValidationError):
        AgentVerifierIdentity.model_validate(
            {
                "verifier_id": "judge-verifier",
                "verifier_kind": verifier_kind,
                "implementation_sha256": SHA_A,
                "configuration_sha256": SHA_B,
                "output_schema_sha256": SHA_C,
                "judge_model_fingerprint": SHA_D,
            }
        )


@pytest.mark.parametrize(
    "verifier_kind",
    ["llm_judge", "hybrid"],
)
def test_judge_backed_verifier_accepts_complete_judge_identity(
    verifier_kind: str,
) -> None:
    identity = AgentVerifierIdentity.model_validate(
        {
            "verifier_id": "judge-verifier",
            "verifier_kind": verifier_kind,
            "implementation_sha256": SHA_A,
            "configuration_sha256": SHA_B,
            "output_schema_sha256": SHA_C,
            "judge_model_fingerprint": SHA_D,
            "judge_prompt_sha256": SHA_E,
        }
    )

    assert identity.judge_model_fingerprint == SHA_D
    assert identity.judge_prompt_sha256 == SHA_E


def test_task_binds_environment_and_verifier_fingerprints() -> None:
    environment = make_environment_identity()
    verifier = make_deterministic_verifier()

    task = AgentTaskIdentity(
        task_id="task-0001",
        environment_fingerprint=environment.fingerprint(),
        verifier_fingerprint=verifier.fingerprint(),
        task_definition_sha256=SHA_C,
        initial_state_sha256=SHA_D,
        allowed_actions_sha256=SHA_E,
        success_criteria_sha256=SHA_F,
    )

    assert task.environment_fingerprint == environment.fingerprint()
    assert task.verifier_fingerprint == verifier.fingerprint()
    assert task.contract_type == "agent_task_identity"


def test_rollout_policy_identity_is_backend_independent_data() -> None:
    identity = AgentRolloutPolicyIdentity(
        policy_id="reference-rollout",
        generation_config_sha256=SHA_A,
        agent_loop_config_sha256=SHA_B,
        sampling_policy_sha256=SHA_C,
        seed_policy_sha256=SHA_D,
    )

    assert identity.contract_type == "agent_rollout_policy_identity"
    assert len(identity.fingerprint()) == 64


def test_agentic_contracts_are_not_exported_from_root_package() -> None:
    assert not hasattr(huyawo_adapt, "AgentHarnessIdentity")
    assert not hasattr(huyawo_adapt, "AgentToolIdentity")
    assert not hasattr(huyawo_adapt, "AgentEnvironmentIdentity")


def test_identity_schema_contains_contract_type() -> None:
    schema = AgentHarnessIdentity.model_json_schema()
    contract_type = schema["properties"]["contract_type"]

    assert contract_type["const"] == "agent_harness_identity"
