from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

import huyawo_adapt.agentic as agentic_root
from huyawo_adapt.agentic.trajectory import (
    AgentFailureRecord,
    AgentRewardComponent,
    AgentStateTransitionRecord,
    AgentToolCallRecord,
    AgentToolResultRecord,
    AgentTrajectory,
    AgentTrajectoryPayload,
    AgentTrajectoryStep,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def payload(content: str = "  hello\n") -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(
        media_type="text/plain",
        content=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def tool_call() -> AgentToolCallRecord:
    return AgentToolCallRecord(
        call_id="call-1",
        tool_fingerprint=SHA_A,
        arguments=payload('{"query":"x"}'),
    )


def tool_result() -> AgentToolResultRecord:
    return AgentToolResultRecord(
        call_id="call-1",
        tool_fingerprint=SHA_A,
        result=payload('{"rows":[]}'),
        is_error=False,
    )


def step(index: int = 0, status: str = "success") -> AgentTrajectoryStep:
    values = {
        "step_index": index,
        "observation": payload("observation"),
        "model_input": payload("  prompt\n"),
        "model_input_token_ids": (0, 27, 65535),
        "generated_output": payload("  response\n"),
        "generated_token_ids": (1, 2, 3),
        "terminal_status": status,
        "provenance_sha256": SHA_B,
    }
    if status in ("failure", "error"):
        values["failure"] = AgentFailureRecord(
            failure_type="tool_failure",
            message="  original failure\n",
            retryable=False,
        )
    return AgentTrajectoryStep.model_validate(values)


def trajectory(steps: tuple[AgentTrajectoryStep, ...] | None = None) -> AgentTrajectory:
    return AgentTrajectory(
        trajectory_id="run-001",
        model_fingerprint=SHA_A,
        tokenizer_fingerprint=SHA_B,
        harness_fingerprint=SHA_C,
        tool_set_fingerprint=SHA_D,
        environment_fingerprint=SHA_A,
        verifier_fingerprint=SHA_B,
        task_fingerprint=SHA_C,
        runtime_fingerprint=SHA_D,
        rollout_policy_fingerprint=SHA_A,
        steps=(step(),) if steps is None else steps,
        capture_provenance_sha256=SHA_B,
    )


def invalid_step(**overrides: object) -> None:
    values = step().model_dump()
    values.update(overrides)
    with pytest.raises(ValidationError):
        AgentTrajectoryStep.model_validate(values)


def invalid_trajectory(**overrides: object) -> None:
    values = trajectory().model_dump()
    values.update(overrides)
    with pytest.raises(ValidationError):
        AgentTrajectory.model_validate(values)


def test_payload_preserves_exact_utf8_content_and_empty_content() -> None:
    exact = "  olá\n\t"
    actual = payload(exact)
    assert actual.content == exact
    assert actual.sha256 == hashlib.sha256(exact.encode("utf-8")).hexdigest()
    assert payload("").content == ""
    assert payload("").sha256 == hashlib.sha256(b"").hexdigest()


def test_payload_rejects_wrong_digest() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(media_type="text/plain", content="abc", sha256=SHA_A)


@pytest.mark.parametrize("bad_sha", ["abc", "A" * 64, "x" * 64])
def test_payload_rejects_malformed_sha(bad_sha: str) -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(media_type="text/plain", content="abc", sha256=bad_sha)


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload.model_validate({**payload().model_dump(), "extra": 1})
    invalid_step(unknown_field="not allowed")
    invalid_trajectory(unknown_field="not allowed")


def test_all_records_frozen() -> None:
    for record in (payload(), tool_call(), tool_result(), step(), trajectory()):
        with pytest.raises(ValidationError):
            record.__setattr__("unexpected", 123)


def test_media_type_and_identifiers_nonempty() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(media_type="  ", content="", sha256=payload("").sha256)
    with pytest.raises(ValidationError):
        AgentToolCallRecord(call_id=" ", tool_fingerprint=SHA_A, arguments=payload())
    invalid_trajectory(trajectory_id=" ")


def test_tool_result_requires_matching_call() -> None:
    invalid_step(tool_result=tool_result())
    invalid_step(
        tool_call=tool_call(),
        tool_result={**tool_result().model_dump(), "call_id": "other"},
    )
    invalid_step(
        tool_call=tool_call(),
        tool_result={**tool_result().model_dump(), "tool_fingerprint": SHA_C},
    )


def test_tool_records_preserve_exact_content_and_error_flag() -> None:
    values = step().model_dump()
    values.update({"tool_call": tool_call(), "tool_result": tool_result()})
    actual = AgentTrajectoryStep.model_validate(values)
    assert actual.tool_call == tool_call()
    assert actual.tool_result == tool_result()
    assert actual.tool_result is not None and actual.tool_result.is_error is False


def test_environment_transition_and_reward_components() -> None:
    transition = AgentStateTransitionRecord(
        before_state_sha256=SHA_A,
        after_state_sha256=SHA_B,
        transition_evidence_sha256=SHA_C,
    )
    reward = AgentRewardComponent(component_id="task", value=0.5, source_fingerprint=SHA_D)
    values = step().model_dump()
    values.update({"state_transition": transition, "reward_components": (reward,)})
    actual = AgentTrajectoryStep.model_validate(values)
    assert actual.state_transition == transition
    assert actual.reward_components == (reward,)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reward_value_must_be_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        AgentRewardComponent(component_id="task", value=value, source_fingerprint=SHA_A)


def test_reward_rejects_invalid_evidence_hash() -> None:
    with pytest.raises(ValidationError):
        AgentRewardComponent(component_id="task", value=1.0, source_fingerprint="invalid")


@pytest.mark.parametrize("name", ["model_input_token_ids", "generated_token_ids"])
def test_token_ids_nonempty_nonnegative_strict(name: str) -> None:
    invalid_step(**{name: ()})
    invalid_step(**{name: (-1,)})
    invalid_step(**{name: (True,)})
    invalid_step(**{name: (1.2,)})


def test_optional_token_ids_and_exact_ids_roundtrip() -> None:
    values = step().model_dump()
    values.update({"model_input_token_ids": None, "generated_token_ids": None})
    actual = AgentTrajectoryStep.model_validate(values)
    assert actual.model_input_token_ids is None
    assert actual.generated_token_ids is None
    restored = AgentTrajectory.from_canonical_json(trajectory().canonical_bytes())
    assert restored.steps[0].model_input_token_ids == (0, 27, 65535)
    assert restored.steps[0].generated_token_ids == (1, 2, 3)


def test_failure_status_invariants() -> None:
    failure = AgentFailureRecord(failure_type="timeout", message="  timeout \n", retryable=True)
    assert failure.message == "  timeout \n"
    invalid_step(terminal_status="failure")
    invalid_step(terminal_status="error")
    invalid_step(terminal_status="continue", failure=failure)
    invalid_step(terminal_status="success", failure=failure)
    assert step(status="failure").failure is not None
    assert step(status="error").failure is not None
    truncated = AgentTrajectoryStep.model_validate(
        {**step().model_dump(), "terminal_status": "truncated"}
    )
    assert truncated.failure is None
    assert (
        AgentTrajectoryStep.model_validate(
            {**step().model_dump(), "terminal_status": "truncated", "failure": failure}
        ).failure
        == failure
    )


def test_trajectory_step_sequence_rules() -> None:
    invalid_trajectory(steps=())
    invalid_trajectory(steps=(step(1, "success"),))
    invalid_trajectory(steps=(step(0, "success"), step(1, "success")))
    invalid_trajectory(steps=(step(0, "continue"),))
    invalid_trajectory(steps=(step(0, "continue"), step(2, "success")))
    actual = trajectory((step(0, "continue"), step(1, "success")))
    assert tuple(value.step_index for value in actual.steps) == (0, 1)


def test_lineage_fingerprints_are_an_indivisible_pair() -> None:
    invalid_trajectory(parent_trajectory_fingerprint=SHA_A)
    invalid_trajectory(transformation_fingerprint=SHA_B)
    values = trajectory().model_dump()
    values.update({"parent_trajectory_fingerprint": SHA_A, "transformation_fingerprint": SHA_B})
    actual = AgentTrajectory.model_validate(values)
    assert actual.parent_trajectory_fingerprint == SHA_A
    assert actual.transformation_fingerprint == SHA_B


def test_canonical_json_roundtrip_and_digest_stability() -> None:
    record = trajectory((step(0, "continue"), step(1, "success")))
    restored = AgentTrajectory.from_canonical_json(record.canonical_bytes())
    assert restored == record
    assert restored.canonical_bytes() == record.canonical_bytes()
    assert restored.fingerprint() == record.fingerprint()
    assert record.fingerprint() == hashlib.sha256(record.canonical_bytes()).hexdigest()
    assert record.canonical_json() == json.dumps(
        record.canonical_data(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_backend_independent_serialization_and_whitespace() -> None:
    record = trajectory()
    encoded = record.canonical_json()
    restored = AgentTrajectory.from_canonical_json(encoded)
    assert restored.steps[0].model_input.content == "  prompt\n"
    assert restored.steps[0].generated_output.content == "  response\n"
    assert "harness_fingerprint" in encoded
    assert "runtime_fingerprint" in encoded
    assert "tool_set_fingerprint" in encoded
    assert "rollout_policy_fingerprint" in encoded
    assert not hasattr(agentic_root, "AgentTrajectory")


def test_fingerprint_changes_with_evidence_and_transformation() -> None:
    original = trajectory()
    changed_step = AgentTrajectoryStep.model_validate(
        {**step().model_dump(), "generated_output": payload("different")}
    )
    changed = trajectory((changed_step,))
    assert changed.fingerprint() != original.fingerprint()
    derived = AgentTrajectory.model_validate(
        {
            **original.model_dump(),
            "parent_trajectory_fingerprint": original.fingerprint(),
            "transformation_fingerprint": SHA_C,
        }
    )
    assert derived.fingerprint() != original.fingerprint()


def test_contract_schema_and_strict_validation() -> None:
    schema = AgentTrajectory.model_json_schema()
    assert schema["properties"]["contract_type"]["const"] == "agent_trajectory"
    invalid_trajectory(steps=[step()])
    invalid_step(step_index=True)
    with pytest.raises(ValidationError):
        AgentToolResultRecord(
            call_id="call-1",
            tool_fingerprint=SHA_A,
            result=payload(),
            is_error=1,
        )
