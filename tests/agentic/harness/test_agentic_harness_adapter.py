from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

import huyawo_adapt.agentic as agentic_root
from huyawo_adapt.agentic.contracts import AgentHarnessIdentity, AgentToolSetIdentity
from huyawo_adapt.agentic.harness import (
    AgentHarnessAdapter,
    CanonicalHarnessCaptureAdapter,
    HarnessCaptureContext,
)
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


def payload(content: str = "  olá\n\t") -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(
        media_type="text/plain",
        content=content,
        sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def context(tools: tuple[str, ...] = (SHA_D,)) -> HarnessCaptureContext:
    return HarnessCaptureContext(
        trajectory_id="capture-001",
        model_fingerprint=SHA_A,
        tokenizer_fingerprint=SHA_B,
        harness_identity=AgentHarnessIdentity(
            harness_id="harness-1",
            source_revision="source-commit",
            implementation_sha256=SHA_A,
            configuration_sha256=SHA_B,
            chat_template_sha256=SHA_C,
            observation_format_sha256=SHA_D,
            control_flow_sha256=SHA_A,
            stopping_behavior_sha256=SHA_B,
        ),
        tool_set_identity=AgentToolSetIdentity(tool_fingerprints=tools),
        environment_fingerprint=SHA_C,
        verifier_fingerprint=SHA_D,
        task_fingerprint=SHA_A,
        runtime_fingerprint=SHA_B,
        rollout_policy_fingerprint=SHA_C,
        capture_provenance_sha256=SHA_D,
    )


def step(
    index: int = 0,
    status: str = "success",
    *,
    with_tool: bool = False,
    tool_fingerprint: str = SHA_D,
    with_result: bool = True,
) -> AgentTrajectoryStep:
    values: dict[str, object] = {
        "step_index": index,
        "observation": payload(" observed\n"),
        "model_input": payload("  prompt olá\n\t"),
        "model_input_token_ids": (0, 21, 65000),
        "generated_output": payload("  response\n"),
        "generated_token_ids": (3, 4, 5),
        "terminal_status": status,
        "provenance_sha256": SHA_A,
    }
    if with_tool:
        values["tool_call"] = AgentToolCallRecord(
            call_id="call-1", tool_fingerprint=tool_fingerprint, arguments=payload('{"q":"á"}')
        )
        if with_result:
            values["tool_result"] = AgentToolResultRecord(
                call_id="call-1",
                tool_fingerprint=tool_fingerprint,
                result=payload('{"ok":true}'),
                is_error=False,
            )
    if status in ("failure", "error"):
        values["failure"] = AgentFailureRecord(
            failure_type="tool_failure", message=" original error \n", retryable=False
        )
    return AgentTrajectoryStep.model_validate(values)


def assembler() -> CanonicalHarnessCaptureAdapter:
    return CanonicalHarnessCaptureAdapter()


def test_capture_returns_canonical_trajectory_and_exact_identities() -> None:
    source = context()
    actual = assembler().capture(source, (step(),))
    assert isinstance(actual, AgentTrajectory)
    assert actual.trajectory_id == "capture-001"
    assert actual.harness_fingerprint == source.harness_identity.fingerprint()
    assert actual.tool_set_fingerprint == source.tool_set_identity.fingerprint()
    for name in (
        "model_fingerprint",
        "tokenizer_fingerprint",
        "environment_fingerprint",
        "verifier_fingerprint",
        "task_fingerprint",
        "runtime_fingerprint",
        "rollout_policy_fingerprint",
        "capture_provenance_sha256",
    ):
        assert getattr(actual, name) == getattr(source, name)
    assert actual.parent_trajectory_fingerprint is None
    assert actual.transformation_fingerprint is None


def test_context_is_strict_frozen_and_forbids_unknown_fields() -> None:
    source = context()
    with pytest.raises(ValidationError):
        HarnessCaptureContext.model_validate({**source.model_dump(), "unknown": "x"})
    with pytest.raises(ValidationError):
        source.trajectory_id = "changed"
    with pytest.raises(ValidationError):
        HarnessCaptureContext.model_validate({**source.model_dump(), "model_fingerprint": "x"})
    with pytest.raises(ValidationError):
        HarnessCaptureContext.model_validate({**source.model_dump(), "trajectory_id": " "})


def test_context_does_not_accept_competing_derived_fingerprints() -> None:
    values = context().model_dump()
    values["harness_fingerprint"] = SHA_A
    with pytest.raises(ValidationError):
        HarnessCaptureContext.model_validate(values)
    values.pop("harness_fingerprint")
    values["tool_set_fingerprint"] = SHA_B
    with pytest.raises(ValidationError):
        HarnessCaptureContext.model_validate(values)


def test_capture_rejects_invalid_context_before_iteration() -> None:
    consumed = False

    def events() -> Iterator[AgentTrajectoryStep]:
        nonlocal consumed
        consumed = True
        yield step()

    with pytest.raises(TypeError, match="HarnessCaptureContext"):
        assembler().capture({"invalid": True}, events())  # type: ignore[arg-type]
    assert consumed is False


def test_capture_rejects_forged_context_on_revalidation() -> None:
    fake = context().model_copy(update={"model_fingerprint": "INVALID"})
    with pytest.raises(ValidationError):
        assembler().capture(fake, (step(),))


def test_capture_preserves_order_and_single_pass_generator() -> None:
    consumed: list[int] = []

    def records() -> Iterator[AgentTrajectoryStep]:
        for candidate in (step(0, "continue"), step(1, "success")):
            consumed.append(candidate.step_index)
            yield candidate

    result = assembler().capture(context(), records())
    assert consumed == [0, 1]
    assert [x.step_index for x in result.steps] == [0, 1]
    assert [x.terminal_status for x in result.steps] == ["continue", "success"]


def test_capture_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assembler().capture(context(), ())


@pytest.mark.parametrize("indexes", [(1,), (0, 2), (0, 0), (1, 0)])
def test_capture_rejects_reordering_or_gaps(indexes: tuple[int, ...]) -> None:
    records = tuple(
        step(i, "continue" if n < len(indexes) - 1 else "success") for n, i in enumerate(indexes)
    )
    with pytest.raises(ValueError, match="contiguous"):
        assembler().capture(context(), records)


def test_capture_rejects_nonterminal_final_step() -> None:
    with pytest.raises(ValueError, match="terminal status"):
        assembler().capture(context(), (step(0, "continue"),))


@pytest.mark.parametrize("status", ["success", "failure", "truncated", "error"])
def test_capture_accepts_all_explicit_terminal_statuses(status: str) -> None:
    actual = assembler().capture(context(), (step(0, status),))
    assert actual.steps[0].terminal_status == status


def test_capture_rejects_steps_after_terminal_without_reordering() -> None:
    with pytest.raises(ValueError, match="after a terminal"):
        assembler().capture(context(), (step(0, "success"), step(1, "success")))


def test_capture_rejects_plain_dict_or_noncanonical_event() -> None:
    with pytest.raises(TypeError, match="canonical AgentTrajectoryStep"):
        assembler().capture(context(), (step().model_dump(),))  # type: ignore[list-item]


def test_capture_revalidates_forged_step_instead_of_silently_accepting() -> None:
    invalid = step().model_copy(update={"step_index": True})
    with pytest.raises(ValidationError):
        assembler().capture(context(), (invalid,))


def test_capture_validates_declared_tool_call_and_result_membership() -> None:
    actual = assembler().capture(context(), (step(with_tool=True),))
    assert actual.steps[0].tool_call is not None
    assert actual.steps[0].tool_call.tool_fingerprint == SHA_D
    assert actual.steps[0].tool_result is not None
    assert actual.steps[0].tool_result.tool_fingerprint == SHA_D


def test_capture_accepts_tool_call_without_result() -> None:
    result = assembler().capture(context(), (step(with_tool=True, with_result=False),))
    assert result.steps[0].tool_call is not None
    assert result.steps[0].tool_result is None


def test_capture_rejects_tool_fingerprint_outside_declared_set() -> None:
    with pytest.raises(ValueError, match="tool_call.*declared tool set"):
        assembler().capture(context(), (step(with_tool=True, tool_fingerprint=SHA_A),))


def test_capture_rejects_tool_result_outside_set_even_if_call_was_forged() -> None:
    original = step(with_tool=True)
    forged_call = original.tool_call.model_copy(update={"tool_fingerprint": SHA_A})
    forged_result = original.tool_result.model_copy(update={"tool_fingerprint": SHA_A})
    invalid = original.model_copy(update={"tool_call": forged_call, "tool_result": forged_result})
    with pytest.raises(ValueError, match="tool_call.*declared tool set"):
        assembler().capture(context(), (invalid,))


def test_capture_keeps_exact_utf8_whitespace_tokens_and_payload_hash() -> None:
    original = step(with_tool=True)
    result = assembler().capture(context(), [original])
    captured = result.steps[0]
    assert captured == original
    assert captured.model_input.content == "  prompt olá\n\t"
    assert captured.model_input.sha256 == original.model_input.sha256
    assert captured.model_input_token_ids == (0, 21, 65000)
    assert captured.generated_token_ids == (3, 4, 5)
    assert captured.tool_call == original.tool_call
    assert captured.tool_result == original.tool_result


def test_capture_preserves_absent_token_ids_without_retokenizing() -> None:
    original = step().model_copy(
        update={"model_input_token_ids": None, "generated_token_ids": None}
    )
    captured = assembler().capture(context(), (original,)).steps[0]
    assert captured.model_input_token_ids is None
    assert captured.generated_token_ids is None


def test_capture_preserves_state_transition_and_reward_evidence() -> None:
    original = step().model_copy(
        update={
            "state_transition": AgentStateTransitionRecord(
                before_state_sha256=SHA_A,
                after_state_sha256=SHA_B,
                transition_evidence_sha256=SHA_C,
            ),
            "reward_components": (
                AgentRewardComponent(component_id="reward", value=0.25, source_fingerprint=SHA_D),
            ),
        }
    )
    captured = assembler().capture(context(), (original,)).steps[0]
    assert captured.state_transition == original.state_transition
    assert captured.reward_components == original.reward_components


def test_capture_preserves_terminal_failure_details() -> None:
    original = step(status="error")
    captured = assembler().capture(context(), (original,)).steps[0]
    assert captured.failure == original.failure
    assert captured.failure is not None
    assert captured.failure.message == " original error \n"


def test_capture_is_canonical_deterministic_across_independent_runs() -> None:
    source = context()
    inputs = (step(0, "continue", with_tool=True), step(1, "success"))
    first = assembler().capture(source, iter(inputs))
    second = assembler().capture(source, iter(inputs))
    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() == hashlib.sha256(first.canonical_bytes()).hexdigest()
    assert AgentTrajectory.from_canonical_json(first.canonical_bytes()) == first


def test_capture_provenance_and_all_source_evidence_affect_fingerprint() -> None:
    first = assembler().capture(context(), (step(),))
    second = assembler().capture(
        context().model_copy(update={"capture_provenance_sha256": SHA_A}), (step(),)
    )
    changed = step().model_copy(update={"generated_output": payload("modified")})
    third = assembler().capture(context(), (changed,))
    assert first.fingerprint() != second.fingerprint()
    assert first.fingerprint() != third.fingerprint()


def test_reference_adapter_satisfies_public_protocol() -> None:
    adapter: AgentHarnessAdapter = CanonicalHarnessCaptureAdapter()
    assert isinstance(adapter.capture(context(), (step(),)), AgentTrajectory)
    assert not hasattr(agentic_root, "HarnessCaptureContext")
