"""Contract-only adversarial tests for the M6 environment exchange boundary."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.contracts import AgentEnvironmentIdentity, AgentTaskIdentity
from huyawo_adapt.agentic.environment import (
    AgentEnvironmentAction,
    AgentEnvironmentObservation,
    AgentEnvironmentResetRequest,
    AgentEnvironmentResetResult,
    AgentEnvironmentStepRequest,
    AgentEnvironmentStepResult,
    ExecutableAgentEnvironment,
    validate_reset_exchange,
    validate_step_exchange,
)
from huyawo_adapt.agentic.trajectory import AgentStateTransitionRecord, AgentTrajectoryPayload


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def payload(value: str) -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(media_type="text/plain", content=value, sha256=digest(value))


def environment() -> AgentEnvironmentIdentity:
    return AgentEnvironmentIdentity(
        environment_id="local-fixture",
        definition_sha256=digest("definition"),
        state_schema_sha256=digest("schema"),
        reset_semantics_sha256=digest("reset-rules"),
        transition_semantics_sha256=digest("step-rules"),
        terminal_conditions_sha256=digest("terminal-rules"),
    )


def task(env: AgentEnvironmentIdentity) -> AgentTaskIdentity:
    return AgentTaskIdentity(
        task_id="fixture-task",
        environment_fingerprint=env.fingerprint(),
        verifier_fingerprint=digest("verifier"),
        task_definition_sha256=digest("task"),
        initial_state_sha256=digest("initial-state"),
        allowed_actions_sha256=digest("allowlist"),
        success_criteria_sha256=digest("criteria"),
    )


def observation(
    index: int, state: str, *, episode: str = "episode-1", content: str = "visible"
) -> AgentEnvironmentObservation:
    return AgentEnvironmentObservation(
        episode_id=episode,
        observation_index=index,
        public_observation=payload(content),
        state_sha256=digest(state),
        state_provenance_sha256=digest("state-provenance"),
        visibility_policy_sha256=digest("visibility"),
    )


def reset_pair() -> tuple[AgentEnvironmentResetRequest, AgentEnvironmentResetResult]:
    env = environment()
    identity = task(env)
    request = AgentEnvironmentResetRequest(
        environment_identity=env, task_identity=identity, episode_id="episode-1", seed=23
    )
    result = AgentEnvironmentResetResult(
        environment_fingerprint=env.fingerprint(),
        task_fingerprint=identity.fingerprint(),
        episode_id="episode-1",
        seed=23,
        observation=observation(0, "initial-state"),
        reset_evidence_sha256=digest("reset-evidence"),
    )
    return request, result


def step_pair(
    *,
    previous: AgentEnvironmentResetResult | AgentEnvironmentStepResult | None = None,
    terminal_status: str = "continue",
) -> tuple[AgentEnvironmentStepRequest, AgentEnvironmentStepResult]:
    reset_request, reset_result = reset_pair()
    last = previous or reset_result
    step_index = last.observation.observation_index
    action = AgentEnvironmentAction(
        action_id="inspect",
        arguments=payload('{"target":"file"}'),
        action_provenance_sha256=digest("action-provenance"),
        tool_fingerprint=digest("tool"),
    )
    request = AgentEnvironmentStepRequest(
        environment_identity=reset_request.environment_identity,
        task_identity=reset_request.task_identity,
        episode_id="episode-1",
        step_index=step_index,
        before_state_sha256=last.observation.state_sha256,
        action=action,
    )
    terminal_evidence = digest("terminal-evidence") if terminal_status != "continue" else None
    result = AgentEnvironmentStepResult(
        environment_fingerprint=request.environment_identity.fingerprint(),
        task_fingerprint=request.task_identity.fingerprint(),
        episode_id="episode-1",
        step_index=step_index,
        action_fingerprint=action.fingerprint(),
        observation=observation(step_index + 1, f"state-{step_index + 1}"),
        transition=AgentStateTransitionRecord(
            before_state_sha256=request.before_state_sha256,
            after_state_sha256=digest(f"state-{step_index + 1}"),
            transition_evidence_sha256=digest("transition-evidence"),
        ),
        terminal_status=terminal_status,
        terminal_evidence_sha256=terminal_evidence,
    )
    return request, result


def changed(model: Any, **changes: Any) -> Any:
    return model.__class__.model_validate({**model.model_dump(mode="python"), **changes})


def test_valid_reset_and_sequential_steps() -> None:
    request, reset = reset_pair()
    validate_reset_exchange(request, reset)
    step0, result0 = step_pair()
    validate_step_exchange(reset, step0, result0)
    step1, result1 = step_pair(previous=result0, terminal_status="terminated")
    validate_step_exchange(result0, step1, result1)
    assert result1.terminal_status == "terminated"
    assert not hasattr(result1, "reward")
    assert not hasattr(result1, "success")


@pytest.mark.parametrize("text", ["", " café\n", "\talpha", "naïve", "line\r\n", " "])
def test_exact_public_utf8_payload_roundtrip(text: str) -> None:
    observed = observation(0, "initial-state", content=text)
    assert observed.public_observation.content == text
    assert observed.public_observation.sha256 == digest(text)
    assert observed.state_sha256 == digest("initial-state")
    assert AgentEnvironmentObservation.from_canonical_json(observed.canonical_bytes()) == observed


@pytest.mark.parametrize("terminal", ["terminated", "truncated", "error"])
def test_terminal_outcomes_have_evidence_but_not_verifier_authority(terminal: str) -> None:
    _, reset = reset_pair()
    request, result = step_pair(terminal_status=terminal)
    validate_step_exchange(reset, request, result)
    assert result.terminal_evidence_sha256 is not None
    assert not hasattr(result, "verdict")


@pytest.mark.parametrize(
    "status,evidence",
    [
        ("continue", digest("wrong")),
        ("terminated", None),
        ("truncated", None),
        ("error", None),
    ],
)
def test_terminal_evidence_shape_is_enforced(status: str, evidence: str | None) -> None:
    _, result = step_pair()
    with pytest.raises(ValidationError):
        changed(result, terminal_status=status, terminal_evidence_sha256=evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_fingerprint", digest("other-environment")),
        ("task_fingerprint", digest("other-task")),
        ("episode_id", "other-episode"),
        ("seed", 1),
        ("observation", observation(1, "initial-state")),
        ("observation", observation(0, "foreign-state")),
        ("observation", observation(0, "initial-state", episode="other-episode")),
    ],
)
def test_reset_cross_record_mismatch_rejected(field: str, value: Any) -> None:
    request, result = reset_pair()
    with pytest.raises(ValueError):
        validate_reset_exchange(request, changed(result, **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_fingerprint", digest("other-environment")),
        ("task_fingerprint", digest("other-task")),
        ("episode_id", "other-episode"),
        ("step_index", 1),
        ("action_fingerprint", digest("other-action")),
        ("observation", observation(2, "state-1")),
        ("observation", observation(1, "state-1", episode="other-episode")),
        (
            "transition",
            AgentStateTransitionRecord(
                before_state_sha256=digest("other-before"),
                after_state_sha256=digest("state-1"),
                transition_evidence_sha256=digest("proof"),
            ),
        ),
        (
            "transition",
            AgentStateTransitionRecord(
                before_state_sha256=digest("initial-state"),
                after_state_sha256=digest("other-after"),
                transition_evidence_sha256=digest("proof"),
            ),
        ),
    ],
)
def test_step_result_conflicts_are_rejected(field: str, value: Any) -> None:
    _, prior = reset_pair()
    request, result = step_pair()
    with pytest.raises(ValueError):
        validate_step_exchange(prior, request, changed(result, **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("episode_id", "other-episode"),
        ("step_index", 1),
        ("before_state_sha256", digest("other-state")),
    ],
)
def test_step_request_conflicts_are_rejected(field: str, value: Any) -> None:
    _, prior = reset_pair()
    request, result = step_pair()
    with pytest.raises(ValueError):
        validate_step_exchange(prior, changed(request, **{field: value}), result)


def test_previous_reset_identity_and_observation_conflicts() -> None:
    _, previous = reset_pair()
    request, result = step_pair()
    for kwargs in (
        {"task_fingerprint": digest("forged-task")},
        {"environment_fingerprint": digest("forged-environment")},
        {"observation": observation(0, "initial-state", episode="other-episode")},
    ):
        with pytest.raises(ValueError):
            validate_step_exchange(changed(previous, **kwargs), request, result)


@pytest.mark.parametrize("terminal", ["terminated", "truncated", "error"])
def test_steps_after_terminal_are_rejected(terminal: str) -> None:
    _, reset = reset_pair()
    request, terminal_result = step_pair(terminal_status=terminal)
    validate_step_exchange(reset, request, terminal_result)
    next_request, next_result = step_pair(previous=terminal_result)
    with pytest.raises(ValueError, match="terminal"):
        validate_step_exchange(terminal_result, next_request, next_result)


def test_previous_reset_initial_commitment_is_checked_by_step_validator() -> None:
    _, previous = reset_pair()
    request, result = step_pair()
    foreign = changed(previous, observation=observation(0, "other-state"))
    with pytest.raises(ValueError, match="initial-state"):
        validate_step_exchange(foreign, request, result)


def test_preceding_step_result_cannot_forge_its_index_or_after_state() -> None:
    _, reset = reset_pair()
    request, first = step_pair()
    validate_step_exchange(reset, request, first)
    next_request, next_result = step_pair(previous=first)
    inconsistent_index = changed(first, observation=observation(4, "state-1"))
    with pytest.raises(ValueError, match="previous step observation index"):
        validate_step_exchange(inconsistent_index, next_request, next_result)
    inconsistent_state = changed(first, observation=observation(1, "forged-state"))
    with pytest.raises(ValueError, match="previous step state transition"):
        validate_step_exchange(inconsistent_state, next_request, next_result)


def test_skipped_steps_and_incorrect_previous_observation_index_rejected() -> None:
    _, reset = reset_pair()
    request, result = step_pair()
    with pytest.raises(ValueError):
        validate_step_exchange(
            changed(reset, observation=observation(1, "initial-state")), request, result
        )


def test_task_environment_swaps_rejected_at_request_construction() -> None:
    request, _ = reset_pair()
    foreign_env = changed(environment(), definition_sha256=digest("foreign"))
    with pytest.raises(ValidationError):
        changed(request, environment_identity=foreign_env)
    step_request, _ = step_pair()
    with pytest.raises(ValidationError):
        changed(step_request, environment_identity=foreign_env)


@pytest.mark.parametrize(
    "model,field,value",
    [
        ("reset", "seed", -1),
        ("reset", "seed", True),
        ("reset", "seed", 1.5),
        ("step", "step_index", -1),
        ("step", "step_index", False),
        ("observation", "observation_index", -1),
        ("observation", "observation_index", "0"),
    ],
)
def test_strict_nonnegative_numeric_constraints(model: str, field: str, value: Any) -> None:
    reset, _ = reset_pair()
    step, _ = step_pair()
    obj = {"reset": reset, "step": step, "observation": observation(0, "initial-state")}[model]
    with pytest.raises(ValidationError):
        changed(obj, **{field: value})


def test_invalid_arguments_digest_and_binary_input_rejected() -> None:
    action, _ = step_pair()
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(content="bytes", media_type="text/plain", sha256=digest("other"))
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(content=b"bytes", media_type="text/plain", sha256=digest("bytes"))
    with pytest.raises(ValidationError):
        changed(
            action.action,
            arguments={
                "media_type": "text/plain",
                "content": "wrong",
                "sha256": digest("different"),
            },
        )


def test_models_are_frozen_strict_and_reject_reward_hidden_state_fields() -> None:
    reset, result = reset_pair()
    step, step_result = step_pair()
    for obj in (reset, result, step, step.action, result.observation, step_result):
        with pytest.raises(ValidationError):
            obj.__class__.model_validate({**obj.model_dump(mode="python"), "unexpected": True})
        with pytest.raises(ValidationError):
            obj.contract_type = "tampered"
    for field in ("hidden_state", "full_state", "reward", "verdict", "training_label"):
        with pytest.raises(ValidationError):
            changed(result.observation, **{field: "secret"})
        with pytest.raises(ValidationError):
            changed(step_result, **{field: True})


def test_all_contracts_canonical_roundtrip_and_hash_stability() -> None:
    reset, reset_result = reset_pair()
    step, result = step_pair()
    for obj in (reset, reset_result, step, result, step.action, result.observation):
        restored = obj.__class__.from_canonical_json(obj.canonical_bytes())
        assert restored == obj
        assert obj.fingerprint() == hashlib.sha256(obj.canonical_bytes()).hexdigest()
        assert restored.fingerprint() == obj.fingerprint()
    before = (
        reset.canonical_bytes(),
        reset_result.canonical_bytes(),
        step.canonical_bytes(),
        result.canonical_bytes(),
    )
    validate_reset_exchange(reset, reset_result)
    validate_step_exchange(reset_result, step, result)
    validate_reset_exchange(reset, reset_result)
    validate_step_exchange(reset_result, step, result)
    assert before == (
        reset.canonical_bytes(),
        reset_result.canonical_bytes(),
        step.canonical_bytes(),
        result.canonical_bytes(),
    )


def test_valid_looking_forged_sha_does_not_prove_execution() -> None:
    request, reset_result = reset_pair()
    validate_reset_exchange(request, reset_result)
    altered = changed(reset_result, reset_evidence_sha256=digest("untrusted-claim"))
    validate_reset_exchange(request, altered)
    assert not hasattr(altered, "trusted_execution")
    assert altered.fingerprint() != reset_result.fingerprint()


def test_valid_reset_and_step_are_not_runnable_environment() -> None:
    reset, reset_result = reset_pair()
    step, step_result = step_pair()

    class FakeEnvironment:
        def reset(self, request: AgentEnvironmentResetRequest) -> AgentEnvironmentResetResult:
            return reset_result

        def step(self, request: AgentEnvironmentStepRequest) -> AgentEnvironmentStepResult:
            return step_result

    fake: ExecutableAgentEnvironment = FakeEnvironment()
    assert fake.reset(reset) == reset_result
    assert fake.step(step) == step_result


@pytest.mark.parametrize(
    "which", ["reset_request", "reset_result", "step_request", "step_result", "prior"]
)
def test_exchange_validators_reject_raw_dict_inputs(which: str) -> None:
    request, previous = reset_pair()
    step, result = step_pair()
    inputs: list[Any] = [previous, step, result]
    if which == "reset_request":
        with pytest.raises(TypeError):
            validate_reset_exchange(request.model_dump(), previous)
    elif which == "reset_result":
        with pytest.raises(TypeError):
            validate_reset_exchange(request, previous.model_dump())
    else:
        index = {"prior": 0, "step_request": 1, "step_result": 2}[which]
        inputs[index] = inputs[index].model_dump()
        with pytest.raises(TypeError):
            validate_step_exchange(*inputs)


def test_model_copy_tampering_revalidated_fail_closed() -> None:
    request, reset_result = reset_pair()
    altered = reset_result.model_copy(update={"seed": -7})
    with pytest.raises(ValidationError):
        validate_reset_exchange(request, altered)
    step, result = step_pair()
    tampered = result.model_copy(update={"terminal_evidence_sha256": digest("forged")})
    with pytest.raises(ValidationError):
        validate_step_exchange(reset_result, step, tampered)
