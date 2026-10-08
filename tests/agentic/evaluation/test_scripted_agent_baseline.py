"""Deterministic scripted action-replay evaluator and fail-closed authority tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

import huyawo_adapt.agentic.evaluation.baseline as baseline
from huyawo_adapt.agentic.environment.interface import AgentEnvironmentAction
from huyawo_adapt.agentic.environment.workspace import WorkspaceFixture
from huyawo_adapt.agentic.evaluation import (
    AgentEvaluationStepEvidence,
    AgentScriptedEvaluationCase,
    AgentScriptedEvaluationResult,
    ScriptedAgentBaselineEvaluator,
)
from huyawo_adapt.agentic.trajectory import AgentTrajectoryPayload


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fixture(*, max_steps: int = 16, seed: int = 23) -> WorkspaceFixture:
    return WorkspaceFixture.create(
        verifier_fingerprint=sha("independent-verifier"),
        success_criteria_sha256=sha("caller-provided-target"),
        seed=seed,
        max_steps=max_steps,
    )


def action(name: str, arguments: dict[str, str] | str) -> AgentEnvironmentAction:
    text = (
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if isinstance(arguments, dict)
        else arguments
    )
    return AgentEnvironmentAction(
        action_id=name,
        arguments=AgentTrajectoryPayload(
            media_type="application/json",
            content=text,
            sha256=sha(text),
        ),
        action_provenance_sha256=sha("scripted-fixture-action-source"),
        tool_fingerprint=None,
    )


def make_case(
    *actions: AgentEnvironmentAction,
    max_steps: int = 16,
    seed: int = 23,
    episode_id: str = "episode-alpha",
) -> AgentScriptedEvaluationCase:
    return AgentScriptedEvaluationCase(
        fixture=fixture(max_steps=max_steps, seed=seed),
        episode_id=episode_id,
        actions=tuple(actions),
        case_provenance_sha256=sha("independent-case-authority"),
    )


def put(key: str = "alpha", value: str = "private-value") -> AgentEnvironmentAction:
    return action("put", {"key": key, "value": value})


def finish() -> AgentEnvironmentAction:
    return action("finish", {})


def evaluate(*actions: AgentEnvironmentAction, **kwargs: Any) -> AgentScriptedEvaluationResult:
    return ScriptedAgentBaselineEvaluator().evaluate(make_case(*actions, **kwargs))


def test_import_is_distinct_from_nonagent_baseline() -> None:
    assert ScriptedAgentBaselineEvaluator.__module__ == ("huyawo_adapt.agentic.evaluation.baseline")
    assert not hasattr(ScriptedAgentBaselineEvaluator(), "score_model")


def test_strict_case_roundtrip_and_fingerprint() -> None:
    case = make_case(put(), finish())
    assert case.declared_source == "scripted_local_fixture"
    assert case == AgentScriptedEvaluationCase.from_canonical_json(case.canonical_bytes())
    assert (
        case.fingerprint()
        == AgentScriptedEvaluationCase.from_canonical_json(case.canonical_json()).fingerprint()
    )
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationCase.model_validate({**case.canonical_data(), "reward": 1.0})
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationCase.model_validate(
            {**case.canonical_data(), "declared_source": "model_generated"}
        )


def test_empty_and_oversized_action_tapes_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_case()
    with pytest.raises(ValidationError):
        make_case(*(finish() for _ in range(17)))


def test_mutated_case_and_wrong_type_are_rejected() -> None:
    case = make_case(finish())
    with pytest.raises(TypeError):
        ScriptedAgentBaselineEvaluator().evaluate(case.canonical_data())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ScriptedAgentBaselineEvaluator().evaluate(case.model_copy(update={"episode_id": ""}))


def test_terminated_replay_is_deterministic_but_not_success() -> None:
    case = make_case(
        put("alpha", "private-value"),
        put("beta", "naïve"),
        action("delete", {"key": "alpha"}),
        finish(),
    )
    evaluator = ScriptedAgentBaselineEvaluator()
    first = evaluator.evaluate(case)
    second = evaluator.evaluate(case)
    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fingerprint() == second.fingerprint()
    assert first.replay_status == "reproducible"
    assert first.reason == "terminal_replay_match"
    assert first.terminal_status == "terminated"
    assert first.executed_action_count == first.attempted_action_count == 4
    assert [step.step_index for step in first.steps] == [0, 1, 2, 3]
    assert [step.terminal_status for step in first.steps] == [
        "continue",
        "continue",
        "continue",
        "terminated",
    ]
    assert first.authority_scope == "scripted_local_replay_only"
    for forbidden in ("reward", "score", "success", "verdict", "model_input"):
        assert forbidden not in type(first).model_fields
    assert "private-value" not in first.canonical_json()
    assert "naïve" not in first.canonical_json()
    assert first.final_state_sha256 != first.final_public_observation_sha256
    assert first.steps[-1].after_state_sha256 == first.final_state_sha256


def test_successful_truncation_does_not_promote_task_success() -> None:
    result = evaluate(put(), max_steps=1)
    assert result.replay_status == "reproducible"
    assert result.terminal_status == "truncated"
    assert result.steps[0].terminal_status == "truncated"
    assert "success" not in result.canonical_json()


def test_incomplete_action_tape_preserves_actual_state_hash() -> None:
    result = evaluate(put())
    assert result.replay_status == "incomplete"
    assert result.reason == "action_tape_incomplete"
    assert result.terminal_status == "continue"
    assert result.executed_action_count == result.attempted_action_count == 1
    assert result.steps[0].after_state_sha256 == result.final_state_sha256


def test_postterminal_action_is_blocked_without_fake_step() -> None:
    result = evaluate(finish(), put())
    assert result.replay_status == "blocked"
    assert result.reason == "step_after_terminal"
    assert result.terminal_status == "terminated"
    assert result.attempted_action_count == 2
    assert result.executed_action_count == 1
    assert len(result.steps) == 1


@pytest.mark.parametrize(
    "bad_action",
    [
        action("unknown", {}),
        action("delete", {"key": "missing"}),
        action("put", {"key": "BAD", "value": "x"}),
        action("put", {"key": "alpha", "value": "x" * 65}),
        action("put", {"key": "a" * 17, "value": "x"}),
        action("put", '{"key":"alpha", "value":"x"}'),
        action("put", '{"key":"alpha","key":"beta","value":"x"}'),
        action("put", "not-json"),
        action("put", {"key": "alpha"}),
        action("finish", {"unexpected": "x"}),
    ],
)
def test_bad_action_is_blocked_without_inventing_state(bad_action: AgentEnvironmentAction) -> None:
    result = evaluate(bad_action)
    assert result.replay_status == "blocked"
    assert result.reason == "action_rejected"
    assert result.executed_action_count == 0
    assert result.attempted_action_count == 1
    assert not result.steps
    assert result.terminal_status == "continue"
    assert result.final_state_sha256 is not None


def test_wrong_tool_fingerprint_is_rejected() -> None:
    forged = put().model_copy(update={"tool_fingerprint": sha("outside-tool")})
    result = evaluate(forged)
    assert result.replay_status == "blocked"
    assert result.reason == "action_rejected"


def test_key_capacity_rejected_after_only_eight_transitions() -> None:
    actions = [put(f"key{i}", f"value{i}") for i in range(9)]
    result = evaluate(*actions)
    assert result.replay_status == "blocked"
    assert result.reason == "action_rejected"
    assert result.executed_action_count == 8
    assert result.attempted_action_count == 9
    assert all(e.terminal_status == "continue" for e in result.steps)


def test_early_terminal_at_max_step_is_preserved() -> None:
    result = evaluate(finish(), max_steps=1)
    assert result.terminal_status == "terminated"
    assert result.replay_status == "reproducible"


def test_script_source_and_episode_are_bound_in_fingerprint() -> None:
    one = evaluate(finish(), episode_id="episode-one")
    two = evaluate(finish(), episode_id="episode-two")
    three = evaluate(finish(), seed=24)
    assert len({one.case_fingerprint, two.case_fingerprint, three.case_fingerprint}) == 3
    assert len({one.replay_evidence_sha256, two.replay_evidence_sha256}) == 2


def test_result_validation_rejects_false_success_and_inconsistent_counts() -> None:
    good = evaluate(finish())
    base = good.canonical_data()
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationResult.model_validate({**base, "reward": 1.0})
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationResult.model_validate({**base, "authority_scope": "benchmark"})
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationResult.model_validate({**base, "executed_action_count": 2})
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationResult.model_validate({**base, "terminal_status": "success"})
    with pytest.raises(ValidationError):
        AgentScriptedEvaluationResult.model_validate({**base, "reason": "action_rejected"})
    assert good == AgentScriptedEvaluationResult.from_canonical_json(good.canonical_bytes())


def test_replayed_step_evidence_is_frozen_and_strict() -> None:
    step = evaluate(finish()).steps[0]
    assert step == AgentEvaluationStepEvidence.from_canonical_json(step.canonical_bytes())
    with pytest.raises(ValidationError):
        AgentEvaluationStepEvidence.model_validate({**step.canonical_data(), "reward": 1})
    with pytest.raises(ValidationError):
        AgentEvaluationStepEvidence.model_validate(
            {**step.canonical_data(), "terminal_status": "success"}
        )


def test_independent_reset_evidence_conflict_blocks() -> None:
    original = baseline.BoundedWorkspaceEnvironment
    count = 0

    class DivergentReset:
        def __init__(self, config: WorkspaceFixture) -> None:
            nonlocal count
            count += 1
            self.inner = original(config)
            self.second = count == 2

        def reset(self, request: Any) -> Any:
            record = self.inner.reset(request)
            if self.second:
                return record.model_copy(update={"reset_evidence_sha256": sha("divergent")})
            return record

        def step(self, request: Any) -> Any:
            return self.inner.step(request)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline, "BoundedWorkspaceEnvironment", DivergentReset)
        result = evaluate(finish())
    assert result.replay_status == "blocked"
    assert result.reason == "environment_replay_conflict"
    assert result.executed_action_count == 0


def test_independent_step_evidence_conflict_blocks() -> None:
    original = baseline.BoundedWorkspaceEnvironment
    count = 0

    class DivergentStep:
        def __init__(self, config: WorkspaceFixture) -> None:
            nonlocal count
            count += 1
            self.inner = original(config)
            self.second = count == 2

        def reset(self, request: Any) -> Any:
            return self.inner.reset(request)

        def step(self, request: Any) -> Any:
            record = self.inner.step(request)
            if self.second:
                return record.model_copy(update={"terminal_evidence_sha256": sha("divergent")})
            return record

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline, "BoundedWorkspaceEnvironment", DivergentStep)
        result = evaluate(finish())
    assert result.replay_status == "blocked"
    assert result.reason == "environment_replay_conflict"
    assert result.executed_action_count == 0


def test_no_external_model_or_synthetic_m3_trajectory() -> None:
    record = evaluate(put(), finish())
    assert not hasattr(record, "trajectory")
    assert not hasattr(record, "model_output")
    assert not hasattr(record, "trainer_eligibility")
    assert not hasattr(record, "verifier_verdict")
    assert all(step.terminal_status != "success" for step in record.steps)
    assert record.authority_scope == "scripted_local_replay_only"


def test_single_sided_tampered_step_is_blocked_without_accepted_step() -> None:
    original = baseline.BoundedWorkspaceEnvironment
    count = 0

    class ForgedIndex:
        def __init__(self, config: WorkspaceFixture) -> None:
            nonlocal count
            count += 1
            self.inner = original(config)
            self.second = count == 2

        def reset(self, request: Any) -> Any:
            return self.inner.reset(request)

        def step(self, request: Any) -> Any:
            record = self.inner.step(request)
            return record.model_copy(update={"step_index": 10}) if self.second else record

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline, "BoundedWorkspaceEnvironment", ForgedIndex)
        record = evaluate(finish())
    assert record.replay_status == "blocked"
    assert record.reason == "evidence_conflict"
    assert record.executed_action_count == 0


def test_unexpected_environment_exception_fails_closed() -> None:
    original = baseline.BoundedWorkspaceEnvironment

    class FatalError:
        def __init__(self, config: WorkspaceFixture) -> None:
            self.inner = original(config)

        def reset(self, request: Any) -> Any:
            return self.inner.reset(request)

        def step(self, request: Any) -> Any:
            raise RuntimeError("unexpected runtime integrity failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline, "BoundedWorkspaceEnvironment", FatalError)
        with pytest.raises(RuntimeError, match="unexpected runtime integrity failure"):
            evaluate(finish())


def test_evaluation_never_uses_host_io_or_external_tools() -> None:
    import socket
    import subprocess
    import urllib.request

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("external side effects are forbidden in scripted evaluation")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess, "run", forbidden)
        patch.setattr(socket, "socket", forbidden)
        patch.setattr(urllib.request, "urlopen", forbidden)
        record = evaluate(put(), finish())
    assert record.replay_status == "reproducible"
