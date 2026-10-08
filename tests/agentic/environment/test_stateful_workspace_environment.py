"""Behavioral and adversarial tests for the first M7 in-memory workspace."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.environment.interface import (
    AgentEnvironmentAction,
    AgentEnvironmentResetRequest,
    AgentEnvironmentStepRequest,
    ExecutableAgentEnvironment,
)
from huyawo_adapt.agentic.environment.workspace import (
    BoundedWorkspaceEnvironment,
    WorkspaceFixture,
)
from huyawo_adapt.agentic.trajectory import AgentTrajectoryPayload


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fixture(*, seed: int = 23, max_steps: int = 16) -> WorkspaceFixture:
    return WorkspaceFixture.create(
        verifier_fingerprint=digest("independent-verifier"),
        success_criteria_sha256=digest("external-success-criteria"),
        seed=seed,
        max_steps=max_steps,
    )


def reset_request(
    config: WorkspaceFixture, episode: str = "episode-1"
) -> AgentEnvironmentResetRequest:
    return AgentEnvironmentResetRequest(
        environment_identity=config.environment_identity,
        task_identity=config.task_identity,
        seed=config.seed,
        episode_id=episode,
    )


def action(name: str, arguments: str | dict[str, Any], **changes: Any) -> AgentEnvironmentAction:
    text = canonical(arguments) if isinstance(arguments, dict) else arguments
    return AgentEnvironmentAction(
        action_id=name,
        arguments=AgentTrajectoryPayload(
            media_type=changes.get("media_type", "application/json"),
            content=text,
            sha256=digest(text),
        ),
        action_provenance_sha256=changes.get("provenance", digest("caller-evidence")),
        tool_fingerprint=changes.get("tool_fingerprint"),
    )


def step_request(
    config: WorkspaceFixture,
    result: Any,
    operation: AgentEnvironmentAction,
    **changes: Any,
) -> AgentEnvironmentStepRequest:
    return AgentEnvironmentStepRequest(
        environment_identity=changes.get("environment_identity", config.environment_identity),
        task_identity=changes.get("task_identity", config.task_identity),
        episode_id=changes.get("episode_id", result.episode_id),
        step_index=changes.get("step_index", result.observation.observation_index),
        before_state_sha256=changes.get("before_state_sha256", result.observation.state_sha256),
        action=operation,
    )


def started(config: WorkspaceFixture | None = None) -> tuple[Any, WorkspaceFixture, Any]:
    config = config or fixture()
    environment = BoundedWorkspaceEnvironment(config)
    initial = environment.reset(reset_request(config))
    return environment, config, initial


def state_sha(entries: dict[str, str], steps: int, terminal: str) -> str:
    return digest(canonical({"entries": entries, "steps": steps, "terminal": terminal}))


def changed(model: Any, **changes: Any) -> Any:
    return model.__class__.model_validate({**model.model_dump(mode="python"), **changes})


def public(result: Any) -> dict[str, Any]:
    return json.loads(result.observation.public_observation.content)


def test_fixture_identifies_runtime_semantics_without_verifier_execution() -> None:
    config = fixture()
    assert config.task_identity.initial_state_sha256 == state_sha({}, 0, "continue")
    assert config.task_identity.environment_fingerprint == config.environment_identity.fingerprint()
    assert config.task_identity.verifier_fingerprint == digest("independent-verifier")
    assert config.task_identity.success_criteria_sha256 == digest("external-success-criteria")
    assert config == WorkspaceFixture.from_canonical_json(config.canonical_bytes())
    restored = WorkspaceFixture.from_canonical_json(config.canonical_json())
    assert config.fingerprint() == restored.fingerprint()
    assert not hasattr(config, "expected_final_state")
    assert not hasattr(config, "reward")


def test_structural_protocol_and_reset_projection() -> None:
    environment, config, result = started()
    typed_environment: ExecutableAgentEnvironment = environment
    assert callable(typed_environment.reset) and callable(typed_environment.step)
    assert result.observation.state_sha256 == config.task_identity.initial_state_sha256
    assert public(result) == {"keys": [], "remaining_steps": 16, "terminal": "continue"}
    assert result.observation.public_observation.sha256 != result.observation.state_sha256
    assert result.observation.observation_index == 0
    assert result.reset_evidence_sha256 != result.observation.state_provenance_sha256
    assert result.observation.visibility_policy_sha256 != result.observation.state_sha256
    assert not hasattr(result, "success")
    assert not hasattr(result, "reward")


def test_put_update_delete_and_finish_are_real_transitions() -> None:
    environment, config, last = started()
    entries: dict[str, str] = {}
    for index, (name, args) in enumerate(
        [
            ("put", {"key": "alpha", "value": "hidden-value-1"}),
            ("put", {"key": "alpha", "value": "hidden-value-2"}),
            ("put", {"key": "beta", "value": "naïve text"}),
            ("delete", {"key": "alpha"}),
            ("finish", {}),
        ]
    ):
        before = last.observation.state_sha256
        last = environment.step(step_request(config, last, action(name, args)))
        if name == "put":
            entries[args["key"]] = args["value"]
        if name == "delete":
            del entries[args["key"]]
        expected_terminal = "terminated" if name == "finish" else "continue"
        assert last.observation.state_sha256 == state_sha(entries, index + 1, expected_terminal)
        assert last.transition.before_state_sha256 == before
        assert last.transition.after_state_sha256 == last.observation.state_sha256
        assert last.observation.observation_index == index + 1
        assert last.terminal_status == expected_terminal
        assert public(last) == {
            "keys": sorted(entries),
            "remaining_steps": config.max_steps - index - 1,
            "terminal": expected_terminal,
        }
        assert "hidden-value" not in last.canonical_json()
        assert "naïve text" not in last.canonical_json()
        assert not hasattr(last, "verdict")
        assert not hasattr(last, "reward")
    assert last.terminal_evidence_sha256 is not None
    with pytest.raises(ValueError, match="terminal"):
        environment.step(step_request(config, last, action("finish", {})))


@pytest.mark.parametrize("max_steps", [1, 2, 16])
def test_limit_truncates_and_rejects_more_steps(max_steps: int) -> None:
    environment, config, last = started(fixture(max_steps=max_steps))
    for index in range(max_steps):
        operation = action("put", {"key": "a", "value": str(index)})
        last = environment.step(step_request(config, last, operation))
        assert last.terminal_status == ("truncated" if index + 1 == max_steps else "continue")
    assert public(last)["remaining_steps"] == 0
    assert last.terminal_evidence_sha256 is not None
    with pytest.raises(ValueError, match="terminal"):
        environment.step(step_request(config, last, action("finish", {})))


def test_finish_at_limit_is_terminated_not_truncated() -> None:
    environment, config, last = started(fixture(max_steps=1))
    last = environment.step(step_request(config, last, action("finish", {})))
    assert last.terminal_status == "terminated"
    assert last.terminal_evidence_sha256 is not None


def test_reset_restores_initial_state_and_invalidates_other_episode() -> None:
    environment, config, first = started()
    previous = environment.step(
        step_request(config, first, action("put", {"key": "a", "value": "x"}))
    )
    fresh = environment.reset(reset_request(config, "episode-2"))
    assert fresh.observation.state_sha256 == first.observation.state_sha256
    assert fresh.reset_evidence_sha256 != first.reset_evidence_sha256
    assert fresh.observation.state_provenance_sha256 != first.observation.state_provenance_sha256
    with pytest.raises(ValueError):
        environment.step(step_request(config, previous, action("finish", {})))
    assert environment.step(step_request(config, fresh, action("finish", {}))).terminal_status == (
        "terminated"
    )
    after = environment.reset(reset_request(config, "episode-3"))
    assert after.observation.observation_index == 0
    assert public(after)["keys"] == []


def test_exact_byte_for_byte_replay_across_fresh_instances() -> None:
    config = fixture(max_steps=4)
    sequences = []
    for _ in range(2):
        environment = BoundedWorkspaceEnvironment(config)
        last = environment.reset(reset_request(config, "episode-replay"))
        records = [last.canonical_bytes()]
        for name, args in [
            ("put", {"key": "first", "value": "naïve"}),
            ("put", {"key": "second", "value": ""}),
            ("delete", {"key": "second"}),
            ("finish", {}),
        ]:
            last = environment.step(step_request(config, last, action(name, args)))
            records.append(last.canonical_bytes())
        sequences.append(records)
    assert sequences[0] == sequences[1]


@pytest.mark.parametrize(
    "seed,max_steps", [(0, 1), (2, 16), (23, 0), (-1, 2), (True, 2), (2, True)]
)
def test_bad_fixture_bounds_rejected(seed: Any, max_steps: Any) -> None:
    if type(seed) is int and seed >= 0 and type(max_steps) is int and 1 <= max_steps <= 16:
        config = fixture(seed=seed, max_steps=max_steps)
        assert config.seed == seed
    else:
        with pytest.raises((ValidationError, ValueError)):
            fixture(seed=seed, max_steps=max_steps)


def test_fixture_cannot_forge_semantic_identity_links() -> None:
    config = fixture()
    with pytest.raises(ValidationError):
        changed(config, max_steps=2)
    with pytest.raises(ValidationError):
        changed(config, seed=5)
    with pytest.raises(ValidationError):
        changed(
            config,
            task_identity=changed(
                config.task_identity, allowed_actions_sha256=digest("foreign-allowlist")
            ),
        )
    with pytest.raises(ValidationError):
        changed(
            config,
            environment_identity=changed(
                config.environment_identity, reset_semantics_sha256=digest("foreign-reset")
            ),
        )
    with pytest.raises(ValidationError):
        changed(config, extra_field=True)
    with pytest.raises(ValidationError):
        changed(config, contract_type="untrusted")
    with pytest.raises(ValidationError):
        changed(config, verifier_fingerprint=digest("extra"))
    with pytest.raises(ValidationError):
        config.seed = 9


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("unknown", {}),
        ("put", '{"key":"a","key":"b","value":"c"}'),
        ("put", '{"key":"a","value":"v","value":"other"}'),
        ("put", '{"key": "a", "value": "v"}'),
        ("put", '{"value":"v","key":"a"}'),
        ("put", '{"key":"a","value":"v","extra":1}'),
        ("put", '{"key":"a"}'),
        ("put", '{"key":"a","value":2}'),
        ("put", '{"key":"a","value":null}'),
        ("put", '{"key":"a","value":NaN}'),
        ("put", '{"key":"A","value":"v"}'),
        ("put", '{"key":"a-b","value":"v"}'),
        ("put", '{"key":"a b","value":"v"}'),
        ("put", '{"key":"é","value":"v"}'),
        ("put", '{"key":"a1234567890123456","value":"v"}'),
        ("put", '{"key":"a","value":"' + "v" * 65 + '"}'),
        ("put", "[" + "0" * 260 + "]"),
        ("delete", {}),
        ("delete", '{"key":"absent"}'),
        ("finish", '{"anything":true}'),
        ("finish", "{ }"),
        ("put", '"not-an-object"'),
        ("put", '{"key":"a","value":{}}'),
        ("put", '{"key":"a","value":"v"} trailing'),
    ],
)
def test_bad_action_rejected_atomically(name: str, arguments: str) -> None:
    environment, config, initial = started()
    before = initial.canonical_bytes()
    with pytest.raises((ValueError, TypeError)):
        environment.step(step_request(config, initial, action(name, arguments)))
    last = environment.step(step_request(config, initial, action("finish", {})))
    assert last.step_index == 0
    assert last.transition.before_state_sha256 == initial.observation.state_sha256
    assert initial.canonical_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"media_type": "text/plain"},
        {"tool_fingerprint": digest("unapproved-tool")},
    ],
)
def test_unregistered_tool_or_wrong_media_type_rejected(changes: dict[str, Any]) -> None:
    environment, config, initial = started()
    with pytest.raises(ValueError):
        operation = action("put", {"key": "a", "value": "b"}, **changes)
        environment.step(step_request(config, initial, operation))
    assert environment.step(step_request(config, initial, action("finish", {}))).step_index == 0


def test_exact_value_byte_limits_and_eight_key_capacity() -> None:
    environment, config, last = started()
    for index in range(8):
        last = environment.step(
            step_request(config, last, action("put", {"key": f"k{index}", "value": "é" * 32}))
        )
    before = last.canonical_bytes()
    with pytest.raises(ValueError, match="capacity"):
        environment.step(step_request(config, last, action("put", {"key": "extra", "value": ""})))
    assert before == last.canonical_bytes()
    assert (
        environment.step(
            step_request(config, last, action("put", {"key": "k0", "value": "x"}))
        ).step_index
        == 8
    )


def test_sixteen_byte_key_and_hidden_unicode_value() -> None:
    environment, config, initial = started()
    key = "a123456789012345"
    value = "é" * 32
    result = environment.step(
        step_request(config, initial, action("put", {"key": key, "value": value}))
    )
    assert public(result)["keys"] == [key]
    assert value not in result.canonical_json()
    assert result.observation.state_sha256 == state_sha({key: value}, 1, "continue")


@pytest.mark.parametrize(
    "kind",
    ["step_index", "before_state_sha256", "episode_id", "task_identity", "environment_identity"],
)
def test_stale_or_cross_episode_requests_do_not_mutate_state(kind: str) -> None:
    environment, config, initial = started()
    changes: dict[str, Any] = {
        "step_index": 1,
        "before_state_sha256": digest("foreign-state"),
        "episode_id": "foreign-episode",
        "task_identity": changed(
            config.task_identity, verifier_fingerprint=digest("foreign-verifier")
        ),
        "environment_identity": changed(
            config.environment_identity, definition_sha256=digest("foreign-definition")
        ),
    }
    request_kwargs = {kind: changes[kind]}
    if kind == "environment_identity":
        request_kwargs["task_identity"] = changed(
            config.task_identity, environment_fingerprint=changes[kind].fingerprint()
        )
    with pytest.raises(ValueError):
        environment.step(step_request(config, initial, action("finish", {}), **request_kwargs))
    result = environment.step(step_request(config, initial, action("finish", {})))
    assert result.step_index == 0
    assert result.terminal_status == "terminated"


def test_reset_rejections_preserve_previous_live_state() -> None:
    environment, config, initial = started()
    last = environment.step(
        step_request(config, initial, action("put", {"key": "a", "value": "v"}))
    )
    bad_seed = changed(reset_request(config), seed=config.seed + 1)
    wrong_task = changed(
        reset_request(config),
        task_identity=changed(config.task_identity, verifier_fingerprint=digest("foreign")),
    )
    for request in [bad_seed, wrong_task]:
        with pytest.raises(ValueError):
            environment.reset(request)
    next_step = environment.step(step_request(config, last, action("finish", {})))
    assert public(next_step)["keys"] == ["a"]
    assert next_step.step_index == 1


def test_wrong_types_and_model_copy_forgery_are_rejected() -> None:
    environment, config, initial = started()
    with pytest.raises(TypeError):
        BoundedWorkspaceEnvironment(config.model_dump(mode="python"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        environment.reset(reset_request(config).model_dump(mode="python"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        environment.step({})  # type: ignore[arg-type]
    forged = step_request(config, initial, action("finish", {})).model_copy(
        update={"before_state_sha256": "not-a-digest"}
    )
    with pytest.raises(ValidationError):
        environment.step(forged)
    assert environment.step(step_request(config, initial, action("finish", {}))).step_index == 0


def test_action_provenance_binds_transition_but_cannot_authenticate_caller() -> None:
    config = fixture()
    sha = []
    for evidence in [digest("source-one"), digest("source-two")]:
        environment = BoundedWorkspaceEnvironment(config)
        initial = environment.reset(reset_request(config, "episode"))
        result = environment.step(
            step_request(config, initial, action("finish", {}, provenance=evidence))
        )
        sha.append(result.transition.transition_evidence_sha256)
    assert sha[0] != sha[1]


def test_episode_rename_changes_evidence_but_not_identical_state_digest() -> None:
    environment, config, first = started()
    second = environment.reset(reset_request(config, "episode-next"))
    assert first.observation.state_sha256 == second.observation.state_sha256
    assert first.observation.state_provenance_sha256 != second.observation.state_provenance_sha256
    assert first.reset_evidence_sha256 != second.reset_evidence_sha256


def test_inprocess_reference_has_no_outside_io(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import os
    import socket
    import subprocess

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("external I/O forbidden by M7 reference environment")

    config = fixture()
    environment = BoundedWorkspaceEnvironment(config)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    first = environment.reset(reset_request(config))
    second = environment.step(
        step_request(config, first, action("put", {"key": "a", "value": "v"}))
    )
    assert second.terminal_status == "continue"
    assert environment.step(step_request(config, second, action("finish", {}))).terminal_status == (
        "terminated"
    )
