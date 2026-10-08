"""Bounded in-process stateful workspace implementing the M6 environment protocol."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, Self, cast

from pydantic import Field, model_validator

from huyawo_adapt.agentic.contracts import AgentEnvironmentIdentity, AgentTaskIdentity
from huyawo_adapt.agentic.environment.interface import (
    AgentEnvironmentAction,
    AgentEnvironmentObservation,
    AgentEnvironmentResetRequest,
    AgentEnvironmentResetResult,
    AgentEnvironmentStepRequest,
    AgentEnvironmentStepResult,
    validate_reset_exchange,
    validate_step_exchange,
)
from huyawo_adapt.agentic.trajectory import AgentStateTransitionRecord, AgentTrajectoryPayload
from huyawo_adapt.contracts.base import ContractModel, Sha256Digest

_VERSION = "bounded-workspace-v1"
_MAX_KEYS = 8
_MAX_KEY_BYTES = 16
_MAX_VALUE_BYTES = 64
_MAX_STEPS = 16
_MAX_ACTION_BYTES = 256
_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)
_ACTIONS = ("delete", "finish", "put")
_TERMINAL = Literal["continue", "terminated", "truncated"]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _evidence(domain: str, values: object) -> str:
    return _sha(_canonical({"domain": f"{_VERSION}.{domain}", "values": values}))


def _state_bytes(entries: dict[str, str], steps: int, terminal: _TERMINAL) -> bytes:
    return _canonical({"entries": entries, "steps": steps, "terminal": terminal})


def _allowed_actions_digest() -> str:
    return _evidence(
        "allowed_actions",
        {
            "delete": {"key": "ascii-key"},
            "finish": {},
            "put": {"key": "ascii-key", "value": "utf8-text"},
            "tool_fingerprint": None,
        },
    )


def _visibility_digest() -> str:
    return _evidence(
        "visibility_policy",
        {"keys": "sorted-public-names", "remaining_steps": "int", "terminal": "status"},
    )


def _environment_identity(environment_id: str) -> AgentEnvironmentIdentity:
    return AgentEnvironmentIdentity(
        environment_id=environment_id,
        definition_sha256=_evidence(
            "environment_definition",
            {
                "environment_id": environment_id,
                "max_keys": _MAX_KEYS,
                "max_key_bytes": _MAX_KEY_BYTES,
                "max_value_bytes": _MAX_VALUE_BYTES,
                "max_steps": _MAX_STEPS,
                "max_action_bytes": _MAX_ACTION_BYTES,
                "actions": _ACTIONS,
            },
        ),
        state_schema_sha256=_evidence(
            "state_schema", {"entries": "dict[str,str]", "steps": "int", "terminal": "status"}
        ),
        reset_semantics_sha256=_evidence(
            "reset_semantics", {"initial": "empty", "seed": "fixed", "active_episodes": 1}
        ),
        transition_semantics_sha256=_evidence(
            "transition_semantics",
            {"actions": _ACTIONS, "atomic": True, "missing_delete": "reject"},
        ),
        terminal_conditions_sha256=_evidence(
            "terminal_semantics", {"finish": "terminated", "step_limit": "truncated"}
        ),
    )


def _task_identity(
    task_id: str,
    env: AgentEnvironmentIdentity,
    verifier_fingerprint: str,
    success_criteria_sha256: str,
    seed: int,
    max_steps: int,
) -> AgentTaskIdentity:
    return AgentTaskIdentity(
        task_id=task_id,
        environment_fingerprint=env.fingerprint(),
        verifier_fingerprint=verifier_fingerprint,
        task_definition_sha256=_evidence(
            "task_definition",
            {"task_id": task_id, "seed": seed, "max_steps": max_steps},
        ),
        initial_state_sha256=_sha(_state_bytes({}, 0, "continue")),
        allowed_actions_sha256=_allowed_actions_digest(),
        success_criteria_sha256=success_criteria_sha256,
    )


class WorkspaceFixture(ContractModel):
    """Frozen local task setup; verifier and success criteria remain externally supplied."""

    contract_type: Literal["workspace_fixture"] = "workspace_fixture"
    environment_identity: AgentEnvironmentIdentity
    task_identity: AgentTaskIdentity
    seed: int = Field(ge=0)
    max_steps: int = Field(ge=1, le=_MAX_STEPS)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        env = _environment_identity(self.environment_identity.environment_id)
        if self.environment_identity != env:
            raise ValueError("workspace environment semantics mismatch")
        task = _task_identity(
            self.task_identity.task_id,
            env,
            self.task_identity.verifier_fingerprint,
            self.task_identity.success_criteria_sha256,
            self.seed,
            self.max_steps,
        )
        if self.task_identity != task:
            raise ValueError("workspace task semantics mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        verifier_fingerprint: Sha256Digest,
        success_criteria_sha256: Sha256Digest,
        seed: int = 0,
        max_steps: int = _MAX_STEPS,
        environment_id: str = "bounded-workspace-v1",
        task_id: str = "bounded-workspace-task-v1",
    ) -> WorkspaceFixture:
        """Create identities for a fixture without generating an expected answer."""
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
            raise ValueError("max_steps must be within 1..16")
        env = _environment_identity(environment_id)
        task = _task_identity(
            task_id, env, verifier_fingerprint, success_criteria_sha256, seed, max_steps
        )
        return cls(
            environment_identity=env,
            task_identity=task,
            seed=seed,
            max_steps=max_steps,
        )


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON property")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _action_arguments(action: AgentEnvironmentAction) -> dict[str, str]:
    if action.tool_fingerprint is not None:
        raise ValueError("workspace actions cannot invoke external tools")
    if action.action_id not in _ACTIONS:
        raise ValueError("action is not in the workspace allowlist")
    if action.arguments.media_type != "application/json":
        raise ValueError("action arguments must be application/json")
    raw = action.arguments.content
    if len(raw.encode("utf-8")) > _MAX_ACTION_BYTES:
        raise ValueError("workspace action arguments exceed byte limit")
    decoded = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    if type(decoded) is not dict or _canonical(decoded).decode("utf-8") != raw:
        raise ValueError("workspace action arguments must be a canonical JSON object")
    expected = {"put": {"key", "value"}, "delete": {"key"}, "finish": set()}[action.action_id]
    if set(decoded) != expected:
        raise ValueError("workspace action argument fields mismatch")
    if any(type(v) is not str for v in decoded.values()):
        raise ValueError("workspace action arguments must contain exact strings")
    if action.action_id != "finish":
        key = decoded["key"]
        if not _KEY_PATTERN.fullmatch(key) or len(key.encode("utf-8")) > _MAX_KEY_BYTES:
            raise ValueError("invalid workspace key")
    if action.action_id == "put" and len(decoded["value"].encode("utf-8")) > _MAX_VALUE_BYTES:
        raise ValueError("workspace value exceeds byte limit")
    return cast(dict[str, str], decoded)


class BoundedWorkspaceEnvironment:
    """Single-episode transactional reference environment; not a security sandbox."""

    def __init__(self, fixture: WorkspaceFixture) -> None:
        if type(fixture) is not WorkspaceFixture:
            raise TypeError("expected WorkspaceFixture")
        self._fixture = WorkspaceFixture.model_validate(fixture.model_dump(mode="python"))
        self._entries: dict[str, str] = {}
        self._steps = 0
        self._terminal: _TERMINAL = "continue"
        self._prior: AgentEnvironmentResetResult | AgentEnvironmentStepResult | None = None

    def _observation(
        self, episode_id: str, entries: dict[str, str], steps: int, terminal: _TERMINAL
    ) -> AgentEnvironmentObservation:
        state_sha = _sha(_state_bytes(entries, steps, terminal))
        public = _canonical(
            {
                "keys": sorted(entries),
                "remaining_steps": self._fixture.max_steps - steps,
                "terminal": terminal,
            }
        ).decode("utf-8")
        return AgentEnvironmentObservation(
            episode_id=episode_id,
            observation_index=steps,
            public_observation=AgentTrajectoryPayload(
                media_type="application/json", content=public, sha256=_sha(public.encode("utf-8"))
            ),
            state_sha256=state_sha,
            state_provenance_sha256=_evidence(
                "state_provenance",
                {
                    "episode_id": episode_id,
                    "observation_index": steps,
                    "state_sha256": state_sha,
                    "task_fingerprint": self._fixture.task_identity.fingerprint(),
                    "visibility_policy_sha256": _visibility_digest(),
                },
            ),
            visibility_policy_sha256=_visibility_digest(),
        )

    def reset(self, request: AgentEnvironmentResetRequest) -> AgentEnvironmentResetResult:
        """Replace the active episode only after complete reset-exchange validation."""
        if type(request) is not AgentEnvironmentResetRequest:
            raise TypeError("expected AgentEnvironmentResetRequest")
        checked = AgentEnvironmentResetRequest.model_validate(request.model_dump(mode="python"))
        if (
            checked.environment_identity != self._fixture.environment_identity
            or checked.task_identity != self._fixture.task_identity
            or checked.seed != self._fixture.seed
        ):
            raise ValueError("reset fixture identity or seed mismatch")
        entries: dict[str, str] = {}
        observation = self._observation(checked.episode_id, entries, 0, "continue")
        result = AgentEnvironmentResetResult(
            environment_fingerprint=self._fixture.environment_identity.fingerprint(),
            task_fingerprint=self._fixture.task_identity.fingerprint(),
            episode_id=checked.episode_id,
            seed=checked.seed,
            observation=observation,
            reset_evidence_sha256=_evidence(
                "reset_evidence",
                {
                    "episode_id": checked.episode_id,
                    "seed": checked.seed,
                    "state_sha256": observation.state_sha256,
                    "task_fingerprint": self._fixture.task_identity.fingerprint(),
                    "visibility_policy_sha256": observation.visibility_policy_sha256,
                },
            ),
        )
        validate_reset_exchange(checked, result)
        self._entries = entries
        self._steps = 0
        self._terminal = "continue"
        self._prior = result
        return result

    def step(self, request: AgentEnvironmentStepRequest) -> AgentEnvironmentStepResult:
        """Admit, evaluate and validate a tentative mutation before committing it."""
        if type(request) is not AgentEnvironmentStepRequest:
            raise TypeError("expected AgentEnvironmentStepRequest")
        checked = AgentEnvironmentStepRequest.model_validate(request.model_dump(mode="python"))
        prior = self._prior
        if prior is None:
            raise ValueError("reset required before step")
        if self._terminal != "continue":
            raise ValueError("cannot step after terminal")
        if (
            checked.environment_identity != self._fixture.environment_identity
            or checked.task_identity != self._fixture.task_identity
            or checked.episode_id != prior.episode_id
            or checked.step_index != self._steps
            or checked.before_state_sha256 != prior.observation.state_sha256
            or checked.before_state_sha256
            != _sha(_state_bytes(self._entries, self._steps, self._terminal))
        ):
            raise ValueError("step identity, index or state mismatch")
        arguments = _action_arguments(checked.action)
        entries = self._entries.copy()
        if checked.action.action_id == "put":
            if arguments["key"] not in entries and len(entries) >= _MAX_KEYS:
                raise ValueError("workspace key capacity exceeded")
            entries[arguments["key"]] = arguments["value"]
        elif checked.action.action_id == "delete":
            if arguments["key"] not in entries:
                raise ValueError("cannot delete a missing workspace key")
            del entries[arguments["key"]]
        next_steps = self._steps + 1
        next_terminal: _TERMINAL = (
            "terminated"
            if checked.action.action_id == "finish"
            else "truncated"
            if next_steps == self._fixture.max_steps
            else "continue"
        )
        observation = self._observation(checked.episode_id, entries, next_steps, next_terminal)
        transition = AgentStateTransitionRecord(
            before_state_sha256=checked.before_state_sha256,
            after_state_sha256=observation.state_sha256,
            transition_evidence_sha256=_evidence(
                "transition_evidence",
                {
                    "episode_id": checked.episode_id,
                    "step_index": checked.step_index,
                    "action_fingerprint": checked.action.fingerprint(),
                    "before_state_sha256": checked.before_state_sha256,
                    "after_state_sha256": observation.state_sha256,
                    "terminal_status": next_terminal,
                    "visibility_policy_sha256": observation.visibility_policy_sha256,
                },
            ),
        )
        terminal_evidence = (
            None
            if next_terminal == "continue"
            else _evidence(
                "terminal_evidence",
                {
                    "episode_id": checked.episode_id,
                    "step_index": checked.step_index,
                    "after_state_sha256": observation.state_sha256,
                    "terminal_status": next_terminal,
                },
            )
        )
        result = AgentEnvironmentStepResult(
            environment_fingerprint=self._fixture.environment_identity.fingerprint(),
            task_fingerprint=self._fixture.task_identity.fingerprint(),
            episode_id=checked.episode_id,
            step_index=checked.step_index,
            action_fingerprint=checked.action.fingerprint(),
            observation=observation,
            transition=transition,
            terminal_status=next_terminal,
            terminal_evidence_sha256=terminal_evidence,
        )
        validate_step_exchange(prior, checked, result)
        self._entries = entries
        self._steps = next_steps
        self._terminal = next_terminal
        self._prior = result
        return result
