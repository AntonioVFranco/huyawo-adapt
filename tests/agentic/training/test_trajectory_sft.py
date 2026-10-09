"""M9-B1 offline projection and split-manifest contract tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
    project_trajectory_decision,
    validate_sft_split_manifest,
)
from huyawo_adapt.agentic.trajectory.schema import (
    AgentFailureRecord,
    AgentRewardComponent,
    AgentStateTransitionRecord,
    AgentToolCallRecord,
    AgentTrajectory,
    AgentTrajectoryPayload,
    AgentTrajectoryStep,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload(content: str, media_type: str = "text/plain") -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(media_type=media_type, content=content, sha256=_digest(content))


def _trajectory(**step_changes: Any) -> AgentTrajectory:
    defaults: dict[str, Any] = {
        "step_index": 0,
        "observation": _payload("visible observation"),
        "model_input": _payload("user input"),
        "model_input_token_ids": (11, 12),
        "generated_output": _payload("answer"),
        "generated_token_ids": (13, 14),
        "terminal_status": "success",
        "provenance_sha256": _digest("step"),
    }
    defaults.update(step_changes)
    step = AgentTrajectoryStep(**defaults)
    return AgentTrajectory(
        trajectory_id="trajectory-1",
        model_fingerprint=_digest("model"),
        tokenizer_fingerprint=_digest("tokenizer"),
        harness_fingerprint=_digest("harness"),
        tool_set_fingerprint=_digest("tools"),
        environment_fingerprint=_digest("env"),
        verifier_fingerprint=_digest("verifier"),
        task_fingerprint=_digest("task"),
        runtime_fingerprint=_digest("runtime"),
        rollout_policy_fingerprint=_digest("rollout"),
        steps=(step,),
        capture_provenance_sha256=_digest("capture"),
    )


def _projection() -> AgentSFTDecisionProjection:
    return project_trajectory_decision(_trajectory(), 0, _digest("projection-policy"))


def _split_entry(index: int, split: str, **changes: Any) -> AgentSFTSplitEntry:
    data: dict[str, Any] = {
        "split": split,
        "projection_fingerprint": _digest(f"projection-{index}"),
        "trajectory_fingerprint": _digest(f"trajectory-{index}"),
        "task_fingerprint": _digest(f"task-{index}"),
        "family_fingerprint": _digest(f"family-{index}"),
        "root_trajectory_fingerprint": _digest(f"root-{index}"),
        "parent_trajectory_fingerprint": None,
        "source_content_sha256": _digest(f"content-{index}"),
    }
    data.update(changes)
    return AgentSFTSplitEntry(**data)


def _manifest(*entries: AgentSFTSplitEntry) -> AgentSFTSplitManifest:
    return AgentSFTSplitManifest(split_policy_sha256=_digest("split-policy"), entries=entries)


def test_projection_exact_mask_and_lineage() -> None:
    trajectory = _trajectory()
    result = project_trajectory_decision(trajectory, 0, _digest("projection-policy"))
    assert result.authority_scope == "structural_projection_only"
    assert result.input_ids == (11, 12, 13, 14)
    assert result.labels == (-100, -100, 13, 14)
    assert (result.target_start, result.target_end) == (2, 4)
    assert result.trajectory_fingerprint == trajectory.fingerprint()
    assert result.step_provenance_sha256 == trajectory.steps[0].provenance_sha256
    assert result.capture_provenance_sha256 == trajectory.capture_provenance_sha256
    assert result.model_input_sha256 == trajectory.steps[0].model_input.sha256
    assert result.generated_output_sha256 == trajectory.steps[0].generated_output.sha256
    assert result.tokenizer_fingerprint == trajectory.tokenizer_fingerprint
    assert "training_eligible" not in AgentSFTDecisionProjection.model_fields
    assert "reward" not in AgentSFTDecisionProjection.model_fields


def test_projection_digest_is_domain_separated_canonical_json() -> None:
    result = _projection()
    payload = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": (13, 14)},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result.target_token_sha256 == _digest(payload)
    assert result.fingerprint() == hashlib.sha256(result.canonical_bytes()).hexdigest()
    assert result == AgentSFTDecisionProjection.from_canonical_json(result.canonical_json())
    assert result.fingerprint() == _projection().fingerprint()


def test_terminal_success_never_proves_training_eligibility() -> None:
    result = _projection()
    with pytest.raises(ValidationError):
        AgentSFTDecisionProjection.model_validate(
            {**result.model_dump(), "training_eligible": True}
        )
    with pytest.raises(ValidationError):
        AgentSFTDecisionProjection.model_validate(
            {**result.model_dump(), "authority_scope": "verified_success"}
        )


def test_projection_is_frozen() -> None:
    result = _projection()
    with pytest.raises((ValidationError, FrozenInstanceError)):
        result.labels = (1,)  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        {"input_ids": (11, -2, 13, 14)},
        {"input_ids": (11, True, 13, 14)},
        {"input_ids": (11, 12, 13)},
        {"labels": (-100, -100)},
        {"labels": (0, -100, 13, 14)},
        {"labels": (-100, 12, 13, 14)},
        {"labels": (-100, -100, -100, 14)},
        {"labels": (-100, -100, 13, -100)},
        {"labels": (-100, -100, 13, True)},
        {"target_start": 0},
        {"target_start": 4},
        {"target_start": 2.0},
        {"target_end": 3},
        {"target_end": 5},
        {"target_token_sha256": _digest("altered-token-target")},
        {"projection_policy_sha256": "malformed"},
        {"unknown_field": "not-allowed"},
    ],
)
def test_reject_invalid_projection_fields(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentSFTDecisionProjection.model_validate({**_projection().model_dump(), **change})


@pytest.mark.parametrize(
    "field",
    ["model_input_token_ids", "generated_token_ids"],
)
def test_absent_token_ids_are_ineligible_for_projection(field: str) -> None:
    with pytest.raises(ValueError, match="token IDs"):
        project_trajectory_decision(_trajectory(**{field: None}), 0, _digest("policy"))


@pytest.mark.parametrize("field", ["model_input_token_ids", "generated_token_ids"])
def test_empty_token_ids_rejected_by_m3(field: str) -> None:
    with pytest.raises(ValidationError):
        _trajectory(**{field: ()})


@pytest.mark.parametrize("field", ["model_input_token_ids", "generated_token_ids"])
def test_boolean_token_ids_rejected_by_strict_m3(field: str) -> None:
    with pytest.raises(ValidationError):
        _trajectory(**{field: (True, 42)})


def test_projection_rejects_nontext_and_tool_control_records() -> None:
    with pytest.raises(ValueError, match="text/plain"):
        project_trajectory_decision(
            _trajectory(generated_output=_payload("{}", "application/json")), 0, _digest("policy")
        )
    call = AgentToolCallRecord(
        call_id="call-1", tool_fingerprint=_digest("tool-1"), arguments=_payload("{}")
    )
    with pytest.raises(ValueError, match="tool/control"):
        project_trajectory_decision(_trajectory(tool_call=call), 0, _digest("policy"))


def test_projection_rejects_state_transition_reward_and_failure() -> None:
    transition = AgentStateTransitionRecord(
        before_state_sha256=_digest("before"),
        after_state_sha256=_digest("after"),
        transition_evidence_sha256=_digest("transition"),
    )
    with pytest.raises(ValueError, match="tool/control"):
        project_trajectory_decision(_trajectory(state_transition=transition), 0, _digest("policy"))
    reward = AgentRewardComponent(
        component_id="fixture-only", value=1.0, source_fingerprint=_digest("fixture")
    )
    with pytest.raises(ValueError, match="tool/control"):
        project_trajectory_decision(_trajectory(reward_components=(reward,)), 0, _digest("policy"))
    with pytest.raises(ValueError, match="incomplete or failed"):
        project_trajectory_decision(_trajectory(terminal_status="truncated"), 0, _digest("policy"))
    failure = AgentFailureRecord(failure_type="error", retryable=False)
    with pytest.raises(ValueError, match="tool/control"):
        project_trajectory_decision(
            _trajectory(terminal_status="failure", failure=failure), 0, _digest("policy")
        )


def test_projection_rejects_ambiguous_multistep_and_indices() -> None:
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="standalone"):
        project_trajectory_decision(trajectory, 1, _digest("policy"))
    with pytest.raises(TypeError):
        project_trajectory_decision(trajectory, True, _digest("policy"))
    invalid: Any = "not-a-trajectory"
    with pytest.raises(TypeError):
        project_trajectory_decision(invalid, 0, _digest("policy"))
    previous = trajectory.steps[0].model_copy(update={"terminal_status": "continue"})
    last = trajectory.steps[0].model_copy(update={"step_index": 1})
    longer = trajectory.model_copy(update={"steps": (previous, last)})
    with pytest.raises(ValueError, match="standalone"):
        project_trajectory_decision(longer, 0, _digest("policy"))


def test_projection_rejects_bad_policy_and_preserves_origin_and_parent() -> None:
    with pytest.raises(ValidationError):
        project_trajectory_decision(_trajectory(), 0, "wrong-digest")
    trajectory = _trajectory().model_copy(
        update={
            "parent_trajectory_fingerprint": _digest("parent"),
            "transformation_fingerprint": _digest("transform"),
        }
    )
    result = project_trajectory_decision(trajectory, 0, _digest("policy"))
    assert result.parent_trajectory_fingerprint == _digest("parent")
    assert result.transformation_fingerprint == _digest("transform")


def test_manifest_valid_distinct_splits_and_determinism() -> None:
    manifest = _manifest(
        _split_entry(1, "train"),
        _split_entry(2, "validation"),
        _split_entry(3, "test"),
    )
    assert validate_sft_split_manifest(manifest) == manifest
    assert manifest == AgentSFTSplitManifest.from_canonical_json(manifest.canonical_json())
    assert manifest.fingerprint() == hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    assert validate_sft_split_manifest(manifest).fingerprint() == manifest.fingerprint()


@pytest.mark.parametrize(
    "overlap",
    [
        "projection_fingerprint",
        "trajectory_fingerprint",
        "task_fingerprint",
        "family_fingerprint",
        "root_trajectory_fingerprint",
        "source_content_sha256",
    ],
)
def test_cross_split_leakage_is_rejected(overlap: str) -> None:
    first = _split_entry(1, "train")
    second = _split_entry(2, "validation", **{overlap: getattr(first, overlap)})
    with pytest.raises(ValidationError):
        _manifest(first, second)


def test_parent_lineage_leakage_is_rejected() -> None:
    first = _split_entry(1, "train")
    second = _split_entry(2, "test", parent_trajectory_fingerprint=first.trajectory_fingerprint)
    with pytest.raises(ValidationError, match="lineage"):
        _manifest(first, second)


def test_shared_parent_lineage_across_splits_rejected() -> None:
    parent = _digest("shared-parent")
    first = _split_entry(1, "train", parent_trajectory_fingerprint=parent)
    second = _split_entry(2, "validation", parent_trajectory_fingerprint=parent)
    with pytest.raises(ValidationError, match="lineage"):
        _manifest(first, second)


def test_duplicate_projection_rejected_within_same_split() -> None:
    first = _split_entry(1, "train")
    second = _split_entry(2, "train", projection_fingerprint=first.projection_fingerprint)
    with pytest.raises(ValidationError, match="duplicate projection"):
        _manifest(first, second)


def test_manifest_rejects_empty_malformed_or_nonrecord_data() -> None:
    with pytest.raises(ValidationError):
        _manifest()
    with pytest.raises(ValidationError):
        AgentSFTSplitEntry.model_validate({**_split_entry(1, "train").model_dump(), "split": "dev"})
    with pytest.raises(ValidationError):
        AgentSFTSplitManifest.model_validate(
            {**_manifest(_split_entry(1, "train")).model_dump(), "unverified": True}
        )
    with pytest.raises(TypeError):
        validate_sft_split_manifest("not a manifest")  # type: ignore[arg-type]


def test_split_entry_cannot_assert_fake_attested_fields() -> None:
    with pytest.raises(ValidationError):
        AgentSFTSplitEntry.model_validate(
            {**_split_entry(1, "train").model_dump(), "held_out_verified": True}
        )


def test_projection_does_not_use_filesystem_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import socket

    trajectory = _trajectory()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("projection may not access external resources")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        result = project_trajectory_decision(trajectory, 0, _digest("policy"))
        manifest = _manifest(_split_entry(1, "train"))
        assert validate_sft_split_manifest(manifest) == manifest
        assert result.authority_scope == "structural_projection_only"
