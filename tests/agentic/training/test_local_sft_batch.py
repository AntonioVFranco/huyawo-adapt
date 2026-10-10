"""Strict local SFT batch collation, synthetic-only C01-C10 acceptance tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.training.local_sft_batch import (
    LocalSFTBatchDiagnostic,
    compile_local_sft_batch,
)
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
    SplitName,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_digest(token_ids: tuple[int, ...]) -> str:
    payload = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": token_ids},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _digest(payload)


def _projection(
    index: int,
    ids: tuple[int, ...] = (11, 12, 13, 14),
    start: int = 2,
    **changes: Any,
) -> AgentSFTDecisionProjection:
    data: dict[str, Any] = {
        "trajectory_fingerprint": _digest(f"trajectory-{index}"),
        "step_index": 0,
        "step_provenance_sha256": _digest(f"step-{index}"),
        "model_fingerprint": _digest("model"),
        "tokenizer_fingerprint": _digest("tokenizer"),
        "harness_fingerprint": _digest("harness"),
        "tool_set_fingerprint": _digest("tool-set"),
        "environment_fingerprint": _digest("environment"),
        "verifier_fingerprint": _digest("verifier"),
        "task_fingerprint": _digest(f"task-{index}"),
        "runtime_fingerprint": _digest("runtime"),
        "rollout_policy_fingerprint": _digest("rollout-policy"),
        "capture_provenance_sha256": _digest(f"capture-{index}"),
        "model_input_sha256": _digest(f"input-{index}"),
        "generated_output_sha256": _digest(f"output-{index}"),
        "projection_policy_sha256": _digest("projection-policy"),
        "input_ids": ids,
        "labels": (-100,) * start + ids[start:],
        "target_start": start,
        "target_end": len(ids),
        "target_token_sha256": _target_digest(ids[start:]),
    }
    data.update(changes)
    return AgentSFTDecisionProjection.model_validate(data)


def _entry(
    row: AgentSFTDecisionProjection,
    index: int,
    split: SplitName = "train",
    **changes: Any,
) -> AgentSFTSplitEntry:
    data: dict[str, Any] = {
        "split": split,
        "projection_fingerprint": row.fingerprint(),
        "trajectory_fingerprint": row.trajectory_fingerprint,
        "task_fingerprint": row.task_fingerprint,
        "family_fingerprint": _digest(f"family-{index}"),
        "root_trajectory_fingerprint": _digest(f"root-{index}"),
        "parent_trajectory_fingerprint": row.parent_trajectory_fingerprint,
        "source_content_sha256": _digest(f"content-{index}"),
    }
    data.update(changes)
    return AgentSFTSplitEntry.model_validate(data)


def _manifest(*entries: AgentSFTSplitEntry) -> AgentSFTSplitManifest:
    return AgentSFTSplitManifest(split_policy_sha256=_digest("split-policy"), entries=entries)


def _compile(
    rows: tuple[AgentSFTDecisionProjection, ...],
    *,
    pad: int = 0,
    limit: int = 4096,
    split: SplitName = "train",
    manifest: AgentSFTSplitManifest | None = None,
) -> LocalSFTBatchDiagnostic:
    if manifest is None:
        manifest = _manifest(*(_entry(row, i) for i, row in enumerate(rows)))
    return compile_local_sft_batch(rows, manifest, split, pad, limit)


def test_c01_exact_single_row_and_nonpromoting_scope() -> None:
    row = _projection(1)
    result = _compile((row,))
    assert result.input_ids == ((11, 12, 13, 14),)
    assert result.labels == ((-100, -100, 13, 14),)
    assert result.attention_mask == ((1, 1, 1, 1),)
    assert result.row_target_starts == (2,)
    assert result.row_target_lengths == (2,)
    assert result.total_loss_tokens == 2
    assert result.projection_fingerprints == (row.fingerprint(),)
    assert result.authority_scope == "local_sft_batch_shape_only"
    assert result.independent_task_outcome_verified is False
    assert result.model_origin_attested is False
    assert result.tokenizer_fidelity_attested is False
    assert result.training_eligible is False


def test_c02_unequal_lengths_right_pad_without_reordering() -> None:
    left = _projection(1, (21, 22, 23), 2)
    right = _projection(2, (31, 32, 33, 34, 35), 3)
    batch = _compile((left, right), pad=99)
    assert batch.padded_sequence_length == 5
    assert batch.input_ids == ((21, 22, 23, 99, 99), (31, 32, 33, 34, 35))
    assert batch.labels == ((-100, -100, 23, -100, -100), (-100, -100, -100, 34, 35))
    assert batch.attention_mask == ((1, 1, 1, 0, 0), (1, 1, 1, 1, 1))
    assert batch.row_target_lengths == (1, 2)
    assert batch.total_loss_tokens == 3


def test_c03_real_pad_valued_token_stays_attended() -> None:
    short = _projection(1, (0, 4, 0), 1)
    long = _projection(2, (6, 7, 8, 9), 2)
    batch = _compile((short, long), pad=0)
    assert batch.input_ids[0] == (0, 4, 0, 0)
    assert batch.attention_mask[0] == (1, 1, 1, 0)
    assert batch.labels[0] == (-100, 4, 0, -100)


def test_c04_ignored_prefix_suffix_and_loss_totals() -> None:
    rows = (_projection(1, (5, 6, 7, 8), 3), _projection(2, (9, 10), 1))
    batch = _compile(rows, pad=42)
    assert batch.row_target_starts == (3, 1)
    assert batch.row_target_lengths == (1, 1)
    assert batch.total_loss_tokens == 2
    for i, row in enumerate(rows):
        width = len(row.input_ids)
        assert batch.labels[i][: row.target_start] == (-100,) * row.target_start
        assert batch.labels[i][row.target_start : width] == row.input_ids[row.target_start :]
        assert batch.labels[i][width:] == (-100,) * (batch.padded_sequence_length - width)


def test_c05_limits_invalid_types_and_no_truncation() -> None:
    row = _projection(1)
    for bad in ((), tuple(_projection(i) for i in range(33))):
        with pytest.raises(ValueError):
            _compile(bad)
    for pad, limit in ((True, 12), (-1, 12), (0, True), (0, 3), (0, 4097), (0, 1)):
        with pytest.raises(ValueError):
            _compile((row,), pad=pad, limit=limit)
    with pytest.raises(ValueError):
        _compile((_projection(1, tuple(range(4097)), 1),), limit=4096)


def test_c06_exact_split_projection_set_and_duplicates() -> None:
    first = _projection(1)
    second = _projection(2)
    train_manifest = _manifest(_entry(first, 1), _entry(second, 2))
    with pytest.raises(ValueError):
        _compile((first,), manifest=train_manifest)
    with pytest.raises(ValueError):
        _compile((first, first), manifest=_manifest(_entry(first, 1)))
    with pytest.raises(ValueError):
        _compile((first, second), manifest=_manifest(_entry(first, 1)))
    with pytest.raises(ValueError):
        _compile((first,), manifest=_manifest(_entry(first, 1, "validation")))
    with pytest.raises(ValidationError):
        _manifest(_entry(first, 1), _entry(first, 2, "validation"))


def test_c07_declared_task_trajectory_parent_and_cross_split_lineage() -> None:
    row = _projection(1)
    for changed in (
        {"task_fingerprint": _digest("different-task")},
        {"trajectory_fingerprint": _digest("different-trajectory")},
        {"parent_trajectory_fingerprint": _digest("different-parent")},
    ):
        with pytest.raises(ValueError):
            _compile((row,), manifest=_manifest(_entry(row, 1, **changed)))
    second = _projection(2)
    with pytest.raises(ValidationError):
        _manifest(
            _entry(row, 1, "train"),
            _entry(second, 2, "test", root_trajectory_fingerprint=row.trajectory_fingerprint),
        )


def test_c08_reject_mixed_model_tokenizer_or_policy() -> None:
    first = _projection(1)
    for field in ("model_fingerprint", "tokenizer_fingerprint", "projection_policy_sha256"):
        second = _projection(2, **{field: _digest("other-" + field)})
        with pytest.raises(ValueError, match="mixed"):
            _compile((first, second))


def test_c09_tamper_rejection_and_order_sensitive_fingerprint() -> None:
    first, second = _projection(1, (2, 3, 4), 2), _projection(2)
    manifest = _manifest(_entry(first, 1), _entry(second, 2))
    batch = _compile((first, second), manifest=manifest, pad=77)
    again = _compile((first, second), manifest=manifest, pad=77)
    assert batch.batch_sha256 == again.batch_sha256
    assert batch == again
    assert _compile((second, first), manifest=manifest, pad=77).batch_sha256 != batch.batch_sha256
    assert _compile((first, second), manifest=manifest, pad=78).batch_sha256 != batch.batch_sha256
    for field, bad in (
        ("attention_mask", ((1, 0, 1, 0), (1, 1, 1, 1))),
        ("labels", ((-100, -100, 999, -100), (-100, -100, 13, 14))),
        ("total_loss_tokens", 100),
        ("batch_sha256", _digest("forged-batch")),
        ("training_eligible", True),
    ):
        with pytest.raises((ValidationError, ValueError)):
            LocalSFTBatchDiagnostic.model_validate({**batch.model_dump(mode="python"), field: bad})


def test_c10_old_contracts_unchanged_and_zero_training_authority() -> None:
    row = _projection(1)
    manifest = _manifest(_entry(row, 1))
    assert AgentSFTSplitManifest.model_validate(manifest.model_dump()) == manifest
    assert AgentSFTDecisionProjection.model_validate(row.model_dump()) == row
    batch = _compile((row,), manifest=manifest)
    assert batch.manifest_fingerprint == manifest.fingerprint()
    assert batch.training_eligible is False
    assert batch.independent_task_outcome_verified is False
    assert batch.model_origin_attested is False
    assert batch.tokenizer_fidelity_attested is False
    with pytest.raises(ValidationError):
        LocalSFTBatchDiagnostic.model_validate(
            {**batch.model_dump(mode="python"), "authority_scope": "training_eligible"}
        )
    with pytest.raises(ValidationError):
        LocalSFTBatchDiagnostic.model_validate(
            {**batch.model_dump(mode="python"), "origin_verified": True}
        )
