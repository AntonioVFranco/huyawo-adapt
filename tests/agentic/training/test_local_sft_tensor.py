"""M9-B5 H01-H10 local CPU tensor handoff tests, without model training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from huyawo_adapt.agentic.training.local_sft_batch import (
    LocalSFTBatchDiagnostic,
    compile_local_sft_batch,
)
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    materialize_local_sft_tensors,
    validate_local_sft_tensors,
)
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _projection(index: int, ids: tuple[int, ...], start: int) -> AgentSFTDecisionProjection:
    target = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    fields: dict[str, Any] = {
        "trajectory_fingerprint": _hash(f"trajectory-{index}"),
        "step_index": 0,
        "step_provenance_sha256": _hash(f"step-{index}"),
        "model_fingerprint": _hash("model"),
        "tokenizer_fingerprint": _hash("tokenizer"),
        "harness_fingerprint": _hash("harness"),
        "tool_set_fingerprint": _hash("tools"),
        "environment_fingerprint": _hash("environment"),
        "verifier_fingerprint": _hash("verifier"),
        "task_fingerprint": _hash(f"task-{index}"),
        "runtime_fingerprint": _hash("runtime"),
        "rollout_policy_fingerprint": _hash("rollout"),
        "capture_provenance_sha256": _hash(f"capture-{index}"),
        "model_input_sha256": _hash(f"input-{index}"),
        "generated_output_sha256": _hash(f"output-{index}"),
        "projection_policy_sha256": _hash("projection-policy"),
        "input_ids": ids,
        "labels": (-100,) * start + ids[start:],
        "target_start": start,
        "target_end": len(ids),
        "target_token_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    }
    return AgentSFTDecisionProjection.model_validate(fields)


def _batch(
    sequences: tuple[tuple[int, ...], ...] = ((11, 12, 13, 14),),
    starts: tuple[int, ...] | None = None,
    pad: int = 0,
) -> LocalSFTBatchDiagnostic:
    if starts is None:
        starts = tuple(len(ids) - 1 for ids in sequences)
    rows = tuple(
        _projection(i, ids, start)
        for i, (ids, start) in enumerate(zip(sequences, starts, strict=True))
    )
    entries = tuple(
        AgentSFTSplitEntry(
            split="train",
            projection_fingerprint=row.fingerprint(),
            trajectory_fingerprint=row.trajectory_fingerprint,
            task_fingerprint=row.task_fingerprint,
            family_fingerprint=_hash(f"family-{i}"),
            root_trajectory_fingerprint=_hash(f"root-{i}"),
            parent_trajectory_fingerprint=row.parent_trajectory_fingerprint,
            source_content_sha256=_hash(f"content-{i}"),
        )
        for i, row in enumerate(rows)
    )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_hash("split-policy"), entries=entries)
    return compile_local_sft_batch(rows, manifest, "train", pad, 4096)


def _tensors(value: LocalSFTTensorDiagnostic) -> tuple[torch.Tensor, ...]:
    return value.input_ids, value.labels, value.attention_mask


def test_h01_exact_cpu_tensors_and_bound_source() -> None:
    source = _batch()
    result = materialize_local_sft_tensors(source)
    assert result.source_batch == source
    assert result.source_batch_sha256 == source.batch_sha256
    assert result.manifest_fingerprint == source.manifest_fingerprint
    assert result.projection_fingerprints == source.projection_fingerprints
    assert result.authority_scope == "local_sft_tensor_shape_only"
    for tensor, values in zip(
        _tensors(result), (source.input_ids, source.labels, source.attention_mask), strict=True
    ):
        assert tensor.tolist() == [list(row) for row in values]
        assert tensor.dtype == torch.int64
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
    validate_local_sft_tensors(result)


def test_h02_pad_valued_real_tokens_and_uneven_rows() -> None:
    source = _batch(((0, 4, 0), (8, 9, 10, 11)), (1, 2), pad=0)
    result = materialize_local_sft_tensors(source)
    assert result.input_ids.tolist() == [[0, 4, 0, 0], [8, 9, 10, 11]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert result.labels.tolist() == [[-100, 4, 0, -100], [-100, -100, 10, 11]]


def test_h03_ignored_prefix_padding_and_loss_tokens() -> None:
    source = _batch(((5, 6, 7), (8, 9, 10, 11)), (2, 3), pad=23)
    result = materialize_local_sft_tensors(source)
    assert result.total_loss_tokens == 2
    assert result.labels.tolist() == [[-100, -100, 7, -100], [-100, -100, -100, 11]]
    assert int((result.labels != -100).sum().item()) == source.total_loss_tokens
    validate_local_sft_tensors(result)


def test_h04_explicit_cpu_int64_no_grad_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.tensor
    devices: list[str] = []

    def checked_tensor(data: Any, **kwargs: Any) -> torch.Tensor:
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == torch.int64
        assert kwargs["requires_grad"] is False
        devices.append(kwargs["device"])
        return original(data, **kwargs)

    monkeypatch.setattr(torch, "tensor", checked_tensor)
    result = materialize_local_sft_tensors(_batch())
    assert devices == ["cpu", "cpu", "cpu"]
    for tensor in _tensors(result):
        assert tensor.device.type == "cpu"
        assert tensor.layout == torch.strided
        assert tensor.requires_grad is False


def test_h05_existing_b4_row_and_width_bounds_are_preserved() -> None:
    many = _batch(tuple((i + 1, i + 2) for i in range(32)))
    result = materialize_local_sft_tensors(many)
    assert result.input_ids.shape == (32, 2)
    long = materialize_local_sft_tensors(_batch((tuple(range(4096)),)))
    assert long.input_ids.shape == (1, 4096)
    assert len(_tensors(long)) == 3
    validate_local_sft_tensors(long)


def test_h06_no_storage_alias_across_columns_or_handoffs() -> None:
    source = _batch()
    left = materialize_local_sft_tensors(source)
    right = materialize_local_sft_tensors(source)
    pointers = [x.untyped_storage().data_ptr() for x in (*_tensors(left), *_tensors(right))]
    assert len(set(pointers)) == 6
    assert all(torch.equal(a, b) for a, b in zip(_tensors(left), _tensors(right), strict=True))
    left.input_ids[0, 0] += 1
    assert right.input_ids[0, 0].item() == source.input_ids[0][0]
    assert source == _batch()
    with pytest.raises(ValueError, match="content"):
        validate_local_sft_tensors(left)


def test_h07_each_in_place_tensor_change_is_rejected() -> None:
    for name in ("input_ids", "labels", "attention_mask"):
        result = materialize_local_sft_tensors(_batch())
        tensor = getattr(result, name)
        tensor[0, 0] += 1
        with pytest.raises(ValueError, match="content"):
            validate_local_sft_tensors(result)


def test_h08_source_metadata_digest_and_authority_spoof_fail_closed() -> None:
    source = _batch()
    with pytest.raises((ValidationError, ValueError)):
        materialize_local_sft_tensors(source.model_copy(update={"batch_sha256": "0" * 64}))
    with pytest.raises(TypeError):
        materialize_local_sft_tensors(object())  # type: ignore[arg-type]
    result = materialize_local_sft_tensors(source)
    for name, forged in (
        ("source_batch_sha256", "0" * 64),
        ("manifest_fingerprint", "0" * 64),
        ("tokenizer_fingerprint", "0" * 64),
        ("training_eligible", True),
        ("model_origin_attested", True),
        ("independent_task_outcome_verified", True),
        ("tokenizer_fidelity_attested", True),
        ("authority_scope", "training_eligible"),
    ):
        fake = replace(result)
        object.__setattr__(fake, name, forged)
        with pytest.raises(ValueError):
            validate_local_sft_tensors(fake)


def test_h09_invalid_dtype_shape_layout_and_alias_are_rejected() -> None:
    good = materialize_local_sft_tensors(_batch(((1, 2, 3, 4), (5, 6, 7, 8))))
    bad_values = (
        torch.zeros((2, 4), dtype=torch.float32),
        torch.zeros((2, 5), dtype=torch.int64),
        torch.zeros((2, 8), dtype=torch.int64)[:, ::2],
        good.labels,
    )
    for bad in bad_values:
        invalid = replace(good, input_ids=bad)
        with pytest.raises((TypeError, ValueError)):
            validate_local_sft_tensors(invalid)
    with pytest.raises(TypeError):
        validate_local_sft_tensors(object())  # type: ignore[arg-type]


def test_h10_b4_scope_stays_false_no_training_or_model_use() -> None:
    source = _batch()
    result = materialize_local_sft_tensors(source)
    assert source.authority_scope == "local_sft_batch_shape_only"
    assert result.authority_scope == "local_sft_tensor_shape_only"
    for name in (
        "independent_task_outcome_verified",
        "model_origin_attested",
        "tokenizer_fidelity_attested",
        "training_eligible",
    ):
        assert getattr(source, name) is False
        assert getattr(result, name) is False
    assert result.total_loss_tokens == source.total_loss_tokens
    validate_local_sft_tensors(result)
