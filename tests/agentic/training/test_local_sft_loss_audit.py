"""M9-B6 E01-E10 synthetic CPU causal loss acceptance candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from huyawo_adapt.agentic.training.local_sft_batch import (
    LocalSFTBatchDiagnostic,
    compile_local_sft_batch,
)
from huyawo_adapt.agentic.training.local_sft_loss_audit import (
    LocalSFTLossDiagnostic,
    audit_local_sft_causal_loss,
)
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    materialize_local_sft_tensors,
)
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _projection(index: int, ids: tuple[int, ...], start: int) -> AgentSFTDecisionProjection:
    payload = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
        allow_nan=False,
        ensure_ascii=False,
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
        "target_token_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    return AgentSFTDecisionProjection.model_validate(fields)


def _batch(
    sequences: tuple[tuple[int, ...], ...] = ((1, 2, 3, 4),),
    starts: tuple[int, ...] | None = None,
    pad: int = 0,
) -> LocalSFTBatchDiagnostic:
    if starts is None:
        starts = tuple(len(row) - 1 for row in sequences)
    rows = tuple(
        _projection(index, ids, start)
        for index, (ids, start) in enumerate(zip(sequences, starts, strict=True))
    )
    entries = tuple(
        AgentSFTSplitEntry(
            split="train",
            projection_fingerprint=row.fingerprint(),
            trajectory_fingerprint=row.trajectory_fingerprint,
            task_fingerprint=row.task_fingerprint,
            family_fingerprint=_hash(f"family-{index}"),
            root_trajectory_fingerprint=_hash(f"root-{index}"),
            parent_trajectory_fingerprint=row.parent_trajectory_fingerprint,
            source_content_sha256=_hash(f"content-{index}"),
        )
        for index, row in enumerate(rows)
    )
    manifest = AgentSFTSplitManifest(
        split_policy_sha256=_hash("split-policy"),
        entries=entries,
    )
    return compile_local_sft_batch(rows, manifest, "train", pad, 4096)


def _handoff(
    sequences: tuple[tuple[int, ...], ...] = ((1, 2, 3, 4),),
    starts: tuple[int, ...] | None = None,
    pad: int = 0,
) -> LocalSFTTensorDiagnostic:
    return materialize_local_sft_tensors(_batch(sequences, starts, pad))


def _logits(handoff: LocalSFTTensorDiagnostic, vocab: int = 16) -> torch.Tensor:
    rows, width = handoff.input_ids.shape
    return torch.zeros((rows, width, vocab), dtype=torch.float32, device="cpu")


def _scalar_nll(scores: list[float], target: int) -> float:
    peak = max(scores)
    stable_log_sum_exp = peak + math.log(math.fsum(math.exp(value - peak) for value in scores))
    return stable_log_sum_exp - scores[target]


def test_e01_exact_causal_t_minus_one_alignment() -> None:
    handoff = _handoff(((1, 2, 3),), (2,))
    correct = _logits(handoff, 8)
    correct[0, 1, 3] = 9.0
    wrong = _logits(handoff, 8)
    wrong[0, 2, 3] = 9.0
    assert audit_local_sft_causal_loss(handoff, correct).mean_nll < 0.01
    assert audit_local_sft_causal_loss(handoff, wrong).mean_nll == pytest.approx(math.log(8))


def test_e02_independent_scalar_log_sum_exp_multiple_targets() -> None:
    handoff = _handoff(((1, 2, 3, 4),), (2,))
    logits = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 11.0
    result = audit_local_sft_causal_loss(handoff, logits)
    expected = sum(
        _scalar_nll(logits[0, target - 1].tolist(), label) for target, label in ((2, 3), (3, 4))
    )
    assert result.total_loss_tokens == 2
    assert result.total_nll == pytest.approx(expected, abs=1e-6)
    assert result.mean_nll == pytest.approx(expected / 2, abs=1e-6)


def test_e03_ignored_prompt_and_padding_logits_do_not_change_loss() -> None:
    handoff = _handoff(((1, 2, 3), (4, 5, 6, 7, 8)), (2, 4))
    baseline = _logits(handoff)
    first = audit_local_sft_causal_loss(handoff, baseline)
    changed = baseline.clone()
    changed[0, 0, :] = 13.0
    changed[0, 2:, :] = -19.0
    changed[1, :3, :] = 27.0
    changed[1, 4, :] = -12.0
    second = audit_local_sft_causal_loss(handoff, changed)
    assert first.mean_nll == pytest.approx(second.mean_nll)


def test_e04_real_pad_valued_target_is_supervised() -> None:
    handoff = _handoff(((1, 2, 0),), (2,), pad=0)
    assert handoff.attention_mask.tolist() == [[1, 1, 1]]
    logits = _logits(handoff, 4)
    logits[0, 1, 0] = 11.0
    result = audit_local_sft_causal_loss(handoff, logits)
    assert result.row_loss_tokens == (1,)
    assert result.total_loss_tokens == 1
    assert result.mean_nll < 0.01


def test_e05_uneven_multirow_token_weighted_mean() -> None:
    handoff = _handoff(((1, 2, 3), (4, 5, 6, 7)), (1, 3))
    logits = _logits(handoff, 8)
    logits[0, 0, 2] = 3.0
    logits[0, 1, 3] = 2.0
    logits[1, 2, 7] = 5.0
    result = audit_local_sft_causal_loss(handoff, logits)
    first = sum(
        _scalar_nll(logits[0, target - 1].tolist(), label) for target, label in ((1, 2), (2, 3))
    )
    second = _scalar_nll(logits[1, 2].tolist(), 7)
    assert result.row_loss_tokens == (2, 1)
    assert result.per_row_nll == pytest.approx((first, second), abs=1e-6)
    assert result.total_nll == pytest.approx(first + second, abs=1e-6)
    assert result.mean_nll == pytest.approx((first + second) / 3, abs=1e-6)


@pytest.mark.parametrize("field", ("input_ids", "labels", "attention_mask"))
def test_e06_b5_mutation_rejected_before_loss(field: str) -> None:
    handoff = _handoff()
    tensor = getattr(handoff, field)
    tensor[0, 0] += 1
    with pytest.raises(ValueError, match="content"):
        audit_local_sft_causal_loss(handoff, _logits(handoff))


def test_e07_reject_invalid_dtype_shape_device_layout_grad_and_nonfinite() -> None:
    handoff = _handoff()
    good = _logits(handoff)
    invalid = (
        good.to(torch.float64),
        good[:, :-1, :],
        good.clone().requires_grad_(True),
        torch.zeros((16, 4, 1), dtype=torch.float32).permute(2, 1, 0),
        good.to_sparse(),
        torch.empty(good.shape, device="meta"),
        good.clone().fill_(float("nan")),
        good.clone().fill_(float("inf")),
    )
    for value in invalid:
        with pytest.raises((ValueError, TypeError)):
            audit_local_sft_causal_loss(handoff, value)


def test_e08_vocabulary_and_cell_cap_fail_closed() -> None:
    handoff = _handoff()
    for vocab in (1, 2, 257):
        with pytest.raises(ValueError, match="vocabulary"):
            audit_local_sft_causal_loss(handoff, _logits(handoff, vocab))
    huge_meta = torch.empty((32, 4096, 9), device="meta")
    with pytest.raises(ValueError, match="cell cap"):
        audit_local_sft_causal_loss(handoff, huge_meta)


def test_e09_negative_authority_and_source_identity() -> None:
    handoff = _handoff()
    result = audit_local_sft_causal_loss(handoff, _logits(handoff))
    assert result.source_batch_sha256 == handoff.source_batch_sha256
    assert result.manifest_fingerprint == handoff.manifest_fingerprint
    assert result.projection_fingerprints == handoff.projection_fingerprints
    assert result.model_fingerprint == handoff.model_fingerprint
    assert result.tokenizer_fingerprint == handoff.tokenizer_fingerprint
    assert result.authority_scope == "local_sft_synthetic_loss_only"
    assert result.independent_task_outcome_verified is False
    assert result.model_origin_attested is False
    assert result.tokenizer_fidelity_attested is False
    assert result.training_eligible is False
    with pytest.raises(ValueError, match="authority"):
        replace(result, training_eligible=True)
    with pytest.raises(ValueError, match="identity"):
        audit_local_sft_causal_loss(
            replace(handoff, source_batch_sha256="0" * 64),
            _logits(handoff),
        )


def test_e10_no_grad_numerical_oracle_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    handoff = _handoff(((1, 2, 3, 4),), (2,))
    logits = _logits(handoff, 8)
    logits[0, 1, 3] = 4.0
    logits[0, 2, 4] = -2.0
    observed: list[str] = []
    original = F.cross_entropy

    def checked_cross_entropy(*args: Any, **kwargs: Any) -> torch.Tensor:
        assert torch.is_grad_enabled() is False
        assert kwargs["ignore_index"] == -100
        assert kwargs["reduction"] == "none"
        observed.append("cpu_cross_entropy")
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", checked_cross_entropy)
    result = audit_local_sft_causal_loss(handoff, logits)
    scalar = _scalar_nll(logits[0, 1].tolist(), 3) + _scalar_nll(logits[0, 2].tolist(), 4)
    assert observed == ["cpu_cross_entropy"]
    assert result.total_nll == pytest.approx(scalar, abs=1e-6)
    assert result.total_loss_tokens == 2
    assert isinstance(result, LocalSFTLossDiagnostic)
