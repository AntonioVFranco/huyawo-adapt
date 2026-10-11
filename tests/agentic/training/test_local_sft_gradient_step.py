"""M9-B7 G01-G10 fake-only CPU autograd and single-step SGD qualification candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_gradient_step import (
    LocalSFTSyntheticGradientStepDiagnostic,
    audit_local_sft_synthetic_gradient_step,
)
from huyawo_adapt.agentic.training.local_sft_loss_audit import audit_local_sft_causal_loss
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
    target = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    values: dict[str, Any] = {
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
    return AgentSFTDecisionProjection.model_validate(values)


def _handoff(
    sequences: tuple[tuple[int, ...], ...] = ((1, 2, 3, 4),),
    starts: tuple[int, ...] | None = None,
    pad: int = 0,
) -> LocalSFTTensorDiagnostic:
    if starts is None:
        starts = tuple(len(row) - 1 for row in sequences)
    projections = tuple(
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
        for i, row in enumerate(projections)
    )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_hash("split-policy"), entries=entries)
    batch = compile_local_sft_batch(projections, manifest, "train", pad, 4096)
    return materialize_local_sft_tensors(batch)


def _logits(handoff: LocalSFTTensorDiagnostic, vocab: int = 16) -> torch.Tensor:
    rows, width = handoff.input_ids.shape
    return torch.zeros((rows, width, vocab), dtype=torch.float32, device="cpu")


def _gradient_capture(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []
    original = torch.optim.SGD.step

    def checked_step(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        assert len(self.param_groups) == 1
        assert self.param_groups[0]["momentum"] == 0.0
        assert self.param_groups[0]["weight_decay"] == 0.0
        assert len(self.param_groups[0]["params"]) == 1
        param = self.param_groups[0]["params"][0]
        assert param.is_leaf
        assert param.device.type == "cpu"
        assert param.grad is not None
        captured.append(param.grad.detach().clone())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", checked_step)
    return captured


def test_g01_bound_identity_and_forged_authority_rejected() -> None:
    handoff = _handoff()
    result = audit_local_sft_synthetic_gradient_step(handoff, _logits(handoff), 0.05)
    assert result.source_batch_sha256 == handoff.source_batch_sha256
    assert result.projection_fingerprints == handoff.projection_fingerprints
    assert result.manifest_fingerprint == handoff.manifest_fingerprint
    assert result.tokenizer_fingerprint == handoff.tokenizer_fingerprint
    forged = replace(handoff)
    object.__setattr__(forged, "model_origin_attested", True)
    with pytest.raises(ValueError, match="authority"):
        audit_local_sft_synthetic_gradient_step(forged, _logits(handoff), 0.05)
    forged_identity = replace(handoff, source_batch_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        audit_local_sft_synthetic_gradient_step(forged_identity, _logits(handoff), 0.05)


def test_g02_exact_t_minus_one_loss_matches_published_b6() -> None:
    handoff = _handoff(((1, 2, 3),), (2,))
    logits = _logits(handoff, 8)
    logits[0, 1, 3] = 3.0
    oracle = audit_local_sft_causal_loss(handoff, logits)
    result = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.1)
    wrong_position = logits.clone()
    wrong_position[0, 1, 3] = 0.0
    wrong_position[0, 2, 3] = 3.0
    wrong = audit_local_sft_synthetic_gradient_step(handoff, wrong_position, 0.1)
    assert result.pre_step_mean_nll == pytest.approx(oracle.mean_nll, abs=2e-6)
    assert result.pre_step_mean_nll < wrong.pre_step_mean_nll
    assert result.total_loss_tokens == 1


def test_g03_gradient_matches_independent_scalar_softmax_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff(((1, 2, 3, 4),), (2,))
    logits = _logits(handoff, 8)
    logits[0, 1, 3] = 0.8
    logits[0, 2, 4] = -0.4
    captured = _gradient_capture(monkeypatch)
    audit_local_sft_synthetic_gradient_step(handoff, logits, 0.05)
    assert len(captured) == 1
    gradient = captured[0]
    for time_index, target in ((1, 3), (2, 4)):
        scores = logits[0, time_index].tolist()
        peak = max(scores)
        exps = [math.exp(score - peak) for score in scores]
        probabilities = [value / math.fsum(exps) for value in exps]
        for token in range(8):
            expected = (probabilities[token] - (1.0 if token == target else 0.0)) / 2.0
            assert gradient[0, time_index, token].item() == pytest.approx(expected, abs=2e-6)


def test_g04_prompt_padding_last_and_real_pad_target_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff(((1, 2, 0), (3, 4, 5, 6, 7)), (2, 3), pad=0)
    logits = _logits(handoff, 8)
    captured = _gradient_capture(monkeypatch)
    result = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.1)
    gradient = captured[0]
    assert handoff.attention_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert result.row_loss_tokens == (1, 2)
    assert result.ignored_gradient_positions == 7
    assert torch.count_nonzero(gradient[0, 0]).item() == 0
    assert torch.count_nonzero(gradient[0, 2:]).item() == 0
    assert torch.count_nonzero(gradient[1, :2]).item() == 0
    assert torch.count_nonzero(gradient[1, 4]).item() == 0
    assert gradient[0, 1, 0].item() < 0.0


def test_g05_unequal_rows_use_token_weighted_not_row_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff(((1, 2, 3), (4, 5, 6, 7)), (1, 3))
    logits = _logits(handoff, 8)
    logits[0, 0, 2] = 2.0
    logits[0, 1, 3] = 1.0
    logits[1, 2, 7] = -1.0
    captured = _gradient_capture(monkeypatch)
    result = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.05)
    assert result.row_loss_tokens == (2, 1)
    assert result.total_loss_tokens == 3
    assert len(captured) == 1
    assert captured[0][1, 2, 7].item() == pytest.approx(
        (float(torch.softmax(logits[1, 2], dim=-1)[7].item()) - 1.0) / 3.0,
        abs=2e-6,
    )
    assert result.pre_step_mean_nll == pytest.approx(
        audit_local_sft_causal_loss(handoff, logits).mean_nll, abs=2e-6
    )


def test_g06_exactly_one_private_sgd_step_reduces_fake_loss_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff(((1, 2, 3, 4),), (2,))
    logits = _logits(handoff, 8)
    original = logits.clone()
    original_ids = handoff.input_ids.clone()
    original_labels = handoff.labels.clone()
    original_mask = handoff.attention_mask.clone()
    captured = _gradient_capture(monkeypatch)
    result = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.1)
    assert len(captured) == 1
    assert result.changed_logit_cells > 0
    assert result.post_step_mean_nll < result.pre_step_mean_nll
    assert torch.equal(logits, original)
    assert torch.equal(handoff.input_ids, original_ids)
    assert torch.equal(handoff.labels, original_labels)
    assert torch.equal(handoff.attention_mask, original_mask)


def test_g07_reject_invalid_types_device_layout_grad_nonfinite_and_lr() -> None:
    handoff = _handoff()
    logits = _logits(handoff)
    bad_values = (
        logits.to(torch.float64),
        logits[:, :-1],
        logits.detach().clone().requires_grad_(True),
        logits.transpose(1, 2),
        logits.to_sparse(),
        torch.empty(logits.shape, device="meta"),
        logits.clone().fill_(float("nan")),
        logits.clone().fill_(float("inf")),
    )
    for value in bad_values:
        with pytest.raises((ValueError, TypeError)):
            audit_local_sft_synthetic_gradient_step(handoff, value, 0.05)
    for bad in (0.0, -0.1, 0.101, float("nan"), float("inf"), True, 1):
        with pytest.raises(ValueError):
            audit_local_sft_synthetic_gradient_step(handoff, logits, bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        audit_local_sft_synthetic_gradient_step(handoff, object(), 0.01)  # type: ignore[arg-type]


def test_g08_vocabulary_cell_cap_and_token_mismatch_fail_closed() -> None:
    handoff = _handoff()
    for vocab in (1, 2, 257):
        with pytest.raises(ValueError):
            audit_local_sft_synthetic_gradient_step(handoff, _logits(handoff, vocab), 0.05)
    huge = torch.empty((32, 4096, 8), device="meta")
    with pytest.raises(ValueError, match="cell cap"):
        audit_local_sft_synthetic_gradient_step(handoff, huge, 0.05)
    shifted = _logits(handoff)[:, :, :16].transpose(1, 2)
    with pytest.raises(ValueError):
        audit_local_sft_synthetic_gradient_step(handoff, shifted, 0.05)


def test_g09_mutated_source_or_nonfinite_optimizer_output_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field in ("input_ids", "labels", "attention_mask"):
        handoff = _handoff()
        getattr(handoff, field)[0, 0] += 1
        with pytest.raises(ValueError, match="content"):
            audit_local_sft_synthetic_gradient_step(handoff, _logits(handoff), 0.05)
    good = _handoff()
    original_step = torch.optim.SGD.step

    def corrupt_private_parameter(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        outcome = original_step(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0][0, 0, 0] = float("nan")
        return outcome

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt_private_parameter)
    with pytest.raises(ValueError, match="nonfinite synthetic poststep"):
        audit_local_sft_synthetic_gradient_step(good, _logits(good), 0.05)


def test_g10_nonpromoting_flags_and_unchanged_b6_oracle() -> None:
    handoff = _handoff()
    logits = _logits(handoff)
    prior = audit_local_sft_causal_loss(handoff, logits)
    result = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.05)
    after = audit_local_sft_causal_loss(handoff, logits)
    assert isinstance(result, LocalSFTSyntheticGradientStepDiagnostic)
    assert result.authority_scope == "local_sft_synthetic_gradient_step_only"
    assert result.independent_task_outcome_verified is False
    assert result.model_origin_attested is False
    assert result.tokenizer_fidelity_attested is False
    assert result.training_eligible is False
    assert prior.mean_nll == after.mean_nll
    with pytest.raises(ValueError, match="positive training authority"):
        replace(result, training_eligible=True)
