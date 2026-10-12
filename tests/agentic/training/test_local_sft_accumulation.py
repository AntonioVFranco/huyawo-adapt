"""M9-B8 A01-A10 bounded synthetic CPU accumulation qualification candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from huyawo_adapt.agentic.training.local_sft_accumulation import (
    LocalSFTSyntheticAccumulationDiagnostic,
    audit_local_sft_synthetic_accumulation,
)
from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_gradient_step import (
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _projection(index: int, ids: tuple[int, ...], start: int) -> AgentSFTDecisionProjection:
    target = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    data: dict[str, Any] = {
        "trajectory_fingerprint": _sha(f"trajectory-{index}"),
        "step_index": 0,
        "step_provenance_sha256": _sha(f"step-{index}"),
        "model_fingerprint": _sha("model"),
        "tokenizer_fingerprint": _sha("tokenizer"),
        "harness_fingerprint": _sha("harness"),
        "tool_set_fingerprint": _sha("tools"),
        "environment_fingerprint": _sha("environment"),
        "verifier_fingerprint": _sha("verifier"),
        "task_fingerprint": _sha(f"task-{index}"),
        "runtime_fingerprint": _sha("runtime"),
        "rollout_policy_fingerprint": _sha("rollout"),
        "capture_provenance_sha256": _sha(f"capture-{index}"),
        "model_input_sha256": _sha(f"input-{index}"),
        "generated_output_sha256": _sha(f"output-{index}"),
        "projection_policy_sha256": _sha("projection-policy"),
        "input_ids": ids,
        "labels": (-100,) * start + ids[start:],
        "target_start": start,
        "target_end": len(ids),
        "target_token_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    }
    return AgentSFTDecisionProjection.model_validate(data)


def _handoff(
    sequences: tuple[tuple[int, ...], ...] = ((1, 2, 3), (4, 5, 6, 7)),
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
            projection_fingerprint=projection.fingerprint(),
            trajectory_fingerprint=projection.trajectory_fingerprint,
            task_fingerprint=projection.task_fingerprint,
            family_fingerprint=_sha(f"family-{i}"),
            root_trajectory_fingerprint=_sha(f"root-{i}"),
            parent_trajectory_fingerprint=projection.parent_trajectory_fingerprint,
            source_content_sha256=_sha(f"content-{i}"),
        )
        for i, projection in enumerate(projections)
    )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_sha("split-policy"), entries=entries)
    return materialize_local_sft_tensors(
        compile_local_sft_batch(projections, manifest, "train", pad, 4096)
    )


def _logits(handoff: LocalSFTTensorDiagnostic, vocab: int = 16) -> torch.Tensor:
    rows, width = handoff.input_ids.shape
    return torch.zeros((rows, width, vocab), dtype=torch.float32, device="cpu")


def test_a01_exact_b5_identity_and_positive_authority_rejected() -> None:
    handoff = _handoff()
    result = audit_local_sft_synthetic_accumulation(handoff, _logits(handoff), 0.05, 1)
    assert type(result) is LocalSFTSyntheticAccumulationDiagnostic
    assert result.source_batch_sha256 == handoff.source_batch_sha256
    assert result.projection_fingerprints == handoff.projection_fingerprints
    assert result.manifest_fingerprint == handoff.manifest_fingerprint
    forged = replace(handoff)
    object.__setattr__(forged, "model_origin_attested", True)
    with pytest.raises(ValueError, match="authority"):
        audit_local_sft_synthetic_accumulation(forged, _logits(handoff), 0.05, 1)
    stale = replace(handoff, source_batch_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        audit_local_sft_synthetic_accumulation(stale, _logits(handoff), 0.05, 1)


def test_a02_global_token_weighted_loss_matches_b6_oracle() -> None:
    handoff = _handoff(((1, 2, 3, 4), (1, 3, 2, 5)), (1, 3))
    logits = _logits(handoff)
    logits[0, 1, 3] = 2.5
    logits[0, 2, 4] = -1.5
    logits[1, 2, 5] = 1.0
    oracle = audit_local_sft_causal_loss(handoff, logits)
    logits_copy = logits.clone()
    result = audit_local_sft_synthetic_accumulation(handoff, logits, 0.05, 1)
    assert result.microbatch_loss_tokens == (3, 1)
    assert result.total_loss_tokens == 4
    assert result.pre_step_mean_nll == pytest.approx(oracle.mean_nll, abs=2e-6)
    assert torch.equal(logits, logits_copy)


def test_a03_accumulated_gradient_equals_independent_full_batch_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff(((1, 2, 3), (4, 5, 6, 7)), (1, 3))
    logits = _logits(handoff)
    logits[0, 1, 3] = 1.0
    captured: list[torch.Tensor] = []
    original_step = torch.optim.SGD.step

    def capture(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        param = self.param_groups[0]["params"][0]
        assert param.grad is not None
        captured.append(param.grad.detach().clone())
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", capture)
    result = audit_local_sft_synthetic_accumulation(handoff, logits, 0.04, 1)
    assert len(captured) == 1
    reference = logits.clone().requires_grad_(True)
    oracle = (
        torch.nn.functional.cross_entropy(
            reference[:, :-1, :].reshape(-1, reference.shape[-1]),
            handoff.labels[:, 1:].reshape(-1),
            reduction="sum",
            ignore_index=-100,
        )
        / handoff.total_loss_tokens
    )
    oracle.backward()
    assert torch.allclose(captured[0], reference.grad, atol=2e-6, rtol=2e-5)
    assert result.max_gradient_parity_abs_error <= 2e-6


def test_a04_ignored_prefix_padding_last_and_real_pad_token() -> None:
    handoff = _handoff(((1, 0, 3), (4, 5, 6, 7, 8)), (2, 3), pad=0)
    logits = _logits(handoff)
    result = audit_local_sft_synthetic_accumulation(handoff, logits, 0.05, 1)
    assert handoff.input_ids[0, 1].item() == 0
    assert handoff.attention_mask[0, 1].item() == 1
    assert handoff.attention_mask[0, 3].item() == 0
    assert result.ignored_gradient_positions == int(
        (handoff.labels[:, 1:] == -100).sum().item() + handoff.labels.shape[0]
    )


def test_a05_irregular_three_microbatch_partition() -> None:
    handoff = _handoff(
        ((1, 2, 3), (1, 2, 3, 4), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3)),
        (2, 2, 2, 1, 1),
    )
    result = audit_local_sft_synthetic_accumulation(handoff, _logits(handoff), 0.02, 2)
    assert result.partition_row_counts == (2, 2, 1)
    assert result.microbatch_loss_tokens == (3, 4, 2)
    assert result.total_loss_tokens == 9
    assert result.max_gradient_parity_abs_error <= 2e-6


def test_a06_exactly_one_sgd_step_fullbatch_parameter_parity_and_immutability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff()
    logits = _logits(handoff)
    logits[0, 1, 3] = 0.2
    untouched = logits.clone()
    snapshot = tuple(t.clone() for t in (handoff.input_ids, handoff.labels, handoff.attention_mask))
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    original_step = torch.optim.SGD.step

    def capture(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        assert group["momentum"] == 0.0 and group["weight_decay"] == 0.0
        assert len(group["params"]) == 1
        parameter = group["params"][0]
        assert parameter.is_leaf and parameter.device.type == "cpu"
        before = parameter.detach().clone()
        grad = parameter.grad
        assert grad is not None
        result = original_step(self, *args, **kwargs)
        calls.append(
            (before, parameter.detach().clone(), grad.detach().clone(), float(group["lr"]))
        )
        return result

    monkeypatch.setattr(torch.optim.SGD, "step", capture)
    result = audit_local_sft_synthetic_accumulation(handoff, logits, 0.05, 1)
    assert len(calls) == 1
    assert calls[0][3] == 0.05
    assert torch.allclose(calls[0][1], calls[0][0] - 0.05 * calls[0][2], atol=2e-6)
    assert result.max_step_parity_abs_error <= 2e-6
    assert result.changed_logit_cells > 0
    assert result.post_step_mean_nll < result.pre_step_mean_nll
    assert torch.equal(logits, untouched)
    assert all(
        torch.equal(t, old)
        for t, old in zip(
            (handoff.input_ids, handoff.labels, handoff.attention_mask), snapshot, strict=True
        )
    )
    assert logits.grad is None


def test_a07_invalid_logits_and_learning_rate_fail_closed() -> None:
    handoff = _handoff()
    good = _logits(handoff)
    wrong_inputs = (
        good.to(torch.float64),
        good.detach().clone().requires_grad_(True),
        good.transpose(0, 1),
        good[:, :, ::2],
        good[:, :, :4],
        good.to_sparse(),
        handoff.input_ids.view(torch.float32).reshape(
            handoff.input_ids.shape[0], handoff.input_ids.shape[1], 2
        ),
        torch.empty(good.shape, device="meta"),
        torch.ones((2, 3, 16), dtype=torch.int64),
    )
    for bad in wrong_inputs:
        with pytest.raises((ValueError, TypeError)):
            audit_local_sft_synthetic_accumulation(handoff, bad, 0.05, 1)
    for rate in (0.0, -0.1, 0.101, math.inf, math.nan, True, 1):
        with pytest.raises((ValueError, TypeError)):
            audit_local_sft_synthetic_accumulation(handoff, good, rate, 1)
    nonfinite = good.clone()
    nonfinite[0, 0, 0] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        audit_local_sft_synthetic_accumulation(handoff, nonfinite, 0.05, 1)


def test_a08_reject_row_partition_vocabulary_and_cell_bounds() -> None:
    handoff = _handoff()
    for count in (0, 2, 3, True, 1.0, -1):
        with pytest.raises((ValueError, TypeError)):
            audit_local_sft_synthetic_accumulation(handoff, _logits(handoff), 0.05, count)
    for vocab in (1, 257):
        with pytest.raises(ValueError, match="vocabulary"):
            audit_local_sft_synthetic_accumulation(handoff, _logits(handoff, vocab), 0.05, 1)
    oversized = torch.empty((2, 600, 256), dtype=torch.float32)
    with pytest.raises(ValueError, match="cell cap"):
        audit_local_sft_synthetic_accumulation(handoff, oversized, 0.05, 1)
    huge_rows = torch.empty((33, 2, 16), dtype=torch.float32)
    with pytest.raises(ValueError, match="partition"):
        audit_local_sft_synthetic_accumulation(handoff, huge_rows, 0.05, 1)
    with pytest.raises(ValueError, match="vocabulary"):
        audit_local_sft_synthetic_accumulation(handoff, _logits(handoff, 2), 0.05, 1)


def test_a09_source_tamper_and_invalid_poststep_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff()
    handoff.input_ids[0, 0] = 12
    with pytest.raises(ValueError, match="content"):
        audit_local_sft_synthetic_accumulation(handoff, _logits(handoff), 0.05, 1)
    clean = _handoff()
    original_step = torch.optim.SGD.step

    def corrupt(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        output = original_step(self, *args, **kwargs)
        self.param_groups[0]["params"][0].data[0, 0, 0] = math.inf
        return output

    monkeypatch.setattr(torch.optim.SGD, "step", corrupt)
    with pytest.raises(ValueError, match="nonfinite synthetic private poststep"):
        audit_local_sft_synthetic_accumulation(clean, _logits(clean), 0.05, 1)


def test_a10_negative_only_authority_and_previous_b7_unchanged() -> None:
    handoff = _handoff()
    logits = _logits(handoff)
    b7 = audit_local_sft_synthetic_gradient_step(handoff, logits, 0.05)
    new = audit_local_sft_synthetic_accumulation(handoff, logits, 0.05, 1)
    assert new.authority_scope == "local_sft_synthetic_accumulation_only"
    assert new.independent_task_outcome_verified is False
    assert new.model_origin_attested is False
    assert new.tokenizer_fidelity_attested is False
    assert new.training_eligible is False
    assert b7.authority_scope == "local_sft_synthetic_gradient_step_only"
    assert new.pre_step_mean_nll == pytest.approx(b7.pre_step_mean_nll, abs=2e-6)
    assert new.post_step_mean_nll == pytest.approx(b7.post_step_mean_nll, abs=2e-6)
    with pytest.raises(ValueError, match="authority"):
        replace(new, training_eligible=True)
