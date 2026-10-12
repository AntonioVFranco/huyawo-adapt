"""M9-B9 P01-P10 fake CPU trainable-parameter contract tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    materialize_local_sft_tensors,
)
from huyawo_adapt.agentic.training.local_sft_tiny_model import (
    LocalSFTSyntheticTinyParameterDiagnostic,
    _causal_logits,
    audit_local_sft_synthetic_tiny_parameter_step,
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
        "trajectory_fingerprint": _sha(f"b9-trajectory-{index}"),
        "step_index": 0,
        "step_provenance_sha256": _sha(f"b9-step-{index}"),
        "model_fingerprint": _sha("b9-model"),
        "tokenizer_fingerprint": _sha("b9-tokenizer"),
        "harness_fingerprint": _sha("b9-harness"),
        "tool_set_fingerprint": _sha("b9-tools"),
        "environment_fingerprint": _sha("b9-environment"),
        "verifier_fingerprint": _sha("b9-verifier"),
        "task_fingerprint": _sha(f"b9-task-{index}"),
        "runtime_fingerprint": _sha("b9-runtime"),
        "rollout_policy_fingerprint": _sha("b9-rollout"),
        "capture_provenance_sha256": _sha(f"b9-capture-{index}"),
        "model_input_sha256": _sha(f"b9-input-{index}"),
        "generated_output_sha256": _sha(f"b9-output-{index}"),
        "projection_policy_sha256": _sha("b9-policy"),
        "input_ids": ids,
        "labels": (-100,) * start + ids[start:],
        "target_start": start,
        "target_end": len(ids),
        "target_token_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
    }
    return AgentSFTDecisionProjection.model_validate(data)


def _handoff(
    sequences: tuple[tuple[int, ...], ...] = ((1, 2, 3, 4), (5, 6, 7, 8)),
    starts: tuple[int, ...] | None = None,
    pad: int = 0,
) -> LocalSFTTensorDiagnostic:
    if starts is None:
        starts = tuple(len(row) - 1 for row in sequences)
    projections = tuple(
        _projection(index, ids, start)
        for index, (ids, start) in enumerate(zip(sequences, starts, strict=True))
    )
    entries = tuple(
        AgentSFTSplitEntry(
            split="train",
            projection_fingerprint=projection.fingerprint(),
            trajectory_fingerprint=projection.trajectory_fingerprint,
            task_fingerprint=projection.task_fingerprint,
            family_fingerprint=_sha(f"b9-family-{index}"),
            root_trajectory_fingerprint=_sha(f"b9-root-{index}"),
            parent_trajectory_fingerprint=projection.parent_trajectory_fingerprint,
            source_content_sha256=_sha(f"b9-content-{index}"),
        )
        for index, projection in enumerate(projections)
    )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_sha("b9-split"), entries=entries)
    return materialize_local_sft_tensors(
        compile_local_sft_batch(projections, manifest, "train", pad, 64)
    )


def _audit(
    handoff: LocalSFTTensorDiagnostic,
    *,
    vocab: int = 16,
    dim: int = 8,
    seed: int = 19,
    lr: float = 0.05,
    microbatch: int = 1,
) -> LocalSFTSyntheticTinyParameterDiagnostic:
    return audit_local_sft_synthetic_tiny_parameter_step(handoff, vocab, dim, seed, lr, microbatch)


def test_p01_exact_b5_binding_and_negative_authority() -> None:
    handoff = _handoff()
    result = _audit(handoff)
    assert type(result) is LocalSFTSyntheticTinyParameterDiagnostic
    assert result.source_batch_sha256 == handoff.source_batch_sha256
    assert result.projection_fingerprints == handoff.projection_fingerprints
    assert result.manifest_fingerprint == handoff.manifest_fingerprint
    assert result.model_fingerprint == handoff.model_fingerprint
    assert result.tokenizer_fingerprint == handoff.tokenizer_fingerprint
    assert result.projection_policy_sha256 == handoff.projection_policy_sha256
    forged = replace(handoff)
    object.__setattr__(forged, "training_eligible", True)
    with pytest.raises(ValueError, match="authority"):
        _audit(forged)


def test_p02_both_trainable_parameter_gradient_paths() -> None:
    result = _audit(_handoff(), seed=99)
    assert result.embedding_gradient_l2_norm > 0
    assert result.projection_gradient_l2_norm > 0
    assert result.changed_embedding_cells > 0
    assert result.changed_projection_cells > 0
    assert result.max_embedding_gradient_parity_abs_error < 3e-6
    assert result.max_projection_gradient_parity_abs_error < 3e-6


def test_p03_globally_weighted_unequal_row_token_counts() -> None:
    handoff = _handoff(((1, 2, 3, 4), (5, 6, 7, 8)), (1, 3))
    result = _audit(handoff, seed=21)
    assert result.microbatch_loss_tokens == (3, 1)
    assert result.total_loss_tokens == 4
    assert result.max_embedding_gradient_parity_abs_error <= 3e-6
    assert result.max_projection_gradient_parity_abs_error <= 3e-6
    assert result.pre_step_mean_nll > 0


def test_p04_irregular_three_microbatch_partition() -> None:
    handoff = _handoff(
        ((1, 2, 3), (4, 5, 6, 7), (7, 8, 9, 10), (1, 4, 6, 7, 8, 9), (2, 5, 8)),
        (1, 3, 2, 5, 1),
    )
    result = _audit(handoff, vocab=16, microbatch=2, seed=7)
    assert result.partition_row_counts == (2, 2, 1)
    assert result.microbatch_loss_tokens == (3, 3, 2)
    assert result.total_loss_tokens == 8
    assert result.max_embedding_step_parity_abs_error <= 3e-6
    assert result.max_projection_step_parity_abs_error <= 3e-6


def test_p05_causal_prefix_does_not_leak_future_tokens() -> None:
    embedding = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 100
    projection = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 111
    attention = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)
    original = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    changed_future = torch.tensor([[1, 2, 9, 4]], dtype=torch.int64)
    before = _causal_logits(original, attention, embedding, projection)
    after = _causal_logits(changed_future, attention, embedding, projection)
    assert torch.equal(before[:, :2], after[:, :2])
    assert not torch.equal(before[:, 2], after[:, 2])


def test_p06_real_pad_like_token_attended_and_padding_masked() -> None:
    handoff = _handoff(((2, 0, 3), (4, 5, 6, 7)), (2, 3), pad=0)
    result = _audit(handoff)
    assert result.total_loss_tokens == 2
    assert handoff.input_ids[0, 1].item() == 0
    assert handoff.attention_mask[0, 1].item() == 1
    assert handoff.attention_mask[0, 3].item() == 0
    embedding = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 40
    projection = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 101
    source = torch.tensor([[2, 0, 3, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
    attended_changed = source.clone()
    attended_changed[0, 1] = 8
    padded_changed = source.clone()
    padded_changed[0, 3] = 8
    baseline = _causal_logits(source, mask, embedding, projection)
    attended = _causal_logits(attended_changed, mask, embedding, projection)
    padded = _causal_logits(padded_changed, mask, embedding, projection)
    assert not torch.equal(baseline[:, 1:3], attended[:, 1:3])
    assert torch.equal(baseline, padded)


def test_p07_one_sgd_step_per_private_branch_and_source_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _handoff()
    copies = tuple(t.clone() for t in (handoff.input_ids, handoff.labels, handoff.attention_mask))
    original_step = torch.optim.SGD.step
    calls: list[int] = []

    def capture(self: torch.optim.SGD, *args: Any, **kwargs: Any) -> Any:
        params = self.param_groups[0]["params"]
        assert len(params) == 2
        assert all(isinstance(param, torch.Tensor) and param.grad is not None for param in params)
        calls.append(len(params))
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", capture)
    result = _audit(handoff)
    assert calls == [2, 2]
    assert result.accumulation_optimizer_steps == 1
    assert result.reference_optimizer_steps == 1
    assert result.max_embedding_step_parity_abs_error < 3e-6
    assert result.max_projection_step_parity_abs_error < 3e-6
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (handoff.input_ids, handoff.labels, handoff.attention_mask), copies, strict=True
        )
    )


def test_p08_repeatable_private_seed_and_untouched_global_rng() -> None:
    handoff = _handoff()
    original_rng = torch.get_rng_state().clone()
    first = _audit(handoff, seed=124)
    second = _audit(handoff, seed=124)
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert (first.initial_embedding_sha256, first.initial_projection_sha256) == (
        second.initial_embedding_sha256,
        second.initial_projection_sha256,
    )
    assert (first.updated_embedding_sha256, first.updated_projection_sha256) == (
        second.updated_embedding_sha256,
        second.updated_projection_sha256,
    )
    assert first.initial_embedding_sha256 != first.updated_embedding_sha256
    assert first.initial_projection_sha256 != first.updated_projection_sha256


def test_p09_fail_closed_types_bounds_and_stale_b5() -> None:
    handoff = _handoff()
    for invalid in (
        {"vocab": True},
        {"vocab": 1},
        {"vocab": 4},
        {"dim": True},
        {"dim": 17},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"lr": 0},
        {"lr": float("nan")},
        {"lr": 0.051},
        {"microbatch": 0},
        {"microbatch": 2},
    ):
        with pytest.raises((ValueError, TypeError)):
            _audit(handoff, **invalid)
    stale = replace(handoff, source_batch_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        _audit(stale)
    wrong_dtype = replace(handoff, input_ids=handoff.input_ids.to(torch.int32))
    with pytest.raises(ValueError, match="tensor"):
        _audit(wrong_dtype)
    handoff.input_ids[0, 0] = 10
    with pytest.raises(ValueError, match="content"):
        _audit(handoff)


def test_p10_only_local_synthetic_diagnostic_no_training_artifacts() -> None:
    result = _audit(_handoff())
    assert result.authority_scope == "local_sft_synthetic_tiny_parameter_only"
    assert result.independent_task_outcome_verified is False
    assert result.model_origin_attested is False
    assert result.tokenizer_fidelity_attested is False
    assert result.training_eligible is False
    assert not hasattr(result, "checkpoint")
    assert not hasattr(result, "run_id")
    with pytest.raises(ValueError, match="authority"):
        replace(result, training_eligible=True)
