"""S01-S10 bounded local synthetic state continuity checks, not real training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from huyawo_adapt.agentic.training import local_sft_tiny_state_chain as chain
from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_tensor import materialize_local_sft_tensors
from huyawo_adapt.agentic.training.local_sft_tiny_model import _causal_logits
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _handoff(batch_number: int, model_suffix: str = "same"):
    sequences = (
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (2, 0, 3, 4),
    )
    starts = (2, 3, 2)
    projections = []
    entries = []
    for row_index, (ids, start) in enumerate(zip(sequences, starts, strict=True)):
        identity = f"b10-{batch_number}-{row_index}"
        target = json.dumps(
            {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        projection = AgentSFTDecisionProjection(
            trajectory_fingerprint=_sha(f"trajectory-{identity}"),
            step_index=0,
            step_provenance_sha256=_sha(f"step-{identity}"),
            model_fingerprint=_sha(f"model-{model_suffix}"),
            tokenizer_fingerprint=_sha("tokenizer-b10"),
            harness_fingerprint=_sha("harness-b10"),
            tool_set_fingerprint=_sha("tools-b10"),
            environment_fingerprint=_sha("environment-b10"),
            verifier_fingerprint=_sha("verifier-b10"),
            task_fingerprint=_sha(f"task-{identity}"),
            runtime_fingerprint=_sha("runtime-b10"),
            rollout_policy_fingerprint=_sha("rollout-b10"),
            capture_provenance_sha256=_sha(f"capture-{identity}"),
            model_input_sha256=_sha(f"input-{identity}"),
            generated_output_sha256=_sha(f"output-{identity}"),
            projection_policy_sha256=_sha("policy-b10"),
            input_ids=ids,
            labels=(-100,) * start + ids[start:],
            target_start=start,
            target_end=len(ids),
            target_token_sha256=hashlib.sha256(target.encode("utf-8")).hexdigest(),
        )
        projections.append(projection)
        entries.append(
            AgentSFTSplitEntry(
                split="train",
                projection_fingerprint=projection.fingerprint(),
                trajectory_fingerprint=projection.trajectory_fingerprint,
                task_fingerprint=projection.task_fingerprint,
                family_fingerprint=_sha(f"family-{identity}"),
                root_trajectory_fingerprint=_sha(f"root-{identity}"),
                parent_trajectory_fingerprint=projection.parent_trajectory_fingerprint,
                source_content_sha256=_sha(f"content-{identity}"),
            )
        )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_sha("split-b10"), entries=tuple(entries))
    batch = compile_local_sft_batch(
        tuple(projections), manifest, "train", pad_token_id=0, max_sequence_length=16
    )
    return materialize_local_sft_tensors(batch)


def _handoffs(n: int = 3):
    return tuple(_handoff(index) for index in range(n))


def _audit(handoffs=None, resume: int = 1, **changes):
    if handoffs is None:
        handoffs = _handoffs()
    options = dict(
        synthetic_vocabulary_size=16,
        embedding_dim=4,
        seed=184,
        learning_rate=0.03,
        microbatch_rows=2,
        resume_after_step=resume,
    )
    options.update(changes)
    return chain.audit_local_sft_synthetic_state_chain(handoffs, **options)


def test_s01_two_ordered_updates_both_weight_families_change() -> None:
    handoffs = _handoffs(2)
    result = _audit(handoffs)
    assert result.total_steps == 2
    assert result.ordered_source_batch_sha256 == tuple(h.source_batch_sha256 for h in handoffs)
    assert result.initial_embedding_sha256 != result.final_embedding_sha256
    assert result.initial_projection_sha256 != result.final_projection_sha256
    assert len(result.mean_nll_after) == 2


def test_s02_distinct_four_batches_and_cross_identity_reject() -> None:
    result = _audit(_handoffs(4), resume=2)
    assert result.total_steps == 4
    with pytest.raises(ValueError, match="duplicate"):
        _audit((_handoff(0), _handoff(0)))
    with pytest.raises(ValueError, match="mixed"):
        _audit((_handoff(0), _handoff(1, model_suffix="other")))


def test_s03_unequal_token_counts_and_full_batch_parity() -> None:
    result = _audit()
    assert result.microbatch_loss_tokens == ((3, 2), (3, 2), (3, 2))
    assert max(result.max_gradient_parity_error) < 3e-6
    assert max(result.max_weight_parity_error) < 3e-6
    assert all(v >= 0 for v in result.mean_nll_before)


def test_s04_resumed_and_uninterrupted_reference_final_match() -> None:
    earlier = _audit(resume=1)
    later = _audit(resume=2)
    assert earlier.final_embedding_sha256 == later.final_embedding_sha256
    assert earlier.final_projection_sha256 == later.final_projection_sha256
    assert earlier.intermediate_state_sha256 != later.intermediate_state_sha256


def test_s05_corrupted_state_and_stale_binding_reject() -> None:
    embed = torch.arange(64, dtype=torch.float32).reshape(16, 4).contiguous() / 101
    projection = torch.arange(64, dtype=torch.float32).reshape(16, 4).contiguous() / 77
    bind = _sha("prefix")
    sealed = chain.seal_local_synthetic_state(embed, projection, 1, bind)
    restored = chain.restore_local_synthetic_state(sealed, 1, bind, 16, 4)
    assert torch.equal(restored[0].detach(), embed)
    assert restored[0].untyped_storage().data_ptr() != embed.untyped_storage().data_ptr()
    bad = (
        replace(sealed, payload=sealed.payload[:-1]),
        replace(sealed, payload=sealed.payload[:-1] + b"x"),
        replace(sealed, payload_sha256="0" * 64),
        replace(sealed, version=2),
        replace(sealed, completed_steps=2),
        replace(sealed, vocabulary_size=17),
        replace(sealed, dtype="float64"),
        replace(sealed, prefix_sha256=_sha("wrong")),
    )
    for candidate in bad:
        with pytest.raises(ValueError):
            chain.restore_local_synthetic_state(candidate, 1, bind, 16, 4)
    with pytest.raises(ValueError):
        chain.restore_local_synthetic_state(sealed, 2, bind, 16, 4)
    with pytest.raises(ValueError):
        chain.restore_local_synthetic_state(sealed, 1, _sha("other"), 16, 4)


def test_s06_local_digest_is_not_attestation_and_flags_are_immutable() -> None:
    embed = torch.ones((16, 4), dtype=torch.float32)
    projection = torch.ones((16, 4), dtype=torch.float32)
    bind = _sha("prefix")
    old = chain.seal_local_synthetic_state(embed, projection, 1, bind)
    changed = chain.seal_local_synthetic_state(embed + 1, projection, 1, bind)
    assert old.payload_sha256 != changed.payload_sha256
    chain.restore_local_synthetic_state(changed, 1, bind, 16, 4)
    result = _audit()
    assert result.training_eligible is False
    assert result.independent_task_outcome_verified is False
    assert result.model_origin_attested is False
    with pytest.raises(ValueError, match="authority"):
        replace(result, training_eligible=True)


def test_s07_handoffs_immutable_and_global_rng_untouched() -> None:
    handoffs = _handoffs()
    snapshots = tuple(
        tuple(value.detach().clone() for value in (h.input_ids, h.labels, h.attention_mask))
        for h in handoffs
    )
    rng = torch.get_rng_state().clone()
    _audit(handoffs)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(actual, old)
        for h, group in zip(handoffs, snapshots, strict=True)
        for actual, old in zip((h.input_ids, h.labels, h.attention_mask), group, strict=True)
    )


def test_s08_causal_prefix_pad_like_token_and_masked_tail() -> None:
    handoffs = _handoffs()
    result = _audit(handoffs)
    assert result.total_steps == 3
    source = torch.tensor([[2, 0, 3, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
    embed = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 40
    proj = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 101
    future = source.clone()
    future[0, 2] = 7
    ignored = source.clone()
    ignored[0, 3] = 8
    middle = source.clone()
    middle[0, 1] = 8
    baseline = _causal_logits(source, mask, embed, proj)
    assert torch.equal(baseline[:, :2], _causal_logits(future, mask, embed, proj)[:, :2])
    assert torch.equal(baseline, _causal_logits(ignored, mask, embed, proj))
    assert not torch.equal(baseline[:, 1], _causal_logits(middle, mask, embed, proj)[:, 1])


def test_s09_invalid_bounds_nonfinite_state_and_interruption(monkeypatch) -> None:
    for mutation in (
        {"seed": True},
        {"seed": -1},
        {"embedding_dim": 17},
        {"synthetic_vocabulary_size": 4},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.06},
        {"microbatch_rows": 3},
        {"resume_after_step": 3},
    ):
        with pytest.raises((ValueError, TypeError)):
            _audit(**mutation)
    embed = torch.ones((16, 4), dtype=torch.float32)
    embed[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        chain.seal_local_synthetic_state(embed, torch.ones_like(embed), 1, _sha("prefix"))

    def fail_restore(*args, **kwargs):
        raise RuntimeError("simulated local state interruption")

    monkeypatch.setattr(chain, "restore_local_synthetic_state", fail_restore)
    with pytest.raises(RuntimeError, match="interruption"):
        _audit()


def test_s10_no_checkpoint_run_id_or_real_training_authority() -> None:
    result = _audit()
    assert result.authority_scope == "local_sft_synthetic_state_chain_only"
    assert result.training_eligible is False
    assert result.tokenizer_fidelity_attested is False
    assert not hasattr(result, "checkpoint")
    assert not hasattr(result, "run_id")
    assert not hasattr(result, "model")
