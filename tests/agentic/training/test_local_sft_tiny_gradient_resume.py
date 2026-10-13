"""G01-G10 adversarial synthetic in-flight microbatch gradient resume tests."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace

import pytest
import torch

from huyawo_adapt.agentic.training import local_sft_tiny_gradient_resume as resume
from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_tensor import materialize_local_sft_tensors
from huyawo_adapt.agentic.training.local_sft_tiny_model import _causal_logits, _shifted_sum
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _handoff():
    sequences = ((1, 2, 3, 4), (5, 6, 7, 8), (2, 0, 3, 4), (3, 9, 4))
    starts = (2, 3, 2, 1)
    projections = []
    entries = []
    for index, (ids, start) in enumerate(zip(sequences, starts, strict=True)):
        identity = f"b11-{index}"
        token_bytes = json.dumps(
            {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        projection = AgentSFTDecisionProjection(
            trajectory_fingerprint=_sha(f"trajectory-{identity}"),
            step_index=0,
            step_provenance_sha256=_sha(f"step-{identity}"),
            model_fingerprint=_sha("model-b11"),
            tokenizer_fingerprint=_sha("tokenizer-b11"),
            harness_fingerprint=_sha("harness-b11"),
            tool_set_fingerprint=_sha("tools-b11"),
            environment_fingerprint=_sha("environment-b11"),
            verifier_fingerprint=_sha("verifier-b11"),
            task_fingerprint=_sha(f"task-{identity}"),
            runtime_fingerprint=_sha("runtime-b11"),
            rollout_policy_fingerprint=_sha("rollout-b11"),
            capture_provenance_sha256=_sha(f"capture-{identity}"),
            model_input_sha256=_sha(f"input-{identity}"),
            generated_output_sha256=_sha(f"output-{identity}"),
            projection_policy_sha256=_sha("policy-b11"),
            input_ids=ids,
            labels=(-100,) * start + ids[start:],
            target_start=start,
            target_end=len(ids),
            target_token_sha256=hashlib.sha256(token_bytes).hexdigest(),
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
    manifest = AgentSFTSplitManifest(split_policy_sha256=_sha("split-b11"), entries=tuple(entries))
    batch = compile_local_sft_batch(
        tuple(projections), manifest, "train", pad_token_id=0, max_sequence_length=16
    )
    return materialize_local_sft_tensors(batch)


def _audit(interrupt: int = 1, **changes):
    options = dict(
        synthetic_vocabulary_size=16,
        embedding_dim=4,
        seed=184,
        learning_rate=0.03,
        microbatch_rows=1,
        interrupt_after_parts=interrupt,
    )
    options.update(changes)
    return resume.audit_local_sft_synthetic_gradient_resume(_handoff(), **options)


def _sealed():
    handoff = _handoff()
    parts = ((0, 1), (1, 2), (2, 3), (3, 4))
    tokens = handoff.source_batch.row_target_lengths
    binding = resume._binding(handoff, 16, 4, 184, 0.03, parts, tokens)
    weights = torch.arange(64, dtype=torch.float32).reshape(16, 4).contiguous() / 101
    outputs = torch.arange(64, dtype=torch.float32).reshape(16, 4).contiguous() / 77
    embedding = weights.detach().clone().requires_grad_(True)
    projection = outputs.detach().clone().requires_grad_(True)
    embedding.grad = torch.ones_like(embedding) / 9
    projection.grad = torch.ones_like(projection) / 7
    state = resume.seal_local_synthetic_gradient_state(embedding, projection, 1, binding)
    return state, binding, (embedding, projection)


def test_g01_three_remaining_partitions_single_update() -> None:
    result = _audit(1)
    assert result.interrupted_after_parts == 1
    assert len(result.partition_boundaries) == 4
    assert result.initial_embedding_sha256 != result.final_embedding_sha256
    assert result.initial_projection_sha256 != result.final_projection_sha256
    assert result.completed_loss_tokens == result.partition_loss_tokens[0]


def test_g02_middle_boundaries_and_invalid_cursors() -> None:
    outcomes = (_audit(1), _audit(2), _audit(3))
    assert len({entry.intermediate_state_sha256 for entry in outcomes}) == 3
    assert len({entry.final_embedding_sha256 for entry in outcomes}) == 1
    assert len({entry.final_projection_sha256 for entry in outcomes}) == 1
    for cursor in (0, 4, -1, True):
        with pytest.raises(ValueError):
            _audit(cursor)
    state, binding, _ = _sealed()
    with pytest.raises(ValueError):
        resume.restore_local_synthetic_gradient_state(state, binding, 2)
    with pytest.raises(ValueError):
        resume.seal_local_synthetic_gradient_state(
            *_sealed()[2],
            1,
            replace(binding, partition_boundaries=((0, 1), (2, 3), (1, 2), (3, 4))),
        )


def test_g03_token_weighted_partial_gradients() -> None:
    result = _audit(2)
    assert result.partition_loss_tokens == (2, 1, 2, 2)
    assert result.total_loss_tokens == 7
    assert result.completed_loss_tokens == 3
    assert result.prefix_gradient_max_error < 3e-6
    assert result.partial_embedding_gradient_sha256 != result.initial_embedding_sha256


def test_g04_full_gradient_and_step_parity() -> None:
    result = _audit(1)
    assert result.resumed_microbatch_gradient_max_error < 3e-6
    assert result.resumed_fullbatch_gradient_max_error < 3e-6
    assert result.resumed_microbatch_step_max_error < 3e-6
    assert result.resumed_fullbatch_step_max_error < 3e-6
    assert result.mean_nll_after >= 0


def test_g05_invalid_state_bytes_schema_and_finite_values() -> None:
    state, binding, _ = _sealed()
    corrupted = (
        replace(state, payload=state.payload[:-1]),
        replace(state, payload=state.payload[:-4] + b"\x00\x00\x00\x00"),
        replace(state, payload_sha256="0" * 64),
        replace(state, version=2),
        replace(state, dtype="float64"),
        replace(state, byte_order="big"),
        replace(state, vocabulary_size=15),
        replace(state, embedding_dim=5),
        replace(state, completed_loss_tokens=5),
    )
    for candidate in corrupted:
        with pytest.raises(ValueError):
            resume.restore_local_synthetic_gradient_state(candidate, binding, 1)
    bad_float = bytearray(state.payload)
    gradient_byte_offset = 2 * 16 * 4 * 4
    bad_float[gradient_byte_offset : gradient_byte_offset + 4] = struct.pack("<f", float("nan"))
    bad = replace(state, payload=bytes(bad_float))
    bad = replace(bad, payload_sha256=resume._state_hash(bad))
    with pytest.raises(ValueError, match="nonfinite"):
        resume.restore_local_synthetic_gradient_state(bad, binding, 1)


def test_g06_identity_and_partition_binding_reject() -> None:
    state, binding, _ = _sealed()
    changed = (
        replace(binding, source_batch_sha256=_sha("stale-source")),
        replace(binding, operation_sha256=_sha("mixed-model-or-tokenizer")),
        replace(binding, seed=185),
        replace(binding, learning_rate=0.02),
        replace(binding, partition_loss_tokens=(1, 1, 2, 3)),
        replace(binding, partition_boundaries=((0, 2), (2, 3), (3, 4), (4, 5))),
    )
    for altered in changed:
        with pytest.raises(ValueError):
            resume.restore_local_synthetic_gradient_state(state, altered, 1)
    assert state.operation_sha256 != _sha("independent issuer")


def test_g07_no_input_or_rng_mutation_and_no_alias() -> None:
    handoff = _handoff()
    before = [
        tensor.clone() for tensor in (handoff.input_ids, handoff.labels, handoff.attention_mask)
    ]
    rng = torch.get_rng_state().clone()
    result = resume.audit_local_sft_synthetic_gradient_resume(handoff, 16, 4, 184, 0.03, 1, 1)
    assert result.training_eligible is False
    assert torch.equal(torch.get_rng_state(), rng)
    for old, new in zip(
        before, (handoff.input_ids, handoff.labels, handoff.attention_mask), strict=True
    ):
        assert torch.equal(old, new)
    state, binding, original = _sealed()
    weights = resume.restore_local_synthetic_gradient_state(state, binding, 1)
    assert all(
        a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
        for a, b in zip(weights, original, strict=True)
    )
    assert len({t.untyped_storage().data_ptr() for w in weights for t in (w, w.grad)}) == 4
    with pytest.raises(ValueError, match="aliased"):
        resume.seal_local_synthetic_gradient_state(original[0], original[0], 1, binding)


def test_g08_causal_prefix_padding_and_mask_invariance() -> None:
    handoff = _handoff()
    embedding = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 101
    projection = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 77
    first = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    changed = handoff.input_ids.clone()
    changed[0, 3] = 15
    future = _causal_logits(changed, handoff.attention_mask, embedding, projection)
    assert torch.equal(first[0, :3, :], future[0, :3, :])
    masked = handoff.input_ids.clone()
    masked[0, 3] = 0
    assert not torch.equal(
        first[0, 3:, :],
        _causal_logits(masked, handoff.attention_mask, embedding, projection)[0, 3:, :],
    )
    assert handoff.input_ids[2, 1].item() == 0
    assert handoff.attention_mask[2, 1].item() == 1
    assert handoff.attention_mask[3, 3].item() == 0
    ignored = handoff.input_ids.clone()
    ignored[3, 3] = 15
    assert torch.equal(
        first[3], _causal_logits(ignored, handoff.attention_mask, embedding, projection)[3]
    )
    assert torch.isfinite(_shifted_sum(first, handoff.labels))
    assert _audit(2).resumed_fullbatch_gradient_max_error < 3e-6


def test_g09_boundaries_fault_and_resume_fail_closed(monkeypatch) -> None:
    for changes in (
        dict(seed=True),
        dict(seed=2**32),
        dict(learning_rate=float("nan")),
        dict(learning_rate=0.5),
        dict(embedding_dim=17),
        dict(synthetic_vocabulary_size=257),
        dict(microbatch_rows=4),
    ):
        with pytest.raises(ValueError):
            _audit(1, **changes)

    def fail_resume(*args, **kwargs):
        raise RuntimeError("injected local interruption")

    monkeypatch.setattr(resume, "restore_local_synthetic_gradient_state", fail_resume)
    with pytest.raises(RuntimeError, match="injected"):
        _audit(1)


def test_g10_nonpromoting_local_diagnostics_no_checkpoints() -> None:
    result = _audit(1)
    assert result.authority_scope == "local_sft_synthetic_inflight_gradient_only"
    assert result.training_eligible is False
    assert result.model_origin_attested is False
    assert result.tokenizer_fidelity_attested is False
    assert result.independent_task_outcome_verified is False
    for field in ("run_id", "checkpoint", "model", "model_weights", "authentic_training"):
        assert not hasattr(result, field)
    with pytest.raises(ValueError):
        replace(result, training_eligible=True)
