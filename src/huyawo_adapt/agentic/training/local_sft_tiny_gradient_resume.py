"""Bounded CPU-only in-flight toy gradient resume, without training authority."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, replace
from typing import Literal

import torch

from huyawo_adapt.agentic.training.local_sft_loss_audit import audit_local_sft_causal_loss
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    validate_local_sft_tensors,
)
from huyawo_adapt.agentic.training.local_sft_tiny_model import (
    _causal_logits,
    _close,
    _fingerprint,
    _shifted_sum,
)

_STATE_VERSION = 1
_MAX_LOGIT_CELLS = 32_768
_MAX_STATE_BYTES = 16_384


def _sha(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalSyntheticGradientBinding:
    """Exact synthetic operation identity; no external-origin attestation."""

    source_batch_sha256: str
    operation_sha256: str
    vocabulary_size: int
    embedding_dim: int
    seed: int
    learning_rate: float
    partition_boundaries: tuple[tuple[int, int], ...]
    partition_loss_tokens: tuple[int, ...]
    total_loss_tokens: int


@dataclass(frozen=True, slots=True)
class LocalSyntheticGradientState:
    """Untrusted, bounded in-memory float32 weights and partial gradients."""

    version: int
    operation_sha256: str
    source_batch_sha256: str
    completed_parts: int
    completed_loss_tokens: int
    total_loss_tokens: int
    vocabulary_size: int
    embedding_dim: int
    seed: int
    learning_rate: float
    partition_boundaries: tuple[tuple[int, int], ...]
    partition_loss_tokens: tuple[int, ...]
    dtype: Literal["float32"]
    byte_order: Literal["little"]
    payload: bytes
    payload_sha256: str


def _state_header(state: LocalSyntheticGradientState) -> tuple[object, ...]:
    return (
        "haa-b11-gradient-resume-v1",
        state.version,
        state.operation_sha256,
        state.source_batch_sha256,
        state.completed_parts,
        state.completed_loss_tokens,
        state.total_loss_tokens,
        state.vocabulary_size,
        state.embedding_dim,
        state.seed,
        state.learning_rate,
        state.partition_boundaries,
        state.partition_loss_tokens,
        state.dtype,
        state.byte_order,
    )


def _state_hash(state: LocalSyntheticGradientState) -> str:
    header = json.dumps(_state_header(state), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(header.encode("utf-8") + b"\x00" + state.payload).hexdigest()


def _tensor_ok(tensor: torch.Tensor, shape: tuple[int, int]) -> bool:
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.device.index is None
        and tensor.layout == torch.strided
        and tensor.dtype == torch.float32
        and tensor.is_contiguous()
        and tuple(tensor.shape) == shape
        and bool(torch.isfinite(tensor).all().item())
    )


def _binding(
    handoff: LocalSFTTensorDiagnostic,
    vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    partition_boundaries: tuple[tuple[int, int], ...],
    partition_loss_tokens: tuple[int, ...],
) -> LocalSyntheticGradientBinding:
    identity = (
        "haa-b11-operation-v1",
        handoff.source_batch_sha256,
        handoff.split,
        handoff.manifest_fingerprint,
        handoff.projection_fingerprints,
        handoff.model_fingerprint,
        handoff.tokenizer_fingerprint,
        handoff.projection_policy_sha256,
        handoff.source_batch.pad_token_id,
        handoff.source_batch.padded_sequence_length,
        vocabulary_size,
        embedding_dim,
        seed,
        learning_rate,
        partition_boundaries,
        partition_loss_tokens,
        handoff.total_loss_tokens,
    )
    return LocalSyntheticGradientBinding(
        source_batch_sha256=handoff.source_batch_sha256,
        operation_sha256=_json_hash(identity),
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        seed=seed,
        learning_rate=learning_rate,
        partition_boundaries=partition_boundaries,
        partition_loss_tokens=partition_loss_tokens,
        total_loss_tokens=handoff.total_loss_tokens,
    )


def seal_local_synthetic_gradient_state(
    embedding: torch.Tensor,
    projection: torch.Tensor,
    completed_parts: int,
    binding: LocalSyntheticGradientBinding,
) -> LocalSyntheticGradientState:
    """Seal toy parameters and accumulated gradients, without a disk checkpoint."""
    if type(binding) is not LocalSyntheticGradientBinding:
        raise TypeError("exact in-flight binding required")
    count = len(binding.partition_boundaries)
    if (
        type(completed_parts) is not int
        or not 1 <= completed_parts < count <= 8
        or not _sha(binding.operation_sha256)
        or not _sha(binding.source_batch_sha256)
        or type(binding.vocabulary_size) is not int
        or not 2 <= binding.vocabulary_size <= 64
        or type(binding.embedding_dim) is not int
        or not 2 <= binding.embedding_dim <= 16
        or type(binding.seed) is not int
        or not 0 <= binding.seed < 2**32
        or type(binding.learning_rate) is not float
        or not math.isfinite(binding.learning_rate)
        or not 0 < binding.learning_rate <= 0.05
        or type(binding.partition_boundaries) is not tuple
        or type(binding.partition_loss_tokens) is not tuple
        or len(binding.partition_loss_tokens) != count
        or any(type(v) is not int or v <= 0 for v in binding.partition_loss_tokens)
        or sum(binding.partition_loss_tokens) != binding.total_loss_tokens
    ):
        raise ValueError("invalid synthetic gradient binding")
    cursor = 0
    for part in binding.partition_boundaries:
        if (
            type(part) is not tuple
            or len(part) != 2
            or type(part[0]) is not int
            or type(part[1]) is not int
        ):
            raise ValueError("invalid synthetic partition type")
        start, stop = part
        if start != cursor or stop <= start or stop > 8:
            raise ValueError("noncanonical synthetic partitions")
        cursor = stop
    shape = (binding.vocabulary_size, binding.embedding_dim)
    weights = (embedding, projection)
    if any(
        type(weight) is not torch.Tensor or not weight.is_leaf or not weight.requires_grad
        for weight in weights
    ):
        raise ValueError("invalid synthetic leaf parameters")
    embedding_gradient = embedding.grad
    projection_gradient = projection.grad
    if embedding_gradient is None or projection_gradient is None:
        raise ValueError("missing synthetic partial gradient")
    matrices = (embedding, projection, embedding_gradient, projection_gradient)
    if any(not _tensor_ok(matrix, shape) for matrix in matrices):
        raise ValueError("invalid synthetic matrix")
    if len({matrix.untyped_storage().data_ptr() for matrix in matrices}) != 4:
        raise ValueError("aliased synthetic gradient state")
    values = tuple(
        float(value) for matrix in matrices for value in matrix.detach().reshape(-1).tolist()
    )
    payload = struct.pack(f"<{len(values)}f", *values)
    if (
        len(payload) != 4 * binding.vocabulary_size * binding.embedding_dim * 4
        or len(payload) > _MAX_STATE_BYTES
    ):
        raise ValueError("oversize synthetic gradient state")
    state = LocalSyntheticGradientState(
        version=_STATE_VERSION,
        operation_sha256=binding.operation_sha256,
        source_batch_sha256=binding.source_batch_sha256,
        completed_parts=completed_parts,
        completed_loss_tokens=sum(binding.partition_loss_tokens[:completed_parts]),
        total_loss_tokens=binding.total_loss_tokens,
        vocabulary_size=binding.vocabulary_size,
        embedding_dim=binding.embedding_dim,
        seed=binding.seed,
        learning_rate=binding.learning_rate,
        partition_boundaries=binding.partition_boundaries,
        partition_loss_tokens=binding.partition_loss_tokens,
        dtype="float32",
        byte_order="little",
        payload=payload,
        payload_sha256="",
    )
    return replace(state, payload_sha256=_state_hash(state))


def restore_local_synthetic_gradient_state(
    state: LocalSyntheticGradientState,
    expected_binding: LocalSyntheticGradientBinding,
    expected_completed_parts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore new non-aliased leaf weights with previously accumulated `.grad`."""
    if (
        type(state) is not LocalSyntheticGradientState
        or type(expected_binding) is not LocalSyntheticGradientBinding
    ):
        raise TypeError("exact local state and binding types required")
    b = expected_binding
    if (
        type(expected_completed_parts) is not int
        or not 1 <= expected_completed_parts < len(b.partition_boundaries)
        or type(state.version) is not int
        or state.version != _STATE_VERSION
        or state.operation_sha256 != b.operation_sha256
        or not _sha(state.operation_sha256)
        or state.source_batch_sha256 != b.source_batch_sha256
        or not _sha(state.source_batch_sha256)
        or type(state.completed_parts) is not int
        or state.completed_parts != expected_completed_parts
        or type(state.completed_loss_tokens) is not int
        or state.completed_loss_tokens != sum(b.partition_loss_tokens[:expected_completed_parts])
        or type(state.total_loss_tokens) is not int
        or state.total_loss_tokens != b.total_loss_tokens
        or type(state.vocabulary_size) is not int
        or state.vocabulary_size != b.vocabulary_size
        or type(state.embedding_dim) is not int
        or state.embedding_dim != b.embedding_dim
        or type(state.seed) is not int
        or state.seed != b.seed
        or type(state.learning_rate) is not float
        or state.learning_rate != b.learning_rate
        or state.partition_boundaries != b.partition_boundaries
        or state.partition_loss_tokens != b.partition_loss_tokens
        or state.dtype != "float32"
        or state.byte_order != "little"
        or type(state.payload) is not bytes
        or len(state.payload) != 4 * b.vocabulary_size * b.embedding_dim * 4
        or len(state.payload) > _MAX_STATE_BYTES
        or not _sha(state.payload_sha256)
        or _state_hash(state) != state.payload_sha256
    ):
        raise ValueError("invalid or stale in-flight synthetic state")
    cells = b.vocabulary_size * b.embedding_dim
    values = struct.unpack(f"<{4 * cells}f", state.payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite synthetic gradient payload")
    matrices = tuple(
        torch.tensor(values[index * cells : (index + 1) * cells], dtype=torch.float32, device="cpu")
        .reshape(b.vocabulary_size, b.embedding_dim)
        .contiguous()
        for index in range(4)
    )
    embedding = matrices[0].requires_grad_(True)
    projection = matrices[1].requires_grad_(True)
    embedding.grad = matrices[2]
    projection.grad = matrices[3]
    if len({matrix.untyped_storage().data_ptr() for matrix in matrices}) != 4:
        raise ValueError("aliased restored synthetic state")
    return embedding, projection


@dataclass(frozen=True, slots=True)
class LocalSFTSyntheticGradientResumeDiagnostic:
    """Local toy gradients only; all real-training authority remains denied."""

    source_batch_sha256: str
    operation_sha256: str
    partition_boundaries: tuple[tuple[int, int], ...]
    partition_loss_tokens: tuple[int, ...]
    interrupted_after_parts: int
    completed_loss_tokens: int
    total_loss_tokens: int
    partial_embedding_gradient_sha256: str
    partial_projection_gradient_sha256: str
    intermediate_state_sha256: str
    prefix_gradient_max_error: float
    resumed_microbatch_gradient_max_error: float
    resumed_fullbatch_gradient_max_error: float
    resumed_microbatch_step_max_error: float
    resumed_fullbatch_step_max_error: float
    mean_nll_before: float
    mean_nll_after: float
    initial_embedding_sha256: str
    initial_projection_sha256: str
    final_embedding_sha256: str
    final_projection_sha256: str
    authority_scope: Literal["local_sft_synthetic_inflight_gradient_only"] = (
        "local_sft_synthetic_inflight_gradient_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_inflight_gradient_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
            or not _sha(self.source_batch_sha256)
            or not _sha(self.operation_sha256)
            or not _sha(self.intermediate_state_sha256)
            or not 2 <= len(self.partition_boundaries) <= 8
            or not 1 <= self.interrupted_after_parts < len(self.partition_boundaries)
            or sum(self.partition_loss_tokens) != self.total_loss_tokens
            or self.completed_loss_tokens
            != sum(self.partition_loss_tokens[: self.interrupted_after_parts])
            or any(
                not math.isfinite(x) or x < 0
                for x in (
                    self.prefix_gradient_max_error,
                    self.resumed_microbatch_gradient_max_error,
                    self.resumed_fullbatch_gradient_max_error,
                    self.resumed_microbatch_step_max_error,
                    self.resumed_fullbatch_step_max_error,
                    self.mean_nll_before,
                    self.mean_nll_after,
                )
            )
        ):
            raise ValueError("invalid synthetic gradient diagnostic")


def audit_local_sft_synthetic_gradient_resume(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
    interrupt_after_parts: int,
) -> LocalSFTSyntheticGradientResumeDiagnostic:
    """Compare in-flight resumed toy SGD with uninterrupted and full-batch copies."""
    validate_local_sft_tensors(handoff)
    rows, width = (int(value) for value in handoff.input_ids.shape)
    if (
        type(synthetic_vocabulary_size) is not int
        or not 2 <= synthetic_vocabulary_size <= 64
        or type(embedding_dim) is not int
        or not 2 <= embedding_dim <= 16
        or type(seed) is not int
        or not 0 <= seed < 2**32
        or type(learning_rate) is not float
        or not math.isfinite(learning_rate)
        or not 0.0 < learning_rate <= 0.05
        or type(microbatch_rows) is not int
        or not 1 <= microbatch_rows < rows
        or not 2 <= rows <= 8
        or not 2 <= width <= 64
        or rows * width * synthetic_vocabulary_size > _MAX_LOGIT_CELLS
        or bool(torch.any(handoff.input_ids >= synthetic_vocabulary_size).item())
        or bool(
            torch.any(
                (handoff.labels != -100) & (handoff.labels >= synthetic_vocabulary_size)
            ).item()
        )
    ):
        raise ValueError("invalid bounded in-flight synthetic input")
    counts = tuple(int(v) for v in (handoff.labels[:, 1:] != -100).sum(1).tolist())
    if (
        any(count < 1 for count in counts)
        or counts != handoff.source_batch.row_target_lengths
        or sum(counts) != handoff.total_loss_tokens
    ):
        raise ValueError("inconsistent supervised target counts")
    parts = tuple(
        (start, min(rows, start + microbatch_rows)) for start in range(0, rows, microbatch_rows)
    )
    tokens = tuple(sum(counts[start:stop]) for start, stop in parts)
    if (
        type(interrupt_after_parts) is not int
        or not 2 <= len(parts) <= 8
        or not 1 <= interrupt_after_parts < len(parts)
        or any(count < 1 for count in tokens)
    ):
        raise ValueError("invalid in-flight interruption boundary")
    binding = _binding(
        handoff, synthetic_vocabulary_size, embedding_dim, seed, learning_rate, parts, tokens
    )
    inputs = tuple(
        tensor.detach().clone()
        for tensor in (handoff.input_ids, handoff.labels, handoff.attention_mask)
    )
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starting_embedding = torch.empty(
        (synthetic_vocabulary_size, embedding_dim), dtype=torch.float32, device="cpu"
    ).uniform_(-0.2, 0.2, generator=generator)
    starting_projection = torch.empty(
        (synthetic_vocabulary_size, embedding_dim), dtype=torch.float32, device="cpu"
    ).uniform_(-0.2, 0.2, generator=generator)
    initial_hashes = (_fingerprint(starting_embedding), _fingerprint(starting_projection))

    def copy_weights() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            starting_embedding.detach().clone().requires_grad_(True),
            starting_projection.detach().clone().requires_grad_(True),
        )

    embedding, projection = copy_weights()
    uninterrupted_embedding, uninterrupted_projection = copy_weights()
    full_embedding, full_projection = copy_weights()
    all_private = (
        embedding,
        projection,
        uninterrupted_embedding,
        uninterrupted_projection,
        full_embedding,
        full_projection,
        *inputs,
        handoff.input_ids,
        handoff.labels,
        handoff.attention_mask,
    )
    if len({tensor.untyped_storage().data_ptr() for tensor in all_private}) != len(all_private):
        raise ValueError("aliased synthetic source or private parameters")
    total = handoff.total_loss_tokens
    before_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    before = audit_local_sft_causal_loss(handoff, before_logits.detach().contiguous()).mean_nll

    def backward_rows(start: int, stop: int, weight: torch.Tensor, output: torch.Tensor) -> None:
        logits = _causal_logits(
            handoff.input_ids[start:stop], handoff.attention_mask[start:stop], weight, output
        )
        loss = _shifted_sum(logits, handoff.labels[start:stop]) / total
        loss.backward()  # type: ignore[no-untyped-call]

    for start, stop in parts[:interrupt_after_parts]:
        backward_rows(start, stop, embedding, projection)
    for start, stop in parts[:interrupt_after_parts]:
        backward_rows(start, stop, uninterrupted_embedding, uninterrupted_projection)
    if any(
        weight.grad is None
        for weight in (embedding, projection, uninterrupted_embedding, uninterrupted_projection)
    ):
        raise ValueError("missing synthetic prefix gradient")
    assert embedding.grad is not None and projection.grad is not None
    assert uninterrupted_embedding.grad is not None and uninterrupted_projection.grad is not None
    prefix_error = max(
        _close(embedding.grad, uninterrupted_embedding.grad, "prefix embedding gradient"),
        _close(projection.grad, uninterrupted_projection.grad, "prefix projection gradient"),
    )
    partial_hashes = (_fingerprint(embedding.grad), _fingerprint(projection.grad))
    if (_fingerprint(embedding), _fingerprint(projection)) != initial_hashes:
        raise ValueError("parameters changed before synthetic optimizer step")
    state = seal_local_synthetic_gradient_state(
        embedding, projection, interrupt_after_parts, binding
    )
    embedding, projection = restore_local_synthetic_gradient_state(
        state, binding, interrupt_after_parts
    )
    if (
        embedding.grad is None
        or projection.grad is None
        or (_fingerprint(embedding.grad), _fingerprint(projection.grad)) != partial_hashes
    ):
        raise ValueError("synthetic partial gradient restoration mismatch")
    for start, stop in parts[interrupt_after_parts:]:
        backward_rows(start, stop, embedding, projection)
    for start, stop in parts[interrupt_after_parts:]:
        backward_rows(start, stop, uninterrupted_embedding, uninterrupted_projection)
    full_logits = _causal_logits(
        handoff.input_ids, handoff.attention_mask, full_embedding, full_projection
    )
    full_loss = _shifted_sum(full_logits, handoff.labels) / total
    full_loss.backward()  # type: ignore[no-untyped-call]
    gradients = (
        embedding.grad,
        projection.grad,
        uninterrupted_embedding.grad,
        uninterrupted_projection.grad,
        full_embedding.grad,
        full_projection.grad,
    )
    if any(grad is None for grad in gradients):
        raise ValueError("missing synthetic complete gradient")
    assert embedding.grad is not None and projection.grad is not None
    assert uninterrupted_embedding.grad is not None and uninterrupted_projection.grad is not None
    assert full_embedding.grad is not None and full_projection.grad is not None
    micro_error = max(
        _close(
            embedding.grad, uninterrupted_embedding.grad, "resumed microbatch embedding gradient"
        ),
        _close(
            projection.grad, uninterrupted_projection.grad, "resumed microbatch projection gradient"
        ),
    )
    full_error = max(
        _close(embedding.grad, full_embedding.grad, "resumed full embedding gradient"),
        _close(projection.grad, full_projection.grad, "resumed full projection gradient"),
    )
    if not math.isclose(float(full_loss.item()), before, abs_tol=3e-6, rel_tol=3e-5):
        raise ValueError("synthetic full-batch forward oracle mismatch")
    if any(
        float(torch.linalg.vector_norm(grad).item()) <= 0
        for grad in (embedding.grad, projection.grad)
    ):
        raise ValueError("missing trainable synthetic signal")
    for weight, output in (
        (embedding, projection),
        (uninterrupted_embedding, uninterrupted_projection),
        (full_embedding, full_projection),
    ):
        torch.optim.SGD([weight, output], lr=learning_rate).step()
    micro_step_error = max(
        _close(
            embedding.detach(),
            uninterrupted_embedding.detach(),
            "resumed microbatch embedding step",
        ),
        _close(
            projection.detach(),
            uninterrupted_projection.detach(),
            "resumed microbatch projection step",
        ),
    )
    full_step_error = max(
        _close(embedding.detach(), full_embedding.detach(), "resumed full embedding step"),
        _close(projection.detach(), full_projection.detach(), "resumed full projection step"),
    )
    if (
        _fingerprint(embedding) == initial_hashes[0]
        or _fingerprint(projection) == initial_hashes[1]
    ):
        raise ValueError("synthetic optimizer did not change both parameter matrices")
    after_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    after = audit_local_sft_causal_loss(handoff, after_logits.detach().contiguous()).mean_nll
    validate_local_sft_tensors(handoff)
    if any(
        not torch.equal(a, b)
        for a, b in zip(
            inputs, (handoff.input_ids, handoff.labels, handoff.attention_mask), strict=True
        )
    ):
        raise ValueError("synthetic source tensors modified")
    if not torch.equal(torch.get_rng_state(), global_rng):
        raise ValueError("global CPU RNG modified")
    if (_fingerprint(starting_embedding), _fingerprint(starting_projection)) != initial_hashes:
        raise ValueError("initial toy parameters modified")
    return LocalSFTSyntheticGradientResumeDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        operation_sha256=binding.operation_sha256,
        partition_boundaries=parts,
        partition_loss_tokens=tokens,
        interrupted_after_parts=interrupt_after_parts,
        completed_loss_tokens=sum(tokens[:interrupt_after_parts]),
        total_loss_tokens=total,
        partial_embedding_gradient_sha256=partial_hashes[0],
        partial_projection_gradient_sha256=partial_hashes[1],
        intermediate_state_sha256=state.payload_sha256,
        prefix_gradient_max_error=prefix_error,
        resumed_microbatch_gradient_max_error=micro_error,
        resumed_fullbatch_gradient_max_error=full_error,
        resumed_microbatch_step_max_error=micro_step_error,
        resumed_fullbatch_step_max_error=full_step_error,
        mean_nll_before=before,
        mean_nll_after=after,
        initial_embedding_sha256=initial_hashes[0],
        initial_projection_sha256=initial_hashes[1],
        final_embedding_sha256=_fingerprint(embedding),
        final_projection_sha256=_fingerprint(projection),
    )
