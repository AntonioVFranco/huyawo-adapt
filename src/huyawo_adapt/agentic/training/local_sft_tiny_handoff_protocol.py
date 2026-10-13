"""Bounded two-invocation CPU toy gradient handoff without training authority."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Literal

import torch

from huyawo_adapt.agentic.training.local_sft_loss_audit import audit_local_sft_causal_loss
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    validate_local_sft_tensors,
)
from huyawo_adapt.agentic.training.local_sft_tiny_gradient_resume import (
    LocalSyntheticGradientBinding,
    LocalSyntheticGradientState,
    _binding,
    restore_local_synthetic_gradient_state,
    seal_local_synthetic_gradient_state,
)
from huyawo_adapt.agentic.training.local_sft_tiny_model import (
    _causal_logits,
    _close,
    _fingerprint,
    _shifted_sum,
)

_MAGIC = b"HAA-B12\x01"
_MAX_ENVELOPE = 32_768
_MAX_METADATA = 8_192
_MAX_PAYLOAD = 16_384
_FIELDS = frozenset(LocalSyntheticGradientState.__dataclass_fields__) - {"payload"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate envelope metadata key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class LocalSyntheticHandoffEnvelope:
    """Local untrusted transport bytes, never a checkpoint or an issuer receipt."""

    payload: bytes
    envelope_sha256: str
    authority_scope: Literal["local_sft_synthetic_two_phase_only"] = (
        "local_sft_synthetic_two_phase_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.payload) is not bytes
            or not 0 < len(self.payload) <= _MAX_ENVELOPE
            or type(self.envelope_sha256) is not str
            or _digest(self.payload) != self.envelope_sha256
            or self.authority_scope != "local_sft_synthetic_two_phase_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
        ):
            raise ValueError("invalid nonpromoting handoff envelope")


@dataclass(frozen=True, slots=True)
class LocalSyntheticTwoPhaseDiagnostic:
    """Toy parity after exactly one local resumed optimizer settlement."""

    source_batch_sha256: str
    operation_sha256: str
    envelope_sha256: str
    partition_boundaries: tuple[tuple[int, int], ...]
    partition_loss_tokens: tuple[int, ...]
    completed_parts: int
    completed_loss_tokens: int
    total_loss_tokens: int
    prefix_gradient_max_error: float
    microbatch_gradient_max_error: float
    fullbatch_gradient_max_error: float
    microbatch_step_max_error: float
    fullbatch_step_max_error: float
    mean_nll_before: float
    mean_nll_after: float
    initial_embedding_sha256: str
    initial_projection_sha256: str
    final_embedding_sha256: str
    final_projection_sha256: str
    authority_scope: Literal["local_sft_synthetic_two_phase_only"] = (
        "local_sft_synthetic_two_phase_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_two_phase_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
            or not 1 <= self.completed_parts < len(self.partition_boundaries) <= 8
            or self.completed_loss_tokens != sum(self.partition_loss_tokens[: self.completed_parts])
            or self.total_loss_tokens != sum(self.partition_loss_tokens)
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.prefix_gradient_max_error,
                    self.microbatch_gradient_max_error,
                    self.fullbatch_gradient_max_error,
                    self.microbatch_step_max_error,
                    self.fullbatch_step_max_error,
                    self.mean_nll_before,
                    self.mean_nll_after,
                )
            )
        ):
            raise ValueError("invalid two-phase diagnostic")


def _operation(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
    interrupt_after_parts: int,
) -> tuple[LocalSyntheticGradientBinding, tuple[tuple[int, int], ...], tuple[int, ...]]:
    validate_local_sft_tensors(handoff)
    rows, width = (int(v) for v in handoff.input_ids.shape)
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
        or rows * width * synthetic_vocabulary_size > 32_768
        or bool(torch.any(handoff.input_ids >= synthetic_vocabulary_size).item())
        or bool(
            torch.any(
                (handoff.labels != -100) & (handoff.labels >= synthetic_vocabulary_size)
            ).item()
        )
    ):
        raise ValueError("invalid bounded synthetic handoff inputs")
    counts = tuple(int(v) for v in (handoff.labels[:, 1:] != -100).sum(1).tolist())
    if (
        any(v < 1 for v in counts)
        or counts != handoff.source_batch.row_target_lengths
        or sum(counts) != handoff.total_loss_tokens
    ):
        raise ValueError("invalid supervised token counts")
    parts = tuple((s, min(rows, s + microbatch_rows)) for s in range(0, rows, microbatch_rows))
    tokens = tuple(sum(counts[s:e]) for s, e in parts)
    if (
        type(interrupt_after_parts) is not int
        or not 2 <= len(parts) <= 8
        or not 1 <= interrupt_after_parts < len(parts)
        or any(value < 1 for value in tokens)
    ):
        raise ValueError("invalid synthetic partition cursor")
    binding = _binding(
        handoff, synthetic_vocabulary_size, embedding_dim, seed, learning_rate, parts, tokens
    )
    return binding, parts, tokens


def _initial(vocab: int, dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    embedding = torch.empty((vocab, dim), dtype=torch.float32, device="cpu").uniform_(
        -0.2, 0.2, generator=generator
    )
    projection = torch.empty((vocab, dim), dtype=torch.float32, device="cpu").uniform_(
        -0.2, 0.2, generator=generator
    )
    return embedding, projection


def _backward(
    handoff: LocalSFTTensorDiagnostic,
    first: int,
    last: int,
    embedding: torch.Tensor,
    projection: torch.Tensor,
) -> None:
    logits = _causal_logits(
        handoff.input_ids[first:last], handoff.attention_mask[first:last], embedding, projection
    )
    loss = _shifted_sum(logits, handoff.labels[first:last]) / handoff.total_loss_tokens
    loss.backward()  # type: ignore[no-untyped-call]


def _encode(state: LocalSyntheticGradientState) -> bytes:
    metadata = {name: getattr(state, name) for name in _FIELDS}
    meta = _canonical(metadata)
    if len(meta) > _MAX_METADATA or len(state.payload) > _MAX_PAYLOAD:
        raise ValueError("oversize gradient handoff")
    body = (
        _MAGIC
        + struct.pack("<I", len(meta))
        + meta
        + struct.pack("<I", len(state.payload))
        + state.payload
    )
    sealed = body + hashlib.sha256(body).digest()
    if len(sealed) > _MAX_ENVELOPE:
        raise ValueError("oversize two-phase envelope")
    return sealed


def _decode(value: bytes) -> LocalSyntheticGradientState:
    if type(value) is not bytes or not 0 < len(value) <= _MAX_ENVELOPE:
        raise ValueError("invalid two-phase envelope length or type")
    header_end = len(_MAGIC) + 4
    if len(value) < header_end + 4 + 32 or value[: len(_MAGIC)] != _MAGIC:
        raise ValueError("invalid two-phase envelope header")
    meta_length = struct.unpack("<I", value[len(_MAGIC) : header_end])[0]
    if meta_length == 0 or meta_length > _MAX_METADATA:
        raise ValueError("invalid handoff metadata length")
    payload_length_offset = header_end + meta_length
    if payload_length_offset + 4 + 32 > len(value):
        raise ValueError("truncated handoff metadata")
    payload_length = struct.unpack("<I", value[payload_length_offset : payload_length_offset + 4])[
        0
    ]
    if (
        payload_length == 0
        or payload_length > _MAX_PAYLOAD
        or payload_length_offset + 4 + payload_length + 32 != len(value)
    ):
        raise ValueError("invalid handoff payload length")
    body = value[:-32]
    if hashlib.sha256(body).digest() != value[-32:]:
        raise ValueError("invalid handoff envelope digest")
    metadata_bytes = value[header_end:payload_length_offset]
    try:
        metadata = json.loads(
            metadata_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid handoff JSON") from error
    if type(metadata) is not dict or set(metadata) != _FIELDS:
        raise ValueError("unknown or missing handoff metadata fields")
    if _canonical(metadata) != metadata_bytes:
        raise ValueError("noncanonical handoff JSON")
    fields = ("partition_boundaries", "partition_loss_tokens")
    for key in fields:
        if type(metadata[key]) is not list or not 2 <= len(metadata[key]) <= 8:
            raise ValueError("invalid handoff partition metadata")
    raw_parts = metadata["partition_boundaries"]
    if any(
        type(part) is not list or len(part) != 2 or any(type(number) is not int for number in part)
        for part in raw_parts
    ):
        raise ValueError("invalid handoff partition boundary types")
    raw_tokens = metadata["partition_loss_tokens"]
    if any(type(number) is not int or number < 1 for number in raw_tokens):
        raise ValueError("invalid handoff token count types")
    integer_fields = (
        "version",
        "completed_parts",
        "completed_loss_tokens",
        "total_loss_tokens",
        "vocabulary_size",
        "embedding_dim",
        "seed",
    )
    if any(type(metadata[key]) is not int for key in integer_fields):
        raise ValueError("invalid integer field in handoff")
    if type(metadata["learning_rate"]) is not float:
        raise ValueError("invalid floating-point learning rate")
    if any(
        type(metadata[key]) is not str
        for key in (
            "operation_sha256",
            "source_batch_sha256",
            "dtype",
            "byte_order",
            "payload_sha256",
        )
    ):
        raise ValueError("invalid string field in handoff")
    return LocalSyntheticGradientState(
        version=metadata["version"],
        operation_sha256=metadata["operation_sha256"],
        source_batch_sha256=metadata["source_batch_sha256"],
        completed_parts=metadata["completed_parts"],
        completed_loss_tokens=metadata["completed_loss_tokens"],
        total_loss_tokens=metadata["total_loss_tokens"],
        vocabulary_size=metadata["vocabulary_size"],
        embedding_dim=metadata["embedding_dim"],
        seed=metadata["seed"],
        learning_rate=metadata["learning_rate"],
        partition_boundaries=tuple((part[0], part[1]) for part in raw_parts),
        partition_loss_tokens=tuple(raw_tokens),
        dtype=metadata["dtype"],
        byte_order=metadata["byte_order"],
        payload=value[payload_length_offset + 4 : -32],
        payload_sha256=metadata["payload_sha256"],
    )


def prepare_local_sft_synthetic_handoff(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
    interrupt_after_parts: int,
) -> LocalSyntheticHandoffEnvelope:
    """Compute prefix gradients only, returning a bounded untrusted byte envelope."""
    binding, parts, _ = _operation(
        handoff,
        synthetic_vocabulary_size,
        embedding_dim,
        seed,
        learning_rate,
        microbatch_rows,
        interrupt_after_parts,
    )
    global_rng = torch.get_rng_state().clone()
    sources = tuple(
        tensor.detach().clone()
        for tensor in (handoff.input_ids, handoff.labels, handoff.attention_mask)
    )
    initial_e, initial_p = _initial(synthetic_vocabulary_size, embedding_dim, seed)
    initial_hashes = (_fingerprint(initial_e), _fingerprint(initial_p))
    embedding = initial_e.detach().clone().requires_grad_(True)
    projection = initial_p.detach().clone().requires_grad_(True)
    for first, last in parts[:interrupt_after_parts]:
        _backward(handoff, first, last, embedding, projection)
    if (_fingerprint(embedding), _fingerprint(projection)) != initial_hashes:
        raise ValueError("optimizer modified prefix parameters")
    state = seal_local_synthetic_gradient_state(
        embedding, projection, interrupt_after_parts, binding
    )
    serialized = _encode(state)
    validate_local_sft_tensors(handoff)
    if any(
        not torch.equal(before, after)
        for before, after in zip(
            sources, (handoff.input_ids, handoff.labels, handoff.attention_mask), strict=True
        )
    ) or not torch.equal(torch.get_rng_state(), global_rng):
        raise ValueError("prefix preparation modified caller data or RNG")
    return LocalSyntheticHandoffEnvelope(serialized, _digest(serialized))


def resume_local_sft_synthetic_handoff(
    handoff: LocalSFTTensorDiagnostic,
    envelope_bytes: bytes,
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
    interrupt_after_parts: int,
) -> LocalSyntheticTwoPhaseDiagnostic:
    """Validate untrusted state, resume suffix only, and check independent toy oracles."""
    binding, parts, tokens = _operation(
        handoff,
        synthetic_vocabulary_size,
        embedding_dim,
        seed,
        learning_rate,
        microbatch_rows,
        interrupt_after_parts,
    )
    state = _decode(envelope_bytes)
    embedding, projection = restore_local_synthetic_gradient_state(
        state, binding, interrupt_after_parts
    )
    initial_e, initial_p = _initial(synthetic_vocabulary_size, embedding_dim, seed)
    if not torch.equal(embedding.detach(), initial_e) or not torch.equal(
        projection.detach(), initial_p
    ):
        raise ValueError("noncanonical initial synthetic weights")
    prefix_e = initial_e.detach().clone().requires_grad_(True)
    prefix_p = initial_p.detach().clone().requires_grad_(True)
    for first, last in parts[:interrupt_after_parts]:
        _backward(handoff, first, last, prefix_e, prefix_p)
    if (
        embedding.grad is None
        or projection.grad is None
        or prefix_e.grad is None
        or prefix_p.grad is None
    ):
        raise ValueError("missing saved or oracle prefix gradient")
    prefix_error = max(
        _close(embedding.grad, prefix_e.grad, "two-phase prefix embedding"),
        _close(projection.grad, prefix_p.grad, "two-phase prefix projection"),
    )
    original_inputs = tuple(
        tensor.detach().clone()
        for tensor in (handoff.input_ids, handoff.labels, handoff.attention_mask)
    )
    global_rng = torch.get_rng_state().clone()
    initial_hashes = (_fingerprint(initial_e), _fingerprint(initial_p))
    baseline_e = initial_e.detach().clone().requires_grad_(True)
    baseline_p = initial_p.detach().clone().requires_grad_(True)
    full_e = initial_e.detach().clone().requires_grad_(True)
    full_p = initial_p.detach().clone().requires_grad_(True)
    before_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    before = audit_local_sft_causal_loss(handoff, before_logits.detach().contiguous()).mean_nll
    for first, last in parts[interrupt_after_parts:]:
        _backward(handoff, first, last, embedding, projection)
    for first, last in parts:
        _backward(handoff, first, last, baseline_e, baseline_p)
    full_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, full_e, full_p)
    full_loss = _shifted_sum(full_logits, handoff.labels) / handoff.total_loss_tokens
    full_loss.backward()  # type: ignore[no-untyped-call]
    if not math.isclose(float(full_loss.item()), before, abs_tol=3e-6, rel_tol=3e-5):
        raise ValueError("full-batch forward oracle mismatch")
    gradients = (
        embedding.grad,
        projection.grad,
        baseline_e.grad,
        baseline_p.grad,
        full_e.grad,
        full_p.grad,
    )
    if any(gradient is None for gradient in gradients):
        raise ValueError("missing two-phase complete gradient")
    assert embedding.grad is not None and projection.grad is not None
    assert baseline_e.grad is not None and baseline_p.grad is not None
    assert full_e.grad is not None and full_p.grad is not None
    micro_error = max(
        _close(embedding.grad, baseline_e.grad, "two-phase microbatch embedding"),
        _close(projection.grad, baseline_p.grad, "two-phase microbatch projection"),
    )
    full_error = max(
        _close(embedding.grad, full_e.grad, "two-phase fullbatch embedding"),
        _close(projection.grad, full_p.grad, "two-phase fullbatch projection"),
    )
    if any(
        float(torch.linalg.vector_norm(gradient).item()) <= 0
        for gradient in (embedding.grad, projection.grad)
    ):
        raise ValueError("missing synthetic training signal")
    for weight, output in ((embedding, projection), (baseline_e, baseline_p), (full_e, full_p)):
        torch.optim.SGD([weight, output], lr=learning_rate).step()
    micro_step_error = max(
        _close(embedding.detach(), baseline_e.detach(), "two-phase microbatch embedding step"),
        _close(projection.detach(), baseline_p.detach(), "two-phase microbatch projection step"),
    )
    full_step_error = max(
        _close(embedding.detach(), full_e.detach(), "two-phase fullbatch embedding step"),
        _close(projection.detach(), full_p.detach(), "two-phase fullbatch projection step"),
    )
    if (
        _fingerprint(embedding) == initial_hashes[0]
        or _fingerprint(projection) == initial_hashes[1]
    ):
        raise ValueError("two-phase optimizer did not change both matrices")
    after_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    after = audit_local_sft_causal_loss(handoff, after_logits.detach().contiguous()).mean_nll
    validate_local_sft_tensors(handoff)
    if any(
        not torch.equal(before_tensor, current_tensor)
        for before_tensor, current_tensor in zip(
            original_inputs,
            (handoff.input_ids, handoff.labels, handoff.attention_mask),
            strict=True,
        )
    ) or not torch.equal(torch.get_rng_state(), global_rng):
        raise ValueError("resume modified caller tensors or global RNG")
    if (_fingerprint(initial_e), _fingerprint(initial_p)) != initial_hashes:
        raise ValueError("initial synthetic weights modified")
    return LocalSyntheticTwoPhaseDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        operation_sha256=binding.operation_sha256,
        envelope_sha256=_digest(envelope_bytes),
        partition_boundaries=parts,
        partition_loss_tokens=tokens,
        completed_parts=interrupt_after_parts,
        completed_loss_tokens=sum(tokens[:interrupt_after_parts]),
        total_loss_tokens=handoff.total_loss_tokens,
        prefix_gradient_max_error=prefix_error,
        microbatch_gradient_max_error=micro_error,
        fullbatch_gradient_max_error=full_error,
        microbatch_step_max_error=micro_step_error,
        fullbatch_step_max_error=full_step_error,
        mean_nll_before=before,
        mean_nll_after=after,
        initial_embedding_sha256=initial_hashes[0],
        initial_projection_sha256=initial_hashes[1],
        final_embedding_sha256=_fingerprint(embedding),
        final_projection_sha256=_fingerprint(projection),
    )
