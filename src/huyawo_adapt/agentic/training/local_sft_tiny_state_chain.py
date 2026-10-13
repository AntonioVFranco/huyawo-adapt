"""CPU-only synthetic multi-step parameter continuity without training authority."""

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
from huyawo_adapt.agentic.training.local_sft_tiny_model import (
    _causal_logits,
    _close,
    _fingerprint,
    _shifted_sum,
)

_MAX_LOGIT_CELLS = 32_768
_STATE_VERSION = 1


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _payload_hash(state: LocalSyntheticParameterState) -> str:
    header = (
        "haa-b10-local-state-v1",
        state.version,
        state.completed_steps,
        state.prefix_sha256,
        state.vocabulary_size,
        state.embedding_dim,
        state.dtype,
    )
    return hashlib.sha256(
        json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\x00" + state.payload
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalSyntheticParameterState:
    """Untrusted bounded in-memory toy weights; never an authentic checkpoint."""

    version: int
    completed_steps: int
    prefix_sha256: str
    vocabulary_size: int
    embedding_dim: int
    dtype: Literal["float32"]
    payload: bytes
    payload_sha256: str


def seal_local_synthetic_state(
    embedding: torch.Tensor,
    projection: torch.Tensor,
    completed_steps: int,
    prefix_sha256: str,
) -> LocalSyntheticParameterState:
    """Make a local deterministic integrity-checked synthetic handoff."""
    if type(completed_steps) is not int or not 1 <= completed_steps <= 3 or not _sha(prefix_sha256):
        raise ValueError("invalid synthetic state binding")
    if type(embedding) is not torch.Tensor or type(projection) is not torch.Tensor:
        raise TypeError("exact synthetic parameter tensors required")
    if embedding.shape != projection.shape or embedding.ndim != 2:
        raise ValueError("inconsistent synthetic parameter shapes")
    vocab, dim = (int(value) for value in embedding.shape)
    if not 2 <= vocab <= 64 or not 2 <= dim <= 16:
        raise ValueError("synthetic weight shapes outside bound")
    if any(
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
        for value in (embedding, projection)
    ):
        raise ValueError("nonfinite or unsupported synthetic state tensor")
    if embedding.untyped_storage().data_ptr() == projection.untyped_storage().data_ptr():
        raise ValueError("aliased synthetic parameter state")
    values = tuple(
        float(v) for t in (embedding, projection) for v in t.detach().reshape(-1).tolist()
    )
    payload = struct.pack(f"<{len(values)}f", *values)
    state = LocalSyntheticParameterState(
        version=_STATE_VERSION,
        completed_steps=completed_steps,
        prefix_sha256=prefix_sha256,
        vocabulary_size=vocab,
        embedding_dim=dim,
        dtype="float32",
        payload=payload,
        payload_sha256="",
    )
    return LocalSyntheticParameterState(
        version=state.version,
        completed_steps=state.completed_steps,
        prefix_sha256=state.prefix_sha256,
        vocabulary_size=state.vocabulary_size,
        embedding_dim=state.embedding_dim,
        dtype=state.dtype,
        payload=state.payload,
        payload_sha256=_payload_hash(state),
    )


def restore_local_synthetic_state(
    state: LocalSyntheticParameterState,
    expected_steps: int,
    expected_prefix_sha256: str,
    expected_vocabulary_size: int,
    expected_embedding_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check local bytes and reconstruct non-aliased CPU float32 toy weights."""
    if type(state) is not LocalSyntheticParameterState:
        raise TypeError("exact local state type required")
    if (
        type(expected_steps) is not int
        or not 1 <= expected_steps <= 3
        or not _sha(expected_prefix_sha256)
        or type(expected_vocabulary_size) is not int
        or not 2 <= expected_vocabulary_size <= 64
        or type(expected_embedding_dim) is not int
        or not 2 <= expected_embedding_dim <= 16
        or type(state.version) is not int
        or state.version != _STATE_VERSION
        or type(state.completed_steps) is not int
        or state.completed_steps != expected_steps
        or state.prefix_sha256 != expected_prefix_sha256
        or not _sha(state.prefix_sha256)
        or type(state.vocabulary_size) is not int
        or state.vocabulary_size != expected_vocabulary_size
        or type(state.embedding_dim) is not int
        or state.embedding_dim != expected_embedding_dim
        or state.dtype != "float32"
        or type(state.payload) is not bytes
        or len(state.payload) != 2 * expected_vocabulary_size * expected_embedding_dim * 4
        or not _sha(state.payload_sha256)
        or _payload_hash(state) != state.payload_sha256
    ):
        raise ValueError("invalid or stale synthetic state")
    cells = expected_vocabulary_size * expected_embedding_dim
    floats = struct.unpack(f"<{2 * cells}f", state.payload)
    if any(not math.isfinite(value) for value in floats):
        raise ValueError("nonfinite synthetic state content")
    embedding = torch.tensor(floats[:cells], dtype=torch.float32, device="cpu").reshape(
        expected_vocabulary_size, expected_embedding_dim
    )
    projection = torch.tensor(floats[cells:], dtype=torch.float32, device="cpu").reshape(
        expected_vocabulary_size, expected_embedding_dim
    )
    return embedding.requires_grad_(True), projection.requires_grad_(True)


@dataclass(frozen=True, slots=True)
class LocalSFTSyntheticStateChainDiagnostic:
    """Synthetic local parameter parity, never permission for real training."""

    ordered_source_batch_sha256: tuple[str, ...]
    ordered_projection_fingerprints: tuple[tuple[str, ...], ...]
    source_chain_sha256: str
    total_steps: int
    resume_after_step: int
    microbatch_loss_tokens: tuple[tuple[int, ...], ...]
    mean_nll_before: tuple[float, ...]
    mean_nll_after: tuple[float, ...]
    max_gradient_parity_error: tuple[float, ...]
    max_weight_parity_error: tuple[float, ...]
    initial_embedding_sha256: str
    initial_projection_sha256: str
    intermediate_state_sha256: str
    final_embedding_sha256: str
    final_projection_sha256: str
    authority_scope: Literal["local_sft_synthetic_state_chain_only"] = (
        "local_sft_synthetic_state_chain_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_state_chain_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
            or not 2 <= self.total_steps <= 4
            or not 1 <= self.resume_after_step < self.total_steps
            or len(self.ordered_source_batch_sha256) != self.total_steps
            or len(self.ordered_projection_fingerprints) != self.total_steps
            or any(
                len(item) != self.total_steps
                for item in (
                    self.microbatch_loss_tokens,
                    self.mean_nll_before,
                    self.mean_nll_after,
                    self.max_gradient_parity_error,
                    self.max_weight_parity_error,
                )
            )
        ):
            raise ValueError("invalid local-only chain diagnostic authority")


def audit_local_sft_synthetic_state_chain(
    handoffs: tuple[LocalSFTTensorDiagnostic, ...],
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
    resume_after_step: int,
) -> LocalSFTSyntheticStateChainDiagnostic:
    """Compare resumed toy parameter evolution with uninterrupted full batches."""
    if type(handoffs) is not tuple or not 2 <= len(handoffs) <= 4:
        raise ValueError("expected 2 to 4 distinct synthetic handoffs")
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
        or microbatch_rows < 1
        or type(resume_after_step) is not int
        or not 1 <= resume_after_step < len(handoffs)
    ):
        raise ValueError("invalid bounded local parameter chain input")
    identities: set[str] = set()
    projections: set[str] = set()
    common: tuple[str, ...] | None = None
    snapshots: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    rows_per_handoff: list[tuple[int, ...]] = []
    for handoff in handoffs:
        validate_local_sft_tensors(handoff)
        rows, width = (int(value) for value in handoff.input_ids.shape)
        if (
            not 2 <= rows <= 8
            or not 2 <= width <= 64
            or microbatch_rows >= rows
            or rows * width * synthetic_vocabulary_size > _MAX_LOGIT_CELLS
            or bool(torch.any(handoff.input_ids >= synthetic_vocabulary_size).item())
            or bool(
                torch.any(
                    (handoff.labels != -100) & (handoff.labels >= synthetic_vocabulary_size)
                ).item()
            )
        ):
            raise ValueError("synthetic handoff size or vocabulary mismatch")
        row_counts = tuple(int(v) for v in (handoff.labels[:, 1:] != -100).sum(1).tolist())
        if (
            any(value < 1 for value in row_counts)
            or sum(row_counts) != handoff.total_loss_tokens
            or row_counts != handoff.source_batch.row_target_lengths
        ):
            raise ValueError("invalid synthetic supervision counts")
        binding = (
            handoff.split,
            handoff.model_fingerprint,
            handoff.tokenizer_fingerprint,
            handoff.projection_policy_sha256,
        )
        if common is None:
            common = binding
        if binding != common or handoff.source_batch_sha256 in identities:
            raise ValueError("mixed or duplicate synthetic batch identity")
        if any(value in projections for value in handoff.projection_fingerprints):
            raise ValueError("reused synthetic projection identity")
        identities.add(handoff.source_batch_sha256)
        projections.update(handoff.projection_fingerprints)
        rows_per_handoff.append(row_counts)
        snapshots.append(
            (
                handoff.input_ids.detach().clone(),
                handoff.labels.detach().clone(),
                handoff.attention_mask.detach().clone(),
            )
        )

    source_ids = tuple(handoff.source_batch_sha256 for handoff in handoffs)
    projection_ids = tuple(handoff.projection_fingerprints for handoff in handoffs)

    def prefix(index: int) -> str:
        return _digest(
            (
                "haa-b10-state-prefix-v1",
                common,
                tuple(zip(source_ids[:index], projection_ids[:index], strict=True)),
            )
        )

    rng = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_embed = torch.empty(
        (synthetic_vocabulary_size, embedding_dim), dtype=torch.float32, device="cpu"
    ).uniform_(-0.2, 0.2, generator=generator)
    initial_proj = torch.empty_like(initial_embed).uniform_(-0.2, 0.2, generator=generator)
    embedding = initial_embed.detach().clone().requires_grad_(True)
    projection = initial_proj.detach().clone().requires_grad_(True)
    reference_embed = initial_embed.detach().clone().requires_grad_(True)
    reference_proj = initial_proj.detach().clone().requires_grad_(True)
    starting_hashes = (_fingerprint(initial_embed), _fingerprint(initial_proj))
    token_parts: list[tuple[int, ...]] = []
    before_losses: list[float] = []
    after_losses: list[float] = []
    grad_errors: list[float] = []
    step_errors: list[float] = []
    state_digest = ""

    for index, (handoff, row_counts) in enumerate(zip(handoffs, rows_per_handoff, strict=True)):
        rows = len(row_counts)
        parts = tuple(
            (start, min(rows, start + microbatch_rows)) for start in range(0, rows, microbatch_rows)
        )
        part_tokens = tuple(sum(row_counts[start:stop]) for start, stop in parts)
        if len(parts) < 2 or any(value < 1 for value in part_tokens):
            raise ValueError("invalid synthetic gradient partitions")
        total = handoff.total_loss_tokens
        logits_before = _causal_logits(
            handoff.input_ids, handoff.attention_mask, embedding, projection
        )
        oracle = audit_local_sft_causal_loss(handoff, logits_before.detach().contiguous())
        for start, stop in parts:
            logits = _causal_logits(
                handoff.input_ids[start:stop],
                handoff.attention_mask[start:stop],
                embedding,
                projection,
            )
            (_shifted_sum(logits, handoff.labels[start:stop]) / total).backward()  # type: ignore[no-untyped-call]
        full_logits = _causal_logits(
            handoff.input_ids, handoff.attention_mask, reference_embed, reference_proj
        )
        full_loss = _shifted_sum(full_logits, handoff.labels) / total
        full_loss.backward()  # type: ignore[no-untyped-call]
        if not math.isclose(float(full_loss.item()), oracle.mean_nll, abs_tol=3e-6, rel_tol=3e-5):
            raise ValueError("synthetic step forward oracle mismatch")
        if (
            embedding.grad is None
            or projection.grad is None
            or reference_embed.grad is None
            or reference_proj.grad is None
        ):
            raise ValueError("missing synthetic gradient")
        grad_error = max(
            _close(embedding.grad, reference_embed.grad, "chain embedding gradient"),
            _close(projection.grad, reference_proj.grad, "chain projection gradient"),
        )
        if (
            float(torch.linalg.vector_norm(embedding.grad).item()) <= 0
            or float(torch.linalg.vector_norm(projection.grad).item()) <= 0
        ):
            raise ValueError("missing trainable synthetic signal")
        torch.optim.SGD([embedding, projection], lr=learning_rate).step()
        torch.optim.SGD([reference_embed, reference_proj], lr=learning_rate).step()
        step_error = max(
            _close(embedding.detach(), reference_embed.detach(), "chain embedding step"),
            _close(projection.detach(), reference_proj.detach(), "chain projection step"),
        )
        if _fingerprint(embedding) == starting_hashes[0] and index == 0:
            raise ValueError("unchanged synthetic embedding")
        if _fingerprint(projection) == starting_hashes[1] and index == 0:
            raise ValueError("unchanged synthetic projection")
        after_logits = _causal_logits(
            handoff.input_ids, handoff.attention_mask, embedding, projection
        )
        after = audit_local_sft_causal_loss(handoff, after_logits.detach().contiguous())
        token_parts.append(part_tokens)
        before_losses.append(oracle.mean_nll)
        after_losses.append(after.mean_nll)
        grad_errors.append(grad_error)
        step_errors.append(step_error)
        if index + 1 == resume_after_step:
            sealed = seal_local_synthetic_state(embedding, projection, index + 1, prefix(index + 1))
            embedding, projection = restore_local_synthetic_state(
                sealed, index + 1, prefix(index + 1), synthetic_vocabulary_size, embedding_dim
            )
            state_digest = sealed.payload_sha256
        else:
            embedding = embedding.detach().clone().requires_grad_(True)
            projection = projection.detach().clone().requires_grad_(True)
        reference_embed = reference_embed.detach().clone().requires_grad_(True)
        reference_proj = reference_proj.detach().clone().requires_grad_(True)

    for handoff, (before_ids, before_labels, before_mask) in zip(handoffs, snapshots, strict=True):
        validate_local_sft_tensors(handoff)
        if (
            not torch.equal(handoff.input_ids, before_ids)
            or not torch.equal(handoff.labels, before_labels)
            or not torch.equal(handoff.attention_mask, before_mask)
        ):
            raise ValueError("source tensors mutated during synthetic state chain")
    if not torch.equal(torch.get_rng_state(), rng):
        raise ValueError("global CPU RNG changed during synthetic chain")
    if not state_digest:
        raise ValueError("missing synthetic intermediate state")
    _close(embedding.detach(), reference_embed.detach(), "final resumed embedding")
    _close(projection.detach(), reference_proj.detach(), "final resumed projection")
    return LocalSFTSyntheticStateChainDiagnostic(
        ordered_source_batch_sha256=source_ids,
        ordered_projection_fingerprints=projection_ids,
        source_chain_sha256=prefix(len(handoffs)),
        total_steps=len(handoffs),
        resume_after_step=resume_after_step,
        microbatch_loss_tokens=tuple(token_parts),
        mean_nll_before=tuple(before_losses),
        mean_nll_after=tuple(after_losses),
        max_gradient_parity_error=tuple(grad_errors),
        max_weight_parity_error=tuple(step_errors),
        initial_embedding_sha256=starting_hashes[0],
        initial_projection_sha256=starting_hashes[1],
        intermediate_state_sha256=state_digest,
        final_embedding_sha256=_fingerprint(embedding),
        final_projection_sha256=_fingerprint(projection),
    )
