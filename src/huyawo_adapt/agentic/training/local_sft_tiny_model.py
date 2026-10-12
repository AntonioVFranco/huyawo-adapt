"""Bounded CPU synthetic trainable-parameter SFT parity, never genuine training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from huyawo_adapt.agentic.training.local_sft_loss_audit import audit_local_sft_causal_loss
from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    validate_local_sft_tensors,
)
from huyawo_adapt.agentic.training.sft import SplitName

_GRAD_ATOL = 3e-6
_GRAD_RTOL = 3e-5
_LOSS_ATOL = 3e-6
_LOSS_RTOL = 3e-5
_MAX_LOGIT_CELLS = 32_768


@dataclass(frozen=True, slots=True, eq=False)
class LocalSFTSyntheticTinyParameterDiagnostic:
    """Numeric parity for private fake trainable parameters, without SFT authority."""

    source_batch_sha256: str
    split: SplitName
    manifest_fingerprint: str
    projection_fingerprints: tuple[str, ...]
    model_fingerprint: str
    tokenizer_fingerprint: str
    projection_policy_sha256: str
    synthetic_vocabulary_size: int
    embedding_dim: int
    seed: int
    partition_row_counts: tuple[int, ...]
    microbatch_loss_tokens: tuple[int, ...]
    total_loss_tokens: int
    learning_rate: float
    pre_step_mean_nll: float
    post_step_mean_nll: float
    embedding_gradient_l2_norm: float
    projection_gradient_l2_norm: float
    max_embedding_gradient_parity_abs_error: float
    max_projection_gradient_parity_abs_error: float
    max_embedding_step_parity_abs_error: float
    max_projection_step_parity_abs_error: float
    changed_embedding_cells: int
    changed_projection_cells: int
    initial_embedding_sha256: str
    initial_projection_sha256: str
    updated_embedding_sha256: str
    updated_projection_sha256: str
    accumulation_optimizer_steps: Literal[1] = 1
    reference_optimizer_steps: Literal[1] = 1
    authority_scope: Literal["local_sft_synthetic_tiny_parameter_only"] = (
        "local_sft_synthetic_tiny_parameter_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_tiny_parameter_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
            or self.accumulation_optimizer_steps != 1
            or self.reference_optimizer_steps != 1
        ):
            raise ValueError("positive training authority or step count is forbidden")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.synthetic_vocabulary_size) is not int
            or not 2 <= self.synthetic_vocabulary_size <= 64
            or type(self.embedding_dim) is not int
            or not 2 <= self.embedding_dim <= 16
            or len(self.partition_row_counts) < 2
            or len(self.partition_row_counts) != len(self.microbatch_loss_tokens)
            or any(type(n) is not int or n <= 0 for n in self.partition_row_counts)
            or any(type(n) is not int or n <= 0 for n in self.microbatch_loss_tokens)
            or type(self.total_loss_tokens) is not int
            or self.total_loss_tokens != sum(self.microbatch_loss_tokens)
            or type(self.changed_embedding_cells) is not int
            or self.changed_embedding_cells <= 0
            or type(self.changed_projection_cells) is not int
            or self.changed_projection_cells <= 0
        ):
            raise ValueError("invalid synthetic parameter accounting")
        digests = (
            self.initial_embedding_sha256,
            self.initial_projection_sha256,
            self.updated_embedding_sha256,
            self.updated_projection_sha256,
        )
        if any(
            len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            for digest in digests
        ):
            raise ValueError("invalid synthetic parameter fingerprint")
        numbers = (
            self.pre_step_mean_nll,
            self.post_step_mean_nll,
            self.embedding_gradient_l2_norm,
            self.projection_gradient_l2_norm,
            self.max_embedding_gradient_parity_abs_error,
            self.max_projection_gradient_parity_abs_error,
            self.max_embedding_step_parity_abs_error,
            self.max_projection_step_parity_abs_error,
        )
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or not 0.0 < self.learning_rate <= 0.05
            or any(not math.isfinite(n) or n < 0 for n in numbers)
            or self.embedding_gradient_l2_norm <= 0
            or self.projection_gradient_l2_norm <= 0
        ):
            raise ValueError("invalid synthetic parameter numeric evidence")


def _fingerprint(weight: torch.Tensor) -> str:
    """Fingerprint local float32 bytes only, not an authenticated model artifact."""
    return hashlib.sha256(weight.detach().contiguous().numpy().tobytes()).hexdigest()


def _causal_logits(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    projection_weight: torch.Tensor,
) -> torch.Tensor:
    attended = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
    vectors = F.embedding(input_ids, embedding_weight) * attended
    prefix_sum = vectors.cumsum(dim=1)
    prefix_count = attended.cumsum(dim=1).clamp_min(1)
    prefix_mean = prefix_sum / prefix_count
    return F.linear(prefix_mean, projection_weight)


def _shifted_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _close(left: torch.Tensor, right: torch.Tensor, kind: str) -> float:
    if not bool(torch.isfinite(left).all().item()) or not bool(torch.isfinite(right).all().item()):
        raise ValueError(f"nonfinite synthetic {kind}")
    error = float(torch.max(torch.abs(left - right)).item())
    if not torch.allclose(left, right, rtol=_GRAD_RTOL, atol=_GRAD_ATOL):
        raise ValueError(f"synthetic {kind} parity mismatch")
    return error


def audit_local_sft_synthetic_tiny_parameter_step(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_vocabulary_size: int,
    embedding_dim: int,
    seed: int,
    learning_rate: float,
    microbatch_rows: int,
) -> LocalSFTSyntheticTinyParameterDiagnostic:
    """Compare globally weighted parameter accumulation with private full-batch SGD."""
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
        or not 2 <= rows <= 8
        or not 2 <= width <= 64
        or not 1 <= microbatch_rows < rows
        or rows * width * synthetic_vocabulary_size > _MAX_LOGIT_CELLS
    ):
        raise ValueError("invalid bounded synthetic parameter input")
    if bool(torch.any(handoff.input_ids >= synthetic_vocabulary_size).item()) or bool(
        torch.any((handoff.labels != -100) & (handoff.labels >= synthetic_vocabulary_size)).item()
    ):
        raise ValueError("synthetic vocabulary excludes a source or target ID")

    row_counts = tuple(int(v) for v in (handoff.labels[:, 1:] != -100).sum(1).tolist())
    if (
        any(v <= 0 for v in row_counts)
        or row_counts != handoff.source_batch.row_target_lengths
        or sum(row_counts) != handoff.total_loss_tokens
    ):
        raise ValueError("invalid global supervised target count")
    parts = tuple(
        (start, min(rows, start + microbatch_rows)) for start in range(0, rows, microbatch_rows)
    )
    part_rows = tuple(stop - start for start, stop in parts)
    part_tokens = tuple(sum(row_counts[start:stop]) for start, stop in parts)
    if len(parts) < 2 or any(value < 1 for value in part_tokens):
        raise ValueError("missing synthetic microbatch supervision")

    input_original = handoff.input_ids.detach().clone()
    labels_original = handoff.labels.detach().clone()
    mask_original = handoff.attention_mask.detach().clone()
    global_rng_original = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_embedding = torch.empty(
        (synthetic_vocabulary_size, embedding_dim), dtype=torch.float32, device="cpu"
    ).uniform_(-0.2, 0.2, generator=generator)
    initial_projection = torch.empty(
        (synthetic_vocabulary_size, embedding_dim), dtype=torch.float32, device="cpu"
    ).uniform_(-0.2, 0.2, generator=generator)
    embedding = initial_embedding.detach().clone().requires_grad_(True)
    projection = initial_projection.detach().clone().requires_grad_(True)
    reference_embedding = initial_embedding.detach().clone().requires_grad_(True)
    reference_projection = initial_projection.detach().clone().requires_grad_(True)
    if (
        len(
            {
                x.untyped_storage().data_ptr()
                for x in (
                    embedding,
                    projection,
                    reference_embedding,
                    reference_projection,
                    handoff.input_ids,
                    handoff.labels,
                    handoff.attention_mask,
                )
            }
        )
        != 7
    ):
        raise ValueError("synthetic parameter storage is aliased")

    total = handoff.total_loss_tokens
    for start, stop in parts:
        logits = _causal_logits(
            handoff.input_ids[start:stop],
            handoff.attention_mask[start:stop],
            embedding,
            projection,
        )
        part_loss = _shifted_sum(logits, handoff.labels[start:stop]) / total
        part_loss.backward()  # type: ignore[no-untyped-call]
    embedding_grad = embedding.grad
    projection_grad = projection.grad
    if embedding_grad is None or projection_grad is None:
        raise ValueError("missing trainable synthetic gradient")
    accumulation_embedding_gradient = embedding_grad.detach().clone()
    accumulation_projection_gradient = projection_grad.detach().clone()

    full_logits = _causal_logits(
        handoff.input_ids, handoff.attention_mask, reference_embedding, reference_projection
    )
    full_loss = _shifted_sum(full_logits, handoff.labels) / total
    full_loss.backward()  # type: ignore[no-untyped-call]
    ref_embed_grad = reference_embedding.grad
    ref_proj_grad = reference_projection.grad
    if ref_embed_grad is None or ref_proj_grad is None:
        raise ValueError("missing full-batch synthetic gradient")
    ref_embedding_gradient = ref_embed_grad.detach().clone()
    ref_projection_gradient = ref_proj_grad.detach().clone()
    embedding_error = _close(
        accumulation_embedding_gradient, ref_embedding_gradient, "embedding gradient"
    )
    projection_error = _close(
        accumulation_projection_gradient, ref_projection_gradient, "projection gradient"
    )
    embedding_norm = float(torch.linalg.vector_norm(accumulation_embedding_gradient).item())
    projection_norm = float(torch.linalg.vector_norm(accumulation_projection_gradient).item())
    if (
        not math.isfinite(embedding_norm)
        or not math.isfinite(projection_norm)
        or min(embedding_norm, projection_norm) <= 0
    ):
        raise ValueError("no learnable signal in synthetic fixture")

    before = audit_local_sft_causal_loss(handoff, full_logits.detach().contiguous())
    if not math.isclose(
        float(full_loss.item()), before.mean_nll, abs_tol=_LOSS_ATOL, rel_tol=_LOSS_RTOL
    ):
        raise ValueError("synthetic B6 forward-loss oracle mismatch")
    initial_embedding_sha = _fingerprint(initial_embedding)
    initial_projection_sha = _fingerprint(initial_projection)

    torch.optim.SGD(
        [embedding, projection], lr=learning_rate, momentum=0.0, weight_decay=0.0
    ).step()
    torch.optim.SGD(
        [reference_embedding, reference_projection],
        lr=learning_rate,
        momentum=0.0,
        weight_decay=0.0,
    ).step()
    embedding_step_error = _close(
        embedding.detach(), reference_embedding.detach(), "embedding SGD step"
    )
    projection_step_error = _close(
        projection.detach(), reference_projection.detach(), "projection SGD step"
    )
    changed_embedding = int(torch.count_nonzero(embedding.detach() != initial_embedding).item())
    changed_projection = int(torch.count_nonzero(projection.detach() != initial_projection).item())
    if changed_embedding < 1 or changed_projection < 1:
        raise ValueError("synthetic SGD did not change both parameter matrices")
    after_logits = _causal_logits(handoff.input_ids, handoff.attention_mask, embedding, projection)
    after = audit_local_sft_causal_loss(handoff, after_logits.detach().contiguous())

    validate_local_sft_tensors(handoff)
    if (
        not torch.equal(handoff.input_ids, input_original)
        or not torch.equal(handoff.labels, labels_original)
        or not torch.equal(handoff.attention_mask, mask_original)
        or not torch.equal(torch.get_rng_state(), global_rng_original)
        or _fingerprint(initial_embedding) != initial_embedding_sha
        or _fingerprint(initial_projection) != initial_projection_sha
    ):
        raise ValueError("synthetic operation changed caller, initial weights or global RNG")

    return LocalSFTSyntheticTinyParameterDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        split=handoff.split,
        manifest_fingerprint=handoff.manifest_fingerprint,
        projection_fingerprints=handoff.projection_fingerprints,
        model_fingerprint=handoff.model_fingerprint,
        tokenizer_fingerprint=handoff.tokenizer_fingerprint,
        projection_policy_sha256=handoff.projection_policy_sha256,
        synthetic_vocabulary_size=synthetic_vocabulary_size,
        embedding_dim=embedding_dim,
        seed=seed,
        partition_row_counts=part_rows,
        microbatch_loss_tokens=part_tokens,
        total_loss_tokens=total,
        learning_rate=learning_rate,
        pre_step_mean_nll=before.mean_nll,
        post_step_mean_nll=after.mean_nll,
        embedding_gradient_l2_norm=embedding_norm,
        projection_gradient_l2_norm=projection_norm,
        max_embedding_gradient_parity_abs_error=embedding_error,
        max_projection_gradient_parity_abs_error=projection_error,
        max_embedding_step_parity_abs_error=embedding_step_error,
        max_projection_step_parity_abs_error=projection_step_error,
        changed_embedding_cells=changed_embedding,
        changed_projection_cells=changed_projection,
        initial_embedding_sha256=initial_embedding_sha,
        initial_projection_sha256=initial_projection_sha,
        updated_embedding_sha256=_fingerprint(embedding),
        updated_projection_sha256=_fingerprint(projection),
    )
