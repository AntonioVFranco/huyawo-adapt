"""CPU-only synthetic token-weighted SFT accumulation parity, never real training."""

from __future__ import annotations

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

_MAX_GRADIENT_CELLS = 262_144
_MAX_SYNTHETIC_VOCAB = 256
_MAX_ROWS = 32
_LOSS_ABS_TOL = 2e-6
_LOSS_REL_TOL = 2e-5
_GRAD_ABS_TOL = 2e-6
_GRAD_REL_TOL = 2e-5
_STEP_ABS_TOL = 2e-6
_STEP_REL_TOL = 2e-5


@dataclass(frozen=True, slots=True, eq=False)
class LocalSFTSyntheticAccumulationDiagnostic:
    """Private fake-logit parity evidence without model or training authority."""

    source_batch_sha256: str
    split: SplitName
    manifest_fingerprint: str
    projection_fingerprints: tuple[str, ...]
    model_fingerprint: str
    tokenizer_fingerprint: str
    projection_policy_sha256: str
    synthetic_vocabulary_size: int
    partition_row_counts: tuple[int, ...]
    microbatch_loss_tokens: tuple[int, ...]
    total_loss_tokens: int
    learning_rate: float
    pre_step_mean_nll: float
    post_step_mean_nll: float
    gradient_l2_norm: float
    max_gradient_parity_abs_error: float
    max_step_parity_abs_error: float
    ignored_gradient_positions: int
    changed_logit_cells: int
    authority_scope: Literal["local_sft_synthetic_accumulation_only"] = (
        "local_sft_synthetic_accumulation_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_accumulation_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
        ):
            raise ValueError("positive training authority is forbidden")
        if (
            len(self.partition_row_counts) < 2
            or len(self.microbatch_loss_tokens) != len(self.partition_row_counts)
            or any(type(n) is not int or n < 1 for n in self.partition_row_counts)
            or any(type(n) is not int or n < 1 for n in self.microbatch_loss_tokens)
            or type(self.total_loss_tokens) is not int
            or self.total_loss_tokens != sum(self.microbatch_loss_tokens)
            or not 2 <= self.synthetic_vocabulary_size <= _MAX_SYNTHETIC_VOCAB
            or type(self.ignored_gradient_positions) is not int
            or self.ignored_gradient_positions < 1
            or type(self.changed_logit_cells) is not int
            or self.changed_logit_cells < 1
        ):
            raise ValueError("invalid synthetic accumulation accounting")
        if type(self.learning_rate) is not float or not 0.0 < self.learning_rate <= 0.1:
            raise ValueError("invalid synthetic learning rate")
        numeric = (
            self.learning_rate,
            self.pre_step_mean_nll,
            self.post_step_mean_nll,
            self.gradient_l2_norm,
            self.max_gradient_parity_abs_error,
            self.max_step_parity_abs_error,
        )
        if not all(math.isfinite(x) and x >= 0 for x in numeric) or self.gradient_l2_norm <= 0:
            raise ValueError("invalid synthetic accumulation numerical result")


def audit_local_sft_synthetic_accumulation(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_logits: torch.Tensor,
    learning_rate: float,
    microbatch_rows: int,
) -> LocalSFTSyntheticAccumulationDiagnostic:
    """Backpropagate globally weighted fake microbatches, then take one private step."""
    validate_local_sft_tensors(handoff)
    if type(learning_rate) is not float or not math.isfinite(learning_rate):
        raise ValueError("finite exact-float learning rate required")
    if not 0.0 < learning_rate <= 0.1:
        raise ValueError("learning rate outside synthetic bound")
    if type(microbatch_rows) is not int:
        raise TypeError("exact integer microbatch row count required")
    if type(synthetic_logits) is not torch.Tensor:
        raise TypeError("exact synthetic tensor required")
    if synthetic_logits.ndim != 3:
        raise ValueError("rank-three synthetic logits required")
    rows, width, vocab = (int(n) for n in synthetic_logits.shape)
    if not 2 <= rows <= _MAX_ROWS or not 1 <= microbatch_rows < rows:
        raise ValueError("invalid microbatch partition")
    if not 2 <= vocab <= _MAX_SYNTHETIC_VOCAB:
        raise ValueError("synthetic vocabulary bound exceeded")
    if rows * width * vocab > _MAX_GRADIENT_CELLS:
        raise ValueError("synthetic accumulation cell cap exceeded")
    if (rows, width) != tuple(handoff.input_ids.shape):
        raise ValueError("synthetic logits do not match B5 dimensions")
    if (
        synthetic_logits.device.type != "cpu"
        or synthetic_logits.device.index is not None
        or synthetic_logits.dtype != torch.float32
        or synthetic_logits.layout != torch.strided
        or synthetic_logits.requires_grad
        or not synthetic_logits.is_contiguous()
    ):
        raise ValueError("contiguous non-grad CPU float32 synthetic logits required")
    if synthetic_logits.untyped_storage().data_ptr() in {
        handoff.input_ids.untyped_storage().data_ptr(),
        handoff.labels.untyped_storage().data_ptr(),
        handoff.attention_mask.untyped_storage().data_ptr(),
    }:
        raise ValueError("synthetic logits alias B5 source storage")
    if not bool(torch.isfinite(synthetic_logits).all().item()):
        raise ValueError("nonfinite synthetic logits")
    if bool(torch.any(handoff.input_ids >= vocab).item()) or bool(
        torch.any((handoff.labels != -100) & (handoff.labels >= vocab)).item()
    ):
        raise ValueError("synthetic vocabulary excludes source or target token IDs")

    original = synthetic_logits.detach().clone()
    oracle_before = audit_local_sft_causal_loss(handoff, original)
    global_tokens = handoff.total_loss_tokens
    shifted_labels = handoff.labels[:, 1:]
    per_row_tokens = tuple(int(n) for n in (shifted_labels != -100).sum(1).tolist())
    if (
        not per_row_tokens
        or any(n < 1 for n in per_row_tokens)
        or per_row_tokens != handoff.source_batch.row_target_lengths
        or global_tokens != sum(per_row_tokens)
        or global_tokens != oracle_before.total_loss_tokens
    ):
        raise ValueError("B5/B6 supervised-token accounting disagreement")

    partition_counts: list[int] = []
    microbatch_counts: list[int] = []
    for start in range(0, rows, microbatch_rows):
        stop = min(start + microbatch_rows, rows)
        partition_counts.append(stop - start)
        microbatch_counts.append(sum(per_row_tokens[start:stop]))
    if len(partition_counts) < 2 or any(n < 1 for n in microbatch_counts):
        raise ValueError("each synthetic microbatch must carry supervised targets")

    positions = torch.zeros((rows, width), dtype=torch.bool, device="cpu")
    positions[:, :-1] = shifted_labels != -100
    ignored_count = int((~positions).sum().item())

    with torch.enable_grad():
        private = original.detach().clone().requires_grad_(True)
        accumulated_losses: list[float] = []
        for start in range(0, rows, microbatch_rows):
            stop = min(start + microbatch_rows, rows)
            partial_logits = private[start:stop, :-1, :].contiguous().reshape(-1, vocab)
            partial_targets = shifted_labels[start:stop].contiguous().reshape(-1)
            partial_sum = F.cross_entropy(
                partial_logits,
                partial_targets,
                ignore_index=-100,
                reduction="sum",
            )
            scaled_loss = partial_sum / global_tokens
            if not bool(torch.isfinite(scaled_loss.detach()).item()):
                raise ValueError("nonfinite synthetic partial loss")
            accumulated_losses.append(float(scaled_loss.detach().item()))
            scaled_loss.backward()

        reference = original.detach().clone().requires_grad_(True)
        full_loss = (
            F.cross_entropy(
                reference[:, :-1, :].contiguous().reshape(-1, vocab),
                shifted_labels.contiguous().reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            / global_tokens
        )
        if not bool(torch.isfinite(full_loss.detach()).item()):
            raise ValueError("nonfinite synthetic full-batch loss")
        full_loss.backward()

    accumulation_grad = private.grad
    reference_grad = reference.grad
    if accumulation_grad is None or reference_grad is None:
        raise ValueError("missing synthetic accumulation/full-batch gradient")
    if not bool(torch.isfinite(accumulation_grad).all().item()) or not bool(
        torch.isfinite(reference_grad).all().item()
    ):
        raise ValueError("nonfinite synthetic gradient")
    if bool(torch.count_nonzero(accumulation_grad[~positions]).item()) or bool(
        torch.count_nonzero(reference_grad[~positions]).item()
    ):
        raise ValueError("ignored synthetic positions received gradients")
    max_grad_error = float(torch.max(torch.abs(accumulation_grad - reference_grad)).item())
    if not bool(
        torch.allclose(accumulation_grad, reference_grad, atol=_GRAD_ABS_TOL, rtol=_GRAD_REL_TOL)
    ):
        raise ValueError("microbatch/full-batch gradient parity failed")
    pre_mean = math.fsum(accumulated_losses)
    full_mean = float(full_loss.detach().item())
    if not math.isclose(pre_mean, full_mean, rel_tol=_LOSS_REL_TOL, abs_tol=_LOSS_ABS_TOL):
        raise ValueError("microbatch/full-batch loss parity failed")
    if not math.isclose(
        pre_mean, oracle_before.mean_nll, rel_tol=_LOSS_REL_TOL, abs_tol=_LOSS_ABS_TOL
    ):
        raise ValueError("microbatch/B6 loss parity failed")
    gradient_norm = float(torch.linalg.vector_norm(accumulation_grad).item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("invalid synthetic gradient norm")

    with torch.no_grad():
        expected_step = original - learning_rate * reference_grad
    optimizer = torch.optim.SGD([private], lr=learning_rate, momentum=0.0, weight_decay=0.0)
    optimizer.step()
    with torch.no_grad():
        if not bool(torch.isfinite(private).all().item()):
            raise ValueError("nonfinite synthetic private poststep logits")
        max_step_error = float(torch.max(torch.abs(private - expected_step)).item())
        if not bool(torch.allclose(private, expected_step, atol=_STEP_ABS_TOL, rtol=_STEP_REL_TOL)):
            raise ValueError("synthetic SGD full-batch step parity failed")
        if bool(torch.count_nonzero((private - original)[~positions]).item()):
            raise ValueError("synthetic SGD changed an ignored position")
        changed = int(torch.count_nonzero(private != original).item())
        if changed < 1:
            raise ValueError("synthetic optimizer changed no private logits")
        post_oracle = audit_local_sft_causal_loss(handoff, private.detach())
        if post_oracle.mean_nll > pre_mean + _LOSS_ABS_TOL:
            raise ValueError("synthetic SGD increased the loss")
        if not bool(torch.equal(synthetic_logits, original)):
            raise ValueError("caller logits changed during synthetic accumulation")
        validate_local_sft_tensors(handoff)

    return LocalSFTSyntheticAccumulationDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        split=handoff.split,
        manifest_fingerprint=handoff.manifest_fingerprint,
        projection_fingerprints=handoff.projection_fingerprints,
        model_fingerprint=handoff.model_fingerprint,
        tokenizer_fingerprint=handoff.tokenizer_fingerprint,
        projection_policy_sha256=handoff.projection_policy_sha256,
        synthetic_vocabulary_size=vocab,
        partition_row_counts=tuple(partition_counts),
        microbatch_loss_tokens=tuple(microbatch_counts),
        total_loss_tokens=global_tokens,
        learning_rate=learning_rate,
        pre_step_mean_nll=pre_mean,
        post_step_mean_nll=post_oracle.mean_nll,
        gradient_l2_norm=gradient_norm,
        max_gradient_parity_abs_error=max_grad_error,
        max_step_parity_abs_error=max_step_error,
        ignored_gradient_positions=ignored_count,
        changed_logit_cells=changed,
    )
