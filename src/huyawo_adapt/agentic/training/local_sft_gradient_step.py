"""CPU-only synthetic SFT gradient and single-step SGD qualification, without training authority."""

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
_LOSS_ATOL = 2e-6
_LOSS_RTOL = 2e-5
_GRADIENT_ATOL = 2e-6
_GRADIENT_RTOL = 2e-5


@dataclass(frozen=True, slots=True, eq=False)
class LocalSFTSyntheticGradientStepDiagnostic:
    """Numerical evidence from private fake logits; never a model-training receipt."""

    source_batch_sha256: str
    split: SplitName
    manifest_fingerprint: str
    projection_fingerprints: tuple[str, ...]
    model_fingerprint: str
    tokenizer_fingerprint: str
    projection_policy_sha256: str
    synthetic_vocabulary_size: int
    total_loss_tokens: int
    row_loss_tokens: tuple[int, ...]
    learning_rate: float
    pre_step_mean_nll: float
    post_step_mean_nll: float
    gradient_l2_norm: float
    ignored_gradient_positions: int
    changed_logit_cells: int
    authority_scope: Literal["local_sft_synthetic_gradient_step_only"] = (
        "local_sft_synthetic_gradient_step_only"
    )
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_gradient_step_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
        ):
            raise ValueError("positive training authority is forbidden")
        if (
            not self.row_loss_tokens
            or self.total_loss_tokens != sum(self.row_loss_tokens)
            or self.total_loss_tokens < 1
            or any(type(count) is not int or count < 1 for count in self.row_loss_tokens)
            or not 2 <= self.synthetic_vocabulary_size <= _MAX_SYNTHETIC_VOCAB
            or type(self.ignored_gradient_positions) is not int
            or self.ignored_gradient_positions < 1
            or type(self.changed_logit_cells) is not int
            or self.changed_logit_cells < 1
        ):
            raise ValueError("invalid synthetic gradient accounting")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or not 0.0 < self.learning_rate <= 0.1
            or not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.pre_step_mean_nll,
                    self.post_step_mean_nll,
                    self.gradient_l2_norm,
                )
            )
            or self.gradient_l2_norm <= 0.0
        ):
            raise ValueError("invalid synthetic gradient values")


def audit_local_sft_synthetic_gradient_step(
    handoff: LocalSFTTensorDiagnostic,
    synthetic_logits: torch.Tensor,
    learning_rate: float,
) -> LocalSFTSyntheticGradientStepDiagnostic:
    """Validate one private synthetic autograd/SGD update against the B6 loss oracle."""
    validate_local_sft_tensors(handoff)
    if type(learning_rate) is not float or not math.isfinite(learning_rate):
        raise ValueError("finite exact float learning rate required")
    if not 0.0 < learning_rate <= 0.1:
        raise ValueError("learning rate outside synthetic bound")
    if type(synthetic_logits) is not torch.Tensor:
        raise TypeError("exact synthetic logits tensor required")
    if synthetic_logits.ndim != 3:
        raise ValueError("synthetic logits require rank three")
    rows, width, vocab = (int(value) for value in synthetic_logits.shape)
    if not 2 <= vocab <= _MAX_SYNTHETIC_VOCAB:
        raise ValueError("synthetic vocabulary bound exceeded")
    if rows * width * vocab > _MAX_GRADIENT_CELLS:
        raise ValueError("synthetic gradient cell cap exceeded")
    if (rows, width) != tuple(handoff.input_ids.shape):
        raise ValueError("synthetic logits do not match the B5 shape")
    if (
        synthetic_logits.device.type != "cpu"
        or synthetic_logits.device.index is not None
        or synthetic_logits.dtype != torch.float32
        or synthetic_logits.layout != torch.strided
        or synthetic_logits.requires_grad
        or not synthetic_logits.is_contiguous()
    ):
        raise ValueError("synthetic logits require contiguous CPU float32 without grad")
    if synthetic_logits.untyped_storage().data_ptr() in {
        handoff.input_ids.untyped_storage().data_ptr(),
        handoff.labels.untyped_storage().data_ptr(),
        handoff.attention_mask.untyped_storage().data_ptr(),
    }:
        raise ValueError("synthetic logits must not alias source storage")
    if not bool(torch.isfinite(synthetic_logits).all().item()):
        raise ValueError("nonfinite synthetic logits")
    if bool(torch.any(handoff.input_ids >= vocab).item()):
        raise ValueError("synthetic vocabulary does not cover source input IDs")
    if bool(torch.any((handoff.labels != -100) & (handoff.labels >= vocab)).item()):
        raise ValueError("synthetic vocabulary does not cover target IDs")

    original_logits = synthetic_logits.detach().clone()
    oracle_before = audit_local_sft_causal_loss(handoff, original_logits)
    count = handoff.total_loss_tokens
    shifted_labels = handoff.labels[:, 1:]
    supervised = shifted_labels != -100
    positions = torch.zeros((rows, width), dtype=torch.bool, device="cpu")
    positions[:, :-1] = supervised
    ignored_count = int((~positions).sum().item())
    if (
        not bool(supervised.any().item())
        or oracle_before.row_loss_tokens != handoff.source_batch.row_target_lengths
        or oracle_before.total_loss_tokens != count
    ):
        raise ValueError("supervised token accounting mismatch")

    with torch.enable_grad():
        private_logits = original_logits.detach().clone().requires_grad_(True)
        shifted_logits = private_logits[:, :-1, :].contiguous().reshape(-1, vocab)
        flat_labels = shifted_labels.contiguous().reshape(-1)
        summed_loss = F.cross_entropy(
            shifted_logits,
            flat_labels,
            ignore_index=-100,
            reduction="sum",
        )
        mean_loss = summed_loss / count
        if not bool(torch.isfinite(mean_loss.detach()).item()):
            raise ValueError("nonfinite synthetic forward loss")
        pre_mean = float(mean_loss.detach().item())
        if not math.isclose(
            pre_mean, oracle_before.mean_nll, rel_tol=_LOSS_RTOL, abs_tol=_LOSS_ATOL
        ):
            raise ValueError("B6 forward oracle disagreement")
        mean_loss.backward()

    gradient = private_logits.grad
    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
        raise ValueError("synthetic gradient unavailable or nonfinite")
    if bool(torch.count_nonzero(gradient[~positions]).item()):
        raise ValueError("ignored positions have nonzero gradient")
    with torch.no_grad():
        reference = torch.zeros_like(private_logits)
        row_index, position_index = supervised.nonzero(as_tuple=True)
        probabilities = torch.softmax(private_logits.detach()[row_index, position_index], dim=-1)
        expected_values = probabilities.clone()
        expected_values[torch.arange(len(row_index)), shifted_labels[supervised]] -= 1.0
        reference[row_index, position_index] = expected_values / count
        if not bool(torch.allclose(gradient, reference, rtol=_GRADIENT_RTOL, atol=_GRADIENT_ATOL)):
            raise ValueError("synthetic analytical gradient disagreement")
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("invalid synthetic gradient norm")

    optimizer = torch.optim.SGD([private_logits], lr=learning_rate, momentum=0.0, weight_decay=0.0)
    optimizer.step()
    with torch.no_grad():
        if not bool(torch.isfinite(private_logits).all().item()):
            raise ValueError("nonfinite synthetic poststep logits")
        changed_cells = int(torch.count_nonzero(private_logits != original_logits).item())
        if changed_cells < 1:
            raise ValueError("synthetic step changed no private logit")
        oracle_after = audit_local_sft_causal_loss(handoff, private_logits.detach())
        post_mean = oracle_after.mean_nll
        if post_mean > pre_mean + _LOSS_ATOL:
            raise ValueError("synthetic SGD step unexpectedly increases the loss")
        if not bool(torch.equal(synthetic_logits, original_logits)):
            raise ValueError("caller-owned synthetic logits were modified")
        validate_local_sft_tensors(handoff)

    return LocalSFTSyntheticGradientStepDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        split=handoff.split,
        manifest_fingerprint=handoff.manifest_fingerprint,
        projection_fingerprints=handoff.projection_fingerprints,
        model_fingerprint=handoff.model_fingerprint,
        tokenizer_fingerprint=handoff.tokenizer_fingerprint,
        projection_policy_sha256=handoff.projection_policy_sha256,
        synthetic_vocabulary_size=vocab,
        total_loss_tokens=count,
        row_loss_tokens=oracle_before.row_loss_tokens,
        learning_rate=learning_rate,
        pre_step_mean_nll=pre_mean,
        post_step_mean_nll=post_mean,
        gradient_l2_norm=gradient_norm,
        ignored_gradient_positions=ignored_count,
        changed_logit_cells=changed_cells,
    )
