"""Synthetic CPU-only causal SFT loss audit without training authority."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from huyawo_adapt.agentic.training.local_sft_tensor import (
    LocalSFTTensorDiagnostic,
    validate_local_sft_tensors,
)
from huyawo_adapt.agentic.training.sft import SplitName

_MAX_LOGIT_CELLS = 1_048_576
_MAX_SYNTHETIC_VOCAB = 256


@dataclass(frozen=True, slots=True, eq=False)
class LocalSFTLossDiagnostic:
    """Numerical diagnostic from untrusted fake logits, never a training receipt."""

    source_batch_sha256: str
    split: SplitName
    manifest_fingerprint: str
    projection_fingerprints: tuple[str, ...]
    model_fingerprint: str
    tokenizer_fingerprint: str
    projection_policy_sha256: str
    synthetic_vocabulary_size: int
    row_loss_tokens: tuple[int, ...]
    total_loss_tokens: int
    per_row_nll: tuple[float, ...]
    total_nll: float
    mean_nll: float
    authority_scope: Literal["local_sft_synthetic_loss_only"] = "local_sft_synthetic_loss_only"
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.authority_scope != "local_sft_synthetic_loss_only"
            or self.independent_task_outcome_verified is not False
            or self.model_origin_attested is not False
            or self.tokenizer_fidelity_attested is not False
            or self.training_eligible is not False
        ):
            raise ValueError("positive authority is forbidden")
        if (
            not self.row_loss_tokens
            or len(self.row_loss_tokens) != len(self.per_row_nll)
            or any(type(value) is not int or value < 1 for value in self.row_loss_tokens)
            or sum(self.row_loss_tokens) != self.total_loss_tokens
            or not 2 <= self.synthetic_vocabulary_size <= _MAX_SYNTHETIC_VOCAB
        ):
            raise ValueError("invalid local loss accounting")
        if (
            any(not math.isfinite(value) or value < 0.0 for value in self.per_row_nll)
            or not math.isfinite(self.total_nll)
            or self.total_nll < 0.0
            or not math.isfinite(self.mean_nll)
            or self.mean_nll < 0.0
        ):
            raise ValueError("nonfinite local loss")


def audit_local_sft_causal_loss(
    handoff: LocalSFTTensorDiagnostic,
    logits: torch.Tensor,
) -> LocalSFTLossDiagnostic:
    """Compare fake causal next-token logits against an independently checked B5 handoff."""
    validate_local_sft_tensors(handoff)
    if type(logits) is not torch.Tensor:
        raise TypeError("exact synthetic logits tensor required")
    if logits.ndim != 3:
        raise ValueError("synthetic logits require rank three")

    rows, width, vocab = (int(value) for value in logits.shape)
    if not 2 <= vocab <= _MAX_SYNTHETIC_VOCAB:
        raise ValueError("synthetic vocabulary bound exceeded")
    if rows * width * vocab > _MAX_LOGIT_CELLS:
        raise ValueError("synthetic logits cell cap exceeded")
    if (rows, width) != tuple(handoff.input_ids.shape):
        raise ValueError("synthetic logits do not match the B5 shape")
    if (
        logits.device.type != "cpu"
        or logits.device.index is not None
        or logits.dtype != torch.float32
        or logits.layout != torch.strided
        or logits.requires_grad
        or not logits.is_contiguous()
    ):
        raise ValueError("synthetic logits must be contiguous CPU float32 without grad")
    if logits.untyped_storage().data_ptr() in {
        handoff.input_ids.untyped_storage().data_ptr(),
        handoff.labels.untyped_storage().data_ptr(),
        handoff.attention_mask.untyped_storage().data_ptr(),
    }:
        raise ValueError("synthetic logits must not alias source tensor storage")

    with torch.no_grad():
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("nonfinite synthetic logits")
        if bool(torch.any(handoff.input_ids >= vocab).item()):
            raise ValueError("synthetic vocabulary does not cover input and padding IDs")
        if bool(torch.any((handoff.labels != -100) & (handoff.labels >= vocab)).item()):
            raise ValueError("synthetic vocabulary does not cover supervised targets")

        shifted_labels = handoff.labels[:, 1:]
        row_counts = tuple(int(value) for value in (shifted_labels != -100).sum(dim=1).tolist())
        if (
            row_counts != handoff.source_batch.row_target_lengths
            or sum(row_counts) != handoff.total_loss_tokens
        ):
            raise ValueError("shifted target count mismatch")

        shifted_logits = logits[:, :-1, :].contiguous().reshape(-1, vocab)
        flat_labels = shifted_labels.contiguous().reshape(-1)
        losses = F.cross_entropy(
            shifted_logits,
            flat_labels,
            ignore_index=-100,
            reduction="none",
        )
        row_nll = tuple(
            float(value) for value in losses.reshape(rows, width - 1).sum(dim=1).tolist()
        )
    if any(not math.isfinite(value) or value < 0.0 for value in row_nll):
        raise ValueError("invalid per-row negative log-likelihood")
    total_nll = math.fsum(row_nll)
    mean_nll = total_nll / handoff.total_loss_tokens
    if not math.isfinite(total_nll) or not math.isfinite(mean_nll):
        raise ValueError("invalid total negative log-likelihood")

    return LocalSFTLossDiagnostic(
        source_batch_sha256=handoff.source_batch_sha256,
        split=handoff.split,
        manifest_fingerprint=handoff.manifest_fingerprint,
        projection_fingerprints=handoff.projection_fingerprints,
        model_fingerprint=handoff.model_fingerprint,
        tokenizer_fingerprint=handoff.tokenizer_fingerprint,
        projection_policy_sha256=handoff.projection_policy_sha256,
        synthetic_vocabulary_size=vocab,
        row_loss_tokens=row_counts,
        total_loss_tokens=handoff.total_loss_tokens,
        per_row_nll=row_nll,
        total_nll=total_nll,
        mean_nll=mean_nll,
    )
