"""CPU-only local SFT tensor handoff without training authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from huyawo_adapt.agentic.training.local_sft_batch import LocalSFTBatchDiagnostic
from huyawo_adapt.agentic.training.sft import SplitName


@dataclass(frozen=True, slots=True, eq=False)
class LocalSFTTensorDiagnostic:
    """Local shape diagnostic; the tensors themselves remain mutable."""

    source_batch: LocalSFTBatchDiagnostic
    source_batch_sha256: str
    split: SplitName
    manifest_fingerprint: str
    projection_fingerprints: tuple[str, ...]
    model_fingerprint: str
    tokenizer_fingerprint: str
    projection_policy_sha256: str
    total_loss_tokens: int
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    authority_scope: Literal["local_sft_tensor_shape_only"] = "local_sft_tensor_shape_only"
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False


def _checked_batch(batch: LocalSFTBatchDiagnostic) -> LocalSFTBatchDiagnostic:
    if type(batch) is not LocalSFTBatchDiagnostic:
        raise TypeError("exact local batch diagnostic required")
    verified = LocalSFTBatchDiagnostic.model_validate(batch.model_dump(mode="python"))
    if verified != batch:
        raise ValueError("source batch revalidation mismatch")
    return verified


def validate_local_sft_tensors(handoff: LocalSFTTensorDiagnostic) -> None:
    """Fail closed if the local source, metadata or any tensor has changed."""
    if type(handoff) is not LocalSFTTensorDiagnostic:
        raise TypeError("exact local tensor diagnostic required")
    if (
        handoff.authority_scope != "local_sft_tensor_shape_only"
        or handoff.independent_task_outcome_verified is not False
        or handoff.model_origin_attested is not False
        or handoff.tokenizer_fidelity_attested is not False
        or handoff.training_eligible is not False
    ):
        raise ValueError("positive training authority is forbidden")

    source = _checked_batch(handoff.source_batch)
    actual_identity = (
        handoff.source_batch_sha256,
        handoff.split,
        handoff.manifest_fingerprint,
        handoff.projection_fingerprints,
        handoff.model_fingerprint,
        handoff.tokenizer_fingerprint,
        handoff.projection_policy_sha256,
        handoff.total_loss_tokens,
    )
    source_identity = (
        source.batch_sha256,
        source.split,
        source.manifest_fingerprint,
        source.projection_fingerprints,
        source.model_fingerprint,
        source.tokenizer_fingerprint,
        source.projection_policy_sha256,
        source.total_loss_tokens,
    )
    if actual_identity != source_identity:
        raise ValueError("tensor identity mismatch")

    dimensions = (len(source.input_ids), source.padded_sequence_length)
    matrices = (
        (handoff.input_ids, source.input_ids),
        (handoff.labels, source.labels),
        (handoff.attention_mask, source.attention_mask),
    )
    storage_addresses: set[int] = set()
    for tensor, expected in matrices:
        if type(tensor) is not torch.Tensor:
            raise TypeError("tensor replacement is not trusted")
        if (
            tensor.device.type != "cpu"
            or tensor.device.index is not None
            or tensor.dtype != torch.int64
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.ndim != 2
            or tuple(tensor.shape) != dimensions
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid tensor properties")
        storage_address = tensor.untyped_storage().data_ptr()
        if storage_address in storage_addresses:
            raise ValueError("aliased tensor storage")
        storage_addresses.add(storage_address)
        if tensor.tolist() != [list(row) for row in expected]:
            raise ValueError("tensor content mismatch")


def materialize_local_sft_tensors(
    batch: LocalSFTBatchDiagnostic,
) -> LocalSFTTensorDiagnostic:
    """Copy validated B4 matrices onto the CPU without training a model."""
    source = _checked_batch(batch)
    result = LocalSFTTensorDiagnostic(
        source_batch=source,
        source_batch_sha256=source.batch_sha256,
        split=source.split,
        manifest_fingerprint=source.manifest_fingerprint,
        projection_fingerprints=source.projection_fingerprints,
        model_fingerprint=source.model_fingerprint,
        tokenizer_fingerprint=source.tokenizer_fingerprint,
        projection_policy_sha256=source.projection_policy_sha256,
        total_loss_tokens=source.total_loss_tokens,
        input_ids=torch.tensor(
            source.input_ids, dtype=torch.int64, device="cpu", requires_grad=False
        ),
        labels=torch.tensor(source.labels, dtype=torch.int64, device="cpu", requires_grad=False),
        attention_mask=torch.tensor(
            source.attention_mask, dtype=torch.int64, device="cpu", requires_grad=False
        ),
    )
    validate_local_sft_tensors(result)
    return result
