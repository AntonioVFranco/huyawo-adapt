"""Local-only structural SFT batch collation without training authority."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitManifest,
    SplitName,
    validate_sft_split_manifest,
)
from huyawo_adapt.contracts.base import ContractModel, Sha256Digest

StrictInteger = Annotated[int, Field(strict=True)]
PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
TokenInteger = Annotated[int, Field(strict=True, ge=0)]
SequenceLimit = Annotated[int, Field(strict=True, ge=2, le=4096)]
IntegerMatrix = tuple[tuple[StrictInteger, ...], ...]

_MAX_ROWS = 32
_MAX_TOKENS = 4096
_MAX_CELLS = 131072
_DOMAIN = "haa-local-sft-batch-v1"


def _batch_digest(value: object) -> str:
    data = json.dumps(
        {"domain": _DOMAIN, "record": value},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class LocalSFTBatchDiagnostic(ContractModel):
    """Frozen local shape diagnostic; never an eligible training dataset."""

    contract_type: Literal["local_sft_batch_diagnostic"] = "local_sft_batch_diagnostic"
    authority_scope: Literal["local_sft_batch_shape_only"] = "local_sft_batch_shape_only"
    split: SplitName
    manifest_fingerprint: Sha256Digest
    projection_fingerprints: tuple[Sha256Digest, ...]
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    projection_policy_sha256: Sha256Digest
    pad_token_id: TokenInteger
    max_sequence_length: SequenceLimit
    padded_sequence_length: SequenceLimit
    row_target_starts: tuple[PositiveInteger, ...]
    row_target_lengths: tuple[PositiveInteger, ...]
    total_loss_tokens: PositiveInteger
    input_ids: IntegerMatrix
    labels: IntegerMatrix
    attention_mask: IntegerMatrix
    batch_sha256: Sha256Digest
    independent_task_outcome_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    tokenizer_fidelity_attested: Literal[False] = False
    training_eligible: Literal[False] = False

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        count = len(self.projection_fingerprints)
        width = self.padded_sequence_length

        if not 1 <= count <= _MAX_ROWS or len(set(self.projection_fingerprints)) != count:
            raise ValueError("invalid or repeated batch projection")
        if not 2 <= width <= self.max_sequence_length or count * width > _MAX_CELLS:
            raise ValueError("invalid batch width or size")

        if any(
            len(group) != count
            for group in (
                self.row_target_starts,
                self.row_target_lengths,
                self.input_ids,
                self.labels,
                self.attention_mask,
            )
        ):
            raise ValueError("inconsistent row count")

        total = 0
        for index in range(count):
            ids = self.input_ids[index]
            labels = self.labels[index]
            mask = self.attention_mask[index]
            start = self.row_target_starts[index]
            target_length = self.row_target_lengths[index]

            if any(len(row) != width for row in (ids, labels, mask)):
                raise ValueError("nonrectangular matrix")
            if any(type(value) is not int or value < 0 for value in ids):
                raise ValueError("invalid token ID")
            if any(type(value) is not int for value in (*labels, *mask)):
                raise ValueError("invalid label or mask type")
            if any(value not in (0, 1) for value in mask):
                raise ValueError("invalid attention mask value")

            actual = sum(mask)
            if not 2 <= actual <= _MAX_TOKENS:
                raise ValueError("invalid actual length")
            if mask != (1,) * actual + (0,) * (width - actual):
                raise ValueError("attention must be a contiguous prefix")
            if not 0 < start < actual or target_length != actual - start:
                raise ValueError("invalid target boundary")
            if any(value != -100 for value in labels[:start]):
                raise ValueError("unmasked input prefix")
            if labels[start:actual] != ids[start:actual]:
                raise ValueError("incorrect target suffix")
            if any(value != -100 for value in labels[actual:]):
                raise ValueError("unmasked padding")
            if any(value != self.pad_token_id for value in ids[actual:]):
                raise ValueError("incorrect pad token ID")

            total += target_length

        if total != self.total_loss_tokens:
            raise ValueError("incorrect loss token total")

        if self.batch_sha256 != _batch_digest(
            self.model_dump(mode="json", exclude={"batch_sha256"})
        ):
            raise ValueError("batch digest mismatch")

        return self


def compile_local_sft_batch(
    projections: tuple[AgentSFTDecisionProjection, ...],
    manifest: AgentSFTSplitManifest,
    split: SplitName,
    pad_token_id: int,
    max_sequence_length: int,
) -> LocalSFTBatchDiagnostic:
    """Build right-padded fake-token matrices from caller-declared contracts."""
    if type(projections) is not tuple or not 1 <= len(projections) <= _MAX_ROWS:
        raise ValueError("invalid projection tuple")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("invalid padding ID")
    if type(max_sequence_length) is not int or not 2 <= max_sequence_length <= _MAX_TOKENS:
        raise ValueError("invalid maximum sequence length")
    if split not in ("train", "validation", "test"):
        raise ValueError("invalid split")

    checked_manifest = validate_sft_split_manifest(manifest)
    checked: list[AgentSFTDecisionProjection] = []

    for row in projections:
        if type(row) is not AgentSFTDecisionProjection:
            raise TypeError("invalid projection type")
        checked.append(AgentSFTDecisionProjection.model_validate(row.model_dump(mode="python")))

    rows = tuple(checked)
    fingerprints = tuple(row.fingerprint() for row in rows)

    if len(set(fingerprints)) != len(rows):
        raise ValueError("duplicate projection")

    selected = tuple(entry for entry in checked_manifest.entries if entry.split == split)
    lookup = {entry.projection_fingerprint: entry for entry in selected}

    if len(selected) != len(rows) or set(lookup) != set(fingerprints):
        raise ValueError("incomplete or inconsistent manifest split")

    identity = (
        rows[0].model_fingerprint,
        rows[0].tokenizer_fingerprint,
        rows[0].projection_policy_sha256,
    )

    for row, fingerprint in zip(rows, fingerprints, strict=True):
        entry = lookup[fingerprint]

        if (
            row.task_fingerprint != entry.task_fingerprint
            or row.trajectory_fingerprint != entry.trajectory_fingerprint
            or row.parent_trajectory_fingerprint != entry.parent_trajectory_fingerprint
        ):
            raise ValueError("projection versus manifest lineage mismatch")

        if (
            row.model_fingerprint,
            row.tokenizer_fingerprint,
            row.projection_policy_sha256,
        ) != identity:
            raise ValueError("mixed projection identities")

        if not 2 <= len(row.input_ids) <= max_sequence_length:
            raise ValueError("truncation forbidden")

    width = max(len(row.input_ids) for row in rows)
    if len(rows) * width > _MAX_CELLS:
        raise ValueError("batch cell limit exceeded")

    fields: dict[str, object] = {
        "schema_version": "1.0",
        "contract_type": "local_sft_batch_diagnostic",
        "authority_scope": "local_sft_batch_shape_only",
        "split": split,
        "manifest_fingerprint": checked_manifest.fingerprint(),
        "projection_fingerprints": fingerprints,
        "model_fingerprint": identity[0],
        "tokenizer_fingerprint": identity[1],
        "projection_policy_sha256": identity[2],
        "pad_token_id": pad_token_id,
        "max_sequence_length": max_sequence_length,
        "padded_sequence_length": width,
        "row_target_starts": tuple(row.target_start for row in rows),
        "row_target_lengths": tuple(row.target_end - row.target_start for row in rows),
        "total_loss_tokens": sum(row.target_end - row.target_start for row in rows),
        "input_ids": tuple(
            row.input_ids + (pad_token_id,) * (width - len(row.input_ids)) for row in rows
        ),
        "labels": tuple(row.labels + (-100,) * (width - len(row.labels)) for row in rows),
        "attention_mask": tuple(
            (1,) * len(row.input_ids) + (0,) * (width - len(row.input_ids)) for row in rows
        ),
        "independent_task_outcome_verified": False,
        "model_origin_attested": False,
        "tokenizer_fidelity_attested": False,
        "training_eligible": False,
    }

    return LocalSFTBatchDiagnostic.model_validate({**fields, "batch_sha256": _batch_digest(fields)})
