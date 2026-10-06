from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from huyawo_adapt.contracts.base import (
    ArtifactDigest,
    ContractModel,
    NonEmptyString,
    Sha256Digest,
)

NonEmptyArtifactTuple = Annotated[
    tuple[ArtifactDigest, ...],
    Field(min_length=1),
]


def _normalize_artifacts(
    artifacts: tuple[ArtifactDigest, ...],
) -> tuple[ArtifactDigest, ...]:
    names = [artifact.name for artifact in artifacts]

    if len(names) != len(set(names)):
        raise ValueError("artifact names must be unique")

    return tuple(sorted(artifacts, key=lambda artifact: artifact.name))


class ModelIdentity(ContractModel):
    """Immutable identity for a base or adapted model artifact set."""

    contract_type: Literal["model_identity"] = "model_identity"
    source: NonEmptyString
    model_id: NonEmptyString
    revision: NonEmptyString
    config_sha256: Sha256Digest
    weight_artifacts: NonEmptyArtifactTuple
    architecture: NonEmptyString | None = None
    parameter_count: int | None = Field(default=None, gt=0)
    trust_remote_code: bool = False

    @field_validator("weight_artifacts")
    @classmethod
    def normalize_weight_artifacts(
        cls,
        artifacts: tuple[ArtifactDigest, ...],
    ) -> tuple[ArtifactDigest, ...]:
        """Normalize model artifacts into deterministic logical-name order."""
        return _normalize_artifacts(artifacts)


class TokenizerIdentity(ContractModel):
    """Immutable identity for tokenizer artifacts and relevant metadata."""

    contract_type: Literal["tokenizer_identity"] = "tokenizer_identity"
    source: NonEmptyString
    tokenizer_id: NonEmptyString
    revision: NonEmptyString
    artifacts: NonEmptyArtifactTuple
    chat_template_sha256: Sha256Digest | None = None
    vocab_size: int | None = Field(default=None, gt=0)
    model_max_length: int | None = Field(default=None, gt=0)

    @field_validator("artifacts")
    @classmethod
    def normalize_artifacts(
        cls,
        artifacts: tuple[ArtifactDigest, ...],
    ) -> tuple[ArtifactDigest, ...]:
        """Normalize tokenizer artifacts into deterministic logical-name order."""
        return _normalize_artifacts(artifacts)


class DatasetIdentity(ContractModel):
    """Evidence-first immutable identity for a training or evaluation dataset."""

    contract_type: Literal["dataset_identity"] = "dataset_identity"
    source_uri: NonEmptyString
    license_expression: NonEmptyString
    revision: NonEmptyString
    content_sha256: Sha256Digest
    schema_sha256: Sha256Digest
    preprocessing_sha256: Sha256Digest
    deduplication_sha256: Sha256Digest
    filtering_sha256: Sha256Digest
    contamination_check_sha256: Sha256Digest
    loss_mask_policy_sha256: Sha256Digest
    chat_template_sha256: Sha256Digest | None = None
    sample_count: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)


class DataSplitIdentity(ContractModel):
    """Immutable identity for a deterministic dataset split manifest."""

    contract_type: Literal["data_split_identity"] = "data_split_identity"
    dataset_fingerprint: Sha256Digest
    split_name: NonEmptyString
    manifest_sha256: Sha256Digest
    selection_policy_sha256: Sha256Digest
    sample_count: int = Field(ge=0)
    seed: int | None = None
