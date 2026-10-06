from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, Self, cast

from pydantic import Field, field_validator, model_validator

from datasets import Dataset  # type: ignore[import-untyped]
from huyawo_adapt.contracts import DataSplitIdentity
from huyawo_adapt.contracts.base import (
    NonEmptyString,
    Sha256Digest,
    StrictFrozenModel,
)

SPLIT_MEMBERSHIP_ALGORITHM: Final[Literal["huyawo_sha256_rank_partition_v1"]] = (
    "huyawo_sha256_rank_partition_v1"
)
SPLIT_ALLOCATION_ALGORITHM: Final[Literal["integer_largest_remainder_declared_order_v1"]] = (
    "integer_largest_remainder_declared_order_v1"
)
SPLIT_OUTPUT_ORDER: Final[Literal["source_index_ascending_v1"]] = "source_index_ascending_v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class _CanonicalOperationalModel(StrictFrozenModel):
    def canonical_data(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_canonical_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> Self:
        return cls.model_validate_json(payload)


class SplitWeight(StrictFrozenModel):
    name: NonEmptyString
    weight: int = Field(gt=0)


class DeterministicSplitPolicy(_CanonicalOperationalModel):
    policy_version: Literal["1.0"] = "1.0"
    membership_algorithm: Literal["huyawo_sha256_rank_partition_v1"] = SPLIT_MEMBERSHIP_ALGORITHM
    allocation_algorithm: Literal["integer_largest_remainder_declared_order_v1"] = (
        SPLIT_ALLOCATION_ALGORITHM
    )
    output_order: Literal["source_index_ascending_v1"] = SPLIT_OUTPUT_ORDER
    splits: tuple[SplitWeight, ...]

    @field_validator("splits")
    @classmethod
    def validate_splits(
        cls,
        splits: tuple[SplitWeight, ...],
    ) -> tuple[SplitWeight, ...]:
        if len(splits) < 2:
            raise ValueError("at least two output splits are required")

        names = [split.name for split in splits]

        if len(names) != len(set(names)):
            raise ValueError("split names must be unique")

        return splits


class DeterministicSplitManifest(_CanonicalOperationalModel):
    manifest_version: Literal["1.0"] = "1.0"
    dataset_fingerprint: Sha256Digest
    selection_policy_sha256: Sha256Digest
    seed: int = Field(ge=0)
    split_name: NonEmptyString
    source_sample_count: int = Field(gt=0)
    sample_count: int = Field(ge=0)
    source_indices: tuple[int, ...]

    @field_validator("source_indices")
    @classmethod
    def validate_source_indices(
        cls,
        indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(index < 0 for index in indices):
            raise ValueError("source indices must be non-negative")

        if len(indices) != len(set(indices)):
            raise ValueError("source indices must be unique")

        if indices != tuple(sorted(indices)):
            raise ValueError("source indices must be sorted ascending")

        return indices

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if len(self.source_indices) != self.sample_count:
            raise ValueError("sample_count must equal the number of source indices")

        if any(index >= self.source_sample_count for index in self.source_indices):
            raise ValueError("source index is outside the source dataset range")

        return self


class DeterministicSplitPartition(_CanonicalOperationalModel):
    dataset_fingerprint: Sha256Digest
    source_sample_count: int = Field(gt=0)
    seed: int = Field(ge=0)
    policy: DeterministicSplitPolicy
    selection_policy_sha256: Sha256Digest
    manifests: tuple[
        DeterministicSplitManifest,
        ...,
    ]
    identities: tuple[
        DataSplitIdentity,
        ...,
    ]

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        if self.selection_policy_sha256 != self.policy.fingerprint():
            raise ValueError("selection policy fingerprint mismatch")

        expected_names = tuple(split.name for split in self.policy.splits)

        manifest_names = tuple(manifest.split_name for manifest in self.manifests)

        identity_names = tuple(identity.split_name for identity in self.identities)

        if manifest_names != expected_names:
            raise ValueError("manifest order must match declared split order")

        if identity_names != expected_names:
            raise ValueError("identity order must match declared split order")

        if len(self.manifests) != len(self.identities):
            raise ValueError("manifest and identity counts differ")

        seen: set[int] = set()
        total = 0

        for manifest, identity in zip(
            self.manifests,
            self.identities,
            strict=True,
        ):
            if manifest.dataset_fingerprint != self.dataset_fingerprint:
                raise ValueError("manifest dataset fingerprint mismatch")

            if manifest.selection_policy_sha256 != self.selection_policy_sha256:
                raise ValueError("manifest selection policy mismatch")

            if manifest.seed != self.seed:
                raise ValueError("manifest seed mismatch")

            if manifest.source_sample_count != self.source_sample_count:
                raise ValueError("manifest source sample count mismatch")

            if identity.dataset_fingerprint != self.dataset_fingerprint:
                raise ValueError("identity dataset fingerprint mismatch")

            if identity.manifest_sha256 != manifest.fingerprint():
                raise ValueError("identity manifest fingerprint mismatch")

            if identity.selection_policy_sha256 != self.selection_policy_sha256:
                raise ValueError("identity selection policy mismatch")

            if identity.sample_count != manifest.sample_count:
                raise ValueError("identity sample count mismatch")

            if identity.seed != self.seed:
                raise ValueError("identity seed mismatch")

            indices = set(manifest.source_indices)

            if seen.intersection(indices):
                raise ValueError("split manifests overlap")

            seen.update(indices)
            total += manifest.sample_count

        if total != self.source_sample_count:
            raise ValueError("split sample counts do not cover source dataset")

        if seen != set(range(self.source_sample_count)):
            raise ValueError("split manifests do not exactly cover source indices")

        return self


class SplitMaterializationEvidence(_CanonicalOperationalModel):
    split_name: NonEmptyString
    manifest_sha256: Sha256Digest
    source_sample_count: int = Field(gt=0)
    sample_count: int = Field(ge=0)
    source_external_dataset_fingerprint: NonEmptyString
    materialized_external_dataset_fingerprint: NonEmptyString


@dataclass(
    frozen=True,
    slots=True,
)
class SplitMaterializationResult:
    dataset: object
    evidence: SplitMaterializationEvidence


class _SelectableDataset(Protocol):
    _fingerprint: str

    def __len__(self) -> int: ...

    def select(
        self,
        indices: list[int],
        *,
        keep_in_memory: bool = False,
    ) -> object: ...


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _require_sha256(
    value: str,
    label: str,
) -> str:
    normalized = value.strip()

    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    return normalized


def _require_positive_int(
    value: int,
    label: str,
) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")

    return value


def _require_non_negative_int(
    value: int,
    label: str,
) -> int:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")

    return value


def _allocate_counts(
    *,
    source_sample_count: int,
    policy: DeterministicSplitPolicy,
) -> tuple[
    tuple[str, int],
    ...,
]:
    total_weight = sum(split.weight for split in policy.splits)

    bases: list[int] = []
    remainders: list[int] = []

    for split in policy.splits:
        numerator = source_sample_count * split.weight

        bases.append(numerator // total_weight)
        remainders.append(numerator % total_weight)

    remaining = source_sample_count - sum(bases)

    priority = sorted(
        range(len(policy.splits)),
        key=lambda index: (
            -remainders[index],
            index,
        ),
    )

    counts = list(bases)

    for index in priority[:remaining]:
        counts[index] += 1

    result = tuple(
        (
            split.name,
            counts[index],
        )
        for index, split in enumerate(policy.splits)
    )

    if sum(count for _, count in result) != source_sample_count:
        raise RuntimeError("split allocation does not cover source samples")

    return result


def _rank_digest(
    *,
    dataset_fingerprint: str,
    seed: int,
    source_index: int,
) -> bytes:
    material = {
        "algorithm": (SPLIT_MEMBERSHIP_ALGORITHM),
        "dataset_fingerprint": (dataset_fingerprint),
        "seed": seed,
        "source_index": source_index,
    }

    return hashlib.sha256(_canonical_json_bytes(material)).digest()


def build_deterministic_partition(
    *,
    dataset_fingerprint: str,
    source_sample_count: int,
    seed: int,
    policy: DeterministicSplitPolicy,
) -> DeterministicSplitPartition:
    normalized_fingerprint = _require_sha256(
        dataset_fingerprint,
        "dataset_fingerprint",
    )

    normalized_sample_count = _require_positive_int(
        source_sample_count,
        "source_sample_count",
    )

    normalized_seed = _require_non_negative_int(
        seed,
        "seed",
    )

    selection_policy_sha256 = policy.fingerprint()

    counts = dict(
        _allocate_counts(
            source_sample_count=(normalized_sample_count),
            policy=policy,
        )
    )

    ranked_indices = sorted(
        range(normalized_sample_count),
        key=lambda source_index: (
            _rank_digest(
                dataset_fingerprint=(normalized_fingerprint),
                seed=normalized_seed,
                source_index=source_index,
            ),
            source_index,
        ),
    )

    manifests: list[DeterministicSplitManifest] = []

    identities: list[DataSplitIdentity] = []

    cursor = 0

    for split in policy.splits:
        split_count = counts[split.name]

        membership = ranked_indices[cursor : cursor + split_count]

        cursor += split_count

        source_indices = tuple(sorted(membership))

        manifest = DeterministicSplitManifest(
            dataset_fingerprint=(normalized_fingerprint),
            selection_policy_sha256=(selection_policy_sha256),
            seed=normalized_seed,
            split_name=split.name,
            source_sample_count=(normalized_sample_count),
            sample_count=len(source_indices),
            source_indices=(source_indices),
        )

        identity = DataSplitIdentity(
            dataset_fingerprint=(normalized_fingerprint),
            split_name=split.name,
            manifest_sha256=(manifest.fingerprint()),
            selection_policy_sha256=(selection_policy_sha256),
            sample_count=(manifest.sample_count),
            seed=normalized_seed,
        )

        manifests.append(manifest)
        identities.append(identity)

    if cursor != normalized_sample_count:
        raise RuntimeError("partition cursor does not cover source samples")

    return DeterministicSplitPartition(
        dataset_fingerprint=(normalized_fingerprint),
        source_sample_count=(normalized_sample_count),
        seed=normalized_seed,
        policy=policy,
        selection_policy_sha256=(selection_policy_sha256),
        manifests=tuple(manifests),
        identities=tuple(identities),
    )


def materialize_split(
    dataset: object,
    manifest: DeterministicSplitManifest,
) -> SplitMaterializationResult:
    if not isinstance(
        dataset,
        Dataset,
    ):
        raise TypeError("split materialization requires a materialized Dataset")

    source = cast(
        _SelectableDataset,
        dataset,
    )

    if len(source) != manifest.source_sample_count:
        raise ValueError("source dataset length does not match manifest")

    source_fingerprint = str(source._fingerprint).strip()

    if not source_fingerprint:
        raise ValueError("source Dataset._fingerprint must be non-empty")

    materialized_object = source.select(
        list(manifest.source_indices),
        keep_in_memory=True,
    )

    if not isinstance(
        materialized_object,
        Dataset,
    ):
        raise TypeError("Dataset.select did not return a materialized Dataset")

    materialized = cast(
        _SelectableDataset,
        materialized_object,
    )

    if len(materialized) != manifest.sample_count:
        raise RuntimeError("materialized split length differs from manifest")

    materialized_fingerprint = str(materialized._fingerprint).strip()

    if not materialized_fingerprint:
        raise ValueError("materialized Dataset._fingerprint must be non-empty")

    evidence = SplitMaterializationEvidence(
        split_name=(manifest.split_name),
        manifest_sha256=(manifest.fingerprint()),
        source_sample_count=(manifest.source_sample_count),
        sample_count=(manifest.sample_count),
        source_external_dataset_fingerprint=(source_fingerprint),
        materialized_external_dataset_fingerprint=(materialized_fingerprint),
    )

    return SplitMaterializationResult(
        dataset=materialized_object,
        evidence=evidence,
    )
