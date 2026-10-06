from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from datasets import Dataset
from huyawo_adapt.datasets import (
    DeterministicSplitManifest,
    DeterministicSplitPolicy,
    SplitWeight,
    build_deterministic_partition,
    materialize_split,
)

DATASET_FINGERPRINT = hashlib.sha256(b"milestone-5-synthetic-upstream-dataset-v1").hexdigest()

SOURCE_SAMPLE_COUNT = 101
SEED = 20261006

EXPECTED_POLICY_SHA256 = "f6a54308279d2d7e35dfa701695c4727929ac62547803fc611de571e650f620d"

EXPECTED_MANIFEST_SHA256 = {
    "train": ("b4fb389dac7a42e5d14a6bbd9ab3ce9327bd221df89b9e8b83378d0889ae0333"),
    "validation": ("e4c9210cb37c3ecbaf51322aa5fde0685800e90654ac40bb020290817bd3a52e"),
    "test": ("3c5860449b2232e01ed69f5f95c475ed8661a630306dfbd9c417905c1713ca2c"),
}

EXPECTED_IDENTITY_FINGERPRINT = {
    "train": ("a8d93637216a429a5f2f46e9f4193128ddcfdb88d7254a9da828fa6c99d0c421"),
    "validation": ("8ab61590bc18b0d4fdc61edd87e80919966d9fe3867ba6ffdea1c8de6af3ebb1"),
    "test": ("d3ba5ac51ddf4503994f34f62fff2bd068873ab9f62dc6209f26f0ca2b8df848"),
}


def _policy() -> DeterministicSplitPolicy:
    return DeterministicSplitPolicy(
        splits=(
            SplitWeight(
                name="train",
                weight=8,
            ),
            SplitWeight(
                name="validation",
                weight=1,
            ),
            SplitWeight(
                name="test",
                weight=1,
            ),
        )
    )


def _partition(
    *,
    seed: int = SEED,
):
    return build_deterministic_partition(
        dataset_fingerprint=(DATASET_FINGERPRINT),
        source_sample_count=(SOURCE_SAMPLE_COUNT),
        seed=seed,
        policy=_policy(),
    )


def _manifest(
    **overrides: object,
) -> DeterministicSplitManifest:
    values: dict[str, object] = {
        "dataset_fingerprint": (DATASET_FINGERPRINT),
        "selection_policy_sha256": (EXPECTED_POLICY_SHA256),
        "seed": SEED,
        "split_name": "train",
        "source_sample_count": 3,
        "sample_count": 2,
        "source_indices": (0, 1),
    }
    values.update(overrides)

    return DeterministicSplitManifest(**values)


def test_policy_fingerprint_matches_preflight() -> None:
    assert _policy().fingerprint() == EXPECTED_POLICY_SHA256


def test_partition_matches_preflight_frozen_fingerprints() -> None:
    partition = _partition()

    manifests = {manifest.split_name: (manifest.fingerprint()) for manifest in partition.manifests}

    identities = {
        identity.split_name: (identity.fingerprint()) for identity in partition.identities
    }

    assert partition.selection_policy_sha256 == EXPECTED_POLICY_SHA256
    assert manifests == EXPECTED_MANIFEST_SHA256
    assert identities == EXPECTED_IDENTITY_FINGERPRINT


def test_partition_has_full_coverage_and_zero_overlap() -> None:
    partition = _partition()

    all_indices = [index for manifest in partition.manifests for index in manifest.source_indices]

    assert len(all_indices) == SOURCE_SAMPLE_COUNT
    assert len(set(all_indices)) == SOURCE_SAMPLE_COUNT
    assert set(all_indices) == set(range(SOURCE_SAMPLE_COUNT))


def test_same_seed_is_deterministic() -> None:
    first = _partition()
    second = _partition()

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fingerprint() == second.fingerprint()


def test_different_seed_changes_membership_not_policy() -> None:
    first = _partition(
        seed=SEED,
    )
    second = _partition(
        seed=SEED + 1,
    )

    assert first.selection_policy_sha256 == second.selection_policy_sha256

    assert tuple(manifest.source_indices for manifest in first.manifests) != tuple(
        manifest.source_indices for manifest in second.manifests
    )

    assert tuple(identity.fingerprint() for identity in first.identities) != tuple(
        identity.fingerprint() for identity in second.identities
    )


def test_declared_order_breaks_equal_remainder_ties() -> None:
    policy_ab = DeterministicSplitPolicy(
        splits=(
            SplitWeight(
                name="a",
                weight=1,
            ),
            SplitWeight(
                name="b",
                weight=1,
            ),
        )
    )

    policy_ba = DeterministicSplitPolicy(
        splits=(
            SplitWeight(
                name="b",
                weight=1,
            ),
            SplitWeight(
                name="a",
                weight=1,
            ),
        )
    )

    first = build_deterministic_partition(
        dataset_fingerprint=(DATASET_FINGERPRINT),
        source_sample_count=1,
        seed=0,
        policy=policy_ab,
    )

    second = build_deterministic_partition(
        dataset_fingerprint=(DATASET_FINGERPRINT),
        source_sample_count=1,
        seed=0,
        policy=policy_ba,
    )

    counts_first = {manifest.split_name: (manifest.sample_count) for manifest in first.manifests}

    counts_second = {manifest.split_name: (manifest.sample_count) for manifest in second.manifests}

    assert counts_first == {
        "a": 1,
        "b": 0,
    }
    assert counts_second == {
        "b": 1,
        "a": 0,
    }


def test_policy_rejects_duplicate_split_names() -> None:
    with pytest.raises(
        ValidationError,
    ):
        DeterministicSplitPolicy(
            splits=(
                SplitWeight(
                    name="train",
                    weight=1,
                ),
                SplitWeight(
                    name="train",
                    weight=1,
                ),
            )
        )


def test_policy_rejects_zero_weight() -> None:
    with pytest.raises(
        ValidationError,
    ):
        SplitWeight(
            name="train",
            weight=0,
        )


def test_policy_rejects_negative_weight() -> None:
    with pytest.raises(
        ValidationError,
    ):
        SplitWeight(
            name="train",
            weight=-1,
        )


def test_policy_requires_two_splits() -> None:
    with pytest.raises(
        ValidationError,
    ):
        DeterministicSplitPolicy(
            splits=(
                SplitWeight(
                    name="train",
                    weight=1,
                ),
            )
        )


def test_build_rejects_invalid_dataset_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="dataset_fingerprint",
    ):
        build_deterministic_partition(
            dataset_fingerprint="invalid",
            source_sample_count=10,
            seed=0,
            policy=_policy(),
        )


def test_build_rejects_non_positive_source_sample_count() -> None:
    with pytest.raises(
        ValueError,
        match="source_sample_count",
    ):
        build_deterministic_partition(
            dataset_fingerprint=(DATASET_FINGERPRINT),
            source_sample_count=0,
            seed=0,
            policy=_policy(),
        )


def test_build_rejects_negative_seed() -> None:
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        build_deterministic_partition(
            dataset_fingerprint=(DATASET_FINGERPRINT),
            source_sample_count=10,
            seed=-1,
            policy=_policy(),
        )


def test_manifest_rejects_duplicate_indices() -> None:
    with pytest.raises(
        ValidationError,
    ):
        _manifest(
            source_indices=(1, 1),
        )


def test_manifest_rejects_unsorted_indices() -> None:
    with pytest.raises(
        ValidationError,
    ):
        _manifest(
            source_indices=(1, 0),
        )


def test_manifest_rejects_out_of_range_index() -> None:
    with pytest.raises(
        ValidationError,
    ):
        _manifest(
            source_indices=(0, 3),
        )


def test_manifest_rejects_sample_count_mismatch() -> None:
    with pytest.raises(
        ValidationError,
    ):
        _manifest(
            sample_count=1,
        )


def test_materialize_split_preserves_manifest_order_and_evidence() -> None:
    dataset = Dataset.from_dict(
        {
            "id": list(range(SOURCE_SAMPLE_COUNT)),
        }
    )

    partition = _partition()
    manifest = partition.manifests[0]

    result = materialize_split(
        dataset,
        manifest,
    )

    assert isinstance(
        result.dataset,
        Dataset,
    )

    assert list(result.dataset["id"]) == list(manifest.source_indices)

    assert result.evidence.split_name == "train"

    assert result.evidence.manifest_sha256 == manifest.fingerprint()

    assert result.evidence.sample_count == manifest.sample_count

    assert result.evidence.source_external_dataset_fingerprint == dataset._fingerprint

    assert result.evidence.materialized_external_dataset_fingerprint == result.dataset._fingerprint


def test_materialize_split_rejects_wrong_source_length() -> None:
    dataset = Dataset.from_dict(
        {
            "id": list(range(100)),
        }
    )

    manifest = _partition().manifests[0]

    with pytest.raises(
        ValueError,
        match="length",
    ):
        materialize_split(
            dataset,
            manifest,
        )


def test_materialize_split_rejects_non_dataset() -> None:
    manifest = _partition().manifests[0]

    with pytest.raises(
        TypeError,
        match="materialized Dataset",
    ):
        materialize_split(
            object(),
            manifest,
        )
