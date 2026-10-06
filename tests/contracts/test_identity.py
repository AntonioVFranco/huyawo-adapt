from __future__ import annotations

import pytest
from pydantic import ValidationError

from huyawo_adapt.contracts import (
    ArtifactDigest,
    DatasetIdentity,
    DataSplitIdentity,
    ModelIdentity,
    TokenizerIdentity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def make_model_identity() -> ModelIdentity:
    return ModelIdentity(
        source="huggingface",
        model_id="organization/model",
        revision="0123456789abcdef",
        config_sha256=SHA_A,
        weight_artifacts=(
            ArtifactDigest(
                name="model-00002.safetensors",
                sha256=SHA_C,
            ),
            ArtifactDigest(
                name="model-00001.safetensors",
                sha256=SHA_B,
            ),
        ),
        architecture="DecoderOnlyForCausalLM",
        parameter_count=1_500_000_000,
    )


def make_dataset_identity() -> DatasetIdentity:
    return DatasetIdentity(
        source_uri="hf://datasets/organization/dataset",
        license_expression="Apache-2.0",
        revision="fedcba9876543210",
        content_sha256=SHA_A,
        schema_sha256=SHA_B,
        preprocessing_sha256=SHA_C,
        deduplication_sha256=SHA_D,
        filtering_sha256=SHA_E,
        contamination_check_sha256=SHA_F,
        loss_mask_policy_sha256=SHA_A,
        sample_count=1024,
        token_count=65536,
    )


def test_model_identity_round_trip_and_fingerprint() -> None:
    identity = make_model_identity()
    restored = ModelIdentity.from_canonical_json(identity.canonical_bytes())

    assert restored == identity
    assert restored.fingerprint() == identity.fingerprint()
    assert identity.contract_type == "model_identity"


def test_model_artifacts_are_normalized_by_name() -> None:
    identity = make_model_identity()

    assert [artifact.name for artifact in identity.weight_artifacts] == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]


def test_model_fingerprint_is_independent_of_artifact_input_order() -> None:
    first = make_model_identity()

    second = ModelIdentity(
        source=first.source,
        model_id=first.model_id,
        revision=first.revision,
        config_sha256=first.config_sha256,
        weight_artifacts=tuple(reversed(first.weight_artifacts)),
        architecture=first.architecture,
        parameter_count=first.parameter_count,
    )

    assert first.fingerprint() == second.fingerprint()


def test_duplicate_artifact_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelIdentity(
            source="huggingface",
            model_id="organization/model",
            revision="0123456789abcdef",
            config_sha256=SHA_A,
            weight_artifacts=(
                ArtifactDigest(
                    name="model.safetensors",
                    sha256=SHA_B,
                ),
                ArtifactDigest(
                    name="model.safetensors",
                    sha256=SHA_C,
                ),
            ),
        )


def test_model_identity_requires_valid_sha256() -> None:
    with pytest.raises(ValidationError):
        ModelIdentity(
            source="huggingface",
            model_id="organization/model",
            revision="0123456789abcdef",
            config_sha256="not-a-sha256",
            weight_artifacts=(
                ArtifactDigest(
                    name="model.safetensors",
                    sha256=SHA_B,
                ),
            ),
        )


def test_model_identity_requires_weight_artifact() -> None:
    with pytest.raises(ValidationError):
        ModelIdentity(
            source="huggingface",
            model_id="organization/model",
            revision="0123456789abcdef",
            config_sha256=SHA_A,
            weight_artifacts=(),
        )


def test_tokenizer_identity_preserves_artifact_identity() -> None:
    identity = TokenizerIdentity(
        source="huggingface",
        tokenizer_id="organization/model",
        revision="0123456789abcdef",
        artifacts=(
            ArtifactDigest(
                name="tokenizer.json",
                sha256=SHA_A,
            ),
            ArtifactDigest(
                name="tokenizer_config.json",
                sha256=SHA_B,
            ),
        ),
        chat_template_sha256=SHA_C,
        vocab_size=32000,
        model_max_length=4096,
    )

    assert identity.contract_type == "tokenizer_identity"
    assert len(identity.artifacts) == 2
    assert len(identity.fingerprint()) == 64


def test_dataset_identity_requires_nonempty_provenance() -> None:
    with pytest.raises(ValidationError):
        DatasetIdentity(
            source_uri="   ",
            license_expression="Apache-2.0",
            revision="fedcba9876543210",
            content_sha256=SHA_A,
            schema_sha256=SHA_B,
            preprocessing_sha256=SHA_C,
            deduplication_sha256=SHA_D,
            filtering_sha256=SHA_E,
            contamination_check_sha256=SHA_F,
            loss_mask_policy_sha256=SHA_A,
            sample_count=1024,
        )


def test_dataset_identity_rejects_string_sample_count() -> None:
    values = make_dataset_identity().model_dump()
    values["sample_count"] = "1024"

    with pytest.raises(ValidationError):
        DatasetIdentity.model_validate(values)


def test_data_split_identity_links_to_dataset_fingerprint() -> None:
    dataset = make_dataset_identity()

    split = DataSplitIdentity(
        dataset_fingerprint=dataset.fingerprint(),
        split_name="train",
        manifest_sha256=SHA_B,
        selection_policy_sha256=SHA_C,
        sample_count=900,
        seed=42,
    )

    assert split.dataset_fingerprint == dataset.fingerprint()
    assert split.contract_type == "data_split_identity"


def test_identity_schema_contains_contract_type() -> None:
    schema = ModelIdentity.model_json_schema()
    contract_type = schema["properties"]["contract_type"]

    assert contract_type["const"] == "model_identity"
