from __future__ import annotations

import pytest
from pydantic import ValidationError

from huyawo_adapt.contracts import ArtifactDigest
from huyawo_adapt.contracts.planning import (
    AdaptRecipe,
    CheckpointIdentity,
    FullSFTConfig,
    LoRAConfig,
    QLoRAConfig,
    TrainingPlan,
    TrainingRunIdentity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def make_lora_recipe() -> AdaptRecipe:
    return AdaptRecipe(
        recipe_id="lora-sft-v1",
        model_fingerprint=SHA_A,
        tokenizer_fingerprint=SHA_B,
        train_split_fingerprint=SHA_C,
        objective_fingerprint=SHA_D,
        method_config=LoRAConfig(
            rank=16,
            alpha=32.0,
            target_modules=(
                "v_proj",
                "q_proj",
            ),
        ),
        sequence_length=2048,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3.0,
        gradient_checkpointing=True,
        assistant_only_loss=True,
        seed=42,
    )


def make_training_plan(
    recipe: AdaptRecipe,
) -> TrainingPlan:
    return TrainingPlan(
        plan_id="plan-v1",
        model_fingerprint=recipe.model_fingerprint,
        tokenizer_fingerprint=recipe.tokenizer_fingerprint,
        train_split_fingerprint=recipe.train_split_fingerprint,
        recipe_fingerprint=recipe.fingerprint(),
        hardware_target_fingerprint=SHA_E,
        runtime_target_fingerprint=SHA_F,
        evaluation_profile_fingerprint=SHA_A,
        regression_profile_fingerprint=SHA_B,
        executor="transformers",
        world_size=1,
        dataloader_num_workers=4,
        precision="bfloat16",
    )


def test_full_sft_recipe_round_trip_is_stable() -> None:
    recipe = AdaptRecipe(
        recipe_id="full-sft-v1",
        model_fingerprint=SHA_A,
        tokenizer_fingerprint=SHA_B,
        train_split_fingerprint=SHA_C,
        objective_fingerprint=SHA_D,
        method_config=FullSFTConfig(),
        sequence_length=2048,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        max_steps=100,
        seed=42,
    )

    restored = AdaptRecipe.from_canonical_json(recipe.canonical_bytes())

    assert restored == recipe
    assert restored.fingerprint() == recipe.fingerprint()
    assert restored.method_config.method == "full_sft"


def test_lora_targets_are_normalized() -> None:
    recipe = make_lora_recipe()

    assert recipe.method_config.method == "lora"
    assert recipe.method_config.target_modules == (
        "q_proj",
        "v_proj",
    )


def test_lora_rejects_duplicate_targets() -> None:
    with pytest.raises(ValidationError):
        LoRAConfig(
            rank=16,
            alpha=32.0,
            target_modules=(
                "q_proj",
                "q_proj",
            ),
        )


def test_lora_recipe_fingerprint_is_target_order_independent() -> None:
    first = make_lora_recipe()

    second = AdaptRecipe(
        recipe_id=first.recipe_id,
        model_fingerprint=first.model_fingerprint,
        tokenizer_fingerprint=first.tokenizer_fingerprint,
        train_split_fingerprint=first.train_split_fingerprint,
        objective_fingerprint=first.objective_fingerprint,
        method_config=LoRAConfig(
            rank=16,
            alpha=32.0,
            target_modules=(
                "q_proj",
                "v_proj",
            ),
        ),
        sequence_length=first.sequence_length,
        micro_batch_size=first.micro_batch_size,
        gradient_accumulation_steps=(first.gradient_accumulation_steps),
        learning_rate=first.learning_rate,
        num_train_epochs=first.num_train_epochs,
        gradient_checkpointing=first.gradient_checkpointing,
        assistant_only_loss=first.assistant_only_loss,
        seed=first.seed,
    )

    assert first.fingerprint() == second.fingerprint()


def test_qlora_configuration_is_explicit() -> None:
    config = QLoRAConfig(
        rank=32,
        alpha=64.0,
        target_modules=(
            "k_proj",
            "q_proj",
            "v_proj",
        ),
        quantization_type="nf4",
        double_quantization=True,
        compute_dtype="bfloat16",
    )

    assert config.method == "qlora"
    assert config.quantization_bits == 4
    assert config.quantization_type == "nf4"
    assert config.compute_dtype == "bfloat16"


def test_recipe_requires_exactly_one_training_budget() -> None:
    with pytest.raises(ValidationError):
        AdaptRecipe(
            recipe_id="invalid-budget",
            model_fingerprint=SHA_A,
            tokenizer_fingerprint=SHA_B,
            train_split_fingerprint=SHA_C,
            objective_fingerprint=SHA_D,
            method_config=FullSFTConfig(),
            sequence_length=1024,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
        )

    with pytest.raises(ValidationError):
        AdaptRecipe(
            recipe_id="invalid-budget",
            model_fingerprint=SHA_A,
            tokenizer_fingerprint=SHA_B,
            train_split_fingerprint=SHA_C,
            objective_fingerprint=SHA_D,
            method_config=FullSFTConfig(),
            sequence_length=1024,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            max_steps=100,
            num_train_epochs=1.0,
        )


def test_recipe_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        AdaptRecipe.model_validate(
            {
                "recipe_id": "invalid-method",
                "model_fingerprint": SHA_A,
                "tokenizer_fingerprint": SHA_B,
                "train_split_fingerprint": SHA_C,
                "objective_fingerprint": SHA_D,
                "method_config": {
                    "method": "unsupported",
                },
                "sequence_length": 1024,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
                "max_steps": 100,
            }
        )


def test_training_plan_round_trip_is_stable() -> None:
    recipe = make_lora_recipe()
    plan = make_training_plan(recipe)

    restored = TrainingPlan.from_canonical_json(plan.canonical_bytes())

    assert restored == plan
    assert restored.fingerprint() == plan.fingerprint()
    assert plan.contract_type == "training_plan"


def test_training_plan_rejects_invalid_world_size() -> None:
    recipe = make_lora_recipe()
    values = make_training_plan(recipe).model_dump()
    values["world_size"] = 0

    with pytest.raises(ValidationError):
        TrainingPlan.model_validate(values)


def test_training_run_identity_round_trip_is_stable() -> None:
    recipe = make_lora_recipe()
    plan = make_training_plan(recipe)

    identity = TrainingRunIdentity(
        run_id="run-000001",
        plan_fingerprint=plan.fingerprint(),
        source_revision="b885bfe8259a147d389522b22361232c36357939",
        package_lock_sha256=SHA_C,
        hardware_snapshot_sha256=SHA_D,
        seed=42,
        attempt=1,
    )

    restored = TrainingRunIdentity.from_canonical_json(identity.canonical_bytes())

    assert restored == identity
    assert restored.fingerprint() == identity.fingerprint()


def test_training_run_identity_rejects_zero_attempt() -> None:
    with pytest.raises(ValidationError):
        TrainingRunIdentity(
            run_id="run-000001",
            plan_fingerprint=SHA_A,
            source_revision="revision",
            package_lock_sha256=SHA_B,
            hardware_snapshot_sha256=SHA_C,
            seed=42,
            attempt=0,
        )


def test_checkpoint_artifacts_are_normalized() -> None:
    checkpoint = CheckpointIdentity(
        checkpoint_id="checkpoint-100",
        training_run_fingerprint=SHA_A,
        step=100,
        checkpoint_kind="weights_only",
        artifacts=(
            ArtifactDigest(
                name="model-00002.safetensors",
                sha256=SHA_C,
            ),
            ArtifactDigest(
                name="model-00001.safetensors",
                sha256=SHA_B,
            ),
        ),
    )

    assert [artifact.name for artifact in checkpoint.artifacts] == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]


def test_checkpoint_rejects_duplicate_artifact_names() -> None:
    with pytest.raises(ValidationError):
        CheckpointIdentity(
            checkpoint_id="checkpoint-100",
            training_run_fingerprint=SHA_A,
            step=100,
            checkpoint_kind="weights_only",
            artifacts=(
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


def test_resumable_checkpoint_requires_all_state_digests() -> None:
    with pytest.raises(ValidationError):
        CheckpointIdentity(
            checkpoint_id="checkpoint-100",
            training_run_fingerprint=SHA_A,
            step=100,
            checkpoint_kind="resumable",
            artifacts=(
                ArtifactDigest(
                    name="model.safetensors",
                    sha256=SHA_B,
                ),
            ),
            trainer_state_sha256=SHA_C,
        )


def test_weights_only_checkpoint_rejects_resume_state() -> None:
    with pytest.raises(ValidationError):
        CheckpointIdentity(
            checkpoint_id="checkpoint-100",
            training_run_fingerprint=SHA_A,
            step=100,
            checkpoint_kind="weights_only",
            artifacts=(
                ArtifactDigest(
                    name="model.safetensors",
                    sha256=SHA_B,
                ),
            ),
            trainer_state_sha256=SHA_C,
        )


def test_resumable_checkpoint_round_trip_is_stable() -> None:
    checkpoint = CheckpointIdentity(
        checkpoint_id="checkpoint-100",
        training_run_fingerprint=SHA_A,
        step=100,
        epoch=1.5,
        checkpoint_kind="resumable",
        artifacts=(
            ArtifactDigest(
                name="model.safetensors",
                sha256=SHA_B,
            ),
        ),
        trainer_state_sha256=SHA_C,
        optimizer_state_sha256=SHA_D,
        rng_state_sha256=SHA_E,
    )

    restored = CheckpointIdentity.from_canonical_json(checkpoint.canonical_bytes())

    assert restored == checkpoint
    assert restored.fingerprint() == checkpoint.fingerprint()


def test_checkpoint_schema_exposes_kind_constraint() -> None:
    schema = CheckpointIdentity.model_json_schema()

    assert schema["properties"]["contract_type"]["const"] == "checkpoint_identity"
