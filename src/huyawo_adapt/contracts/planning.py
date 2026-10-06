from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_adapt.contracts.base import (
    ArtifactDigest,
    ContractModel,
    NonEmptyString,
    Sha256Digest,
    StrictFrozenModel,
)
from huyawo_adapt.contracts.specification import DTypeName

NonEmptyModuleTuple = Annotated[
    tuple[NonEmptyString, ...],
    Field(min_length=1),
]

NonEmptyArtifactTuple = Annotated[
    tuple[ArtifactDigest, ...],
    Field(min_length=1),
]


def _normalize_unique_strings(
    values: tuple[str, ...],
    field_name: str,
) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")

    return tuple(sorted(values))


def _normalize_artifacts(
    artifacts: tuple[ArtifactDigest, ...],
) -> tuple[ArtifactDigest, ...]:
    names = [artifact.name for artifact in artifacts]

    if len(names) != len(set(names)):
        raise ValueError("artifact names must be unique")

    return tuple(
        sorted(
            artifacts,
            key=lambda artifact: artifact.name,
        )
    )


class FullSFTConfig(StrictFrozenModel):
    """Method-specific settings for full supervised fine-tuning."""

    method: Literal["full_sft"] = "full_sft"


class LoRAConfig(StrictFrozenModel):
    """Method-specific settings for LoRA adaptation."""

    method: Literal["lora"] = "lora"
    rank: int = Field(gt=0)
    alpha: float = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: NonEmptyModuleTuple
    bias: Literal[
        "none",
        "all",
        "lora_only",
    ] = "none"
    use_rslora: bool = False

    @field_validator("target_modules")
    @classmethod
    def normalize_target_modules(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize adapter targets into deterministic order."""
        return _normalize_unique_strings(
            values,
            "target_modules",
        )


class QLoRAConfig(StrictFrozenModel):
    """Method-specific settings for quantized LoRA adaptation."""

    method: Literal["qlora"] = "qlora"
    rank: int = Field(gt=0)
    alpha: float = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: NonEmptyModuleTuple
    bias: Literal[
        "none",
        "all",
        "lora_only",
    ] = "none"
    use_rslora: bool = False
    quantization_bits: Literal[4] = 4
    quantization_type: Literal[
        "nf4",
        "fp4",
    ] = "nf4"
    double_quantization: bool = True
    compute_dtype: DTypeName = "bfloat16"

    @field_validator("target_modules")
    @classmethod
    def normalize_target_modules(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize adapter targets into deterministic order."""
        return _normalize_unique_strings(
            values,
            "target_modules",
        )


AdaptationMethodConfig = Annotated[
    FullSFTConfig | LoRAConfig | QLoRAConfig,
    Field(discriminator="method"),
]


class AdaptRecipe(ContractModel):
    """Normalized adaptation recipe bound to immutable inputs."""

    contract_type: Literal["adapt_recipe"] = "adapt_recipe"
    recipe_id: NonEmptyString
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    train_split_fingerprint: Sha256Digest
    objective_fingerprint: Sha256Digest
    method_config: AdaptationMethodConfig
    sequence_length: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_steps: int | None = Field(default=None, gt=0)
    num_train_epochs: float | None = Field(default=None, gt=0)
    gradient_checkpointing: bool = False
    packing: bool = False
    assistant_only_loss: bool = False
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_training_budget(self) -> Self:
        """Require exactly one explicit training budget."""
        has_steps = self.max_steps is not None
        has_epochs = self.num_train_epochs is not None

        if has_steps == has_epochs:
            raise ValueError("exactly one of max_steps or num_train_epochs is required")

        return self


class TrainingPlan(ContractModel):
    """Resolved execution plan binding recipe and execution constraints."""

    contract_type: Literal["training_plan"] = "training_plan"
    plan_id: NonEmptyString
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    train_split_fingerprint: Sha256Digest
    recipe_fingerprint: Sha256Digest
    hardware_target_fingerprint: Sha256Digest
    runtime_target_fingerprint: Sha256Digest
    evaluation_profile_fingerprint: Sha256Digest
    regression_profile_fingerprint: Sha256Digest
    executor: NonEmptyString
    world_size: int = Field(default=1, ge=1)
    dataloader_num_workers: int = Field(default=0, ge=0)
    precision: DTypeName
    compile_enabled: bool = False


class TrainingRunIdentity(ContractModel):
    """Identity of one concrete execution of a training plan."""

    contract_type: Literal["training_run_identity"] = "training_run_identity"
    run_id: NonEmptyString
    plan_fingerprint: Sha256Digest
    source_revision: NonEmptyString
    package_lock_sha256: Sha256Digest
    hardware_snapshot_sha256: Sha256Digest
    seed: int = Field(ge=0)
    attempt: int = Field(default=1, ge=1)


class CheckpointIdentity(ContractModel):
    """Identity and resumability evidence for one training checkpoint."""

    contract_type: Literal["checkpoint_identity"] = "checkpoint_identity"
    checkpoint_id: NonEmptyString
    training_run_fingerprint: Sha256Digest
    step: int = Field(ge=0)
    epoch: float | None = Field(default=None, ge=0)
    checkpoint_kind: Literal[
        "weights_only",
        "resumable",
    ]
    artifacts: NonEmptyArtifactTuple
    trainer_state_sha256: Sha256Digest | None = None
    optimizer_state_sha256: Sha256Digest | None = None
    rng_state_sha256: Sha256Digest | None = None

    @field_validator("artifacts")
    @classmethod
    def normalize_artifacts(
        cls,
        artifacts: tuple[ArtifactDigest, ...],
    ) -> tuple[ArtifactDigest, ...]:
        """Normalize checkpoint artifacts by logical name."""
        return _normalize_artifacts(artifacts)

    @model_validator(mode="after")
    def validate_checkpoint_state(self) -> Self:
        """Validate state requirements for the checkpoint kind."""
        state_values = (
            self.trainer_state_sha256,
            self.optimizer_state_sha256,
            self.rng_state_sha256,
        )

        if self.checkpoint_kind == "resumable":
            if any(value is None for value in state_values):
                raise ValueError(
                    "resumable checkpoints require trainer, optimizer, and RNG state digests"
                )
        elif any(value is not None for value in state_values):
            raise ValueError("weights_only checkpoints must not declare resume state")

        return self
