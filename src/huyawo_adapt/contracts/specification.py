from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from huyawo_adapt.contracts.base import (
    ContractModel,
    NonEmptyString,
    Sha256Digest,
    StrictFrozenModel,
)

DTypeName = Literal[
    "bfloat16",
    "float16",
    "float32",
]

MetricDirection = Literal[
    "maximize",
    "minimize",
]

TaskType = Literal[
    "causal_lm",
    "instruction_following",
]

NonEmptyDTypeTuple = Annotated[
    tuple[DTypeName, ...],
    Field(min_length=1),
]

NonEmptyStringTuple = Annotated[
    tuple[NonEmptyString, ...],
    Field(min_length=1),
]

NonEmptySha256Tuple = Annotated[
    tuple[Sha256Digest, ...],
    Field(min_length=1),
]


def _normalize_unique_strings[StringType: str](
    values: tuple[StringType, ...],
    field_name: str,
) -> tuple[StringType, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")

    return tuple(sorted(values))


class ComputeCapability(StrictFrozenModel):
    """Minimum CUDA compute capability requirement."""

    major: int = Field(ge=1)
    minor: int = Field(ge=0)


class MetricGoal(StrictFrozenModel):
    """Primary optimization goal for one named metric."""

    metric: NonEmptyString
    direction: MetricDirection
    target_value: float | None = None


class MetricConstraint(StrictFrozenModel):
    """Allowed range for one required metric."""

    metric: NonEmptyString
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Validate that the metric constraint defines a valid range."""
        if self.minimum is None and self.maximum is None:
            raise ValueError("metric constraint requires minimum or maximum")

        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("metric constraint minimum must not exceed maximum")

        return self


NonEmptyMetricConstraintTuple = Annotated[
    tuple[MetricConstraint, ...],
    Field(min_length=1),
]


def _normalize_metric_constraints(
    constraints: tuple[MetricConstraint, ...],
) -> tuple[MetricConstraint, ...]:
    metrics = [constraint.metric for constraint in constraints]

    if len(metrics) != len(set(metrics)):
        raise ValueError("metric constraints must use unique metric names")

    return tuple(
        sorted(
            constraints,
            key=lambda constraint: constraint.metric,
        )
    )


class HardwareTarget(ContractModel):
    """Capability-driven hardware requirements for an execution plan."""

    contract_type: Literal["hardware_target"] = "hardware_target"
    accelerator: Literal["cuda"] = "cuda"
    min_gpu_count: int = Field(default=1, ge=1)
    min_vram_bytes_per_gpu: int = Field(gt=0)
    min_compute_capability: ComputeCapability | None = None
    required_dtypes: NonEmptyDTypeTuple = ("float16",)
    required_features: tuple[NonEmptyString, ...] = ()

    @field_validator("required_dtypes")
    @classmethod
    def normalize_required_dtypes(
        cls,
        values: tuple[DTypeName, ...],
    ) -> tuple[DTypeName, ...]:
        """Normalize required dtypes into deterministic order."""
        return _normalize_unique_strings(
            values,
            "required_dtypes",
        )

    @field_validator("required_features")
    @classmethod
    def normalize_required_features(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize hardware features into deterministic order."""
        return _normalize_unique_strings(
            values,
            "required_features",
        )


class RuntimeTarget(ContractModel):
    """Required software runtime for reproducible execution."""

    contract_type: Literal["runtime_target"] = "runtime_target"
    python_version: NonEmptyString
    torch_version: NonEmptyString
    package_lock_sha256: Sha256Digest
    cuda_version: NonEmptyString | None = None
    requires_cuda: bool = True
    determinism_required: bool = True


class AdaptationObjective(ContractModel):
    """Machine-validatable target behavior and hard constraints."""

    contract_type: Literal["adaptation_objective"] = "adaptation_objective"
    objective_id: NonEmptyString
    task_type: TaskType
    primary_metric: MetricGoal
    hard_constraints: tuple[MetricConstraint, ...] = ()
    description: NonEmptyString | None = None

    @field_validator("hard_constraints")
    @classmethod
    def normalize_hard_constraints(
        cls,
        constraints: tuple[MetricConstraint, ...],
    ) -> tuple[MetricConstraint, ...]:
        """Normalize hard constraints into deterministic metric order."""
        return _normalize_metric_constraints(constraints)


class EvaluationProfile(ContractModel):
    """Deterministic evaluation definition for a candidate model."""

    contract_type: Literal["evaluation_profile"] = "evaluation_profile"
    profile_id: NonEmptyString
    dataset_fingerprints: NonEmptySha256Tuple
    metrics: NonEmptyStringTuple
    evaluator_config_sha256: Sha256Digest
    generation_config_sha256: Sha256Digest | None = None
    seed: int = Field(default=0, ge=0)

    @field_validator("dataset_fingerprints")
    @classmethod
    def normalize_dataset_fingerprints(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize evaluation dataset identities."""
        return _normalize_unique_strings(
            values,
            "dataset_fingerprints",
        )

    @field_validator("metrics")
    @classmethod
    def normalize_metrics(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize metric names into deterministic order."""
        return _normalize_unique_strings(
            values,
            "metrics",
        )


class RegressionProfile(ContractModel):
    """Regression constraints evaluated against a reference model."""

    contract_type: Literal["regression_profile"] = "regression_profile"
    profile_id: NonEmptyString
    reference_model_fingerprint: Sha256Digest
    evaluation_profile_fingerprint: Sha256Digest
    constraints: NonEmptyMetricConstraintTuple
    fail_on_missing_metric: bool = True

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(
        cls,
        constraints: tuple[MetricConstraint, ...],
    ) -> tuple[MetricConstraint, ...]:
        """Normalize regression constraints into metric order."""
        return _normalize_metric_constraints(constraints)
