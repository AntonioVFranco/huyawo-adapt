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

BenchmarkStatus = Literal[
    "completed",
    "failed",
]

ComparisonHardwareMode = Literal[
    "same_physical_gpu",
    "controlled_difference",
]

QualificationState = Literal[
    "ELIGIBLE",
    "INELIGIBLE",
    "BLOCKED",
    "PREFERRED",
]

SuperiorityStatus = Literal[
    "NOT_EVALUATED",
    "NOT_PROVEN",
    "SUPERIORITY_PROVEN",
]

MeasurementSource = Literal[
    "observed",
    "derived",
]


class MetricMeasurement(StrictFrozenModel):
    """One named observed or derived metric measurement."""

    name: NonEmptyString
    value: float
    unit: NonEmptyString | None = None
    source: MeasurementSource = "observed"


class SystemsMeasurement(StrictFrozenModel):
    """Systems measurements associated with one benchmark execution."""

    wall_time_seconds: float = Field(gt=0)
    samples_per_second: float | None = Field(default=None, gt=0)
    tokens_per_second: float | None = Field(default=None, gt=0)
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    artifact_size_bytes: int | None = Field(default=None, ge=0)
    convergence_step: int | None = Field(default=None, ge=0)
    stable: bool


NonEmptyMetricTuple = Annotated[
    tuple[MetricMeasurement, ...],
    Field(min_length=1),
]

NonEmptyArtifactTuple = Annotated[
    tuple[ArtifactDigest, ...],
    Field(min_length=1),
]

NonEmptyShaTuple = Annotated[
    tuple[Sha256Digest, ...],
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


def _normalize_measurements(
    measurements: tuple[MetricMeasurement, ...],
    field_name: str,
) -> tuple[MetricMeasurement, ...]:
    names = [measurement.name for measurement in measurements]

    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must use unique metric names")

    return tuple(
        sorted(
            measurements,
            key=lambda measurement: measurement.name,
        )
    )


def _require_measurement_source(
    measurements: tuple[MetricMeasurement, ...],
    expected_source: MeasurementSource,
    field_name: str,
) -> None:
    invalid = [
        measurement.name for measurement in measurements if measurement.source != expected_source
    ]

    if invalid:
        raise ValueError(f"{field_name} requires source={expected_source}: {sorted(invalid)}")


class BenchmarkResult(ContractModel):
    """Target, regression, and systems evidence for one benchmark subject."""

    contract_type: Literal["benchmark_result"] = "benchmark_result"
    benchmark_id: NonEmptyString
    subject_fingerprint: Sha256Digest
    evaluation_profile_fingerprint: Sha256Digest
    regression_profile_fingerprint: Sha256Digest
    status: BenchmarkStatus
    target_metrics: tuple[MetricMeasurement, ...] = ()
    regression_metrics: tuple[MetricMeasurement, ...] = ()
    baseline_benchmark_fingerprint: Sha256Digest | None = None
    baseline_relative_metrics: tuple[MetricMeasurement, ...] = ()
    systems: SystemsMeasurement | None = None
    artifact_valid: bool | None = None
    raw_evidence_artifacts: NonEmptyArtifactTuple
    failure_record_sha256: Sha256Digest | None = None

    @field_validator(
        "target_metrics",
        "regression_metrics",
        "baseline_relative_metrics",
    )
    @classmethod
    def normalize_measurements(
        cls,
        measurements: tuple[MetricMeasurement, ...],
        info: object,
    ) -> tuple[MetricMeasurement, ...]:
        """Normalize benchmark metrics into deterministic order."""
        field_name = getattr(info, "field_name", "measurements")

        return _normalize_measurements(
            measurements,
            field_name,
        )

    @field_validator("raw_evidence_artifacts")
    @classmethod
    def normalize_raw_evidence_artifacts(
        cls,
        artifacts: tuple[ArtifactDigest, ...],
    ) -> tuple[ArtifactDigest, ...]:
        """Normalize raw evidence artifacts by logical name."""
        return _normalize_artifacts(artifacts)

    @model_validator(mode="after")
    def validate_result_state(self) -> Self:
        """Validate benchmark completion and failure evidence."""
        _require_measurement_source(
            self.target_metrics,
            "observed",
            "target_metrics",
        )
        _require_measurement_source(
            self.regression_metrics,
            "observed",
            "regression_metrics",
        )
        _require_measurement_source(
            self.baseline_relative_metrics,
            "derived",
            "baseline_relative_metrics",
        )

        if self.status == "completed":
            if not self.target_metrics:
                raise ValueError("completed benchmark requires target metrics")

            if not self.regression_metrics:
                raise ValueError("completed benchmark requires regression metrics")

            if self.systems is None:
                raise ValueError("completed benchmark requires systems measurements")

            if self.artifact_valid is None:
                raise ValueError("completed benchmark requires artifact validity")

            if self.failure_record_sha256 is not None:
                raise ValueError("completed benchmark must not declare failure evidence")

            if (
                self.baseline_benchmark_fingerprint is not None
                and not self.baseline_relative_metrics
            ):
                raise ValueError(
                    "benchmark linked to a baseline requires baseline-relative metrics"
                )

        if self.status == "failed":
            if self.failure_record_sha256 is None:
                raise ValueError("failed benchmark requires failure evidence")

        if self.baseline_benchmark_fingerprint is None and self.baseline_relative_metrics:
            raise ValueError("baseline-relative metrics require a baseline benchmark")

        return self


class CompetitorResult(ContractModel):
    """Controlled comparison between Huyawo Adapt and a competitor stack."""

    contract_type: Literal["competitor_result"] = "competitor_result"
    comparison_id: NonEmptyString
    competitor_id: NonEmptyString
    competitor_version: NonEmptyString
    method: NonEmptyString
    candidate_benchmark_fingerprint: Sha256Digest
    competitor_benchmark_fingerprint: Sha256Digest
    configuration_sha256: Sha256Digest
    tuning_protocol_sha256: Sha256Digest
    optimization_budget_sha256: Sha256Digest
    environment_lock_sha256: Sha256Digest
    hardware_snapshot_sha256: Sha256Digest
    comparison_hardware_mode: ComparisonHardwareMode
    hardware_difference_protocol_sha256: Sha256Digest | None = None
    comparison_metrics: NonEmptyMetricTuple

    @field_validator("comparison_metrics")
    @classmethod
    def normalize_comparison_metrics(
        cls,
        measurements: tuple[MetricMeasurement, ...],
    ) -> tuple[MetricMeasurement, ...]:
        """Normalize competitor-relative metrics."""
        return _normalize_measurements(
            measurements,
            "comparison_metrics",
        )

    @model_validator(mode="after")
    def validate_comparison_integrity(self) -> Self:
        """Validate hardware and derived-metric comparison evidence."""
        _require_measurement_source(
            self.comparison_metrics,
            "derived",
            "comparison_metrics",
        )

        if self.comparison_hardware_mode == "same_physical_gpu":
            if self.hardware_difference_protocol_sha256 is not None:
                raise ValueError(
                    "same-physical-GPU comparison must not declare a hardware-difference protocol"
                )

        if self.comparison_hardware_mode == "controlled_difference":
            if self.hardware_difference_protocol_sha256 is None:
                raise ValueError(
                    "controlled hardware difference requires a hardware-difference protocol"
                )

        return self


class QualificationResult(ContractModel):
    """Evidence-backed candidate qualification and superiority decision."""

    contract_type: Literal["qualification_result"] = "qualification_result"
    qualification_id: NonEmptyString
    candidate_checkpoint_fingerprint: Sha256Digest
    benchmark_result_fingerprints: NonEmptyShaTuple
    competitor_result_fingerprints: tuple[Sha256Digest, ...] = ()
    state: QualificationState
    superiority_status: SuperiorityStatus
    primary_objective_passed: bool | None = None
    hard_constraints_passed: bool | None = None
    competitor_benchmark_complete: bool = False
    qualification_policy_sha256: Sha256Digest
    superiority_criterion_sha256: Sha256Digest
    pareto_record_sha256: Sha256Digest | None = None
    decision_record_sha256: Sha256Digest

    @field_validator("benchmark_result_fingerprints")
    @classmethod
    def normalize_benchmark_fingerprints(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize benchmark references."""
        return _normalize_unique_strings(
            values,
            "benchmark_result_fingerprints",
        )

    @field_validator("competitor_result_fingerprints")
    @classmethod
    def normalize_competitor_fingerprints(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Normalize competitor references."""
        return _normalize_unique_strings(
            values,
            "competitor_result_fingerprints",
        )

    @model_validator(mode="after")
    def validate_qualification_state(self) -> Self:
        """Validate candidate state and superiority requirements."""
        if self.state in {
            "ELIGIBLE",
            "PREFERRED",
        }:
            if self.primary_objective_passed is not True:
                raise ValueError("eligible or preferred candidates must pass the primary objective")

            if self.hard_constraints_passed is not True:
                raise ValueError("eligible or preferred candidates must pass all hard constraints")

        if self.competitor_benchmark_complete:
            if not self.competitor_result_fingerprints:
                raise ValueError("completed competitor benchmarking requires competitor evidence")

            if self.pareto_record_sha256 is None:
                raise ValueError("completed competitor benchmarking requires a Pareto record")

            if self.superiority_status == "NOT_EVALUATED":
                raise ValueError(
                    "completed competitor benchmarking requires a superiority decision"
                )
        elif self.superiority_status != "NOT_EVALUATED":
            raise ValueError(
                "superiority cannot be evaluated before competitor benchmarking is complete"
            )

        if self.superiority_status == "SUPERIORITY_PROVEN":
            if self.state != "PREFERRED":
                raise ValueError("SUPERIORITY_PROVEN requires PREFERRED state")

            if self.primary_objective_passed is not True:
                raise ValueError("SUPERIORITY_PROVEN requires primary objective pass")

            if self.hard_constraints_passed is not True:
                raise ValueError("SUPERIORITY_PROVEN requires hard-constraint pass")

            if not self.competitor_benchmark_complete:
                raise ValueError("SUPERIORITY_PROVEN requires completed competitor benchmarking")

        if self.state == "BLOCKED" and self.superiority_status == "SUPERIORITY_PROVEN":
            raise ValueError("BLOCKED qualification cannot prove superiority")

        return self


class EvidenceBundle(ContractModel):
    """Canonical internal evidence object for one adaptation candidate."""

    contract_type: Literal["evidence_bundle"] = "evidence_bundle"
    bundle_id: NonEmptyString
    model_identity_fingerprints: NonEmptyShaTuple
    tokenizer_identity_fingerprints: NonEmptyShaTuple
    dataset_identity_fingerprints: NonEmptyShaTuple
    data_split_identity_fingerprints: NonEmptyShaTuple
    adaptation_objective_fingerprint: Sha256Digest
    evaluation_profile_fingerprint: Sha256Digest
    regression_profile_fingerprint: Sha256Digest
    recipe_fingerprint: Sha256Digest
    training_plan_fingerprint: Sha256Digest
    training_run_fingerprint: Sha256Digest
    checkpoint_fingerprints: NonEmptyShaTuple
    benchmark_result_fingerprints: NonEmptyShaTuple
    competitor_result_fingerprints: tuple[Sha256Digest, ...] = ()
    environment_lock_sha256: Sha256Digest
    hardware_snapshot_sha256: Sha256Digest
    raw_evidence_artifacts: NonEmptyArtifactTuple
    derived_evidence_artifacts: tuple[ArtifactDigest, ...] = ()
    failure_record_sha256s: tuple[Sha256Digest, ...] = ()
    reproducibility_record_sha256: Sha256Digest
    qualification: QualificationResult

    @field_validator(
        "model_identity_fingerprints",
        "tokenizer_identity_fingerprints",
        "dataset_identity_fingerprints",
        "data_split_identity_fingerprints",
        "checkpoint_fingerprints",
        "benchmark_result_fingerprints",
        "competitor_result_fingerprints",
        "failure_record_sha256s",
    )
    @classmethod
    def normalize_fingerprint_collections(
        cls,
        values: tuple[str, ...],
        info: object,
    ) -> tuple[str, ...]:
        """Normalize fingerprint collections into deterministic order."""
        field_name = getattr(info, "field_name", "fingerprints")

        return _normalize_unique_strings(
            values,
            field_name,
        )

    @field_validator(
        "raw_evidence_artifacts",
        "derived_evidence_artifacts",
    )
    @classmethod
    def normalize_evidence_artifacts(
        cls,
        artifacts: tuple[ArtifactDigest, ...],
        info: object,
    ) -> tuple[ArtifactDigest, ...]:
        """Normalize evidence artifact collections."""
        if not artifacts:
            return artifacts

        return _normalize_artifacts(artifacts)

    @model_validator(mode="after")
    def validate_evidence_references(self) -> Self:
        """Validate qualification references against bundle contents."""
        if self.qualification.candidate_checkpoint_fingerprint not in self.checkpoint_fingerprints:
            raise ValueError("qualification checkpoint is not present in the Evidence Bundle")

        missing_benchmarks = set(self.qualification.benchmark_result_fingerprints) - set(
            self.benchmark_result_fingerprints
        )

        if missing_benchmarks:
            raise ValueError(
                "qualification references benchmark results not present in the Evidence Bundle"
            )

        missing_competitors = set(self.qualification.competitor_result_fingerprints) - set(
            self.competitor_result_fingerprints
        )

        if missing_competitors:
            raise ValueError(
                "qualification references competitor results not present in the Evidence Bundle"
            )

        return self
