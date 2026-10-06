from __future__ import annotations

import pytest
from pydantic import ValidationError

from huyawo_adapt.contracts import ArtifactDigest
from huyawo_adapt.contracts.evidence import (
    BenchmarkResult,
    CompetitorResult,
    EvidenceBundle,
    MetricMeasurement,
    QualificationResult,
    SystemsMeasurement,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def make_completed_benchmark() -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id="candidate-benchmark-v1",
        subject_fingerprint=SHA_A,
        evaluation_profile_fingerprint=SHA_B,
        regression_profile_fingerprint=SHA_C,
        status="completed",
        target_metrics=(
            MetricMeasurement(
                name="target_accuracy",
                value=0.82,
            ),
        ),
        regression_metrics=(
            MetricMeasurement(
                name="held_out_accuracy",
                value=0.79,
            ),
        ),
        baseline_benchmark_fingerprint=SHA_D,
        baseline_relative_metrics=(
            MetricMeasurement(
                name="target_accuracy_delta",
                value=0.04,
                source="derived",
            ),
        ),
        systems=SystemsMeasurement(
            wall_time_seconds=120.0,
            tokens_per_second=4200.0,
            peak_vram_bytes=12_000_000_000,
            artifact_size_bytes=1_000_000_000,
            stable=True,
        ),
        artifact_valid=True,
        raw_evidence_artifacts=(
            ArtifactDigest(
                name="metrics.json",
                sha256=SHA_E,
            ),
        ),
    )


def make_competitor_result() -> CompetitorResult:
    return CompetitorResult(
        comparison_id="comparison-v1",
        competitor_id="reference-stack",
        competitor_version="1.0",
        method="lora",
        candidate_benchmark_fingerprint=SHA_A,
        competitor_benchmark_fingerprint=SHA_B,
        configuration_sha256=SHA_C,
        tuning_protocol_sha256=SHA_D,
        optimization_budget_sha256=SHA_E,
        environment_lock_sha256=SHA_F,
        hardware_snapshot_sha256=SHA_A,
        comparison_hardware_mode="same_physical_gpu",
        comparison_metrics=(
            MetricMeasurement(
                name="throughput_delta",
                value=0.12,
                source="derived",
            ),
        ),
    )


def make_superior_qualification() -> QualificationResult:
    return QualificationResult(
        qualification_id="qualification-v1",
        candidate_checkpoint_fingerprint=SHA_A,
        benchmark_result_fingerprints=(
            SHA_C,
            SHA_B,
        ),
        competitor_result_fingerprints=(SHA_D,),
        state="PREFERRED",
        superiority_status="SUPERIORITY_PROVEN",
        primary_objective_passed=True,
        hard_constraints_passed=True,
        competitor_benchmark_complete=True,
        qualification_policy_sha256=SHA_E,
        superiority_criterion_sha256=SHA_F,
        pareto_record_sha256=SHA_A,
        decision_record_sha256=SHA_B,
    )


def test_completed_benchmark_round_trip_is_stable() -> None:
    result = make_completed_benchmark()

    restored = BenchmarkResult.from_canonical_json(result.canonical_bytes())

    assert restored == result
    assert restored.fingerprint() == result.fingerprint()


def test_completed_benchmark_requires_target_metrics() -> None:
    values = make_completed_benchmark().model_dump()
    values["target_metrics"] = ()

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(values)


def test_completed_benchmark_requires_regression_metrics() -> None:
    values = make_completed_benchmark().model_dump()
    values["regression_metrics"] = ()

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(values)


def test_completed_benchmark_rejects_failure_record() -> None:
    values = make_completed_benchmark().model_dump()
    values["failure_record_sha256"] = SHA_F

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(values)


def test_failed_benchmark_requires_failure_evidence() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            benchmark_id="failed-benchmark",
            subject_fingerprint=SHA_A,
            evaluation_profile_fingerprint=SHA_B,
            regression_profile_fingerprint=SHA_C,
            status="failed",
            raw_evidence_artifacts=(
                ArtifactDigest(
                    name="failure.log",
                    sha256=SHA_D,
                ),
            ),
        )


def test_failed_benchmark_preserves_partial_evidence() -> None:
    result = BenchmarkResult(
        benchmark_id="failed-benchmark",
        subject_fingerprint=SHA_A,
        evaluation_profile_fingerprint=SHA_B,
        regression_profile_fingerprint=SHA_C,
        status="failed",
        target_metrics=(
            MetricMeasurement(
                name="partial_accuracy",
                value=0.5,
            ),
        ),
        raw_evidence_artifacts=(
            ArtifactDigest(
                name="failure.log",
                sha256=SHA_D,
            ),
        ),
        failure_record_sha256=SHA_E,
    )

    assert result.status == "failed"
    assert result.failure_record_sha256 == SHA_E
    assert len(result.target_metrics) == 1


def test_baseline_relative_metrics_require_baseline() -> None:
    with pytest.raises(ValidationError):
        BenchmarkResult(
            benchmark_id="invalid-baseline",
            subject_fingerprint=SHA_A,
            evaluation_profile_fingerprint=SHA_B,
            regression_profile_fingerprint=SHA_C,
            status="failed",
            baseline_relative_metrics=(
                MetricMeasurement(
                    name="delta",
                    value=0.1,
                    source="derived",
                ),
            ),
            raw_evidence_artifacts=(
                ArtifactDigest(
                    name="failure.log",
                    sha256=SHA_D,
                ),
            ),
            failure_record_sha256=SHA_E,
        )


def test_observed_metrics_reject_derived_source() -> None:
    values = make_completed_benchmark().model_dump()
    values["target_metrics"] = (
        {
            "name": "target_accuracy",
            "value": 0.82,
            "source": "derived",
        },
    )

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(values)


def test_duplicate_benchmark_metrics_are_rejected() -> None:
    values = make_completed_benchmark().model_dump()
    values["target_metrics"] = (
        {
            "name": "accuracy",
            "value": 0.8,
            "source": "observed",
        },
        {
            "name": "accuracy",
            "value": 0.9,
            "source": "observed",
        },
    )

    with pytest.raises(ValidationError):
        BenchmarkResult.model_validate(values)


def test_competitor_result_same_gpu_mode_is_stable() -> None:
    result = make_competitor_result()

    restored = CompetitorResult.from_canonical_json(result.canonical_bytes())

    assert restored == result
    assert restored.fingerprint() == result.fingerprint()
    assert result.hardware_difference_protocol_sha256 is None


def test_controlled_hardware_difference_requires_protocol() -> None:
    values = make_competitor_result().model_dump()
    values["comparison_hardware_mode"] = "controlled_difference"

    with pytest.raises(ValidationError):
        CompetitorResult.model_validate(values)


def test_same_gpu_comparison_rejects_difference_protocol() -> None:
    values = make_competitor_result().model_dump()
    values["hardware_difference_protocol_sha256"] = SHA_F

    with pytest.raises(ValidationError):
        CompetitorResult.model_validate(values)


def test_competitor_metrics_must_be_derived() -> None:
    values = make_competitor_result().model_dump()
    values["comparison_metrics"] = (
        {
            "name": "throughput_delta",
            "value": 0.12,
            "source": "observed",
        },
    )

    with pytest.raises(ValidationError):
        CompetitorResult.model_validate(values)


def test_superiority_requires_preferred_state() -> None:
    values = make_superior_qualification().model_dump()
    values["state"] = "ELIGIBLE"

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(values)


def test_superiority_requires_completed_competitor_benchmark() -> None:
    values = make_superior_qualification().model_dump()
    values["competitor_benchmark_complete"] = False

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(values)


def test_completed_competitor_benchmark_requires_pareto_record() -> None:
    values = make_superior_qualification().model_dump()
    values["pareto_record_sha256"] = None

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(values)


def test_eligible_candidate_requires_hard_constraint_pass() -> None:
    values = make_superior_qualification().model_dump()
    values["state"] = "ELIGIBLE"
    values["superiority_status"] = "NOT_PROVEN"
    values["hard_constraints_passed"] = False

    with pytest.raises(ValidationError):
        QualificationResult.model_validate(values)


def test_incomplete_competitor_benchmark_is_not_evaluated() -> None:
    result = QualificationResult(
        qualification_id="qualification-in-progress",
        candidate_checkpoint_fingerprint=SHA_A,
        benchmark_result_fingerprints=(SHA_B,),
        state="ELIGIBLE",
        superiority_status="NOT_EVALUATED",
        primary_objective_passed=True,
        hard_constraints_passed=True,
        competitor_benchmark_complete=False,
        qualification_policy_sha256=SHA_C,
        superiority_criterion_sha256=SHA_D,
        decision_record_sha256=SHA_E,
    )

    assert result.superiority_status == "NOT_EVALUATED"


def test_qualification_fingerprints_are_normalized() -> None:
    result = make_superior_qualification()

    assert result.benchmark_result_fingerprints == (
        SHA_B,
        SHA_C,
    )


def test_evidence_bundle_round_trip_is_stable() -> None:
    qualification = make_superior_qualification()

    bundle = EvidenceBundle(
        bundle_id="bundle-v1",
        model_identity_fingerprints=(
            SHA_B,
            SHA_A,
        ),
        tokenizer_identity_fingerprints=(SHA_C,),
        dataset_identity_fingerprints=(SHA_D,),
        data_split_identity_fingerprints=(SHA_E,),
        adaptation_objective_fingerprint=SHA_F,
        evaluation_profile_fingerprint=SHA_A,
        regression_profile_fingerprint=SHA_B,
        recipe_fingerprint=SHA_C,
        training_plan_fingerprint=SHA_D,
        training_run_fingerprint=SHA_E,
        checkpoint_fingerprints=(SHA_A,),
        benchmark_result_fingerprints=(
            SHA_C,
            SHA_B,
        ),
        competitor_result_fingerprints=(SHA_D,),
        environment_lock_sha256=SHA_E,
        hardware_snapshot_sha256=SHA_F,
        raw_evidence_artifacts=(
            ArtifactDigest(
                name="metrics.json",
                sha256=SHA_A,
            ),
        ),
        derived_evidence_artifacts=(
            ArtifactDigest(
                name="comparison.json",
                sha256=SHA_B,
            ),
        ),
        reproducibility_record_sha256=SHA_C,
        qualification=qualification,
    )

    restored = EvidenceBundle.from_canonical_json(bundle.canonical_bytes())

    assert restored == bundle
    assert restored.fingerprint() == bundle.fingerprint()
    assert bundle.model_identity_fingerprints == (
        SHA_A,
        SHA_B,
    )


def test_evidence_bundle_rejects_missing_qualification_checkpoint() -> None:
    qualification = make_superior_qualification()

    with pytest.raises(ValidationError):
        EvidenceBundle(
            bundle_id="bundle-v1",
            model_identity_fingerprints=(SHA_A,),
            tokenizer_identity_fingerprints=(SHA_B,),
            dataset_identity_fingerprints=(SHA_C,),
            data_split_identity_fingerprints=(SHA_D,),
            adaptation_objective_fingerprint=SHA_E,
            evaluation_profile_fingerprint=SHA_F,
            regression_profile_fingerprint=SHA_A,
            recipe_fingerprint=SHA_B,
            training_plan_fingerprint=SHA_C,
            training_run_fingerprint=SHA_D,
            checkpoint_fingerprints=(SHA_B,),
            benchmark_result_fingerprints=(
                SHA_B,
                SHA_C,
            ),
            competitor_result_fingerprints=(SHA_D,),
            environment_lock_sha256=SHA_E,
            hardware_snapshot_sha256=SHA_F,
            raw_evidence_artifacts=(
                ArtifactDigest(
                    name="metrics.json",
                    sha256=SHA_A,
                ),
            ),
            reproducibility_record_sha256=SHA_B,
            qualification=qualification,
        )


def test_evidence_bundle_rejects_missing_benchmark_reference() -> None:
    qualification = make_superior_qualification()

    with pytest.raises(ValidationError):
        EvidenceBundle(
            bundle_id="bundle-v1",
            model_identity_fingerprints=(SHA_A,),
            tokenizer_identity_fingerprints=(SHA_B,),
            dataset_identity_fingerprints=(SHA_C,),
            data_split_identity_fingerprints=(SHA_D,),
            adaptation_objective_fingerprint=SHA_E,
            evaluation_profile_fingerprint=SHA_F,
            regression_profile_fingerprint=SHA_A,
            recipe_fingerprint=SHA_B,
            training_plan_fingerprint=SHA_C,
            training_run_fingerprint=SHA_D,
            checkpoint_fingerprints=(SHA_A,),
            benchmark_result_fingerprints=(SHA_B,),
            competitor_result_fingerprints=(SHA_D,),
            environment_lock_sha256=SHA_E,
            hardware_snapshot_sha256=SHA_F,
            raw_evidence_artifacts=(
                ArtifactDigest(
                    name="metrics.json",
                    sha256=SHA_A,
                ),
            ),
            reproducibility_record_sha256=SHA_B,
            qualification=qualification,
        )


def test_evidence_bundle_rejects_missing_competitor_reference() -> None:
    qualification = make_superior_qualification()

    with pytest.raises(ValidationError):
        EvidenceBundle(
            bundle_id="bundle-v1",
            model_identity_fingerprints=(SHA_A,),
            tokenizer_identity_fingerprints=(SHA_B,),
            dataset_identity_fingerprints=(SHA_C,),
            data_split_identity_fingerprints=(SHA_D,),
            adaptation_objective_fingerprint=SHA_E,
            evaluation_profile_fingerprint=SHA_F,
            regression_profile_fingerprint=SHA_A,
            recipe_fingerprint=SHA_B,
            training_plan_fingerprint=SHA_C,
            training_run_fingerprint=SHA_D,
            checkpoint_fingerprints=(SHA_A,),
            benchmark_result_fingerprints=(
                SHA_B,
                SHA_C,
            ),
            environment_lock_sha256=SHA_E,
            hardware_snapshot_sha256=SHA_F,
            raw_evidence_artifacts=(
                ArtifactDigest(
                    name="metrics.json",
                    sha256=SHA_A,
                ),
            ),
            reproducibility_record_sha256=SHA_B,
            qualification=qualification,
        )
