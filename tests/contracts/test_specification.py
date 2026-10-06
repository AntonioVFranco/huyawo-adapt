from __future__ import annotations

import pytest
from pydantic import ValidationError

from huyawo_adapt.contracts.specification import (
    AdaptationObjective,
    ComputeCapability,
    EvaluationProfile,
    HardwareTarget,
    MetricConstraint,
    MetricGoal,
    RegressionProfile,
    RuntimeTarget,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def make_hardware_target() -> HardwareTarget:
    return HardwareTarget(
        min_vram_bytes_per_gpu=16 * 1024**3,
        min_compute_capability=ComputeCapability(
            major=8,
            minor=0,
        ),
        required_dtypes=(
            "float32",
            "bfloat16",
            "float16",
        ),
        required_features=(
            "tensor_cores",
            "scaled_dot_product_attention",
        ),
    )


def make_evaluation_profile() -> EvaluationProfile:
    return EvaluationProfile(
        profile_id="instruction-quality-v1",
        dataset_fingerprints=(
            SHA_C,
            SHA_A,
        ),
        metrics=(
            "target_accuracy",
            "held_out_accuracy",
        ),
        evaluator_config_sha256=SHA_B,
        seed=42,
    )


def test_hardware_target_is_capability_driven() -> None:
    target = make_hardware_target()

    assert target.accelerator == "cuda"
    assert target.required_dtypes == (
        "bfloat16",
        "float16",
        "float32",
    )
    assert target.required_features == (
        "scaled_dot_product_attention",
        "tensor_cores",
    )

    schema = HardwareTarget.model_json_schema()

    assert "gpu_model" not in schema["properties"]
    assert schema["properties"]["contract_type"]["const"] == "hardware_target"


def test_hardware_target_rejects_duplicate_dtypes() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget(
            min_vram_bytes_per_gpu=16 * 1024**3,
            required_dtypes=(
                "float16",
                "float16",
            ),
        )


def test_hardware_target_rejects_invalid_vram_requirement() -> None:
    with pytest.raises(ValidationError):
        HardwareTarget(
            min_vram_bytes_per_gpu=0,
        )


def test_runtime_target_round_trip_is_stable() -> None:
    target = RuntimeTarget(
        python_version="3.12.3",
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        package_lock_sha256=SHA_A,
    )

    restored = RuntimeTarget.from_canonical_json(target.canonical_bytes())

    assert restored == target
    assert restored.fingerprint() == target.fingerprint()


def test_metric_constraint_requires_at_least_one_bound() -> None:
    with pytest.raises(ValidationError):
        MetricConstraint(metric="regression_score")


def test_metric_constraint_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError):
        MetricConstraint(
            metric="regression_score",
            minimum=1.0,
            maximum=0.5,
        )


def test_adaptation_objective_normalizes_constraints() -> None:
    objective = AdaptationObjective(
        objective_id="instruction-sft-v1",
        task_type="instruction_following",
        primary_metric=MetricGoal(
            metric="target_accuracy",
            direction="maximize",
        ),
        hard_constraints=(
            MetricConstraint(
                metric="z_metric",
                minimum=0.8,
            ),
            MetricConstraint(
                metric="a_metric",
                minimum=0.7,
            ),
        ),
    )

    assert [constraint.metric for constraint in objective.hard_constraints] == [
        "a_metric",
        "z_metric",
    ]


def test_adaptation_objective_rejects_duplicate_constraints() -> None:
    with pytest.raises(ValidationError):
        AdaptationObjective(
            objective_id="instruction-sft-v1",
            task_type="instruction_following",
            primary_metric=MetricGoal(
                metric="target_accuracy",
                direction="maximize",
            ),
            hard_constraints=(
                MetricConstraint(
                    metric="regression_score",
                    minimum=0.8,
                ),
                MetricConstraint(
                    metric="regression_score",
                    maximum=1.0,
                ),
            ),
        )


def test_adaptation_objective_rejects_unknown_task() -> None:
    with pytest.raises(ValidationError):
        AdaptationObjective.model_validate(
            {
                "objective_id": "invalid-task",
                "task_type": "classification",
                "primary_metric": {
                    "metric": "accuracy",
                    "direction": "maximize",
                },
            }
        )


def test_evaluation_profile_normalizes_sets_as_tuples() -> None:
    profile = make_evaluation_profile()

    assert profile.dataset_fingerprints == (
        SHA_A,
        SHA_C,
    )
    assert profile.metrics == (
        "held_out_accuracy",
        "target_accuracy",
    )


def test_evaluation_profile_rejects_duplicate_metrics() -> None:
    with pytest.raises(ValidationError):
        EvaluationProfile(
            profile_id="duplicate-metrics",
            dataset_fingerprints=(SHA_A,),
            metrics=(
                "accuracy",
                "accuracy",
            ),
            evaluator_config_sha256=SHA_B,
        )


def test_evaluation_profile_fingerprint_is_order_independent() -> None:
    first = make_evaluation_profile()

    second = EvaluationProfile(
        profile_id=first.profile_id,
        dataset_fingerprints=tuple(reversed(first.dataset_fingerprints)),
        metrics=tuple(reversed(first.metrics)),
        evaluator_config_sha256=first.evaluator_config_sha256,
        seed=first.seed,
    )

    assert first.fingerprint() == second.fingerprint()


def test_regression_profile_requires_constraint() -> None:
    with pytest.raises(ValidationError):
        RegressionProfile(
            profile_id="regression-v1",
            reference_model_fingerprint=SHA_A,
            evaluation_profile_fingerprint=SHA_B,
            constraints=(),
        )


def test_regression_profile_normalizes_constraints() -> None:
    profile = RegressionProfile(
        profile_id="regression-v1",
        reference_model_fingerprint=SHA_A,
        evaluation_profile_fingerprint=SHA_B,
        constraints=(
            MetricConstraint(
                metric="z_metric",
                minimum=-0.1,
            ),
            MetricConstraint(
                metric="a_metric",
                minimum=-0.05,
            ),
        ),
    )

    assert [constraint.metric for constraint in profile.constraints] == [
        "a_metric",
        "z_metric",
    ]


def test_regression_profile_round_trip_is_stable() -> None:
    profile = RegressionProfile(
        profile_id="regression-v1",
        reference_model_fingerprint=SHA_A,
        evaluation_profile_fingerprint=SHA_B,
        constraints=(
            MetricConstraint(
                metric="quality_delta",
                minimum=-0.02,
            ),
        ),
    )

    restored = RegressionProfile.from_canonical_json(profile.canonical_bytes())

    assert restored == profile
    assert restored.fingerprint() == profile.fingerprint()
