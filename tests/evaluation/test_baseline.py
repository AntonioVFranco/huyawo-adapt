from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import huyawo_adapt.evaluation.baseline as baseline
from datasets import Dataset
from huyawo_adapt.contracts import (
    ArtifactDigest,
    DatasetIdentity,
    DataSplitIdentity,
    EvaluationProfile,
    ModelIdentity,
    TokenizerIdentity,
)
from huyawo_adapt.datasets import DeterministicSplitManifest
from huyawo_adapt.evaluation import (
    BaselineEvaluationError,
    BaselineEvaluatorConfig,
    BaselineGenerationConfig,
    evaluate_huggingface_baseline,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
RESOLVED_REVISION = "1" * 40


class FakeIdentityEvidence:
    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint
        self.trust_remote_code = False
        self.eos_token_id = 2
        self.pad_token_id = 0

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is True
        return [1, len(text) + 3]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "yes" if token_ids == [9] else "no"


class FakeModel:
    def __init__(self, *, generated_token: int = 9, fail_generate: bool = False) -> None:
        self.training = True
        self.generated_token = generated_token
        self.fail_generate = fail_generate
        self.device: torch.device | None = None
        self.eval_called = False
        self.inference_mode_seen = False

    def to(self, device: torch.device) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.training = False
        self.eval_called = True
        return self

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        if self.fail_generate:
            raise RuntimeError("generation failed")
        self.inference_mode_seen = torch.is_inference_mode_enabled()
        continuation = torch.tensor(
            [[self.generated_token]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, continuation), dim=1)


class FakeAutoModel:
    model = FakeModel()
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.model = FakeModel()
        cls.calls = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
        cls.calls.append((model_id, kwargs))
        return cls.model


class FakeAutoTokenizer:
    tokenizer = FakeTokenizer()
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.tokenizer = FakeTokenizer()
        cls.calls = []

    @classmethod
    def from_pretrained(cls, tokenizer_id: str, **kwargs: object) -> FakeTokenizer:
        cls.calls.append((tokenizer_id, kwargs))
        return cls.tokenizer


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_model_identity() -> ModelIdentity:
    return ModelIdentity(
        source="huggingface",
        model_id="organization/model",
        revision=RESOLVED_REVISION,
        config_sha256=SHA_A,
        weight_artifacts=(
            ArtifactDigest(
                name="model.safetensors",
                sha256=SHA_B,
                size_bytes=10,
            ),
        ),
        architecture="FakeForCausalLM",
        parameter_count=100,
        trust_remote_code=False,
    )


def make_tokenizer_identity() -> TokenizerIdentity:
    return TokenizerIdentity(
        source="huggingface",
        tokenizer_id="organization/model",
        revision=RESOLVED_REVISION,
        artifacts=(
            ArtifactDigest(
                name="tokenizer.json",
                sha256=SHA_C,
                size_bytes=10,
            ),
        ),
        vocab_size=100,
        model_max_length=1024,
    )


def make_dataset_identity() -> DatasetIdentity:
    return DatasetIdentity(
        source_uri="fixture://baseline",
        license_expression="MIT",
        revision="fixture-v1",
        content_sha256=SHA_A,
        schema_sha256=SHA_B,
        preprocessing_sha256=SHA_C,
        deduplication_sha256=SHA_D,
        filtering_sha256=SHA_E,
        contamination_check_sha256=SHA_F,
        loss_mask_policy_sha256=SHA_A,
        sample_count=2,
    )


def make_fixture() -> tuple[
    DatasetIdentity,
    DataSplitIdentity,
    DeterministicSplitManifest,
    BaselineEvaluatorConfig,
    BaselineGenerationConfig,
    EvaluationProfile,
    Dataset,
]:
    dataset_identity = make_dataset_identity()
    dataset_fingerprint = dataset_identity.fingerprint()
    manifest = DeterministicSplitManifest(
        dataset_fingerprint=dataset_fingerprint,
        selection_policy_sha256=SHA_B,
        seed=7,
        split_name="validation",
        source_sample_count=2,
        sample_count=2,
        source_indices=(0, 1),
    )
    split_identity = DataSplitIdentity(
        dataset_fingerprint=dataset_fingerprint,
        split_name="validation",
        manifest_sha256=manifest.fingerprint(),
        selection_policy_sha256=SHA_B,
        sample_count=2,
        seed=7,
    )
    evaluator_config = BaselineEvaluatorConfig(
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_identity.fingerprint(),
        split_manifest_fingerprint=manifest.fingerprint(),
        prompt_column="prompt",
        reference_column="reference",
        metric_name="exact_match",
    )
    generation_config = BaselineGenerationConfig(max_new_tokens=1, eos_token_id=2, pad_token_id=0)
    profile = EvaluationProfile(
        profile_id="baseline-fixture-v1",
        dataset_fingerprints=(dataset_fingerprint,),
        metrics=("exact_match",),
        evaluator_config_sha256=evaluator_config.fingerprint(),
        generation_config_sha256=generation_config.fingerprint(),
        seed=7,
    )
    dataset = Dataset.from_dict(
        {
            "prompt": ["first", "second"],
            "reference": ["yes", "yes"],
        }
    )
    return (
        dataset_identity,
        split_identity,
        manifest,
        evaluator_config,
        generation_config,
        profile,
        dataset,
    )


def install_success_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
) -> None:
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_model",
        lambda *args, **kwargs: SimpleNamespace(
            identity=model_identity,
            evidence=FakeIdentityEvidence(SHA_D),
        ),
    )
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            identity=tokenizer_identity,
            evidence=FakeIdentityEvidence(SHA_E),
        ),
    )
    FakeAutoModel.reset()
    FakeAutoTokenizer.reset()
    monkeypatch.setattr(baseline, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(baseline, "AutoTokenizer", FakeAutoTokenizer)


def run_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, FakeModel]:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    (
        dataset_identity,
        split_identity,
        manifest,
        evaluator_config,
        generation_config,
        profile,
        dataset,
    ) = make_fixture()
    result = evaluate_huggingface_baseline(
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        dataset_identity=dataset_identity,
        split_identity=split_identity,
        split_manifest=manifest,
        evaluation_profile=profile,
        evaluator_config=evaluator_config,
        generation_config=generation_config,
        dataset=dataset,
        output_dir=tmp_path / "baseline-output",
        device="cpu",
    )
    return result, FakeAutoModel.model


def test_evaluator_config_fingerprint_is_deterministic() -> None:
    fixture = make_fixture()
    first = fixture[3]
    second = BaselineEvaluatorConfig(
        dataset_fingerprint=first.dataset_fingerprint,
        split_fingerprint=first.split_fingerprint,
        split_manifest_fingerprint=first.split_manifest_fingerprint,
        prompt_column=first.prompt_column,
        reference_column=first.reference_column,
        metric_name=first.metric_name,
    )
    assert first.fingerprint() == second.fingerprint()


def test_generation_config_fingerprint_is_deterministic() -> None:
    first = BaselineGenerationConfig(max_new_tokens=8, eos_token_id=2, pad_token_id=0)
    second = BaselineGenerationConfig(max_new_tokens=8, eos_token_id=2, pad_token_id=0)
    assert first.fingerprint() == second.fingerprint()


def test_generation_config_rejects_sampling() -> None:
    with pytest.raises(ValueError):
        BaselineGenerationConfig(max_new_tokens=8, eos_token_id=2, pad_token_id=0, do_sample=True)  # type: ignore[arg-type]


def test_generation_config_rejects_beam_search() -> None:
    with pytest.raises(ValueError):
        BaselineGenerationConfig(max_new_tokens=8, eos_token_id=2, pad_token_id=0, num_beams=2)  # type: ignore[arg-type]


def test_generation_config_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        BaselineGenerationConfig(max_new_tokens=0, eos_token_id=2, pad_token_id=0)


def test_profile_evaluator_config_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    profile = fixture[5].model_copy(update={"evaluator_config_sha256": SHA_F})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=profile,
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "bind_evaluation_profile"
    assert exc_info.value.failure_artifact is not None


def test_profile_generation_config_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    profile = fixture[5].model_copy(update={"generation_config_sha256": SHA_F})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=profile,
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "bind_evaluation_profile"


def test_dataset_profile_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    profile = fixture[5].model_copy(update={"dataset_fingerprints": (SHA_F,)})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=profile,
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "bind_evaluation_profile"


def test_split_dataset_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    split_identity = fixture[1].model_copy(update={"dataset_fingerprint": SHA_F})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=split_identity,
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_dataset_split_binding"


def test_manifest_fingerprint_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    split_identity = fixture[1].model_copy(update={"manifest_sha256": SHA_F})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=split_identity,
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_dataset_split_binding"


def test_split_metadata_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    split_identity = fixture[1].model_copy(update={"sample_count": 1})
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=split_identity,
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_dataset_split_binding"


def test_unsupported_model_source_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity().model_copy(update={"source": "other"})
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_identity_alignment"


def test_identity_fingerprint_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_model = make_model_identity()
    observed_model = expected_model.model_copy(update={"config_sha256": SHA_F})
    tokenizer_identity = make_tokenizer_identity()
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_model",
        lambda *args, **kwargs: SimpleNamespace(
            identity=observed_model,
            evidence=FakeIdentityEvidence(SHA_D),
        ),
    )
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            identity=tokenizer_identity,
            evidence=FakeIdentityEvidence(SHA_E),
        ),
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=expected_model,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "verify_model_identity"


def test_revision_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity().model_copy(update={"revision": "2" * 40})
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_identity_alignment"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", 1),
        ("reference", 1),
    ],
)
def test_non_string_input_fields_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    data = {
        "prompt": ["first", "second"],
        "reference": ["yes", "yes"],
    }
    data[field] = [value, value]
    dataset = Dataset.from_dict(data)
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=dataset,
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "prepare_sample"


def test_exact_match_success_and_failure_are_observed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    assert result.samples[0].score == 1.0
    assert result.samples[1].score == 1.0
    assert result.evidence.target_metrics[0].name == "exact_match"
    assert result.evidence.target_metrics[0].value == 1.0
    assert result.evidence.target_metrics[0].source == "observed"


def test_generated_continuation_excludes_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    assert result.samples[0].input_token_ids == (1, 8)
    assert result.samples[0].generated_token_ids == (9,)


def test_model_eval_and_inference_mode_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, model = run_success(monkeypatch, tmp_path)
    assert model.eval_called is True
    assert model.training is False
    assert model.inference_mode_seen is True


def test_sample_evidence_and_raw_hash_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, _ = run_success(monkeypatch, tmp_path / "first")
    second, _ = run_success(monkeypatch, tmp_path / "second")
    assert tuple(sample.fingerprint() for sample in first.samples) == tuple(
        sample.fingerprint() for sample in second.samples
    )
    assert first.evidence.raw_evidence_artifacts == second.evidence.raw_evidence_artifacts
    assert (
        first.evidence.reproducibility_fingerprint()
        == second.evidence.reproducibility_fingerprint()
    )


def test_systems_measurement_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    assert result.evidence.systems.wall_time_seconds > 0
    assert result.evidence.systems.samples_per_second is not None
    assert result.evidence.systems.samples_per_second > 0
    assert result.evidence.systems.stable is True
    assert result.evidence.runtime_dtype == "float32"
    assert result.evidence.runtime_device_type == "cpu"


def test_generation_failure_is_structured_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    FakeAutoModel.model = FakeModel(fail_generate=True)
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "generate"
    assert exc_info.value.failure_artifact is not None
    assert (tmp_path / "out" / "baseline_failure.json").is_file()


def test_unsupported_tokenizer_source_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity().model_copy(update={"source": "other"})
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "validate_identity_alignment"


def test_tokenizer_identity_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    expected_tokenizer = make_tokenizer_identity()
    observed_tokenizer = expected_tokenizer.model_copy(update={"vocab_size": 101})
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_model",
        lambda *args, **kwargs: SimpleNamespace(
            identity=model_identity,
            evidence=FakeIdentityEvidence(SHA_D),
        ),
    )
    monkeypatch.setattr(
        baseline,
        "inspect_huggingface_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(
            identity=observed_tokenizer,
            evidence=FakeIdentityEvidence(SHA_E),
        ),
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=expected_tokenizer,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "verify_tokenizer_identity"


@pytest.mark.parametrize("missing_column", ["prompt", "reference"])
def test_missing_input_column_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_column: str,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    fixture = make_fixture()
    data = {
        "prompt": ["first", "second"],
        "reference": ["yes", "yes"],
    }
    del data[missing_column]
    dataset = Dataset.from_dict(data)
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=dataset,
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "prepare_sample"


def test_exact_match_failure_scores_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    FakeAutoModel.model = FakeModel(generated_token=8)
    fixture = make_fixture()
    result = evaluate_huggingface_baseline(
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        dataset_identity=fixture[0],
        split_identity=fixture[1],
        split_manifest=fixture[2],
        evaluation_profile=fixture[5],
        evaluator_config=fixture[3],
        generation_config=fixture[4],
        dataset=fixture[6],
        output_dir=tmp_path / "out",
        device="cpu",
    )
    assert result.evidence.target_metrics[0].value == 0.0


def test_scoring_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    install_success_environment(
        monkeypatch,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
    )
    monkeypatch.setattr(
        baseline,
        "_score_exact_match",
        lambda generated_text, reference: (_ for _ in ()).throw(RuntimeError("score failed")),
    )
    fixture = make_fixture()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=tmp_path / "out",
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "score"


def test_output_directory_must_be_new(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_identity = make_model_identity()
    tokenizer_identity = make_tokenizer_identity()
    fixture = make_fixture()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(BaselineEvaluationError) as exc_info:
        evaluate_huggingface_baseline(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            dataset_identity=fixture[0],
            split_identity=fixture[1],
            split_manifest=fixture[2],
            evaluation_profile=fixture[5],
            evaluator_config=fixture[3],
            generation_config=fixture[4],
            dataset=fixture[6],
            output_dir=output_dir,
            device="cpu",
        )
    assert exc_info.value.evidence.stage == "input"


def test_loads_model_and_tokenizer_at_immutable_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    assert result.evidence.model_fingerprint == make_model_identity().fingerprint()
    assert FakeAutoModel.calls[0][0] == "organization/model"
    assert FakeAutoModel.calls[0][1]["revision"] == RESOLVED_REVISION
    assert FakeAutoModel.calls[0][1]["trust_remote_code"] is False
    assert FakeAutoTokenizer.calls[0][0] == "organization/model"
    assert FakeAutoTokenizer.calls[0][1]["revision"] == RESOLVED_REVISION
    assert FakeAutoTokenizer.calls[0][1]["trust_remote_code"] is False


def test_aggregate_evidence_artifact_matches_evidence_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    assert result.evidence_artifact.name == "baseline_evidence.json"
    assert result.evidence_artifact.sha256 == result.evidence.fingerprint()


def test_raw_evidence_artifact_contains_canonical_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, _ = run_success(monkeypatch, tmp_path)
    path = tmp_path / "baseline-output" / "baseline_samples.jsonl"
    assert path.read_text(encoding="utf-8") == "".join(
        sample.canonical_json() + "\n" for sample in result.samples
    )


def test_package_exports_baseline_api() -> None:
    import huyawo_adapt.evaluation as evaluation_package

    expected = {
        "BaselineEvaluationError": baseline.BaselineEvaluationError,
        "BaselineEvaluationEvidence": baseline.BaselineEvaluationEvidence,
        "BaselineEvaluationFailure": baseline.BaselineEvaluationFailure,
        "BaselineEvaluationInspection": baseline.BaselineEvaluationInspection,
        "BaselineEvaluatorConfig": baseline.BaselineEvaluatorConfig,
        "BaselineGenerationConfig": baseline.BaselineGenerationConfig,
        "BaselineSampleEvidence": baseline.BaselineSampleEvidence,
        "evaluate_huggingface_baseline": baseline.evaluate_huggingface_baseline,
    }
    for name, value in expected.items():
        assert name in evaluation_package.__all__
        assert getattr(evaluation_package, name) is value


def test_config_payloads_use_stable_field_sets() -> None:
    fixture = make_fixture()
    assert fixture[3].fingerprint() == canonical_sha256(fixture[3].canonical_data())
    assert fixture[4].fingerprint() == canonical_sha256(fixture[4].canonical_data())
