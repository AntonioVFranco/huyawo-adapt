from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, NoReturn, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from datasets import Dataset  # type: ignore[import-untyped]
from huyawo_adapt.contracts import (
    ArtifactDigest,
    DatasetIdentity,
    DataSplitIdentity,
    EvaluationProfile,
    MetricMeasurement,
    ModelIdentity,
    SystemsMeasurement,
    TokenizerIdentity,
)
from huyawo_adapt.datasets import (
    DeterministicSplitManifest,
    SplitMaterializationEvidence,
    materialize_split,
)
from huyawo_adapt.identity import inspect_huggingface_model, inspect_huggingface_tokenizer

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCORER_ID: Final[Literal["exact_match_utf8_v1"]] = "exact_match_utf8_v1"
_INPUT_MODE: Final[Literal["raw_text_v1"]] = "raw_text_v1"
_OUTPUT_NORMALIZATION: Final[Literal["none_utf8_v1"]] = "none_utf8_v1"
_TIMING_SCOPE = "inference_loop_excludes_load_v1"
_TOKEN_THROUGHPUT_SCOPE = "generated_tokens_only_v1"
_RAW_EVIDENCE_NAME = "baseline_samples.jsonl"
_FAILURE_EVIDENCE_NAME = "baseline_failure.json"
_AGGREGATE_EVIDENCE_NAME = "baseline_evidence.json"


@dataclass(frozen=True, slots=True)
class BaselineEvaluatorConfig:
    """Deterministic operational configuration for baseline evaluation."""

    dataset_fingerprint: str
    split_fingerprint: str
    split_manifest_fingerprint: str
    prompt_column: str
    reference_column: str
    metric_name: str
    scorer_id: Literal["exact_match_utf8_v1"] = _SCORER_ID
    input_mode: Literal["raw_text_v1"] = _INPUT_MODE
    output_normalization: Literal["none_utf8_v1"] = _OUTPUT_NORMALIZATION
    config_version: Literal["1.0"] = "1.0"

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("split_fingerprint", self.split_fingerprint),
            ("split_manifest_fingerprint", self.split_manifest_fingerprint),
        ):
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")

        for label, value in (
            ("prompt_column", self.prompt_column),
            ("reference_column", self.reference_column),
            ("metric_name", self.metric_name),
        ):
            if not value or value != value.strip():
                raise ValueError(f"{label} must be a non-empty trimmed string")

        if self.prompt_column == self.reference_column:
            raise ValueError("prompt_column and reference_column must differ")
        if self.scorer_id != _SCORER_ID:
            raise ValueError("initial baseline evaluator supports exact_match_utf8_v1 only")
        if self.input_mode != _INPUT_MODE:
            raise ValueError("initial baseline evaluator supports raw_text_v1 only")
        if self.output_normalization != _OUTPUT_NORMALIZATION:
            raise ValueError("initial baseline evaluator supports none_utf8_v1 only")
        if self.config_version != "1.0":
            raise ValueError("unsupported evaluator configuration version")

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evaluator configuration."""
        return {
            "config_version": self.config_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "input_mode": self.input_mode,
            "metric_name": self.metric_name,
            "output_normalization": self.output_normalization,
            "prompt_column": self.prompt_column,
            "reference_column": self.reference_column,
            "scorer_id": self.scorer_id,
            "split_fingerprint": self.split_fingerprint,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical evaluator configuration JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return evaluator configuration SHA-256."""
        return _sha256_bytes(self.canonical_json().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class BaselineGenerationConfig:
    """Deterministic generation semantics for the initial causal-LM baseline."""

    max_new_tokens: int
    eos_token_id: int | None
    pad_token_id: int | None
    do_sample: Literal[False] = False
    num_beams: Literal[1] = 1
    min_new_tokens: Literal[0] = 0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: Literal[0] = 0
    length_penalty: float = 1.0
    early_stopping: Literal[False] = False
    renormalize_logits: Literal[False] = False
    forced_bos_token_id: None = None
    forced_eos_token_id: None = None
    config_version: Literal["1.0"] = "1.0"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        for label, value in (
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or None")
        if self.do_sample is not False:
            raise ValueError("initial baseline generation requires do_sample=False")
        if self.num_beams != 1:
            raise ValueError("initial baseline generation requires num_beams=1")
        if self.min_new_tokens != 0:
            raise ValueError("initial baseline generation requires min_new_tokens=0")
        if self.repetition_penalty != 1.0:
            raise ValueError("initial baseline generation requires repetition_penalty=1.0")
        if self.no_repeat_ngram_size != 0:
            raise ValueError("initial baseline generation requires no_repeat_ngram_size=0")
        if self.length_penalty != 1.0:
            raise ValueError("initial baseline generation requires length_penalty=1.0")
        if self.early_stopping is not False:
            raise ValueError("initial baseline generation requires early_stopping=False")
        if self.renormalize_logits is not False:
            raise ValueError("initial baseline generation requires renormalize_logits=False")
        if self.forced_bos_token_id is not None or self.forced_eos_token_id is not None:
            raise ValueError("forced BOS/EOS generation is unsupported in the initial baseline")
        if self.config_version != "1.0":
            raise ValueError("unsupported generation configuration version")

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible generation configuration."""
        return {
            "config_version": self.config_version,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "eos_token_id": self.eos_token_id,
            "forced_bos_token_id": self.forced_bos_token_id,
            "forced_eos_token_id": self.forced_eos_token_id,
            "length_penalty": self.length_penalty,
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "pad_token_id": self.pad_token_id,
            "renormalize_logits": self.renormalize_logits,
            "repetition_penalty": self.repetition_penalty,
        }

    def to_transformers_config(self) -> GenerationConfig:
        """Build an explicit Transformers generation configuration."""
        return GenerationConfig(  # type: ignore[no-untyped-call]
            do_sample=self.do_sample,
            early_stopping=self.early_stopping,
            eos_token_id=self.eos_token_id,
            forced_bos_token_id=self.forced_bos_token_id,
            forced_eos_token_id=self.forced_eos_token_id,
            length_penalty=self.length_penalty,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_beams=self.num_beams,
            pad_token_id=self.pad_token_id,
            renormalize_logits=self.renormalize_logits,
            repetition_penalty=self.repetition_penalty,
        )

    def canonical_json(self) -> str:
        """Return deterministic canonical generation configuration JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return generation configuration SHA-256."""
        return _sha256_bytes(self.canonical_json().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class BaselineSampleEvidence:
    """Deterministic sample-level evidence for one baseline prediction."""

    sample_position: int
    source_index: int
    prompt: str
    reference: str
    input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    generated_text: str
    score: float

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible sample evidence."""
        return {
            "generated_text": self.generated_text,
            "generated_token_ids": list(self.generated_token_ids),
            "input_token_ids": list(self.input_token_ids),
            "prompt": self.prompt,
            "reference": self.reference,
            "sample_position": self.sample_position,
            "score": self.score,
            "source_index": self.source_index,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical sample evidence JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return sample evidence SHA-256."""
        return _sha256_bytes(self.canonical_json().encode("utf-8"))


@dataclass(frozen=True, slots=True)
class BaselineEvaluationFailure:
    """Structured provenance for a failed baseline evaluation."""

    stage: str
    model_id: str
    tokenizer_id: str
    evaluation_profile_fingerprint: str
    error_type: str
    message: str

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible failure evidence."""
        return {
            "error_type": self.error_type,
            "evaluation_profile_fingerprint": self.evaluation_profile_fingerprint,
            "message": self.message,
            "model_id": self.model_id,
            "stage": self.stage,
            "tokenizer_id": self.tokenizer_id,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical failure JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return failure evidence SHA-256."""
        return _sha256_bytes(self.canonical_json().encode("utf-8"))


class BaselineEvaluationError(RuntimeError):
    """Fail-closed baseline evaluation error with structured evidence."""

    def __init__(
        self,
        evidence: BaselineEvaluationFailure,
        failure_artifact: ArtifactDigest | None,
    ) -> None:
        self.evidence = evidence
        self.failure_artifact = failure_artifact
        super().__init__(f"{evidence.stage}: {evidence.message}")


@dataclass(frozen=True, slots=True)
class BaselineEvaluationEvidence:
    """Operational evidence for a completed base-model evaluation."""

    model_fingerprint: str
    tokenizer_fingerprint: str
    dataset_fingerprint: str
    split_fingerprint: str
    split_manifest_fingerprint: str
    evaluation_profile_fingerprint: str
    evaluator_config_sha256: str
    generation_config_sha256: str
    model_identity_evidence_fingerprint: str
    tokenizer_identity_evidence_fingerprint: str
    split_materialization: SplitMaterializationEvidence
    target_metrics: tuple[MetricMeasurement, ...]
    systems: SystemsMeasurement
    artifact_valid: bool
    raw_evidence_artifacts: tuple[ArtifactDigest, ...]
    scorer_id: str
    sample_count: int
    input_token_count: int
    generated_token_count: int
    runtime_dtype: str
    runtime_device_type: str
    runtime_device_index: int | None
    timing_scope: str
    token_throughput_scope: str
    library_versions: tuple[tuple[str, str], ...]

    def canonical_data(self) -> dict[str, object]:
        """Return complete observed evaluation evidence."""
        return {
            "artifact_valid": self.artifact_valid,
            "dataset_fingerprint": self.dataset_fingerprint,
            "evaluation_profile_fingerprint": self.evaluation_profile_fingerprint,
            "evaluator_config_sha256": self.evaluator_config_sha256,
            "generated_token_count": self.generated_token_count,
            "generation_config_sha256": self.generation_config_sha256,
            "input_token_count": self.input_token_count,
            "library_versions": [
                {"name": name, "version": library_version}
                for name, library_version in self.library_versions
            ],
            "model_fingerprint": self.model_fingerprint,
            "model_identity_evidence_fingerprint": self.model_identity_evidence_fingerprint,
            "raw_evidence_artifacts": [
                artifact.model_dump(mode="json", round_trip=True)
                for artifact in self.raw_evidence_artifacts
            ],
            "runtime_device_index": self.runtime_device_index,
            "runtime_device_type": self.runtime_device_type,
            "runtime_dtype": self.runtime_dtype,
            "sample_count": self.sample_count,
            "scorer_id": self.scorer_id,
            "split_fingerprint": self.split_fingerprint,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "split_materialization": self.split_materialization.canonical_data(),
            "systems": self.systems.model_dump(mode="json", round_trip=True),
            "target_metrics": [
                metric.model_dump(mode="json", round_trip=True) for metric in self.target_metrics
            ],
            "timing_scope": self.timing_scope,
            "token_throughput_scope": self.token_throughput_scope,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tokenizer_identity_evidence_fingerprint": (
                self.tokenizer_identity_evidence_fingerprint
            ),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical observed evidence JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return full observed evaluation evidence SHA-256."""
        return _sha256_bytes(self.canonical_json().encode("utf-8"))

    def reproducibility_data(self) -> dict[str, object]:
        """Return deterministic semantic evidence excluding volatile systems timing."""
        payload = self.canonical_data()
        payload.pop("systems")
        return payload

    def reproducibility_fingerprint(self) -> str:
        """Return deterministic semantic evidence SHA-256."""
        return _sha256_bytes(_canonical_json(self.reproducibility_data()).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class BaselineEvaluationInspection:
    """Completed operational baseline evaluation."""

    evidence: BaselineEvaluationEvidence
    samples: tuple[BaselineSampleEvidence, ...]
    evidence_artifact: ArtifactDigest


@dataclass(frozen=True, slots=True)
class _RuntimeSelection:
    device: torch.device
    dtype: torch.dtype
    dtype_name: str
    device_index: int | None


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest(path: Path, *, name: str) -> ArtifactDigest:
    return ArtifactDigest(
        name=name,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _prepare_output_directory(path: str | Path) -> Path:
    output_dir = Path(path)
    if output_dir.exists():
        raise ValueError("output_dir must not already exist")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _persist_failure(
    *,
    output_dir: Path | None,
    failure: BaselineEvaluationFailure,
) -> ArtifactDigest | None:
    if output_dir is None:
        return None

    path = output_dir / _FAILURE_EVIDENCE_NAME
    try:
        path.write_text(failure.canonical_json(), encoding="utf-8")
        return _artifact_digest(path, name=_FAILURE_EVIDENCE_NAME)
    except OSError:
        return None


def _raise_failure(
    *,
    stage: str,
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    evaluation_profile: EvaluationProfile,
    output_dir: Path | None,
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    failure = BaselineEvaluationFailure(
        stage=stage,
        model_id=model_identity.model_id,
        tokenizer_id=tokenizer_identity.tokenizer_id,
        evaluation_profile_fingerprint=evaluation_profile.fingerprint(),
        error_type=(type(cause).__name__ if cause is not None else "BaselineValidationError"),
        message=message,
    )
    failure_artifact = _persist_failure(output_dir=output_dir, failure=failure)
    error = BaselineEvaluationError(failure, failure_artifact)
    if cause is not None:
        raise error from cause
    raise error


def _validate_profile_binding(
    *,
    dataset_identity: DatasetIdentity,
    evaluation_profile: EvaluationProfile,
    evaluator_config: BaselineEvaluatorConfig,
    generation_config: BaselineGenerationConfig,
) -> None:
    dataset_fingerprint = dataset_identity.fingerprint()
    if evaluation_profile.dataset_fingerprints != (dataset_fingerprint,):
        raise ValueError(
            "initial baseline evaluator requires exactly one EvaluationProfile dataset fingerprint"
        )
    if evaluation_profile.evaluator_config_sha256 != evaluator_config.fingerprint():
        raise ValueError("EvaluationProfile evaluator_config_sha256 mismatch")
    if evaluation_profile.generation_config_sha256 != generation_config.fingerprint():
        raise ValueError("EvaluationProfile generation_config_sha256 mismatch")
    if evaluation_profile.metrics != (evaluator_config.metric_name,):
        raise ValueError(
            "initial baseline evaluator requires exactly one metric matching metric_name"
        )
    if evaluator_config.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("Evaluator configuration dataset fingerprint mismatch")


def _validate_identity_alignment(
    *,
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
) -> None:
    if model_identity.source != "huggingface":
        raise ValueError("initial baseline evaluator supports source=huggingface only")
    if tokenizer_identity.source != "huggingface":
        raise ValueError("initial baseline evaluator supports Hugging Face tokenizers only")
    if model_identity.trust_remote_code:
        raise ValueError("initial baseline evaluator requires trust_remote_code=False")
    if model_identity.revision != tokenizer_identity.revision:
        raise ValueError("model and tokenizer immutable revisions must match")


def _validate_dataset_split_binding(
    *,
    dataset_identity: DatasetIdentity,
    split_identity: DataSplitIdentity,
    split_manifest: DeterministicSplitManifest,
    evaluator_config: BaselineEvaluatorConfig,
) -> None:
    dataset_fingerprint = dataset_identity.fingerprint()
    split_fingerprint = split_identity.fingerprint()
    manifest_fingerprint = split_manifest.fingerprint()

    if split_identity.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("DataSplitIdentity dataset fingerprint mismatch")
    if split_manifest.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("split manifest dataset fingerprint mismatch")
    if split_identity.manifest_sha256 != manifest_fingerprint:
        raise ValueError("DataSplitIdentity manifest fingerprint mismatch")
    if split_identity.split_name != split_manifest.split_name:
        raise ValueError("split name mismatch")
    if split_identity.selection_policy_sha256 != split_manifest.selection_policy_sha256:
        raise ValueError("split selection policy mismatch")
    if split_identity.sample_count != split_manifest.sample_count:
        raise ValueError("split sample count mismatch")
    if split_identity.seed != split_manifest.seed:
        raise ValueError("split seed mismatch")
    if split_manifest.sample_count <= 0:
        raise ValueError("baseline evaluation split must contain at least one sample")
    if evaluator_config.split_fingerprint != split_fingerprint:
        raise ValueError("Evaluator configuration split fingerprint mismatch")
    if evaluator_config.split_manifest_fingerprint != manifest_fingerprint:
        raise ValueError("Evaluator configuration split manifest fingerprint mismatch")


def _select_runtime(device: str | None) -> _RuntimeSelection:
    requested = device
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: cpu, cuda")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        runtime_device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            return _RuntimeSelection(
                device=runtime_device,
                dtype=torch.bfloat16,
                dtype_name="bfloat16",
                device_index=torch.cuda.current_device(),
            )
        return _RuntimeSelection(
            device=runtime_device,
            dtype=torch.float16,
            dtype_name="float16",
            device_index=torch.cuda.current_device(),
        )

    return _RuntimeSelection(
        device=torch.device("cpu"),
        dtype=torch.float32,
        dtype_name="float32",
        device_index=None,
    )


def _validate_token_ids(raw_ids: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"{label} must be a non-empty list of token IDs")
    if not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in raw_ids
    ):
        raise ValueError(f"{label} contains an invalid token ID")
    return tuple(cast(list[int], raw_ids))


def _score_exact_match(generated_text: str, reference: str) -> float:
    return 1.0 if generated_text == reference else 0.0


def _persist_samples(
    *,
    output_dir: Path,
    samples: tuple[BaselineSampleEvidence, ...],
) -> ArtifactDigest:
    path = output_dir / _RAW_EVIDENCE_NAME
    payload = "".join(sample.canonical_json() + "\n" for sample in samples)
    path.write_text(payload, encoding="utf-8")
    return _artifact_digest(path, name=_RAW_EVIDENCE_NAME)


def _library_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("datasets", version("datasets")),
                ("torch", version("torch")),
                ("transformers", version("transformers")),
            )
        )
    )


def evaluate_huggingface_baseline(
    *,
    model_identity: ModelIdentity,
    tokenizer_identity: TokenizerIdentity,
    dataset_identity: DatasetIdentity,
    split_identity: DataSplitIdentity,
    split_manifest: DeterministicSplitManifest,
    evaluation_profile: EvaluationProfile,
    evaluator_config: BaselineEvaluatorConfig,
    generation_config: BaselineGenerationConfig,
    dataset: object,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
    device: str | None = None,
) -> BaselineEvaluationInspection:
    """Evaluate an exact Hugging Face base model and preserve operational evidence."""
    evidence_root: Path | None = None

    try:
        evidence_root = _prepare_output_directory(output_dir)
    except Exception as exc:
        _raise_failure(
            stage="input",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=None,
            message=str(exc),
            cause=exc,
        )

    assert evidence_root is not None

    try:
        _validate_profile_binding(
            dataset_identity=dataset_identity,
            evaluation_profile=evaluation_profile,
            evaluator_config=evaluator_config,
            generation_config=generation_config,
        )
    except Exception as exc:
        _raise_failure(
            stage="bind_evaluation_profile",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        observed_model = inspect_huggingface_model(
            model_identity.model_id,
            revision=model_identity.revision,
            cache_dir=cache_dir,
            token=token,
        )
        if observed_model.identity != model_identity:
            raise ValueError("observed ModelIdentity differs from expected ModelIdentity")
    except Exception as exc:
        _raise_failure(
            stage="verify_model_identity",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        observed_tokenizer = inspect_huggingface_tokenizer(
            tokenizer_identity.tokenizer_id,
            revision=tokenizer_identity.revision,
            cache_dir=cache_dir,
            token=token,
        )
        if observed_tokenizer.identity != tokenizer_identity:
            raise ValueError("observed TokenizerIdentity differs from expected TokenizerIdentity")
        if observed_tokenizer.evidence.trust_remote_code:
            raise ValueError("observed tokenizer requires remote code")
    except Exception as exc:
        _raise_failure(
            stage="verify_tokenizer_identity",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        _validate_identity_alignment(
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
        )
    except Exception as exc:
        _raise_failure(
            stage="validate_identity_alignment",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        _validate_dataset_split_binding(
            dataset_identity=dataset_identity,
            split_identity=split_identity,
            split_manifest=split_manifest,
            evaluator_config=evaluator_config,
        )
    except Exception as exc:
        _raise_failure(
            stage="validate_dataset_split_binding",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        materialized = materialize_split(dataset, split_manifest)
        materialized_dataset = materialized.dataset
        if not isinstance(materialized_dataset, Dataset):
            raise TypeError("materialize_split did not return a Dataset")
        evaluation_dataset = materialized_dataset
    except Exception as exc:
        _raise_failure(
            stage="materialize_split",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        runtime = _select_runtime(device)
    except Exception as exc:
        _raise_failure(
            stage="input",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_identity.model_id,
            revision=model_identity.revision,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=False,
            dtype=runtime.dtype,
        )
        model = model.to(runtime.device)  # type: ignore[arg-type]
        model.eval()
        if getattr(model, "training", None) is not False:
            raise RuntimeError("model.eval() did not place the model in evaluation mode")
    except Exception as exc:
        _raise_failure(
            stage="load_model",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        runtime_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_identity.tokenizer_id,
            revision=tokenizer_identity.revision,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=False,
        )
    except Exception as exc:
        _raise_failure(
            stage="load_tokenizer",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        runtime_eos_token_id = getattr(runtime_tokenizer, "eos_token_id", None)
        runtime_pad_token_id = getattr(runtime_tokenizer, "pad_token_id", None)
        if runtime_eos_token_id != generation_config.eos_token_id:
            raise ValueError("runtime tokenizer eos_token_id differs from generation config")
        if runtime_pad_token_id != generation_config.pad_token_id:
            raise ValueError("runtime tokenizer pad_token_id differs from generation config")
        if observed_tokenizer.evidence.eos_token_id != generation_config.eos_token_id:
            raise ValueError("qualified tokenizer eos_token_id differs from generation config")
        if observed_tokenizer.evidence.pad_token_id != generation_config.pad_token_id:
            raise ValueError("qualified tokenizer pad_token_id differs from generation config")
        transformers_generation_config = generation_config.to_transformers_config()
    except Exception as exc:
        _raise_failure(
            stage="validate_identity_alignment",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)

    sample_evidence: list[BaselineSampleEvidence] = []
    input_token_count = 0
    generated_token_count = 0

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    started = time.perf_counter()

    for sample_position, source_index in enumerate(split_manifest.source_indices):
        try:
            row = evaluation_dataset[sample_position]
            if not isinstance(row, dict):
                raise TypeError("evaluation dataset row must be a mapping")
            prompt = row.get(evaluator_config.prompt_column)
            reference = row.get(evaluator_config.reference_column)
            if not isinstance(prompt, str):
                raise TypeError("prompt value must be a string")
            if not isinstance(reference, str):
                raise TypeError("reference value must be a string")
            raw_input_ids = runtime_tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = _validate_token_ids(raw_input_ids, label="input_token_ids")
            if (
                tokenizer_identity.model_max_length is not None
                and len(input_ids) + generation_config.max_new_tokens
                > tokenizer_identity.model_max_length
            ):
                raise ValueError("prompt plus max_new_tokens exceeds tokenizer model_max_length")
            input_tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=runtime.device,
            )
            attention_mask = torch.ones_like(input_tensor)
        except Exception as exc:
            _raise_failure(
                stage="prepare_sample",
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                evaluation_profile=evaluation_profile,
                output_dir=evidence_root,
                message=f"sample_position={sample_position}: {exc}",
                cause=exc,
            )

        try:
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    generation_config=transformers_generation_config,
                    use_cache=True,
                )
            if not torch.is_tensor(generated):
                raise TypeError("model.generate() must return a tensor")
            generated_tensor = generated
            if generated_tensor.ndim != 2 or generated_tensor.shape[0] != 1:
                raise TypeError("model.generate() must return one rank-2 tensor")
            continuation = generated_tensor[0, len(input_ids) :].detach().to("cpu").tolist()
            generated_ids = _validate_token_ids(
                continuation,
                label="generated_token_ids",
            )
        except Exception as exc:
            _raise_failure(
                stage="generate",
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                evaluation_profile=evaluation_profile,
                output_dir=evidence_root,
                message=f"sample_position={sample_position}: {exc}",
                cause=exc,
            )

        try:
            generated_text = runtime_tokenizer.decode(
                list(generated_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(generated_text, str):
                raise TypeError("tokenizer.decode() must return a string")
            score = _score_exact_match(generated_text, reference)
        except Exception as exc:
            _raise_failure(
                stage="score",
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                evaluation_profile=evaluation_profile,
                output_dir=evidence_root,
                message=f"sample_position={sample_position}: {exc}",
                cause=exc,
            )

        sample_evidence.append(
            BaselineSampleEvidence(
                sample_position=sample_position,
                source_index=source_index,
                prompt=prompt,
                reference=reference,
                input_token_ids=input_ids,
                generated_token_ids=generated_ids,
                generated_text=generated_text,
                score=score,
            )
        )
        input_token_count += len(input_ids)
        generated_token_count += len(generated_ids)

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    wall_time_seconds = time.perf_counter() - started

    if wall_time_seconds <= 0:
        _raise_failure(
            stage="construct_evidence",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message="measured inference wall time is not positive",
        )

    samples = tuple(sample_evidence)
    try:
        raw_artifact = _persist_samples(output_dir=evidence_root, samples=samples)
    except Exception as exc:
        _raise_failure(
            stage="persist_raw_evidence",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        mean_score = sum(sample.score for sample in samples) / len(samples)
        target_metrics = (
            MetricMeasurement(
                name=evaluator_config.metric_name,
                value=mean_score,
                source="observed",
            ),
        )
        peak_vram_bytes = (
            torch.cuda.max_memory_allocated(runtime.device)
            if runtime.device.type == "cuda"
            else None
        )
        tokens_per_second = (
            generated_token_count / wall_time_seconds if generated_token_count > 0 else None
        )
        systems = SystemsMeasurement(
            wall_time_seconds=wall_time_seconds,
            samples_per_second=len(samples) / wall_time_seconds,
            tokens_per_second=tokens_per_second,
            peak_vram_bytes=peak_vram_bytes,
            artifact_size_bytes=None,
            convergence_step=None,
            stable=True,
        )
        evidence = BaselineEvaluationEvidence(
            model_fingerprint=model_identity.fingerprint(),
            tokenizer_fingerprint=tokenizer_identity.fingerprint(),
            dataset_fingerprint=dataset_identity.fingerprint(),
            split_fingerprint=split_identity.fingerprint(),
            split_manifest_fingerprint=split_manifest.fingerprint(),
            evaluation_profile_fingerprint=evaluation_profile.fingerprint(),
            evaluator_config_sha256=evaluator_config.fingerprint(),
            generation_config_sha256=generation_config.fingerprint(),
            model_identity_evidence_fingerprint=observed_model.evidence.fingerprint(),
            tokenizer_identity_evidence_fingerprint=observed_tokenizer.evidence.fingerprint(),
            split_materialization=materialized.evidence,
            target_metrics=target_metrics,
            systems=systems,
            artifact_valid=True,
            raw_evidence_artifacts=(raw_artifact,),
            scorer_id=evaluator_config.scorer_id,
            sample_count=len(samples),
            input_token_count=input_token_count,
            generated_token_count=generated_token_count,
            runtime_dtype=runtime.dtype_name,
            runtime_device_type=runtime.device.type,
            runtime_device_index=runtime.device_index,
            timing_scope=_TIMING_SCOPE,
            token_throughput_scope=_TOKEN_THROUGHPUT_SCOPE,
            library_versions=_library_versions(),
        )
    except Exception as exc:
        _raise_failure(
            stage="construct_evidence",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    try:
        evidence_path = evidence_root / _AGGREGATE_EVIDENCE_NAME
        evidence_path.write_text(evidence.canonical_json(), encoding="utf-8")
        evidence_artifact = _artifact_digest(
            evidence_path,
            name=_AGGREGATE_EVIDENCE_NAME,
        )
        if evidence_artifact.sha256 != evidence.fingerprint():
            raise RuntimeError("aggregate evidence artifact fingerprint mismatch")
    except Exception as exc:
        _raise_failure(
            stage="construct_evidence",
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            evaluation_profile=evaluation_profile,
            output_dir=evidence_root,
            message=str(exc),
            cause=exc,
        )

    return BaselineEvaluationInspection(
        evidence=evidence,
        samples=samples,
        evidence_artifact=evidence_artifact,
    )
