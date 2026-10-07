from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import transformers
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from huyawo_adapt.contracts import ArtifactDigest, TokenizerIdentity
from huyawo_adapt.identity.huggingface import (
    HuggingFaceRepositoryFileEvidence,
    HuggingFaceVerifiedArtifactEvidence,
)

_FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_CONFIG_PATH = "tokenizer_config.json"
_OPTIONAL_TOKENIZER_ARTIFACTS = (
    "added_tokens.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
)
_BEHAVIOR_PROBE_INPUTS = (
    "The purpose of a deterministic evaluation baseline is",
    "Deterministic evaluation requires explicit provenance.",
)


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizerIdentityEvidence:
    """Operational evidence for a completed Hugging Face tokenizer inspection."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    repository_files: tuple[HuggingFaceRepositoryFileEvidence, ...]
    repository_inventory_sha256: str
    selected_artifacts: tuple[HuggingFaceVerifiedArtifactEvidence, ...]
    selected_artifact_manifest_sha256: str
    declared_tokenizer_class: str
    runtime_tokenizer_class: str
    vocab_files_names: tuple[tuple[str, str], ...]
    tokenizer_length: int
    vocab_size: int
    model_max_length: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    padding_side: str
    truncation_side: str
    special_tokens_map: tuple[tuple[str, str | tuple[str, ...]], ...]
    added_vocabulary: tuple[tuple[str, int], ...]
    chat_template: str | None
    chat_template_sha256: str | None
    behavior_probe_inputs: tuple[str, ...]
    behavior_probe_token_ids: tuple[tuple[int, ...], ...]
    behavior_sha256: str
    semantics_sha256: str
    trust_remote_code: bool
    library_versions: tuple[tuple[str, str], ...]

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "added_vocabulary": {token: token_id for token, token_id in self.added_vocabulary},
            "behavior_probe_inputs": list(self.behavior_probe_inputs),
            "behavior_probe_token_ids": [
                list(token_ids) for token_ids in self.behavior_probe_token_ids
            ],
            "behavior_sha256": self.behavior_sha256,
            "bos_token_id": self.bos_token_id,
            "chat_template": self.chat_template,
            "chat_template_sha256": self.chat_template_sha256,
            "declared_tokenizer_class": self.declared_tokenizer_class,
            "eos_token_id": self.eos_token_id,
            "library_versions": [
                {"name": name, "version": library_version}
                for name, library_version in self.library_versions
            ],
            "model_max_length": self.model_max_length,
            "pad_token_id": self.pad_token_id,
            "padding_side": self.padding_side,
            "repo_id": self.repo_id,
            "repository_files": [item.canonical_data() for item in self.repository_files],
            "repository_inventory_sha256": self.repository_inventory_sha256,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "runtime_tokenizer_class": self.runtime_tokenizer_class,
            "selected_artifact_manifest_sha256": (self.selected_artifact_manifest_sha256),
            "selected_artifacts": [item.canonical_data() for item in self.selected_artifacts],
            "semantics_sha256": self.semantics_sha256,
            "special_tokens_map": _special_tokens_map_to_dict(self.special_tokens_map),
            "tokenizer_length": self.tokenizer_length,
            "truncation_side": self.truncation_side,
            "trust_remote_code": self.trust_remote_code,
            "vocab_files_names": [
                {"key": key, "path": path} for key, path in self.vocab_files_names
            ],
            "vocab_size": self.vocab_size,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical evidence JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the evidence record."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizerIdentityInspection:
    """Completed operational tokenizer identity inspection."""

    identity: TokenizerIdentity
    evidence: HuggingFaceTokenizerIdentityEvidence


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizerIdentityFailure:
    """Structured provenance for a failed tokenizer identity inspection."""

    stage: str
    repo_id: str
    requested_revision: str
    resolved_revision: str | None
    error_type: str
    message: str

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible failure evidence."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "stage": self.stage,
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical failure JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the failure record."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class HuggingFaceTokenizerIdentityError(RuntimeError):
    """Fail-closed operational tokenizer identity error with structured evidence."""

    def __init__(self, evidence: HuggingFaceTokenizerIdentityFailure) -> None:
        self.evidence = evidence
        super().__init__(f"{evidence.stage}: {evidence.message}")


@dataclass(frozen=True, slots=True)
class _TokenizerSemantics:
    tokenizer_class: str
    vocab_size: int
    tokenizer_length: int
    model_max_length: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    padding_side: str
    truncation_side: str
    special_tokens_map: tuple[tuple[str, str | tuple[str, ...]], ...]
    added_vocabulary: tuple[tuple[str, int], ...]
    chat_template: str | None
    probe_token_ids: tuple[tuple[int, ...], ...]

    def canonical_data(self) -> dict[str, object]:
        return {
            "added_vocabulary": {token: token_id for token, token_id in self.added_vocabulary},
            "bos_token_id": self.bos_token_id,
            "chat_template": self.chat_template,
            "class": self.tokenizer_class,
            "eos_token_id": self.eos_token_id,
            "length": self.tokenizer_length,
            "model_max_length": self.model_max_length,
            "pad_token_id": self.pad_token_id,
            "padding_side": self.padding_side,
            "probe_token_ids": [list(item) for item in self.probe_token_ids],
            "special_tokens_map": _special_tokens_map_to_dict(self.special_tokens_map),
            "truncation_side": self.truncation_side,
            "vocab_size": self.vocab_size,
        }


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


def _raise_failure(
    *,
    stage: str,
    repo_id: str,
    requested_revision: str,
    resolved_revision: str | None,
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    evidence = HuggingFaceTokenizerIdentityFailure(
        stage=stage,
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        error_type=(
            type(cause).__name__ if cause is not None else "TokenizerIdentityValidationError"
        ),
        message=message,
    )

    error = HuggingFaceTokenizerIdentityError(evidence)

    if cause is not None:
        raise error from cause

    raise error


def _validate_input(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")

    return value


def _validate_repository_path(path: str) -> None:
    pure_path = PurePosixPath(path)

    if not path or path.startswith("/") or "\\" in path:
        raise ValueError(f"Invalid repository path: {path!r}")

    if any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError(f"Invalid repository path: {path!r}")


def _repository_file_evidence(sibling: Any) -> HuggingFaceRepositoryFileEvidence:
    path = str(sibling.rfilename)
    _validate_repository_path(path)

    size_raw = getattr(sibling, "size", None)
    blob_raw = getattr(sibling, "blob_id", None)
    lfs_raw = getattr(sibling, "lfs", None)

    size: int | None
    if size_raw is None:
        size = None
    elif isinstance(size_raw, int) and not isinstance(size_raw, bool) and size_raw >= 0:
        size = size_raw
    else:
        raise ValueError(f"Invalid provider size for {path}")

    blob_id = None if blob_raw is None else str(blob_raw)
    if blob_id == "":
        raise ValueError(f"Invalid provider blob ID for {path}")

    lfs_sha256: str | None = None
    if lfs_raw is not None:
        sha_raw = getattr(lfs_raw, "sha256", None)
        if sha_raw is not None:
            lfs_sha256 = str(sha_raw)
            if _SHA256_PATTERN.fullmatch(lfs_sha256) is None:
                raise ValueError(f"Invalid provider LFS SHA-256 for {path}")

    return HuggingFaceRepositoryFileEvidence(
        path=path,
        size=size,
        blob_id=blob_id,
        lfs_sha256=lfs_sha256,
    )


def _inventory_sha256(
    files: tuple[HuggingFaceRepositoryFileEvidence, ...],
) -> str:
    payload = [item.canonical_data() for item in files]
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _download_artifact(
    *,
    repo_id: str,
    path: str,
    resolved_revision: str,
    cache_dir: str | Path | None,
    token: bool | str | None,
) -> Path:
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=path,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
        )
    )

    if not downloaded.is_file():
        raise ValueError(f"Downloaded artifact is not a regular file: {path}")

    return downloaded


def _verify_artifact(
    *,
    repository_file: HuggingFaceRepositoryFileEvidence,
    local_path: Path,
) -> HuggingFaceVerifiedArtifactEvidence:
    local_size = local_path.stat().st_size
    local_sha256 = _sha256_file(local_path)

    if repository_file.size is not None and repository_file.size != local_size:
        raise ValueError(
            f"Provider/local size mismatch for {repository_file.path}: "
            f"provider={repository_file.size}, local={local_size}"
        )

    if repository_file.lfs_sha256 is not None and repository_file.lfs_sha256 != local_sha256:
        raise ValueError(f"Provider/local SHA-256 mismatch for {repository_file.path}")

    return HuggingFaceVerifiedArtifactEvidence(
        path=repository_file.path,
        provider_size=repository_file.size,
        provider_blob_id=repository_file.blob_id,
        provider_sha256=repository_file.lfs_sha256,
        local_size=local_size,
        local_sha256=local_sha256,
    )


def _parse_tokenizer_config(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("tokenizer_config.json must contain a JSON object")
    return cast(dict[str, Any], parsed)


def _resolve_tokenizer_class(
    declared_tokenizer_class: str,
) -> tuple[type[Any], tuple[tuple[str, str], ...]]:
    runtime_class = getattr(transformers, declared_tokenizer_class, None)
    if not isinstance(runtime_class, type):
        raise ValueError(f"Unsupported tokenizer class: {declared_tokenizer_class}")

    raw_mapping = getattr(runtime_class, "vocab_files_names", None)
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError(
            f"Tokenizer class {declared_tokenizer_class} exposes no usable "
            "vocab_files_names mapping"
        )

    normalized: list[tuple[str, str]] = []
    for raw_key, raw_path in raw_mapping.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("Tokenizer vocab_files_names contains an invalid key")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Tokenizer vocab_files_names contains an invalid repository path")
        _validate_repository_path(raw_path)
        normalized.append((raw_key, raw_path))

    return runtime_class, tuple(sorted(normalized))


def _select_artifact_paths(
    *,
    repository_paths: set[str],
    vocab_files_names: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    selected = {_TOKENIZER_CONFIG_PATH}

    for _, path in vocab_files_names:
        if path not in repository_paths:
            raise ValueError(f"Required tokenizer vocabulary artifact is missing: {path}")
        selected.add(path)

    for path in _OPTIONAL_TOKENIZER_ARTIFACTS:
        if path in repository_paths:
            selected.add(path)

    for path in repository_paths:
        if path.startswith("additional_chat_templates/") and path.endswith(".jinja"):
            selected.add(path)

    return tuple(sorted(selected))


def _selected_artifact_manifest_sha256(
    artifacts: tuple[HuggingFaceVerifiedArtifactEvidence, ...],
) -> str:
    payload = [
        {
            "name": item.path,
            "sha256": item.local_sha256,
            "size_bytes": item.local_size,
        }
        for item in artifacts
    ]
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _normalize_special_tokens_map(
    raw: object,
) -> tuple[tuple[str, str | tuple[str, ...]], ...]:
    if not isinstance(raw, dict):
        raise ValueError("special_tokens_map must be a dictionary")

    normalized: list[tuple[str, str | tuple[str, ...]]] = []
    for raw_key, raw_value in raw.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("special_tokens_map contains an invalid key")

        if isinstance(raw_value, str):
            value: str | tuple[str, ...] = raw_value
        elif isinstance(raw_value, (list, tuple)) and all(
            isinstance(item, str) for item in raw_value
        ):
            value = tuple(cast(list[str] | tuple[str, ...], raw_value))
        else:
            raise ValueError(f"special_tokens_map contains an unsupported value for {raw_key}")

        normalized.append((raw_key, value))

    return tuple(sorted(normalized, key=lambda item: item[0]))


def _special_tokens_map_to_dict(
    value: tuple[tuple[str, str | tuple[str, ...]], ...],
) -> dict[str, object]:
    return {key: list(item) if isinstance(item, tuple) else item for key, item in value}


def _normalize_added_vocabulary(raw: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, dict):
        raise ValueError("get_added_vocab() must return a dictionary")

    normalized: list[tuple[str, int]] = []
    for raw_token, raw_id in raw.items():
        if not isinstance(raw_token, str):
            raise ValueError("Added vocabulary contains a non-string token")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id < 0:
            raise ValueError("Added vocabulary contains an invalid token ID")
        normalized.append((raw_token, raw_id))

    return tuple(sorted(normalized, key=lambda item: item[0]))


def _normalize_optional_token_id(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or None")
    return value


def _extract_chat_template(tokenizer: Any) -> str | None:
    raw = getattr(tokenizer, "chat_template", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        raise ValueError("Multiple named chat templates are unsupported")
    raise ValueError("Tokenizer chat_template has an unsupported runtime type")


def _probe_token_ids(tokenizer: Any) -> tuple[tuple[int, ...], ...]:
    probe_results: list[tuple[int, ...]] = []
    for text in _BEHAVIOR_PROBE_INPUTS:
        raw_ids = tokenizer.encode(text, add_special_tokens=True)
        if not isinstance(raw_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in raw_ids
        ):
            raise ValueError("Tokenizer behavior probe returned invalid token IDs")
        probe_results.append(tuple(raw_ids))
    return tuple(probe_results)


def _snapshot_tokenizer(tokenizer: Any) -> _TokenizerSemantics:
    tokenizer_class = type(tokenizer).__name__

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("Tokenizer vocab_size must be a positive integer")

    tokenizer_length = len(tokenizer)
    if tokenizer_length <= 0:
        raise ValueError("Tokenizer length must be positive")

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if (
        not isinstance(model_max_length, int)
        or isinstance(model_max_length, bool)
        or model_max_length <= 0
    ):
        raise ValueError("Tokenizer model_max_length must be a positive integer")

    padding_side = getattr(tokenizer, "padding_side", None)
    truncation_side = getattr(tokenizer, "truncation_side", None)
    if not isinstance(padding_side, str) or not padding_side:
        raise ValueError("Tokenizer padding_side must be a non-empty string")
    if not isinstance(truncation_side, str) or not truncation_side:
        raise ValueError("Tokenizer truncation_side must be a non-empty string")

    special_tokens_map = _normalize_special_tokens_map(
        getattr(tokenizer, "special_tokens_map", None)
    )
    added_vocabulary = _normalize_added_vocabulary(tokenizer.get_added_vocab())
    chat_template = _extract_chat_template(tokenizer)
    probe_token_ids = _probe_token_ids(tokenizer)

    return _TokenizerSemantics(
        tokenizer_class=tokenizer_class,
        vocab_size=vocab_size,
        tokenizer_length=tokenizer_length,
        model_max_length=model_max_length,
        bos_token_id=_normalize_optional_token_id(
            getattr(tokenizer, "bos_token_id", None),
            label="bos_token_id",
        ),
        eos_token_id=_normalize_optional_token_id(
            getattr(tokenizer, "eos_token_id", None),
            label="eos_token_id",
        ),
        pad_token_id=_normalize_optional_token_id(
            getattr(tokenizer, "pad_token_id", None),
            label="pad_token_id",
        ),
        padding_side=padding_side,
        truncation_side=truncation_side,
        special_tokens_map=special_tokens_map,
        added_vocabulary=added_vocabulary,
        chat_template=chat_template,
        probe_token_ids=probe_token_ids,
    )


def _behavior_sha256(semantics: _TokenizerSemantics) -> str:
    payload = {
        "bos_token_id": semantics.bos_token_id,
        "class": semantics.tokenizer_class,
        "eos_token_id": semantics.eos_token_id,
        "length": semantics.tokenizer_length,
        "pad_token_id": semantics.pad_token_id,
        "padding_side": semantics.padding_side,
        "probe_token_ids": [list(item) for item in semantics.probe_token_ids],
        "special_tokens_map": _special_tokens_map_to_dict(semantics.special_tokens_map),
        "truncation_side": semantics.truncation_side,
        "vocab_size": semantics.vocab_size,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _semantics_sha256(semantics: _TokenizerSemantics) -> str:
    return _sha256_bytes(_canonical_json(semantics.canonical_data()).encode("utf-8"))


def _materialize_selected_artifacts(
    *,
    artifacts: tuple[HuggingFaceVerifiedArtifactEvidence, ...],
    local_paths: dict[str, Path],
    destination: Path,
) -> None:
    for artifact in artifacts:
        source = local_paths[artifact.path]
        target = destination / PurePosixPath(artifact.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _library_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("huggingface-hub", version("huggingface-hub")),
                ("transformers", version("transformers")),
            )
        )
    )


def inspect_huggingface_tokenizer(
    repo_id: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
) -> HuggingFaceTokenizerIdentityInspection:
    """Resolve and verify one Hugging Face tokenizer identity fail closed."""
    resolved_revision: str | None = None

    try:
        repo_id = _validate_input(repo_id, label="repo_id")
        revision = _validate_input(revision, label="revision")
    except Exception as exc:
        _raise_failure(
            stage="input",
            repo_id=str(repo_id),
            requested_revision=str(revision),
            resolved_revision=None,
            message=str(exc),
            cause=exc,
        )

    api = HfApi()

    try:
        resolution = api.model_info(
            repo_id,
            revision=revision,
            files_metadata=False,
            token=token,
        )
        raw_revision = getattr(resolution, "sha", None)
        if (
            not isinstance(raw_revision, str)
            or _FULL_COMMIT_PATTERN.fullmatch(raw_revision) is None
        ):
            raise ValueError("Provider did not resolve a full 40-character commit SHA")
        resolved_revision = raw_revision
    except Exception as exc:
        _raise_failure(
            stage="resolve_revision",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        pinned = api.model_info(
            repo_id,
            revision=resolved_revision,
            files_metadata=True,
            token=token,
        )
        pinned_sha = getattr(pinned, "sha", None)
        if pinned_sha != resolved_revision:
            raise ValueError("Pinned repository metadata resolved to a different commit")

        siblings = getattr(pinned, "siblings", None)
        if not isinstance(siblings, list):
            raise ValueError("Pinned repository metadata exposes no file inventory")
    except Exception as exc:
        _raise_failure(
            stage="inventory_repository",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        repository_files = tuple(
            sorted(
                (_repository_file_evidence(item) for item in siblings),
                key=lambda item: item.path,
            )
        )
        repository_paths = [item.path for item in repository_files]
        if len(repository_paths) != len(set(repository_paths)):
            raise ValueError("Pinned repository metadata contains duplicate paths")
        repository_inventory_sha256 = _inventory_sha256(repository_files)
        repository_by_path = {item.path: item for item in repository_files}
    except Exception as exc:
        _raise_failure(
            stage="validate_provider_metadata",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    if list(repository_by_path).count(_TOKENIZER_CONFIG_PATH) != 1:
        _raise_failure(
            stage="select_tokenizer_artifacts",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message="Repository must contain exactly one tokenizer_config.json",
        )

    try:
        tokenizer_config_path = _download_artifact(
            repo_id=repo_id,
            path=_TOKENIZER_CONFIG_PATH,
            resolved_revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as exc:
        _raise_failure(
            stage="download_tokenizer_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        verified_config = _verify_artifact(
            repository_file=repository_by_path[_TOKENIZER_CONFIG_PATH],
            local_path=tokenizer_config_path,
        )
        tokenizer_config = _parse_tokenizer_config(tokenizer_config_path)
        declared_tokenizer_class = tokenizer_config.get("tokenizer_class")
        if (
            not isinstance(declared_tokenizer_class, str)
            or not declared_tokenizer_class
            or declared_tokenizer_class != declared_tokenizer_class.strip()
        ):
            raise ValueError("tokenizer_config.json must declare a non-empty tokenizer_class")
    except Exception as exc:
        _raise_failure(
            stage="verify_tokenizer_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        _, vocab_files_names = _resolve_tokenizer_class(declared_tokenizer_class)
    except Exception as exc:
        _raise_failure(
            stage="resolve_tokenizer_class",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        selected_paths = _select_artifact_paths(
            repository_paths=set(repository_by_path),
            vocab_files_names=vocab_files_names,
        )
    except Exception as exc:
        _raise_failure(
            stage="select_tokenizer_artifacts",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    local_paths: dict[str, Path] = {
        _TOKENIZER_CONFIG_PATH: tokenizer_config_path,
    }
    verified_by_path: dict[str, HuggingFaceVerifiedArtifactEvidence] = {
        _TOKENIZER_CONFIG_PATH: verified_config,
    }

    for selected_path in selected_paths:
        if selected_path == _TOKENIZER_CONFIG_PATH:
            continue

        try:
            local_paths[selected_path] = _download_artifact(
                repo_id=repo_id,
                path=selected_path,
                resolved_revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as exc:
            _raise_failure(
                stage="download_tokenizer_artifact",
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                message=str(exc),
                cause=exc,
            )

        try:
            verified_by_path[selected_path] = _verify_artifact(
                repository_file=repository_by_path[selected_path],
                local_path=local_paths[selected_path],
            )
        except Exception as exc:
            _raise_failure(
                stage="verify_tokenizer_artifact",
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                message=str(exc),
                cause=exc,
            )

    selected_artifacts = tuple(verified_by_path[path] for path in selected_paths)
    selected_artifact_manifest_sha256 = _selected_artifact_manifest_sha256(selected_artifacts)

    try:
        remote_tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=False,
        )
    except Exception as exc:
        _raise_failure(
            stage="load_remote_tokenizer",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        remote_semantics = _snapshot_tokenizer(remote_tokenizer)
        if remote_semantics.tokenizer_class != declared_tokenizer_class:
            raise ValueError(
                "Runtime tokenizer class does not match tokenizer_config.json authority"
            )
    except Exception as exc:
        stage = (
            "validate_chat_template"
            if "chat template" in str(exc).lower()
            else "behavior_probe"
            if "behavior probe" in str(exc).lower()
            else "validate_tokenizer_semantics"
        )
        _raise_failure(
            stage=stage,
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    with tempfile.TemporaryDirectory(
        prefix="huyawo-adapt-tokenizer-rehydration-"
    ) as temporary_directory:
        rehydration_root = Path(temporary_directory)

        try:
            _materialize_selected_artifacts(
                artifacts=selected_artifacts,
                local_paths=local_paths,
                destination=rehydration_root,
            )
        except Exception as exc:
            _raise_failure(
                stage="materialize_selected_artifacts",
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                message=str(exc),
                cause=exc,
            )

        try:
            local_tokenizer = AutoTokenizer.from_pretrained(
                rehydration_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            _raise_failure(
                stage="load_local_tokenizer",
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                message=str(exc),
                cause=exc,
            )

        try:
            local_semantics = _snapshot_tokenizer(local_tokenizer)
        except Exception as exc:
            stage = (
                "validate_chat_template"
                if "chat template" in str(exc).lower()
                else "behavior_probe"
                if "behavior probe" in str(exc).lower()
                else "validate_tokenizer_semantics"
            )
            _raise_failure(
                stage=stage,
                repo_id=repo_id,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                message=str(exc),
                cause=exc,
            )

    if local_semantics != remote_semantics:
        _raise_failure(
            stage="validate_tokenizer_semantics",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message="Pinned remote and selected-artifact local tokenizer semantics differ",
        )

    behavior_sha256 = _behavior_sha256(remote_semantics)
    semantics_sha256 = _semantics_sha256(remote_semantics)
    chat_template_sha256 = (
        None
        if remote_semantics.chat_template is None
        else _sha256_bytes(remote_semantics.chat_template.encode("utf-8"))
    )

    try:
        identity = TokenizerIdentity(
            source="huggingface",
            tokenizer_id=repo_id,
            revision=resolved_revision,
            artifacts=tuple(
                ArtifactDigest(
                    name=item.path,
                    sha256=item.local_sha256,
                    size_bytes=item.local_size,
                )
                for item in selected_artifacts
            ),
            chat_template_sha256=chat_template_sha256,
            vocab_size=remote_semantics.vocab_size,
            model_max_length=remote_semantics.model_max_length,
        )
    except Exception as exc:
        _raise_failure(
            stage="construct_identity",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    evidence = HuggingFaceTokenizerIdentityEvidence(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        repository_files=repository_files,
        repository_inventory_sha256=repository_inventory_sha256,
        selected_artifacts=selected_artifacts,
        selected_artifact_manifest_sha256=selected_artifact_manifest_sha256,
        declared_tokenizer_class=declared_tokenizer_class,
        runtime_tokenizer_class=remote_semantics.tokenizer_class,
        vocab_files_names=vocab_files_names,
        tokenizer_length=remote_semantics.tokenizer_length,
        vocab_size=remote_semantics.vocab_size,
        model_max_length=remote_semantics.model_max_length,
        bos_token_id=remote_semantics.bos_token_id,
        eos_token_id=remote_semantics.eos_token_id,
        pad_token_id=remote_semantics.pad_token_id,
        padding_side=remote_semantics.padding_side,
        truncation_side=remote_semantics.truncation_side,
        special_tokens_map=remote_semantics.special_tokens_map,
        added_vocabulary=remote_semantics.added_vocabulary,
        chat_template=remote_semantics.chat_template,
        chat_template_sha256=chat_template_sha256,
        behavior_probe_inputs=_BEHAVIOR_PROBE_INPUTS,
        behavior_probe_token_ids=remote_semantics.probe_token_ids,
        behavior_sha256=behavior_sha256,
        semantics_sha256=semantics_sha256,
        trust_remote_code=False,
        library_versions=_library_versions(),
    )

    return HuggingFaceTokenizerIdentityInspection(
        identity=identity,
        evidence=evidence,
    )
