from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn, cast

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig

from huyawo_adapt.contracts import ArtifactDigest, ModelIdentity

_FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_PATH = "config.json"
_WEIGHT_PATH = "model.safetensors"
_WEIGHT_INDEX_PATH = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class HuggingFaceRepositoryFileEvidence:
    """Provider metadata for one repository file."""

    path: str
    size: int | None
    blob_id: str | None
    lfs_sha256: str | None

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "blob_id": self.blob_id,
            "lfs_sha256": self.lfs_sha256,
            "path": self.path,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class HuggingFaceVerifiedArtifactEvidence:
    """Provider and local identity for one downloaded artifact."""

    path: str
    provider_size: int | None
    provider_blob_id: str | None
    provider_sha256: str | None
    local_size: int
    local_sha256: str

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "local_sha256": self.local_sha256,
            "local_size": self.local_size,
            "path": self.path,
            "provider_blob_id": self.provider_blob_id,
            "provider_sha256": self.provider_sha256,
            "provider_size": self.provider_size,
        }


@dataclass(frozen=True, slots=True)
class HuggingFaceModelIdentityEvidence:
    """Operational evidence for a completed Hugging Face identity inspection."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    repository_files: tuple[HuggingFaceRepositoryFileEvidence, ...]
    repository_inventory_sha256: str
    config_artifact: HuggingFaceVerifiedArtifactEvidence
    weight_artifact: HuggingFaceVerifiedArtifactEvidence
    model_type: str
    architecture: str
    tensor_count: int
    parameter_count: int
    trust_remote_code: bool
    library_versions: tuple[tuple[str, str], ...]

    def canonical_data(self) -> dict[str, object]:
        """Return deterministic JSON-compatible evidence."""
        return {
            "architecture": self.architecture,
            "config_artifact": self.config_artifact.canonical_data(),
            "library_versions": [
                {
                    "name": name,
                    "version": library_version,
                }
                for name, library_version in self.library_versions
            ],
            "model_type": self.model_type,
            "parameter_count": self.parameter_count,
            "repo_id": self.repo_id,
            "repository_files": [item.canonical_data() for item in self.repository_files],
            "repository_inventory_sha256": (self.repository_inventory_sha256),
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "tensor_count": self.tensor_count,
            "trust_remote_code": self.trust_remote_code,
            "weight_artifact": self.weight_artifact.canonical_data(),
        }

    def canonical_json(self) -> str:
        """Return deterministic canonical evidence JSON."""
        return _canonical_json(self.canonical_data())

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the evidence record."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HuggingFaceModelIdentityInspection:
    """Completed operational model identity inspection."""

    identity: ModelIdentity
    evidence: HuggingFaceModelIdentityEvidence


@dataclass(frozen=True, slots=True)
class HuggingFaceModelIdentityFailure:
    """Structured provenance for a failed identity inspection."""

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


class HuggingFaceModelIdentityError(RuntimeError):
    """Fail-closed operational identity error with structured evidence."""

    def __init__(
        self,
        evidence: HuggingFaceModelIdentityFailure,
    ) -> None:
        self.evidence = evidence
        super().__init__(f"{evidence.stage}: {evidence.message}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
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
    evidence = HuggingFaceModelIdentityFailure(
        stage=stage,
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        error_type=(type(cause).__name__ if cause is not None else "IdentityValidationError"),
        message=message,
    )

    error = HuggingFaceModelIdentityError(evidence)

    if cause is not None:
        raise error from cause

    raise error


def _repository_file_evidence(
    sibling: Any,
) -> HuggingFaceRepositoryFileEvidence:
    lfs_sha256: str | None = None

    lfs = getattr(sibling, "lfs", None)

    if lfs is not None:
        value = getattr(lfs, "sha256", None)

        if value is not None:
            lfs_sha256 = str(value)

    path = str(sibling.rfilename)
    size = sibling.size
    blob_id = sibling.blob_id

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

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _inspect_safetensors(
    path: Path,
) -> tuple[int, int]:
    tensor_count = 0
    parameter_count = 0

    with safe_open(
        str(path),
        framework="numpy",
    ) as raw_handle:
        handle = cast(Any, raw_handle)
        tensor_names = sorted(handle.keys())

        for tensor_name in tensor_names:
            shape = handle.get_slice(tensor_name).get_shape()

            if not isinstance(shape, list):
                raise ValueError("Safetensors tensor shape is not a list")

            if not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape):
                raise ValueError("Safetensors tensor shape contains an invalid dimension")

            tensor_count += 1
            parameter_count += math.prod(shape)

    if tensor_count <= 0:
        raise ValueError("Safetensors artifact exposes no tensors")

    if parameter_count <= 0:
        raise ValueError("Safetensors parameter count is not positive")

    return tensor_count, parameter_count


def inspect_huggingface_model(
    repo_id: str,
    revision: str = "main",
    *,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
) -> HuggingFaceModelIdentityInspection:
    """Build a verified ModelIdentity from a Hugging Face repository."""
    if not repo_id or repo_id != repo_id.strip():
        _raise_failure(
            stage="input",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=None,
            message="repo_id must be non-empty and trimmed",
        )

    if not revision or revision != revision.strip():
        _raise_failure(
            stage="input",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=None,
            message="revision must be non-empty and trimmed",
        )

    api = HfApi()

    try:
        initial_info = api.model_info(
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
    except Exception as exc:
        _raise_failure(
            stage="resolve_revision",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=None,
            message=str(exc),
            cause=exc,
        )

    resolved_revision = initial_info.sha

    if (
        not isinstance(resolved_revision, str)
        or _FULL_COMMIT_PATTERN.fullmatch(resolved_revision) is None
    ):
        _raise_failure(
            stage="resolve_revision",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=(
                resolved_revision
                if isinstance(
                    resolved_revision,
                    str,
                )
                else None
            ),
            message=("Hub did not resolve the requested revision to a full hexadecimal commit SHA"),
        )

    try:
        pinned_info = api.model_info(
            repo_id=repo_id,
            revision=resolved_revision,
            files_metadata=True,
            token=token,
        )
    except Exception as exc:
        _raise_failure(
            stage="inventory_repository",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    if pinned_info.sha != resolved_revision:
        _raise_failure(
            stage="inventory_repository",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Pinned repository metadata changed the resolved commit SHA"),
        )

    siblings = tuple(pinned_info.siblings or ())

    if not siblings:
        _raise_failure(
            stage="inventory_repository",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Resolved repository returned no file inventory"),
        )

    repository_files = tuple(
        sorted(
            (_repository_file_evidence(sibling) for sibling in siblings),
            key=lambda item: item.path,
        )
    )

    paths = [item.path for item in repository_files]

    if len(paths) != len(set(paths)):
        _raise_failure(
            stage="inventory_repository",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Repository inventory contains duplicate paths"),
        )

    if _WEIGHT_INDEX_PATH in paths:
        _raise_failure(
            stage="select_weight_artifact",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Sharded Safetensors checkpoints are not supported by this implementation"),
        )

    config_matches = [item for item in repository_files if item.path == _CONFIG_PATH]

    if len(config_matches) != 1:
        _raise_failure(
            stage="select_config_artifact",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Expected exactly one config.json artifact"),
        )

    weight_matches = [item for item in repository_files if item.path == _WEIGHT_PATH]

    if len(weight_matches) != 1:
        _raise_failure(
            stage="select_weight_artifact",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Expected exactly one unsharded model.safetensors artifact"),
        )

    config_provider = config_matches[0]
    weight_provider = weight_matches[0]

    if (
        weight_provider.lfs_sha256 is not None
        and _SHA256_PATTERN.fullmatch(weight_provider.lfs_sha256) is None
    ):
        _raise_failure(
            stage="validate_provider_metadata",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Provider weight SHA-256 metadata is malformed"),
        )

    try:
        config_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=_CONFIG_PATH,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
    except Exception as exc:
        _raise_failure(
            stage="download_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    try:
        weight_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=_WEIGHT_PATH,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
            )
        )
    except Exception as exc:
        _raise_failure(
            stage="download_weight",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    if not config_path.is_file():
        _raise_failure(
            stage="verify_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Downloaded config path is not a file"),
        )

    if not weight_path.is_file():
        _raise_failure(
            stage="verify_weight",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Downloaded weight path is not a file"),
        )

    config_size = config_path.stat().st_size
    weight_size = weight_path.stat().st_size

    try:
        config_sha256 = _sha256_file(config_path)
        weight_sha256 = _sha256_file(weight_path)
    except OSError as exc:
        _raise_failure(
            stage="hash_local_artifacts",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    if config_provider.size is not None and config_provider.size != config_size:
        _raise_failure(
            stage="verify_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Local config size does not match provider metadata"),
        )

    if weight_provider.size is not None and weight_provider.size != weight_size:
        _raise_failure(
            stage="verify_weight",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Local weight size does not match provider metadata"),
        )

    if weight_provider.lfs_sha256 is not None and weight_provider.lfs_sha256 != weight_sha256:
        _raise_failure(
            stage="verify_weight",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Local weight SHA-256 does not match provider LFS SHA-256"),
        )

    try:
        config = cast(
            Any,
            AutoConfig.from_pretrained(
                config_path.parent,
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
    except Exception as exc:
        _raise_failure(
            stage="parse_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    architectures = getattr(
        config,
        "architectures",
        None,
    )

    if (
        not isinstance(
            architectures,
            (list, tuple),
        )
        or not architectures
        or not isinstance(
            architectures[0],
            str,
        )
        or not architectures[0]
    ):
        _raise_failure(
            stage="parse_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Config does not expose a usable architecture"),
        )

    model_type_value = getattr(
        config,
        "model_type",
        None,
    )

    if (
        not isinstance(
            model_type_value,
            str,
        )
        or not model_type_value
    ):
        _raise_failure(
            stage="parse_config",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=("Config does not expose a usable model_type"),
        )

    architecture = architectures[0]
    model_type = model_type_value

    try:
        (
            tensor_count,
            parameter_count,
        ) = _inspect_safetensors(weight_path)
    except Exception as exc:
        _raise_failure(
            stage="inspect_safetensors",
            repo_id=repo_id,
            requested_revision=revision,
            resolved_revision=resolved_revision,
            message=str(exc),
            cause=exc,
        )

    identity = ModelIdentity(
        source="huggingface",
        model_id=repo_id,
        revision=resolved_revision,
        config_sha256=config_sha256,
        weight_artifacts=(
            ArtifactDigest(
                name=_WEIGHT_PATH,
                sha256=weight_sha256,
                size_bytes=weight_size,
            ),
        ),
        architecture=architecture,
        parameter_count=parameter_count,
        trust_remote_code=False,
    )

    config_evidence = HuggingFaceVerifiedArtifactEvidence(
        path=_CONFIG_PATH,
        provider_size=config_provider.size,
        provider_blob_id=(config_provider.blob_id),
        provider_sha256=(config_provider.lfs_sha256),
        local_size=config_size,
        local_sha256=config_sha256,
    )

    weight_evidence = HuggingFaceVerifiedArtifactEvidence(
        path=_WEIGHT_PATH,
        provider_size=weight_provider.size,
        provider_blob_id=(weight_provider.blob_id),
        provider_sha256=(weight_provider.lfs_sha256),
        local_size=weight_size,
        local_sha256=weight_sha256,
    )

    library_versions = tuple(
        sorted(
            (
                (
                    "huggingface-hub",
                    version("huggingface-hub"),
                ),
                (
                    "safetensors",
                    version("safetensors"),
                ),
                (
                    "transformers",
                    version("transformers"),
                ),
            ),
            key=lambda item: item[0],
        )
    )

    evidence = HuggingFaceModelIdentityEvidence(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=(resolved_revision),
        repository_files=(repository_files),
        repository_inventory_sha256=(_inventory_sha256(repository_files)),
        config_artifact=(config_evidence),
        weight_artifact=(weight_evidence),
        model_type=model_type,
        architecture=architecture,
        tensor_count=tensor_count,
        parameter_count=(parameter_count),
        trust_remote_code=False,
        library_versions=(library_versions),
    )

    return HuggingFaceModelIdentityInspection(
        identity=identity,
        evidence=evidence,
    )
