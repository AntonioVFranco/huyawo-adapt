from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Self, cast

from huggingface_hub import HfApi, hf_hub_download
from pydantic import Field, field_validator

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from huyawo_adapt.contracts import DatasetIdentity
from huyawo_adapt.contracts.base import (
    NonEmptyString,
    Sha256Digest,
    StrictFrozenModel,
)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIMPLE_SPLIT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ROW_LENGTH_BYTES = 8


class _DatasetFeatures(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class _MaterializedDataset(Protocol):
    features: _DatasetFeatures
    info: object
    _fingerprint: str

    def __len__(self) -> int: ...

    def __getitem__(
        self,
        index: int,
    ) -> Mapping[str, Any]: ...


class _OperationalEvidenceModel(StrictFrozenModel):
    def canonical_data(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            round_trip=True,
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_canonical_json(
        cls,
        payload: str | bytes | bytearray,
    ) -> Self:
        return cls.model_validate_json(payload)


class HuggingFaceDatasetRepositoryFileEvidence(_OperationalEvidenceModel):
    path: NonEmptyString
    size: int | None = Field(
        default=None,
        ge=0,
    )
    blob_id: NonEmptyString | None = None
    lfs_sha256: Sha256Digest | None = None


class HuggingFaceDatasetSourceArtifactEvidence(_OperationalEvidenceModel):
    path: NonEmptyString
    provider_size: int | None = Field(
        default=None,
        ge=0,
    )
    provider_blob_id: NonEmptyString | None = None
    provider_sha256: Sha256Digest | None = None
    local_size: int = Field(ge=0)
    local_sha256: Sha256Digest


class HuggingFaceDatasetIdentityEvidence(_OperationalEvidenceModel):
    repo_id: NonEmptyString
    requested_revision: NonEmptyString
    resolved_revision: NonEmptyString
    config_name: NonEmptyString | None = None
    split: NonEmptyString
    license_expression: NonEmptyString
    provider_license: NonEmptyString | None = None
    repository_files: tuple[
        HuggingFaceDatasetRepositoryFileEvidence,
        ...,
    ]
    repository_inventory_sha256: Sha256Digest
    source_artifacts: tuple[
        HuggingFaceDatasetSourceArtifactEvidence,
        ...,
    ]
    external_dataset_fingerprint: NonEmptyString
    schema_sha256: Sha256Digest
    content_sha256: Sha256Digest
    sample_count: int = Field(ge=0)
    token_count: int | None = Field(
        default=None,
        ge=0,
    )
    row_canonicalization: str = "json-sort-keys-length-prefix-u64be-v1"
    schema_canonicalization: str = "features-to-dict-json-sort-keys-v1"
    library_versions: tuple[
        tuple[str, str],
        ...,
    ]

    @field_validator("repository_files")
    @classmethod
    def normalize_repository_files(
        cls,
        files: tuple[
            HuggingFaceDatasetRepositoryFileEvidence,
            ...,
        ],
    ) -> tuple[
        HuggingFaceDatasetRepositoryFileEvidence,
        ...,
    ]:
        paths = [item.path for item in files]

        if len(paths) != len(set(paths)):
            raise ValueError("repository file paths must be unique")

        return tuple(
            sorted(
                files,
                key=lambda item: item.path,
            )
        )

    @field_validator("source_artifacts")
    @classmethod
    def normalize_source_artifacts(
        cls,
        artifacts: tuple[
            HuggingFaceDatasetSourceArtifactEvidence,
            ...,
        ],
    ) -> tuple[
        HuggingFaceDatasetSourceArtifactEvidence,
        ...,
    ]:
        paths = [item.path for item in artifacts]

        if len(paths) != len(set(paths)):
            raise ValueError("source artifact paths must be unique")

        return tuple(
            sorted(
                artifacts,
                key=lambda item: item.path,
            )
        )

    @field_validator("library_versions")
    @classmethod
    def normalize_library_versions(
        cls,
        versions: tuple[
            tuple[str, str],
            ...,
        ],
    ) -> tuple[
        tuple[str, str],
        ...,
    ]:
        names = [name for name, _ in versions]

        if len(names) != len(set(names)):
            raise ValueError("library version names must be unique")

        return tuple(sorted(versions))


class HuggingFaceDatasetIdentityFailure(_OperationalEvidenceModel):
    stage: NonEmptyString
    repo_id: NonEmptyString
    requested_revision: NonEmptyString
    config_name: NonEmptyString | None = None
    split: NonEmptyString
    resolved_revision: NonEmptyString | None = None
    error_type: NonEmptyString
    message: NonEmptyString


class HuggingFaceDatasetIdentityInspection(_OperationalEvidenceModel):
    identity: DatasetIdentity
    evidence: HuggingFaceDatasetIdentityEvidence


class HuggingFaceDatasetIdentityError(RuntimeError):
    def __init__(
        self,
        failure: HuggingFaceDatasetIdentityFailure,
    ) -> None:
        super().__init__(f"{failure.stage}: {failure.error_type}: {failure.message}")
        self.failure = failure


def _canonical_json_bytes(
    value: Any,
) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(
    path: Path,
) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)

    return hasher.hexdigest(), size


def _require_non_empty(
    value: str,
    label: str,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{label} must be non-empty")

    return normalized


def _require_sha256(
    value: str,
    label: str,
) -> str:
    normalized = value.strip()

    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    return normalized


def _require_commit(
    value: object,
) -> str:
    if not isinstance(value, str):
        raise ValueError("dataset revision did not resolve to a string commit SHA")

    if _COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError("dataset revision did not resolve to a full lowercase commit SHA")

    return value


def _normalize_config_name(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    return _require_non_empty(
        value,
        "config_name",
    )


def _normalize_split(
    value: str,
) -> str:
    normalized = _require_non_empty(
        value,
        "split",
    )

    if _SIMPLE_SPLIT_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "split must be a simple source split "
            "name; slice and concatenation "
            "expressions are not supported"
        )

    return normalized


def _normalize_source_files(
    source_files: tuple[str, ...],
) -> tuple[str, ...]:
    if not source_files:
        raise ValueError("source_files must contain at least one repository path")

    normalized: list[str] = []

    for value in source_files:
        path_text = _require_non_empty(
            value,
            "source file",
        )
        path = PurePosixPath(path_text)

        if path.is_absolute() or ".." in path.parts or path.as_posix() != path_text:
            raise ValueError(
                f"source file must be a normalized repository-relative path: {path_text!r}"
            )

        normalized.append(path_text)

    if len(normalized) != len(set(normalized)):
        raise ValueError("source_files must be unique")

    return tuple(sorted(normalized))


def _safe_failure_text(
    value: object,
) -> str:
    if isinstance(value, str):
        normalized = value.strip()

        if normalized:
            return normalized

    return "<invalid>"


def _safe_failure_optional_text(
    value: object,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()

        if normalized:
            return normalized

    return None


def _extract_lfs_sha256(
    lfs: Any,
) -> str | None:
    if lfs is None:
        return None

    if isinstance(lfs, Mapping):
        value = lfs.get("sha256")
    else:
        value = getattr(
            lfs,
            "sha256",
            None,
        )

    if value is None:
        return None

    return _require_sha256(
        str(value),
        "provider LFS sha256",
    )


def _repository_inventory(
    siblings: Any,
) -> tuple[
    HuggingFaceDatasetRepositoryFileEvidence,
    ...,
]:
    files: list[HuggingFaceDatasetRepositoryFileEvidence] = []

    for sibling in siblings or ():
        path = getattr(
            sibling,
            "rfilename",
            None,
        )

        if not isinstance(path, str) or not path.strip():
            raise ValueError("repository file is missing a valid path")

        size = getattr(
            sibling,
            "size",
            None,
        )

        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ValueError(f"invalid provider size for {path}")

        blob_id = getattr(
            sibling,
            "blob_id",
            None,
        )

        if blob_id is not None and not isinstance(blob_id, str):
            raise ValueError(f"invalid provider blob ID for {path}")

        files.append(
            HuggingFaceDatasetRepositoryFileEvidence(
                path=path,
                size=size,
                blob_id=blob_id,
                lfs_sha256=_extract_lfs_sha256(
                    getattr(
                        sibling,
                        "lfs",
                        None,
                    )
                ),
            )
        )

    if not files:
        raise ValueError("dataset repository returned no file inventory")

    paths = [item.path for item in files]

    if len(paths) != len(set(paths)):
        raise ValueError("dataset repository returned duplicate file paths")

    return tuple(
        sorted(
            files,
            key=lambda item: item.path,
        )
    )


def _repository_inventory_sha256(
    files: tuple[
        HuggingFaceDatasetRepositoryFileEvidence,
        ...,
    ],
) -> str:
    payload = [item.canonical_data() for item in files]

    return _sha256_bytes(_canonical_json_bytes(payload))


def _schema_sha256(
    dataset: _MaterializedDataset,
) -> str:
    return _sha256_bytes(_canonical_json_bytes(dataset.features.to_dict()))


def _content_sha256(
    dataset: _MaterializedDataset,
) -> tuple[str, int]:
    hasher = hashlib.sha256()

    for index in range(len(dataset)):
        row = dataset[index]

        payload = _canonical_json_bytes(row)

        hasher.update(
            len(payload).to_bytes(
                _ROW_LENGTH_BYTES,
                byteorder="big",
                signed=False,
            )
        )
        hasher.update(payload)

    return hasher.hexdigest(), len(dataset)


def _provider_license(
    dataset: _MaterializedDataset,
) -> str | None:
    value = getattr(
        dataset.info,
        "license",
        None,
    )

    if not isinstance(value, str):
        return None

    normalized = value.strip()

    return normalized or None


def _failure(
    *,
    stage: str,
    repo_id: object,
    requested_revision: object,
    config_name: object,
    split: object,
    resolved_revision: str | None,
    exc: Exception,
) -> HuggingFaceDatasetIdentityError:
    message = str(exc).strip()

    return HuggingFaceDatasetIdentityError(
        HuggingFaceDatasetIdentityFailure(
            stage=stage,
            repo_id=_safe_failure_text(repo_id),
            requested_revision=(_safe_failure_text(requested_revision)),
            config_name=(_safe_failure_optional_text(config_name)),
            split=_safe_failure_text(split),
            resolved_revision=(resolved_revision),
            error_type=type(exc).__name__,
            message=(message or type(exc).__name__),
        )
    )


def inspect_huggingface_dataset(
    repo_id: str,
    revision: str,
    *,
    split: str,
    source_files: tuple[str, ...],
    license_expression: str,
    preprocessing_sha256: str,
    deduplication_sha256: str,
    filtering_sha256: str,
    contamination_check_sha256: str,
    loss_mask_policy_sha256: str,
    chat_template_sha256: str | None = None,
    config_name: str | None = None,
    cache_dir: str | Path | None = None,
    token: bool | str | None = None,
) -> HuggingFaceDatasetIdentityInspection:
    resolved_revision: str | None = None

    try:
        normalized_repo_id = _require_non_empty(
            repo_id,
            "repo_id",
        )
        normalized_revision = _require_non_empty(
            revision,
            "revision",
        )
        normalized_split = _normalize_split(split)
        normalized_config_name = _normalize_config_name(config_name)
        normalized_source_files = _normalize_source_files(source_files)
        normalized_license = _require_non_empty(
            license_expression,
            "license_expression",
        )
        normalized_preprocessing = _require_sha256(
            preprocessing_sha256,
            "preprocessing_sha256",
        )
        normalized_deduplication = _require_sha256(
            deduplication_sha256,
            "deduplication_sha256",
        )
        normalized_filtering = _require_sha256(
            filtering_sha256,
            "filtering_sha256",
        )
        normalized_contamination = _require_sha256(
            contamination_check_sha256,
            "contamination_check_sha256",
        )
        normalized_loss_mask = _require_sha256(
            loss_mask_policy_sha256,
            "loss_mask_policy_sha256",
        )
        normalized_chat_template = (
            None
            if chat_template_sha256 is None
            else _require_sha256(
                chat_template_sha256,
                "chat_template_sha256",
            )
        )
    except Exception as exc:
        raise _failure(
            stage="input",
            repo_id=repo_id,
            requested_revision=revision,
            config_name=config_name,
            split=split,
            resolved_revision=None,
            exc=exc,
        ) from exc

    cache_path = None if cache_dir is None else str(cache_dir)

    api = HfApi()

    try:
        requested_info = api.dataset_info(
            repo_id=normalized_repo_id,
            revision=normalized_revision,
            token=token,
        )

        resolved_revision = _require_commit(requested_info.sha)
    except Exception as exc:
        raise _failure(
            stage="resolve_revision",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    try:
        pinned_info = api.dataset_info(
            repo_id=normalized_repo_id,
            revision=resolved_revision,
            files_metadata=True,
            token=token,
        )

        pinned_revision = _require_commit(pinned_info.sha)

        if pinned_revision != resolved_revision:
            raise ValueError("pinned dataset metadata returned a different commit SHA")

        repository_files = _repository_inventory(pinned_info.siblings)

        repository_inventory_sha256 = _repository_inventory_sha256(repository_files)
    except Exception as exc:
        raise _failure(
            stage="inventory_repository",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    repository_by_path = {item.path: item for item in repository_files}

    source_artifacts: list[HuggingFaceDatasetSourceArtifactEvidence] = []

    try:
        for source_path in normalized_source_files:
            provider = repository_by_path.get(source_path)

            if provider is None:
                raise ValueError(f"source file is absent from the pinned repository: {source_path}")

            local_path = Path(
                hf_hub_download(
                    repo_id=normalized_repo_id,
                    filename=source_path,
                    repo_type="dataset",
                    revision=(resolved_revision),
                    cache_dir=cache_path,
                    token=token,
                )
            )

            (
                local_sha256,
                local_size,
            ) = _sha256_file(local_path)

            if provider.size is not None and local_size != provider.size:
                raise ValueError(
                    "source file size mismatch "
                    f"for {source_path}: "
                    f"provider={provider.size}, "
                    f"local={local_size}"
                )

            if provider.lfs_sha256 is not None and local_sha256 != provider.lfs_sha256:
                raise ValueError(
                    "source file SHA-256 mismatch "
                    f"for {source_path}: "
                    "provider="
                    f"{provider.lfs_sha256}, "
                    f"local={local_sha256}"
                )

            source_artifacts.append(
                HuggingFaceDatasetSourceArtifactEvidence(
                    path=source_path,
                    provider_size=(provider.size),
                    provider_blob_id=(provider.blob_id),
                    provider_sha256=(provider.lfs_sha256),
                    local_size=local_size,
                    local_sha256=(local_sha256),
                )
            )
    except Exception as exc:
        raise _failure(
            stage="verify_source_artifacts",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    try:
        loaded_dataset = load_dataset(
            normalized_repo_id,
            name=normalized_config_name,
            split=normalized_split,
            revision=resolved_revision,
            cache_dir=cache_path,
            token=token,
            streaming=False,
        )

        if not isinstance(loaded_dataset, Dataset):
            raise TypeError("initial dataset identity support requires a materialized Dataset")
    except Exception as exc:
        raise _failure(
            stage="load_dataset",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    dataset = cast(
        _MaterializedDataset,
        loaded_dataset,
    )

    try:
        schema_sha256 = _schema_sha256(dataset)
    except Exception as exc:
        raise _failure(
            stage="hash_schema",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    try:
        (
            content_sha256,
            sample_count,
        ) = _content_sha256(dataset)
    except Exception as exc:
        raise _failure(
            stage="hash_content",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc

    try:
        external_fingerprint = _require_non_empty(
            str(
                getattr(
                    dataset,
                    "_fingerprint",
                    "",
                )
            ),
            "Dataset._fingerprint",
        )

        identity = DatasetIdentity(
            source_uri=(f"https://huggingface.co/datasets/{normalized_repo_id}"),
            license_expression=(normalized_license),
            revision=resolved_revision,
            content_sha256=(content_sha256),
            schema_sha256=(schema_sha256),
            preprocessing_sha256=(normalized_preprocessing),
            deduplication_sha256=(normalized_deduplication),
            filtering_sha256=(normalized_filtering),
            contamination_check_sha256=(normalized_contamination),
            loss_mask_policy_sha256=(normalized_loss_mask),
            chat_template_sha256=(normalized_chat_template),
            sample_count=sample_count,
            token_count=None,
        )

        evidence = HuggingFaceDatasetIdentityEvidence(
            repo_id=(normalized_repo_id),
            requested_revision=(normalized_revision),
            resolved_revision=(resolved_revision),
            config_name=(normalized_config_name),
            split=normalized_split,
            license_expression=(normalized_license),
            provider_license=(_provider_license(dataset)),
            repository_files=(repository_files),
            repository_inventory_sha256=(repository_inventory_sha256),
            source_artifacts=tuple(source_artifacts),
            external_dataset_fingerprint=(external_fingerprint),
            schema_sha256=(schema_sha256),
            content_sha256=(content_sha256),
            sample_count=(sample_count),
            token_count=None,
            library_versions=(
                (
                    "datasets",
                    version("datasets"),
                ),
                (
                    "huggingface-hub",
                    version("huggingface-hub"),
                ),
            ),
        )

        return HuggingFaceDatasetIdentityInspection(
            identity=identity,
            evidence=evidence,
        )
    except Exception as exc:
        raise _failure(
            stage="build_identity",
            repo_id=normalized_repo_id,
            requested_revision=normalized_revision,
            config_name=normalized_config_name,
            split=normalized_split,
            resolved_revision=(resolved_revision),
            exc=exc,
        ) from exc
