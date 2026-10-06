from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import huyawo_adapt.datasets.huggingface as module
from datasets import Dataset
from huyawo_adapt.datasets import (
    HuggingFaceDatasetIdentityError,
    HuggingFaceDatasetIdentityInspection,
    inspect_huggingface_dataset,
)

REPO_ID = "example/dataset"
REQUESTED_REVISION = "main"
RESOLVED_REVISION = "1" * 40

PREPROCESSING_SHA256 = "a" * 64
DEDUPLICATION_SHA256 = "b" * 64
FILTERING_SHA256 = "c" * 64
CONTAMINATION_SHA256 = "d" * 64
LOSS_MASK_SHA256 = "e" * 64
CHAT_TEMPLATE_SHA256 = "f" * 64

SOURCE_PATH = "data/train.jsonl"


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _expected_content_sha256(
    rows: list[dict[str, object]],
) -> str:
    hasher = hashlib.sha256()

    for row in rows:
        payload = _canonical_json_bytes(row)
        hasher.update(
            len(payload).to_bytes(
                8,
                byteorder="big",
                signed=False,
            )
        )
        hasher.update(payload)

    return hasher.hexdigest()


def _make_dataset(
    rows: list[dict[str, object]],
    *,
    fingerprint: str = "external-fingerprint",
) -> Dataset:
    columns = {key: [row[key] for row in rows] for key in rows[0]}

    dataset = Dataset.from_dict(columns)
    dataset._fingerprint = fingerprint
    dataset.info.license = "apache-2.0"

    return dataset


def _default_rows() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "text": "alpha",
        },
        {
            "id": 2,
            "text": "beta",
        },
    ]


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    dataset: object,
    resolved_revision: str = (RESOLVED_REVISION),
    source_sha256: str | None = None,
    source_present: bool = True,
    resolve_error: Exception | None = None,
    load_error: Exception | None = None,
) -> bytes:
    source_bytes = b'{"id":1,"text":"alpha"}\n{"id":2,"text":"beta"}\n'

    source_file = tmp_path / "provider-source.jsonl"
    source_file.write_bytes(source_bytes)

    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()

    provider_sha256 = actual_sha256 if source_sha256 is None else source_sha256

    siblings = [
        SimpleNamespace(
            rfilename="README.md",
            size=10,
            blob_id="2" * 40,
            lfs=None,
        )
    ]

    if source_present:
        siblings.append(
            SimpleNamespace(
                rfilename=SOURCE_PATH,
                size=len(source_bytes),
                blob_id="3" * 40,
                lfs=SimpleNamespace(sha256=provider_sha256),
            )
        )

    class FakeApi:
        def dataset_info(
            self,
            repo_id: str,
            *,
            revision: str | None = None,
            files_metadata: bool = False,
            token: bool | str | None = None,
        ) -> SimpleNamespace:
            del token

            assert repo_id == REPO_ID

            if resolve_error is not None:
                raise resolve_error

            if revision == REQUESTED_REVISION:
                sha = resolved_revision
            else:
                sha = revision

            return SimpleNamespace(
                sha=sha,
                siblings=(siblings if files_metadata else None),
            )

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        repo_type: str,
        revision: str,
        cache_dir: str | None = None,
        token: bool | str | None = None,
    ) -> str:
        del cache_dir
        del token

        assert repo_id == REPO_ID
        assert filename == SOURCE_PATH
        assert repo_type == "dataset"
        assert revision == RESOLVED_REVISION

        return str(source_file)

    def fake_load_dataset(
        path: str,
        name: str | None = None,
        data_dir: str | None = None,
        data_files: object = None,
        split: str | None = None,
        cache_dir: str | None = None,
        features: object = None,
        download_config: object = None,
        download_mode: object = None,
        verification_mode: object = None,
        keep_in_memory: bool | None = None,
        save_infos: bool = False,
        revision: object = None,
        token: bool | str | None = None,
        streaming: bool = False,
        num_proc: int | None = None,
        storage_options: dict | None = None,
        **config_kwargs: object,
    ) -> object:
        del data_dir
        del data_files
        del cache_dir
        del features
        del download_config
        del download_mode
        del verification_mode
        del keep_in_memory
        del save_infos
        del token
        del num_proc
        del storage_options
        del config_kwargs

        assert path == REPO_ID
        assert name is None
        assert split == "train"
        assert revision == RESOLVED_REVISION
        assert streaming is False

        if load_error is not None:
            raise load_error

        return dataset

    monkeypatch.setattr(
        module,
        "HfApi",
        FakeApi,
    )
    monkeypatch.setattr(
        module,
        "hf_hub_download",
        fake_hf_hub_download,
    )
    monkeypatch.setattr(
        module,
        "load_dataset",
        fake_load_dataset,
    )

    return source_bytes


def _inspect(
    tmp_path: Path,
    **overrides: object,
) -> HuggingFaceDatasetIdentityInspection:
    kwargs: dict[str, object] = {
        "repo_id": REPO_ID,
        "revision": REQUESTED_REVISION,
        "split": "train",
        "source_files": (SOURCE_PATH,),
        "license_expression": ("Apache-2.0"),
        "preprocessing_sha256": (PREPROCESSING_SHA256),
        "deduplication_sha256": (DEDUPLICATION_SHA256),
        "filtering_sha256": (FILTERING_SHA256),
        "contamination_check_sha256": (CONTAMINATION_SHA256),
        "loss_mask_policy_sha256": (LOSS_MASK_SHA256),
        "chat_template_sha256": (CHAT_TEMPLATE_SHA256),
        "cache_dir": tmp_path,
    }

    kwargs.update(overrides)

    return inspect_huggingface_dataset(
        **kwargs,
    )


def test_successful_inspection_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _default_rows()
    dataset = _make_dataset(rows)

    source_bytes = _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    first = _inspect(tmp_path)
    second = _inspect(tmp_path)

    assert first == second
    assert first.identity.fingerprint() == second.identity.fingerprint()
    assert first.evidence.fingerprint() == second.evidence.fingerprint()

    assert first.identity.revision == RESOLVED_REVISION
    assert first.identity.sample_count == 2
    assert first.identity.token_count is None

    expected_source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    assert first.evidence.source_artifacts[0].local_sha256 == expected_source_sha256
    assert first.evidence.provider_license == "apache-2.0"


def test_repository_inventory_is_sorted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    result = _inspect(tmp_path)

    assert [item.path for item in (result.evidence.repository_files)] == [
        "README.md",
        SOURCE_PATH,
    ]


def test_schema_hash_uses_canonical_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    result = _inspect(tmp_path)

    expected = hashlib.sha256(_canonical_json_bytes(dataset.features.to_dict())).hexdigest()

    assert result.identity.schema_sha256 == expected
    assert result.evidence.schema_sha256 == expected


def test_content_hash_matches_length_prefixed_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _default_rows()
    dataset = _make_dataset(rows)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    result = _inspect(tmp_path)

    expected = _expected_content_sha256(rows)

    assert result.identity.content_sha256 == expected
    assert result.evidence.content_sha256 == expected


def test_content_hash_changes_when_row_order_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows_a = _default_rows()
    rows_b = list(reversed(rows_a))

    dataset_a = _make_dataset(rows_a)
    dataset_b = _make_dataset(rows_b)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_a,
    )
    first = _inspect(tmp_path)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_b,
    )
    second = _inspect(tmp_path)

    assert first.identity.content_sha256 != second.identity.content_sha256


def test_content_hash_changes_when_row_value_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows_a = _default_rows()
    rows_b = [
        {
            "id": 1,
            "text": "changed",
        },
        {
            "id": 2,
            "text": "beta",
        },
    ]

    dataset_a = _make_dataset(rows_a)
    dataset_b = _make_dataset(rows_b)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_a,
    )
    first = _inspect(tmp_path)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_b,
    )
    second = _inspect(tmp_path)

    assert first.identity.content_sha256 != second.identity.content_sha256


def test_external_fingerprint_is_evidence_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _default_rows()

    dataset_a = _make_dataset(
        rows,
        fingerprint="external-a",
    )
    dataset_b = _make_dataset(
        rows,
        fingerprint="external-b",
    )

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_a,
    )
    first = _inspect(tmp_path)

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset_b,
    )
    second = _inspect(tmp_path)

    assert first.identity == second.identity
    assert first.identity.fingerprint() == second.identity.fingerprint()
    assert first.evidence.fingerprint() != second.evidence.fingerprint()


def test_missing_license_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(
            tmp_path,
            license_expression="   ",
        )

    assert captured.value.failure.stage == "input"


def test_invalid_resolved_revision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        resolved_revision="main",
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    assert captured.value.failure.stage == "resolve_revision"


def test_provider_resolution_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        resolve_error=RuntimeError("provider unavailable"),
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    failure = captured.value.failure

    assert failure.stage == "resolve_revision"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "provider unavailable"


def test_missing_source_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        source_present=False,
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    assert captured.value.failure.stage == "verify_source_artifacts"


def test_source_sha_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        source_sha256="0" * 64,
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    assert captured.value.failure.stage == "verify_source_artifacts"


def test_dataset_load_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
        load_error=RuntimeError("load failed"),
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    assert captured.value.failure.stage == "load_dataset"
    assert captured.value.failure.message == "load failed"


def test_non_materialized_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=object(),
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(tmp_path)

    assert captured.value.failure.stage == "load_dataset"


def test_invalid_policy_digest_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(
            tmp_path,
            preprocessing_sha256=("not-a-sha256"),
        )

    assert captured.value.failure.stage == "input"


def test_split_expression_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    with pytest.raises(HuggingFaceDatasetIdentityError) as captured:
        _inspect(
            tmp_path,
            split="train[:8]",
        )

    assert captured.value.failure.stage == "input"


def test_identity_round_trip_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    result = _inspect(tmp_path)

    restored = result.identity.from_canonical_json(result.identity.canonical_bytes())

    assert restored == result.identity
    assert restored.fingerprint() == result.identity.fingerprint()


def test_operational_evidence_round_trip_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(_default_rows())

    _install_provider(
        monkeypatch,
        tmp_path,
        dataset=dataset,
    )

    result = _inspect(tmp_path)

    restored = result.evidence.from_canonical_json(result.evidence.canonical_bytes())

    assert restored == result.evidence
    assert restored.fingerprint() == result.evidence.fingerprint()
