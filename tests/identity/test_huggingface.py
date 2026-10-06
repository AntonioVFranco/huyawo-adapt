from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

import huyawo_adapt.identity.huggingface as hf_identity
from huyawo_adapt.identity import (
    HuggingFaceModelIdentityError,
    inspect_huggingface_model,
)

RESOLVED_REVISION = "a" * 40


class FakeLfs:
    def __init__(
        self,
        sha256: str,
    ) -> None:
        self.sha256 = sha256


class FakeSibling:
    def __init__(
        self,
        path: str,
        *,
        size: int | None,
        blob_id: str | None,
        lfs_sha256: str | None = None,
    ) -> None:
        self.rfilename = path
        self.size = size
        self.blob_id = blob_id
        self.lfs = FakeLfs(lfs_sha256) if lfs_sha256 is not None else None


class FakeApi:
    def __init__(
        self,
        *,
        resolved_revision: str,
        siblings: list[FakeSibling],
        fail_resolution: bool = False,
    ) -> None:
        self.resolved_revision = resolved_revision
        self.siblings = siblings
        self.fail_resolution = fail_resolution
        self.calls: list[tuple[str, str | None, bool]] = []

    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
        token: bool | str | None = None,
    ) -> SimpleNamespace:
        del token

        self.calls.append(
            (
                repo_id,
                revision,
                files_metadata,
            )
        )

        if self.fail_resolution:
            raise RuntimeError("provider resolution failed")

        if files_metadata:
            return SimpleNamespace(
                sha=self.resolved_revision,
                siblings=self.siblings,
            )

        return SimpleNamespace(
            sha=self.resolved_revision,
            siblings=None,
        )


class FakeAutoConfig:
    architectures: list[str] = ["Qwen3ForCausalLM"]
    model_type = "qwen3"
    calls: list[tuple[Path, dict[str, object]]] = []

    @classmethod
    def from_pretrained(
        cls,
        path: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        cls.calls.append(
            (
                Path(path),
                kwargs,
            )
        )

        return SimpleNamespace(
            architectures=list(cls.architectures),
            model_type=cls.model_type,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_local_artifacts(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    str,
]:
    config_path = tmp_path / "config.json"

    config_path.write_text(
        '{"model_type":"qwen3"}',
        encoding="utf-8",
    )

    weight_path = tmp_path / "model.safetensors"

    save_file(
        {
            "layer_a": np.zeros(
                (2, 3),
                dtype=np.float32,
            ),
            "layer_b": np.zeros(
                (4,),
                dtype=np.float16,
            ),
        },
        str(weight_path),
    )

    return (
        config_path,
        weight_path,
        sha256_file(weight_path),
    )


def install_success_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    resolved_revision: str = (RESOLVED_REVISION),
    weight_provider_sha256: (str | None) = None,
    include_config: bool = True,
    include_weight: bool = True,
    include_weight_index: bool = False,
    config_provider_size: int | None = None,
    weight_provider_size: int | None = None,
) -> tuple[
    FakeApi,
    list[
        tuple[
            str,
            str,
            str,
        ]
    ],
    Path,
    Path,
]:
    (
        config_path,
        weight_path,
        local_weight_sha256,
    ) = build_local_artifacts(tmp_path)

    siblings: list[FakeSibling] = []

    if include_config:
        siblings.append(
            FakeSibling(
                "config.json",
                size=(
                    config_path.stat().st_size
                    if config_provider_size is None
                    else config_provider_size
                ),
                blob_id="config-blob",
            )
        )

    if include_weight:
        siblings.append(
            FakeSibling(
                "model.safetensors",
                size=(
                    weight_path.stat().st_size
                    if weight_provider_size is None
                    else weight_provider_size
                ),
                blob_id="weight-blob",
                lfs_sha256=(
                    local_weight_sha256
                    if weight_provider_sha256 is None
                    else weight_provider_sha256
                ),
            )
        )

    if include_weight_index:
        siblings.append(
            FakeSibling(
                "model.safetensors.index.json",
                size=100,
                blob_id="index-blob",
            )
        )

    siblings.append(
        FakeSibling(
            "README.md",
            size=50,
            blob_id="readme-blob",
        )
    )

    api = FakeApi(
        resolved_revision=resolved_revision,
        siblings=list(reversed(siblings)),
    )

    monkeypatch.setattr(
        hf_identity,
        "HfApi",
        lambda: api,
    )

    download_calls: list[tuple[str, str, str]] = []

    def fake_download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str | Path | None = None,
        token: bool | str | None = None,
    ) -> str:
        del cache_dir
        del token

        download_calls.append(
            (
                repo_id,
                filename,
                revision,
            )
        )

        if filename == "config.json":
            return str(config_path)

        if filename == "model.safetensors":
            return str(weight_path)

        raise AssertionError(f"Unexpected filename: {filename}")

    monkeypatch.setattr(
        hf_identity,
        "hf_hub_download",
        fake_download,
    )

    FakeAutoConfig.calls = []
    FakeAutoConfig.architectures = ["Qwen3ForCausalLM"]
    FakeAutoConfig.model_type = "qwen3"

    monkeypatch.setattr(
        hf_identity,
        "AutoConfig",
        FakeAutoConfig,
    )

    return (
        api,
        download_calls,
        config_path,
        weight_path,
    )


def test_successful_inspection_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        api,
        download_calls,
        config_path,
        weight_path,
    ) = install_success_environment(
        monkeypatch,
        tmp_path,
    )

    first = inspect_huggingface_model(
        "Qwen/Qwen3-0.6B-Base",
        cache_dir=tmp_path / "cache",
    )

    second = inspect_huggingface_model(
        "Qwen/Qwen3-0.6B-Base",
        cache_dir=tmp_path / "cache",
    )

    assert first.identity == second.identity
    assert first.identity.fingerprint() == second.identity.fingerprint()
    assert first.evidence.fingerprint() == second.evidence.fingerprint()

    assert first.identity.revision == RESOLVED_REVISION
    assert first.identity.architecture == "Qwen3ForCausalLM"
    assert first.identity.parameter_count == 10
    assert first.identity.config_sha256 == sha256_file(config_path)
    assert first.identity.weight_artifacts[0].sha256 == sha256_file(weight_path)

    assert [item.path for item in first.evidence.repository_files] == [
        "README.md",
        "config.json",
        "model.safetensors",
    ]

    assert first.evidence.tensor_count == 2
    assert first.evidence.parameter_count == 10
    assert first.evidence.trust_remote_code is False

    assert api.calls[0] == (
        "Qwen/Qwen3-0.6B-Base",
        "main",
        False,
    )
    assert api.calls[1] == (
        "Qwen/Qwen3-0.6B-Base",
        RESOLVED_REVISION,
        True,
    )

    assert all(call[2] == RESOLVED_REVISION for call in download_calls)

    assert FakeAutoConfig.calls

    config_call = FakeAutoConfig.calls[0]

    assert config_call[0] == (config_path.parent)
    assert config_call[1]["local_files_only"] is True
    assert config_call[1]["trust_remote_code"] is False


def test_failure_record_is_deterministic() -> None:
    failure = hf_identity.HuggingFaceModelIdentityFailure(
        stage="verify_weight",
        repo_id="organization/model",
        requested_revision="main",
        resolved_revision=RESOLVED_REVISION,
        error_type="IdentityValidationError",
        message="digest mismatch",
    )

    assert failure.canonical_json() == failure.canonical_json()
    assert failure.fingerprint() == failure.fingerprint()


def test_invalid_resolved_revision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        resolved_revision="main",
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "resolve_revision"


def test_provider_resolution_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(
        resolved_revision=RESOLVED_REVISION,
        siblings=[],
        fail_resolution=True,
    )

    monkeypatch.setattr(
        hf_identity,
        "HfApi",
        lambda: api,
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    failure = exc_info.value.evidence

    assert failure.stage == "resolve_revision"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "provider resolution failed"


def test_missing_config_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        include_config=False,
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "select_config_artifact"


def test_missing_weight_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        include_weight=False,
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "select_weight_artifact"


def test_sharded_checkpoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        include_weight_index=True,
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "select_weight_artifact"
    assert "Sharded" in (exc_info.value.evidence.message)


def test_weight_size_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        _,
        weight_path,
    ) = install_success_environment(
        monkeypatch,
        tmp_path,
        weight_provider_size=1,
    )

    assert weight_path.stat().st_size != 1

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "verify_weight"


def test_weight_digest_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        weight_provider_sha256=("0" * 64),
    )

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "verify_weight"
    assert "SHA-256" in (exc_info.value.evidence.message)


def test_missing_architecture_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
    )

    FakeAutoConfig.architectures = []

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "parse_config"


def test_invalid_safetensors_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        _,
        weight_path,
    ) = install_success_environment(
        monkeypatch,
        tmp_path,
    )

    weight_path.write_bytes(b"not-a-safetensors-file")

    local_sha = sha256_file(weight_path)

    install_success_environment(
        monkeypatch,
        tmp_path,
        weight_provider_sha256=local_sha,
        weight_provider_size=(weight_path.stat().st_size),
    )

    weight_path.write_bytes(b"not-a-safetensors-file")

    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model("organization/model")

    assert exc_info.value.evidence.stage == "inspect_safetensors"


def test_input_whitespace_fails_closed() -> None:
    with pytest.raises(HuggingFaceModelIdentityError) as exc_info:
        inspect_huggingface_model(" organization/model")

    assert exc_info.value.evidence.stage == "input"
