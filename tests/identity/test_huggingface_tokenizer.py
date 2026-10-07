from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import huyawo_adapt.identity.huggingface_tokenizer as hf_tokenizer
from huyawo_adapt.contracts import TokenizerIdentity
from huyawo_adapt.identity import (
    HuggingFaceTokenizerIdentityError,
    inspect_huggingface_tokenizer,
)

RESOLVED_REVISION = "a" * 40
TOKENIZER_SCHEMA_SHA256 = "f907bf5ff9bbd5cc2168b6bf98c3017135ec71e61ef5ceed7cfb49e454179108"


class FakeLfs:
    def __init__(self, sha256: str) -> None:
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
        self.calls.append((repo_id, revision, files_metadata))

        if self.fail_resolution:
            raise RuntimeError("provider resolution failed")

        return SimpleNamespace(
            sha=self.resolved_revision,
            siblings=self.siblings if files_metadata else None,
        )


class FakeDeclaredTokenizer:
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }


class FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size: int = 100,
        tokenizer_length: int = 104,
        chat_template: str | dict[str, str] | None = "{{ messages }}",
        probe_delta: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self._length = tokenizer_length
        self.model_max_length = 4096
        self.bos_token_id = None
        self.eos_token_id = 101
        self.pad_token_id = 101
        self.padding_side = "right"
        self.truncation_side = "right"
        self.special_tokens_map = {
            "eos_token": "<eos>",
            "pad_token": "<eos>",
            "additional_special_tokens": ["<a>", "<b>"],
        }
        self.chat_template = chat_template
        self._probe_delta = probe_delta

    def __len__(self) -> int:
        return self._length

    def get_added_vocab(self) -> dict[str, int]:
        return {
            "<b>": 103,
            "<a>": 102,
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is True
        base = 10 if text.startswith("The purpose") else 20
        return [base + self._probe_delta, base + 1 + self._probe_delta]


FakeTokenizer.__name__ = "FakeDeclaredTokenizer"


class FakeAutoTokenizer:
    remote_tokenizer = FakeTokenizer()
    local_tokenizer = FakeTokenizer()
    fail_remote = False
    fail_local = False
    calls: list[tuple[object, dict[str, object]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.remote_tokenizer = FakeTokenizer()
        cls.local_tokenizer = FakeTokenizer()
        cls.fail_remote = False
        cls.fail_local = False
        cls.calls = []

    @classmethod
    def from_pretrained(cls, source: object, **kwargs: object) -> FakeTokenizer:
        cls.calls.append((source, kwargs))
        is_local = isinstance(source, Path)

        if is_local and cls.fail_local:
            raise RuntimeError("local tokenizer load failed")
        if not is_local and cls.fail_remote:
            raise RuntimeError("remote tokenizer load failed")

        return cls.local_tokenizer if is_local else cls.remote_tokenizer


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_artifacts(tmp_path: Path) -> dict[str, Path]:
    artifacts = {
        "tokenizer_config.json": tmp_path / "tokenizer_config.json",
        "vocab.json": tmp_path / "vocab.json",
        "merges.txt": tmp_path / "merges.txt",
        "tokenizer.json": tmp_path / "tokenizer.json",
        "special_tokens_map.json": tmp_path / "special_tokens_map.json",
        "added_tokens.json": tmp_path / "added_tokens.json",
        "chat_template.jinja": tmp_path / "chat_template.jinja",
    }
    artifacts["tokenizer_config.json"].write_text(
        '{"tokenizer_class":"FakeDeclaredTokenizer"}',
        encoding="utf-8",
    )
    artifacts["vocab.json"].write_text('{"a":0}', encoding="utf-8")
    artifacts["merges.txt"].write_text("#version: 0.2\na b\n", encoding="utf-8")
    artifacts["tokenizer.json"].write_text("{}", encoding="utf-8")
    artifacts["special_tokens_map.json"].write_text("{}", encoding="utf-8")
    artifacts["added_tokens.json"].write_text("{}", encoding="utf-8")
    artifacts["chat_template.jinja"].write_text("{{ messages }}", encoding="utf-8")
    return artifacts


def install_success_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    resolved_revision: str = RESOLVED_REVISION,
    include_config: bool = True,
    include_vocab: bool = True,
    config_text: str | None = None,
    config_provider_size: int | None = None,
    vocab_provider_sha256: str | None = None,
    malformed_lfs_path: str | None = None,
) -> tuple[FakeApi, list[tuple[str, str, str]], dict[str, Path]]:
    artifacts = build_artifacts(tmp_path)

    if config_text is not None:
        artifacts["tokenizer_config.json"].write_text(config_text, encoding="utf-8")

    include_paths = {
        "merges.txt",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    }
    if include_config:
        include_paths.add("tokenizer_config.json")
    if include_vocab:
        include_paths.add("vocab.json")

    siblings: list[FakeSibling] = []
    for path in sorted(include_paths, reverse=True):
        local_path = artifacts[path]
        lfs_sha256 = sha256_file(local_path)
        if path == "vocab.json" and vocab_provider_sha256 is not None:
            lfs_sha256 = vocab_provider_sha256
        if path == malformed_lfs_path:
            lfs_sha256 = "not-a-sha256"

        size = local_path.stat().st_size
        if path == "tokenizer_config.json" and config_provider_size is not None:
            size = config_provider_size

        siblings.append(
            FakeSibling(
                path,
                size=size,
                blob_id=f"blob-{path}",
                lfs_sha256=lfs_sha256,
            )
        )

    siblings.extend(
        [
            FakeSibling("README.md", size=10, blob_id="readme"),
            FakeSibling("config.json", size=10, blob_id="model-config"),
            FakeSibling("model.safetensors", size=10, blob_id="weight"),
        ]
    )

    api = FakeApi(
        resolved_revision=resolved_revision,
        siblings=siblings,
    )
    monkeypatch.setattr(hf_tokenizer, "HfApi", lambda: api)
    monkeypatch.setattr(
        hf_tokenizer.transformers,
        "FakeDeclaredTokenizer",
        FakeDeclaredTokenizer,
        raising=False,
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
        download_calls.append((repo_id, filename, revision))
        if filename not in artifacts:
            raise RuntimeError(f"missing local fixture: {filename}")
        return str(artifacts[filename])

    monkeypatch.setattr(hf_tokenizer, "hf_hub_download", fake_download)
    FakeAutoTokenizer.reset()
    monkeypatch.setattr(hf_tokenizer, "AutoTokenizer", FakeAutoTokenizer)

    return api, download_calls, artifacts


def test_successful_inspection_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api, download_calls, artifacts = install_success_environment(monkeypatch, tmp_path)

    first = inspect_huggingface_tokenizer(
        "organization/model",
        cache_dir=tmp_path / "cache",
    )
    second = inspect_huggingface_tokenizer(
        "organization/model",
        cache_dir=tmp_path / "cache",
    )

    assert first.identity == second.identity
    assert first.identity.fingerprint() == second.identity.fingerprint()
    assert first.evidence.fingerprint() == second.evidence.fingerprint()
    assert first.identity.revision == RESOLVED_REVISION
    assert first.identity.tokenizer_id == "organization/model"
    assert first.identity.vocab_size == 100
    assert first.identity.model_max_length == 4096
    assert first.evidence.requested_revision == "main"
    assert first.evidence.resolved_revision == RESOLVED_REVISION
    assert first.evidence.trust_remote_code is False
    assert first.evidence.runtime_tokenizer_class == "FakeDeclaredTokenizer"

    assert [item.path for item in first.evidence.repository_files] == sorted(
        item.path for item in first.evidence.repository_files
    )
    assert [item.path for item in first.evidence.selected_artifacts] == [
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    assert all(
        item.path not in {"README.md", "config.json", "model.safetensors"}
        for item in first.evidence.selected_artifacts
    )
    assert api.calls[0] == ("organization/model", "main", False)
    assert api.calls[1] == ("organization/model", RESOLVED_REVISION, True)
    assert all(call[2] == RESOLVED_REVISION for call in download_calls)
    assert first.identity.artifacts[0].sha256 == sha256_file(artifacts["added_tokens.json"])

    remote_call, local_call = FakeAutoTokenizer.calls[:2]
    assert remote_call[1]["revision"] == RESOLVED_REVISION
    assert remote_call[1]["trust_remote_code"] is False
    assert local_call[1]["local_files_only"] is True
    assert local_call[1]["trust_remote_code"] is False


def test_failure_record_is_deterministic() -> None:
    failure = hf_tokenizer.HuggingFaceTokenizerIdentityFailure(
        stage="verify_tokenizer_artifact",
        repo_id="organization/model",
        requested_revision="main",
        resolved_revision=RESOLVED_REVISION,
        error_type="TokenizerIdentityValidationError",
        message="digest mismatch",
    )
    assert failure.canonical_json() == failure.canonical_json()
    assert failure.fingerprint() == failure.fingerprint()


def test_invalid_resolved_revision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path, resolved_revision="main")
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "resolve_revision"


def test_provider_resolution_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi(
        resolved_revision=RESOLVED_REVISION,
        siblings=[],
        fail_resolution=True,
    )
    monkeypatch.setattr(hf_tokenizer, "HfApi", lambda: api)
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "resolve_revision"
    assert exc_info.value.evidence.error_type == "RuntimeError"


def test_missing_tokenizer_config_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path, include_config=False)
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "select_tokenizer_artifacts"


def test_missing_tokenizer_class_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path, config_text="{}")
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "verify_tokenizer_config"


def test_unknown_tokenizer_class_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        config_text='{"tokenizer_class":"DefinitelyMissingTokenizer"}',
    )
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "resolve_tokenizer_class"


def test_missing_required_vocab_artifact_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path, include_vocab=False)
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "select_tokenizer_artifacts"


def test_malformed_provider_sha256_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        malformed_lfs_path="vocab.json",
    )
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "validate_provider_metadata"


def test_provider_local_size_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path, config_provider_size=1)
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "verify_tokenizer_config"


def test_provider_local_sha_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(
        monkeypatch,
        tmp_path,
        vocab_provider_sha256="0" * 64,
    )
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "verify_tokenizer_artifact"


def test_remote_tokenizer_load_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    FakeAutoTokenizer.fail_remote = True
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "load_remote_tokenizer"


def test_local_rehydration_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    FakeAutoTokenizer.fail_local = True
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "load_local_tokenizer"


def test_remote_local_semantic_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    FakeAutoTokenizer.local_tokenizer = FakeTokenizer(probe_delta=1)
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "validate_tokenizer_semantics"


def test_multiple_named_chat_templates_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    FakeAutoTokenizer.remote_tokenizer = FakeTokenizer(chat_template={"default": "{{ messages }}"})
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer("organization/model")
    assert exc_info.value.evidence.stage == "validate_chat_template"


def test_chat_template_digest_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    result = inspect_huggingface_tokenizer("organization/model")
    expected = hashlib.sha256(b"{{ messages }}").hexdigest()
    assert result.identity.chat_template_sha256 == expected
    assert result.evidence.chat_template_sha256 == expected


def test_behavior_payload_uses_frozen_field_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    result = inspect_huggingface_tokenizer("organization/model")
    payload = {
        "class": "FakeDeclaredTokenizer",
        "vocab_size": 100,
        "length": 104,
        "bos_token_id": None,
        "eos_token_id": 101,
        "pad_token_id": 101,
        "padding_side": "right",
        "truncation_side": "right",
        "special_tokens_map": {
            "additional_special_tokens": ["<a>", "<b>"],
            "eos_token": "<eos>",
            "pad_token": "<eos>",
        },
        "probe_token_ids": [[10, 11], [20, 21]],
    }
    assert result.evidence.behavior_sha256 == canonical_sha256(payload)


def test_identity_and_evidence_round_trip_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)
    result = inspect_huggingface_tokenizer("organization/model")
    restored = TokenizerIdentity.model_validate_json(result.identity.model_dump_json())
    assert restored == result.identity
    assert result.evidence.canonical_json() == result.evidence.canonical_json()
    assert result.evidence.fingerprint() == result.evidence.fingerprint()


def test_normative_tokenizer_identity_schema_is_unchanged() -> None:
    schema_sha256 = canonical_sha256(TokenizerIdentity.model_json_schema())
    assert schema_sha256 == TOKENIZER_SCHEMA_SHA256


def test_input_whitespace_fails_closed() -> None:
    with pytest.raises(HuggingFaceTokenizerIdentityError) as exc_info:
        inspect_huggingface_tokenizer(" organization/model")
    assert exc_info.value.evidence.stage == "input"


def test_added_vocabulary_is_canonicalized_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_success_environment(monkeypatch, tmp_path)

    first = inspect_huggingface_tokenizer("organization/model")
    second = inspect_huggingface_tokenizer("organization/model")

    expected = (("<a>", 102), ("<b>", 103))

    assert first.evidence.added_vocabulary == expected
    assert second.evidence.added_vocabulary == expected


def test_identity_package_exports_tokenizer_operational_api() -> None:
    import huyawo_adapt.identity as identity_package

    expected = {
        "HuggingFaceTokenizerIdentityError": (hf_tokenizer.HuggingFaceTokenizerIdentityError),
        "HuggingFaceTokenizerIdentityEvidence": (hf_tokenizer.HuggingFaceTokenizerIdentityEvidence),
        "HuggingFaceTokenizerIdentityFailure": (hf_tokenizer.HuggingFaceTokenizerIdentityFailure),
        "HuggingFaceTokenizerIdentityInspection": (
            hf_tokenizer.HuggingFaceTokenizerIdentityInspection
        ),
        "inspect_huggingface_tokenizer": (hf_tokenizer.inspect_huggingface_tokenizer),
    }

    for name, value in expected.items():
        assert name in identity_package.__all__
        assert getattr(identity_package, name) is value
