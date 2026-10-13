"""H01-H10 CPU-only adversarial two-phase synthetic gradient handoff tests."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from huyawo_adapt.agentic.training import local_sft_tiny_handoff_protocol as phase
from huyawo_adapt.agentic.training.local_sft_batch import compile_local_sft_batch
from huyawo_adapt.agentic.training.local_sft_tensor import materialize_local_sft_tensors
from huyawo_adapt.agentic.training.local_sft_tiny_gradient_resume import _binding
from huyawo_adapt.agentic.training.local_sft_tiny_model import _causal_logits
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    AgentSFTSplitEntry,
    AgentSFTSplitManifest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _handoff():
    sequences = ((1, 2, 3, 4), (5, 6, 7, 8), (2, 0, 3, 4), (3, 9, 4))
    starts = (2, 3, 2, 1)
    projections = []
    entries = []
    for index, (ids, start) in enumerate(zip(sequences, starts, strict=True)):
        identity = f"b12-{index}"
        token_bytes = json.dumps(
            {"domain": "haa-sft-target-token-ids-v1", "token_ids": ids[start:]},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        projection = AgentSFTDecisionProjection(
            trajectory_fingerprint=_sha(f"trajectory-{identity}"),
            step_index=0,
            step_provenance_sha256=_sha(f"step-{identity}"),
            model_fingerprint=_sha("model-b12"),
            tokenizer_fingerprint=_sha("tokenizer-b12"),
            harness_fingerprint=_sha("harness-b12"),
            tool_set_fingerprint=_sha("tools-b12"),
            environment_fingerprint=_sha("environment-b12"),
            verifier_fingerprint=_sha("verifier-b12"),
            task_fingerprint=_sha(f"task-{identity}"),
            runtime_fingerprint=_sha("runtime-b12"),
            rollout_policy_fingerprint=_sha("rollout-b12"),
            capture_provenance_sha256=_sha(f"capture-{identity}"),
            model_input_sha256=_sha(f"input-{identity}"),
            generated_output_sha256=_sha(f"output-{identity}"),
            projection_policy_sha256=_sha("policy-b12"),
            input_ids=ids,
            labels=(-100,) * start + ids[start:],
            target_start=start,
            target_end=len(ids),
            target_token_sha256=hashlib.sha256(token_bytes).hexdigest(),
        )
        projections.append(projection)
        entries.append(
            AgentSFTSplitEntry(
                split="train",
                projection_fingerprint=projection.fingerprint(),
                trajectory_fingerprint=projection.trajectory_fingerprint,
                task_fingerprint=projection.task_fingerprint,
                family_fingerprint=_sha(f"family-{identity}"),
                root_trajectory_fingerprint=_sha(f"root-{identity}"),
                parent_trajectory_fingerprint=projection.parent_trajectory_fingerprint,
                source_content_sha256=_sha(f"content-{identity}"),
            )
        )
    manifest = AgentSFTSplitManifest(split_policy_sha256=_sha("split-b12"), entries=tuple(entries))
    batch = compile_local_sft_batch(
        tuple(projections), manifest, "train", pad_token_id=0, max_sequence_length=16
    )
    return materialize_local_sft_tensors(batch)


_PARAMS = dict(
    synthetic_vocabulary_size=16,
    embedding_dim=4,
    seed=184,
    learning_rate=0.03,
    microbatch_rows=1,
    interrupt_after_parts=1,
)


def _prepare(**changes):
    options = dict(_PARAMS)
    options.update(changes)
    return phase.prepare_local_sft_synthetic_handoff(_handoff(), **options)


def _resume(data: bytes, **changes):
    options = dict(_PARAMS)
    options.update(changes)
    return phase.resume_local_sft_synthetic_handoff(_handoff(), data, **options)


def _reframe(data: bytes, *, edit=None, payload=None) -> bytes:
    state = phase._decode(data)
    metadata = {name: getattr(state, name) for name in phase._FIELDS}
    if edit is not None:
        edit(metadata)
    meta = phase._canonical(metadata)
    raw = state.payload if payload is None else payload
    body = phase._MAGIC + struct.pack("<I", len(meta)) + meta + struct.pack("<I", len(raw)) + raw
    return body + hashlib.sha256(body).digest()


def test_h01_prefix_only_and_canonical_untrusted_envelope(monkeypatch) -> None:
    steps = []
    original_step = torch.optim.SGD.step

    def counted(self, *args, **kwargs):
        steps.append(True)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counted)
    token = _prepare()
    assert type(token.payload) is bytes and 0 < len(token.payload) <= 32_768
    assert token.envelope_sha256 == hashlib.sha256(token.payload).hexdigest()
    state = phase._decode(token.payload)
    assert state.completed_parts == 1 and state.completed_loss_tokens == 2
    assert state.total_loss_tokens == 7
    assert len(state.payload) == 4 * 16 * 4 * 4
    assert not steps


def test_h02_independent_calls_and_one_resumed_step(monkeypatch) -> None:
    data = _prepare().payload
    original = torch.optim.SGD.step
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.SGD, "step", counted)
    result = _resume(data)
    assert result.completed_parts == 1
    assert len(calls) == 3
    assert result.initial_embedding_sha256 != result.final_embedding_sha256
    assert result.initial_projection_sha256 != result.final_projection_sha256


def test_h03_cursor_accounting_and_partition_rejection() -> None:
    outputs = [
        _resume(_prepare(interrupt_after_parts=i).payload, interrupt_after_parts=i)
        for i in (1, 2, 3)
    ]
    assert {o.partition_loss_tokens for o in outputs} == {(2, 1, 2, 2)}
    assert tuple(o.completed_loss_tokens for o in outputs) == (2, 3, 5)
    assert len({o.final_embedding_sha256 for o in outputs}) == 1
    for cursor in (0, 4, -1, True):
        with pytest.raises(ValueError):
            _prepare(interrupt_after_parts=cursor)
    for size in (0, 4, True):
        with pytest.raises(ValueError):
            _prepare(microbatch_rows=size)
    with pytest.raises(ValueError):
        _resume(_prepare().payload, interrupt_after_parts=2)


def test_h04_reference_gradient_and_weight_oracles() -> None:
    result = _resume(_prepare(interrupt_after_parts=2).payload, interrupt_after_parts=2)
    assert result.prefix_gradient_max_error < 3e-6
    assert result.microbatch_gradient_max_error < 3e-6
    assert result.fullbatch_gradient_max_error < 3e-6
    assert result.microbatch_step_max_error < 3e-6
    assert result.fullbatch_step_max_error < 3e-6
    assert result.mean_nll_before >= 0 and result.mean_nll_after >= 0


def test_h05_two_fresh_interpreters_bounded_anonymous_pipe() -> None:
    test_file = str(Path(__file__).resolve())
    shared = 'import runpy,sys; n=runpy.run_path(sys.argv[1]); p=n["_PARAMS"]; h=n["_handoff"](); '
    producer = (
        shared + "from huyawo_adapt.agentic.training.local_sft_tiny_handoff_protocol "
        "import prepare_local_sft_synthetic_handoff as prepare; "
        "sys.stdout.buffer.write(prepare(h,**p).payload)"
    )
    consumer = (
        shared + "from huyawo_adapt.agentic.training.local_sft_tiny_handoff_protocol "
        "import resume_local_sft_synthetic_handoff as resume; "
        "raw=sys.stdin.buffer.read(32769); "
        "assert len(raw)<=32768; "
        "out=resume(h,raw,**p); "
        'sys.stdout.write(out.final_embedding_sha256+" "+out.final_projection_sha256)'
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    produced = subprocess.run(
        [sys.executable, "-B", "-c", producer, test_file],
        capture_output=True,
        check=True,
        timeout=30,
        env=env,
    )
    assert 0 < len(produced.stdout) <= 32_768 and produced.stderr == b""
    consumed = subprocess.run(
        [sys.executable, "-B", "-c", consumer, test_file],
        input=produced.stdout,
        capture_output=True,
        check=True,
        timeout=30,
        env=env,
    )
    reference = _resume(_prepare().payload)
    assert consumed.stdout.decode("ascii") == (
        reference.final_embedding_sha256 + " " + reference.final_projection_sha256
    )
    assert consumed.stderr == b""
    for value in (produced.stdout[:-1], produced.stdout + b"x", b"", b"x" * 32_769):
        bad = subprocess.run(
            [sys.executable, "-B", "-c", consumer, test_file],
            input=value,
            capture_output=True,
            timeout=30,
            env=env,
        )
        assert bad.returncode != 0


def test_h06_malformed_noncanonical_payloads_reject() -> None:
    data = _prepare().payload
    altered = [
        data[:-1],
        data + b"x",
        data[:5] + b"?" + data[6:],
        b"",
        b"x" * 32_769,
        data[:8] + struct.pack("<I", 8193) + data[12:],
        _reframe(data, edit=lambda m: m.update({"unknown": 4})),
        _reframe(data, edit=lambda m: m.update({"version": 9})),
        _reframe(data, edit=lambda m: m.update({"dtype": "float64"})),
        _reframe(data, edit=lambda m: m.update({"byte_order": "big"})),
    ]
    for candidate in altered:
        with pytest.raises(ValueError):
            _resume(candidate)
    state = phase._decode(data)
    corrupt = bytearray(state.payload)
    corrupt[2 * 16 * 4 * 4 : 2 * 16 * 4 * 4 + 4] = struct.pack("<f", float("nan"))
    with pytest.raises(ValueError):
        _resume(_reframe(data, payload=bytes(corrupt)))
    raw = data[12 : 12 + struct.unpack("<I", data[8:12])[0]]
    duplicate = raw[:-1] + b',"seed":184}'
    payload = state.payload
    body = (
        phase._MAGIC
        + struct.pack("<I", len(duplicate))
        + duplicate
        + struct.pack("<I", len(payload))
        + payload
    )
    with pytest.raises(ValueError):
        _resume(body + hashlib.sha256(body).digest())


def test_h07_stale_lineage_seed_lr_and_forged_self_consistent_state() -> None:
    data = _prepare().payload
    for parameter in (
        dict(seed=185),
        dict(learning_rate=0.02),
        dict(embedding_dim=5),
        dict(microbatch_rows=2),
    ):
        with pytest.raises(ValueError):
            _resume(data, **parameter)
    for key in ("operation_sha256", "source_batch_sha256", "total_loss_tokens", "completed_parts"):
        with pytest.raises(ValueError):
            _resume(_reframe(data, edit=lambda m, k=key: m.update({k: 10})))
    # A checksum-consistent attacker-held state must not inject arbitrary weights.
    state = phase._decode(data)
    forged = bytearray(state.payload)
    forged[0:4] = struct.pack("<f", 0.19)
    modified = replace(state, payload=bytes(forged))
    from huyawo_adapt.agentic.training.local_sft_tiny_gradient_resume import _state_hash

    modified = replace(modified, payload_sha256=_state_hash(modified))
    metadata = {name: getattr(modified, name) for name in phase._FIELDS}
    meta = phase._canonical(metadata)
    body = (
        phase._MAGIC
        + struct.pack("<I", len(meta))
        + meta
        + struct.pack("<I", len(modified.payload))
        + modified.payload
    )
    with pytest.raises(ValueError, match="noncanonical initial"):
        _resume(body + hashlib.sha256(body).digest())


def test_h08_source_rng_pad_mask_no_alias() -> None:
    handoff = _handoff()
    originals = tuple(
        t.clone() for t in (handoff.input_ids, handoff.labels, handoff.attention_mask)
    )
    rng = torch.get_rng_state().clone()
    token = phase.prepare_local_sft_synthetic_handoff(handoff, **_PARAMS)
    result = phase.resume_local_sft_synthetic_handoff(handoff, token.payload, **_PARAMS)
    assert result.training_eligible is False
    assert torch.equal(torch.get_rng_state(), rng)
    for left, right in zip(
        originals, (handoff.input_ids, handoff.labels, handoff.attention_mask), strict=True
    ):
        assert torch.equal(left, right)
    assert handoff.input_ids[2, 1].item() == 0
    assert handoff.attention_mask[2, 1].item() == 1
    assert handoff.attention_mask[3, 3].item() == 0
    initial_e, initial_p = phase._initial(16, 4, 184)
    first = _causal_logits(handoff.input_ids, handoff.attention_mask, initial_e, initial_p)
    changed = handoff.input_ids.clone()
    changed[0, 3] = 15
    after = _causal_logits(changed, handoff.attention_mask, initial_e, initial_p)
    assert torch.equal(first[0, :3], after[0, :3])
    assert len({t.untyped_storage().data_ptr() for t in (*originals, initial_e, initial_p)}) == 5
    state = phase._decode(token.payload)
    b = _binding(handoff, 16, 4, 184, 0.03, state.partition_boundaries, state.partition_loss_tokens)
    from huyawo_adapt.agentic.training.local_sft_tiny_gradient_resume import (
        restore_local_synthetic_gradient_state,
    )

    weights = restore_local_synthetic_gradient_state(state, b, 1)
    assert len({x.untyped_storage().data_ptr() for w in weights for x in (w, w.grad)}) == 4


def test_h09_faults_stop_and_replay_is_non_authoritative(monkeypatch) -> None:
    data = _prepare().payload
    original = phase._backward

    def fail(*args, **kwargs):
        raise RuntimeError("controlled fault")

    monkeypatch.setattr(phase, "_backward", fail)
    with pytest.raises(RuntimeError):
        _prepare()
    with pytest.raises(RuntimeError):
        _resume(data)
    monkeypatch.setattr(phase, "_backward", original)
    with pytest.raises(ValueError):
        _resume(data[:-1])
    first, second = _resume(data), _resume(data)
    assert first.final_embedding_sha256 == second.final_embedding_sha256
    assert first.training_eligible is False and second.training_eligible is False


def test_h10_nonpromoting_diagnostics_without_real_receipts() -> None:
    token = _prepare()
    result = _resume(token.payload)
    for value in (token, result):
        assert value.authority_scope == "local_sft_synthetic_two_phase_only"
        assert value.independent_task_outcome_verified is False
        assert value.model_origin_attested is False
        assert value.tokenizer_fidelity_attested is False
        assert value.training_eligible is False
        for prohibited in (
            "checkpoint_path",
            "real_training_run_id",
            "execution_receipt",
            "authentic_model_origin",
        ):
            assert not hasattr(value, prohibited)
        with pytest.raises(ValueError):
            replace(value, training_eligible=True)
