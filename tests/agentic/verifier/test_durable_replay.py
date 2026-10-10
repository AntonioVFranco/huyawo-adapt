"""Synthetic-only durable replay checks, not independent RFC21 qualification."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from huyawo_adapt.agentic.verifier.durable_replay import (
    DurableReplayStore,
    DurableReplayUnavailableError,
)
from huyawo_adapt.agentic.verifier.local_synthetic import (
    LocalSyntheticHarness,
)

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "haa23-20261010-qual"


def new_trial(label: str) -> tuple[str, DurableReplayStore]:
    trial = f"{PREFIX}-{label}-{secrets.token_hex(4)}"
    return trial, DurableReplayStore.create_new(ROOT, trial)


def harness(store: DurableReplayStore, **changes: object) -> LocalSyntheticHarness:
    params: dict[str, Any] = dict(
        task_id="synthetic:task-one",
        episode_id="synthetic:episode-one",
        issuer_id="synthetic:issuer-local",
        private_expected_state="synthetic:done",
        captured_final_state="synthetic:done",
        captured_transcript="synthetic:fake-transcript",
        replay_store=store,
    )
    params.update(changes)
    return LocalSyntheticHarness(**params)


def check(result: object, disposition: str, reason: str) -> None:
    assert result.disposition == disposition
    assert result.reason == reason
    assert result.authority_scope == "LOCAL_SYNTHETIC_NONPROMOTING"
    assert result.independently_verified is False
    assert result.model_origin_attested is False
    assert result.training_eligible is False
    assert "receipt" not in asdict(result)


def child(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-B", "-c", code, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )


def test_d01_new_store_first_local_match_is_nonpromoting() -> None:
    trial, store = new_trial("d01")
    h = harness(store)
    c = h.issue()
    assert len(c.nonce) == 64
    check(h.evaluate(h.candidate_submission(c)), "local_match", "synthetic_state_match_only")
    assert DurableReplayStore.open_existing(ROOT, trial)


def test_d02_replay_rejected_after_distinct_process_restart() -> None:
    trial, store = new_trial("d02")
    h = harness(store)
    c = h.issue()
    report = h.candidate_submission(c)
    check(h.evaluate(report), "local_match", "synthetic_state_match_only")
    code = """
import json, os, sys
from pathlib import Path
from huyawo_adapt.agentic.verifier.durable_replay import DurableReplayStore
from huyawo_adapt.agentic.verifier.local_synthetic import LocalSyntheticHarness, CandidateSubmission
root, trial, payload = Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
store = DurableReplayStore.open_existing(root, trial)
h = LocalSyntheticHarness(task_id='synthetic:task-one', episode_id='synthetic:episode-one',
 issuer_id='synthetic:issuer-local', private_expected_state='synthetic:done',
 captured_final_state='synthetic:done', captured_transcript='synthetic:fake-transcript',
 replay_store=store)
result = h.evaluate(CandidateSubmission(**payload))
print(json.dumps({'pid': os.getpid(), 'disposition': result.disposition,
                  'reason': result.reason, 'trusted': result.training_eligible}))
"""
    output = json.loads(child(code, str(ROOT), trial, json.dumps(asdict(report))).stdout)
    assert output == {
        "pid": output["pid"],
        "disposition": "rejected",
        "reason": "nonce_replayed",
        "trusted": False,
    }
    assert output["pid"] != os.getpid()


def test_d03_unique_nonce_across_restart_and_forced_collision() -> None:
    trial, store = new_trial("d03")
    h = harness(store)
    first = h.issue()
    check(h.evaluate(h.candidate_submission(first)), "local_match", "synthetic_state_match_only")
    code = """
import sys
from pathlib import Path
from huyawo_adapt.agentic.verifier.durable_replay import DurableReplayStore
from huyawo_adapt.agentic.verifier.local_synthetic import LocalSyntheticHarness
s = DurableReplayStore.open_existing(Path(sys.argv[1]), sys.argv[2])
h = LocalSyntheticHarness(task_id='synthetic:task-one', episode_id='synthetic:episode-one',
 issuer_id='synthetic:issuer-local', private_expected_state='synthetic:done',
 captured_final_state='synthetic:done', captured_transcript='synthetic:fake-transcript',
 replay_store=s)
print(h.issue().nonce)
"""
    second = child(code, str(ROOT), trial).stdout.strip()
    assert second != first.nonce
    with patch(
        "huyawo_adapt.agentic.verifier.durable_replay.secrets.token_hex",
        return_value=first.nonce,
    ):
        with pytest.raises(DurableReplayUnavailableError):
            h.issue()
    check(h.evaluate(h.candidate_submission(first)), "rejected", "nonce_replayed")


def test_d04_task_episode_issuer_and_fixture_binding() -> None:
    _, store = new_trial("d04")
    h = harness(store)
    c = h.issue()
    for key in ("task_id", "episode_id", "issuer_id"):
        wrong = replace(h.candidate_submission(c), **{key: "synthetic:forged"})
        check(h.evaluate(wrong), "rejected", "identity_binding_conflict")
        c = h.issue()
    other = harness(store, private_expected_state="synthetic:changed")
    check(other.evaluate(other.candidate_submission(c)), "rejected", "identity_binding_conflict")
    check(h.evaluate(h.candidate_submission(c)), "local_match", "synthetic_state_match_only")


def test_d05_two_real_processes_at_most_one_match() -> None:
    trial, store = new_trial("d05")
    h = harness(store)
    c = h.issue()
    payload = json.dumps(asdict(h.candidate_submission(c)))
    code = """
import json, sys
from pathlib import Path
from huyawo_adapt.agentic.verifier.durable_replay import DurableReplayStore
from huyawo_adapt.agentic.verifier.local_synthetic import LocalSyntheticHarness, CandidateSubmission
s = DurableReplayStore.open_existing(Path(sys.argv[1]), sys.argv[2])
h = LocalSyntheticHarness(task_id='synthetic:task-one', episode_id='synthetic:episode-one',
 issuer_id='synthetic:issuer-local', private_expected_state='synthetic:done',
 captured_final_state='synthetic:done', captured_transcript='synthetic:fake-transcript',
 replay_store=s)
v = h.evaluate(CandidateSubmission(**json.loads(sys.argv[3])))
print(json.dumps([v.disposition,v.reason,v.training_eligible]))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    args = [sys.executable, "-B", "-c", code, str(ROOT), trial, payload]
    p1 = subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    p2 = subprocess.Popen(
        args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    one, err1 = p1.communicate(timeout=10)
    two, err2 = p2.communicate(timeout=10)
    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    results = [json.loads(one), json.loads(two)]
    assert all(r[2] is False for r in results)
    assert sum(r[0] == "local_match" for r in results) <= 1
    assert any(r[0] == "local_match" for r in results)
    assert all(r[0] in {"local_match", "rejected", "blocked"} for r in results)


def test_d06_committed_consume_survives_process_exit() -> None:
    trial, store = new_trial("d06")
    h = harness(store)
    c = h.issue()
    code = """
import os, sys
from pathlib import Path
from huyawo_adapt.agentic.verifier.durable_replay import DurableReplayStore
s = DurableReplayStore.open_existing(Path(sys.argv[1]), sys.argv[2])
result = s.consume(sys.argv[3], 'synthetic:task-one', 'synthetic:episode-one',
 'synthetic:issuer-local', sys.argv[4])
if result != 'consumed': os._exit(9)
os._exit(0)
"""
    child(code, str(ROOT), trial, c.nonce, h.local_fixture_fingerprint)
    reopened = harness(DurableReplayStore.open_existing(ROOT, trial))
    check(reopened.evaluate(reopened.candidate_submission(c)), "rejected", "nonce_replayed")


def test_d07_uncommitted_process_exit_recovers_cleanly() -> None:
    trial, store = new_trial("d07")
    h = harness(store)
    c = h.issue()
    code = """
import os, sqlite3, sys
from pathlib import Path
p = Path(sys.argv[1])/'runs'/'local-synthetic-replay'/sys.argv[2]/'replay.sqlite3'
c = sqlite3.connect(f'file:{p}?mode=rw', uri=True, isolation_level=None)
c.execute('BEGIN IMMEDIATE')
c.execute("UPDATE challenges SET state='consumed' WHERE nonce=?", (sys.argv[3],))
os._exit(0)
"""
    child(code, str(ROOT), trial, c.nonce)
    reopened = harness(DurableReplayStore.open_existing(ROOT, trial))
    check(
        reopened.evaluate(reopened.candidate_submission(c)),
        "local_match",
        "synthetic_state_match_only",
    )
    check(reopened.evaluate(reopened.candidate_submission(c)), "rejected", "nonce_replayed")


def test_d08_missing_corrupt_incompatible_readonly_and_busy_store_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial, store = new_trial("d08")
    h = harness(store)
    c = h.issue()
    database = ROOT / "runs" / "local-synthetic-replay" / trial / "replay.sqlite3"
    original_connect = sqlite3.connect

    def readonly_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        return original_connect(
            f"file:{database}?mode=ro", uri=True, timeout=0.2, isolation_level=None
        )

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite3, "connect", readonly_connect)
        check(h.evaluate(h.candidate_submission(c)), "blocked", "replay_store_unavailable")
    con = original_connect(database, isolation_level=None)
    con.execute("UPDATE metadata SET schema_version=77")
    con.close()
    check(h.evaluate(h.candidate_submission(c)), "blocked", "replay_store_unavailable")
    con = original_connect(database, isolation_level=None)
    con.execute("UPDATE metadata SET schema_version=1")
    con.close()
    database.write_bytes(b"not a sqlite database")
    check(h.evaluate(h.candidate_submission(c)), "blocked", "replay_store_unavailable")
    database.unlink()
    check(h.evaluate(h.candidate_submission(c)), "blocked", "replay_store_unavailable")
    with pytest.raises(DurableReplayUnavailableError):
        DurableReplayStore.open_existing(ROOT, trial)
    _, locked = new_trial("d08-lock")
    other = harness(locked)
    locked_challenge = other.issue()
    lock = sqlite3.connect(locked._database, isolation_level=None)
    lock.execute("BEGIN IMMEDIATE")
    try:
        check(
            other.evaluate(other.candidate_submission(locked_challenge)),
            "blocked",
            "replay_store_unavailable",
        )
    finally:
        lock.execute("ROLLBACK")
        lock.close()


def test_d09_bad_paths_symlinks_and_existing_directory_rejected() -> None:
    trial, _ = new_trial("d09")
    with pytest.raises(DurableReplayUnavailableError):
        DurableReplayStore.create_new(ROOT, trial)
    for invalid in ("../outside", "../../workspace", "UPPER", "nested/path", ""):
        with pytest.raises(DurableReplayUnavailableError):
            DurableReplayStore.create_new(ROOT, invalid)
    with pytest.raises(DurableReplayUnavailableError):
        DurableReplayStore.open_existing(ROOT.parent, trial)
    evil_trial = f"{PREFIX}-d09-link-{secrets.token_hex(3)}"
    link = ROOT / "runs" / "local-synthetic-replay" / evil_trial
    link.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(DurableReplayUnavailableError):
        DurableReplayStore.create_new(ROOT, evil_trial)
    with pytest.raises(DurableReplayUnavailableError):
        DurableReplayStore.open_existing(ROOT, evil_trial)


def test_d10_default_mode_unchanged_and_no_eligibility_promotion() -> None:
    h = LocalSyntheticHarness(
        task_id="synthetic:task-one",
        episode_id="synthetic:episode-one",
        issuer_id="synthetic:issuer-local",
        private_expected_state="synthetic:done",
        captured_final_state="synthetic:done",
        captured_transcript="synthetic:fake-transcript",
    )
    c = h.issue()
    check(h.evaluate(h.candidate_submission(c)), "local_match", "synthetic_state_match_only")
    check(h.evaluate(h.candidate_submission(c)), "rejected", "nonce_replayed")
    _, store = new_trial("d10")
    durable = harness(store)
    challenge = durable.issue()
    check(
        durable.evaluate(
            replace(durable.candidate_submission(challenge), claimed_model_origin=True)
        ),
        "blocked",
        "model_origin_unverified",
    )
