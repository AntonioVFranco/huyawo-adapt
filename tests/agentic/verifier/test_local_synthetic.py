"""RFC-HAA-0021-inspired local simulations, never independent qualification."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from huyawo_adapt.agentic.verifier.local_synthetic import LocalSyntheticHarness


def fixture(**changes: object) -> LocalSyntheticHarness:
    fields: dict[str, Any] = dict(
        task_id="synthetic:task-alpha",
        episode_id="synthetic:episode-one",
        issuer_id="synthetic:issuer-local",
        private_expected_state="synthetic:done",
        captured_final_state="synthetic:done",
        captured_transcript="synthetic:fake-captured-transcript",
    )
    fields.update(changes)
    return LocalSyntheticHarness(**fields)


def check_blocked(verdict: object, disposition: str, reason: str) -> None:
    assert getattr(verdict, "disposition") == disposition
    assert getattr(verdict, "reason") == reason
    assert getattr(verdict, "authority_scope") == "LOCAL_SYNTHETIC_NONPROMOTING"
    assert getattr(verdict, "independently_verified") is False
    assert getattr(verdict, "model_origin_attested") is False
    assert getattr(verdict, "training_eligible") is False
    assert "receipt" not in asdict(verdict)


def test_t01_local_functional_control_only() -> None:
    harness = fixture()
    challenge = harness.issue()
    assert "synthetic:done" not in repr(challenge)
    check_blocked(
        harness.evaluate(harness.candidate_submission(challenge)),
        "local_match",
        "synthetic_state_match_only",
    )


def test_t02_forged_success_and_candidate_reference_rejected() -> None:
    harness = fixture()
    c1 = harness.issue()
    check_blocked(
        harness.evaluate(replace(harness.candidate_submission(c1), claimed_success=True)),
        "rejected",
        "candidate_asserted_success_or_reference",
    )
    c2 = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(
                harness.candidate_submission(c2),
                candidate_supplied_expected_state="synthetic:done",
            )
        ),
        "rejected",
        "candidate_asserted_success_or_reference",
    )


def test_t03_in_memory_replay_rejected_restart_not_tested() -> None:
    harness = fixture()
    challenge = harness.issue()
    report = harness.candidate_submission(challenge)
    check_blocked(harness.evaluate(report), "local_match", "synthetic_state_match_only")
    check_blocked(harness.evaluate(report), "rejected", "nonce_replayed")


def test_t04_task_episode_cross_binding_rejected() -> None:
    harness = fixture()
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(harness.candidate_submission(challenge), episode_id="synthetic:other")
        ),
        "rejected",
        "identity_binding_conflict",
    )


def test_t05_out_of_order_terminal_transition_rejected() -> None:
    harness = fixture()
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(replace(harness.candidate_submission(challenge), terminal_index=2)),
        "rejected",
        "invalid_terminal_transition",
    )


def test_t06_tampered_transcript_or_result_digest_rejected() -> None:
    harness = fixture()
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(harness.candidate_submission(challenge), transcript_digest="0" * 64)
        ),
        "rejected",
        "capture_digest_conflict",
    )
    other = harness.issue()
    check_blocked(
        harness.evaluate(replace(harness.candidate_submission(other), observed_digest="1" * 64)),
        "rejected",
        "capture_digest_conflict",
    )


def test_t07_fake_issuer_identity_rejected() -> None:
    harness = fixture()
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(harness.candidate_submission(challenge), issuer_id="synthetic:forged")
        ),
        "rejected",
        "identity_binding_conflict",
    )


def test_t08_candidate_private_reference_api_request_rejected() -> None:
    harness = fixture(private_expected_state="synthetic:fake-private-value")
    original = harness.local_fixture_fingerprint
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(harness.candidate_submission(challenge), request_private_reference=True)
        ),
        "rejected",
        "forbidden_candidate_operation",
    )
    assert harness.local_fixture_fingerprint == original
    assert "synthetic:fake-private-value" not in repr(challenge)


def test_t09_candidate_reference_and_audit_write_requests_rejected() -> None:
    harness = fixture()
    original = harness.local_fixture_fingerprint
    for changes in (dict(request_reference_write=True), dict(request_audit_write=True)):
        challenge = harness.issue()
        check_blocked(
            harness.evaluate(replace(harness.candidate_submission(challenge), **changes)),
            "rejected",
            "forbidden_candidate_operation",
        )
    assert harness.local_fixture_fingerprint == original
    changed = harness.issue()
    harness._reference = "synthetic:illicit-change"
    check_blocked(
        harness.evaluate(harness.candidate_submission(changed)),
        "blocked",
        "local_fixture_integrity_conflict",
    )


def test_t10_stale_verifier_configuration_blocks() -> None:
    harness = fixture(issuer_fresh=False)
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(harness.candidate_submission(challenge)),
        "blocked",
        "verifier_unavailable_or_stale",
    )


def test_t11_verifier_or_capture_unavailable_blocks() -> None:
    for config, reason in (
        (dict(verifier_available=False), "verifier_unavailable_or_stale"),
        (dict(capture_available=False), "capture_unavailable"),
    ):
        harness = fixture(**config)
        challenge = harness.issue()
        check_blocked(harness.evaluate(harness.candidate_submission(challenge)), "blocked", reason)


def test_t12_fabricated_model_origin_cannot_promote() -> None:
    harness = fixture()
    challenge = harness.issue()
    check_blocked(
        harness.evaluate(
            replace(harness.candidate_submission(challenge), claimed_model_origin=True)
        ),
        "blocked",
        "model_origin_unverified",
    )
