"""In-memory synthetic task checks without independent attestation authority.

This code is not a sandbox, issuer, signer, trusted receipt or training gate.
All fixture contents and identity labels are fake and same-admin controlled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Literal

from huyawo_adapt.agentic.verifier.durable_replay import (
    DurableReplayStore,
    DurableReplayUnavailableError,
)

LOCAL_AUTHORITY: Final[Literal["LOCAL_SYNTHETIC_NONPROMOTING"]] = "LOCAL_SYNTHETIC_NONPROMOTING"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(values: tuple[str, ...]) -> str:
    return _digest(json.dumps(values, ensure_ascii=True, separators=(",", ":")))


@dataclass(frozen=True, slots=True)
class PublicChallenge:
    task_id: str
    episode_id: str
    nonce: str
    issuer_id: str


@dataclass(frozen=True, slots=True)
class CandidateSubmission:
    task_id: str
    episode_id: str
    nonce: str
    issuer_id: str
    terminal_index: int
    observed_final_state: str
    observed_digest: str
    transcript_digest: str
    claimed_success: bool = False
    claimed_model_origin: bool = False
    candidate_supplied_expected_state: str | None = None
    request_private_reference: bool = False
    request_reference_write: bool = False
    request_audit_write: bool = False


@dataclass(frozen=True, slots=True)
class LocalAssessment:
    disposition: Literal["local_match", "rejected", "blocked"]
    reason: str
    authority_scope: Literal["LOCAL_SYNTHETIC_NONPROMOTING"] = LOCAL_AUTHORITY
    independently_verified: Literal[False] = False
    model_origin_attested: Literal[False] = False
    training_eligible: Literal[False] = False


class LocalSyntheticHarness:
    """Single-process fake-fixture comparator, not protected reference custody."""

    def __init__(
        self,
        *,
        task_id: str,
        episode_id: str,
        issuer_id: str,
        private_expected_state: str,
        captured_final_state: str,
        captured_transcript: str,
        verifier_available: bool = True,
        capture_available: bool = True,
        issuer_fresh: bool = True,
        replay_store: DurableReplayStore | None = None,
    ) -> None:
        values = (
            task_id,
            episode_id,
            issuer_id,
            private_expected_state,
            captured_final_state,
            captured_transcript,
        )
        if any(type(value) is not str or not value.startswith("synthetic:") for value in values):
            raise ValueError("all local fixture values must be synthetic-prefixed strings")
        if any(
            type(flag) is not bool for flag in (verifier_available, capture_available, issuer_fresh)
        ):
            raise TypeError("availability flags must be exact booleans")
        if replay_store is not None and type(replay_store) is not DurableReplayStore:
            raise TypeError("replay_store must be an explicit DurableReplayStore")
        self._task_id = task_id
        self._episode_id = episode_id
        self._issuer_id = issuer_id
        self._reference = private_expected_state
        self._captured = captured_final_state
        self._transcript = captured_transcript
        self._transcript_digest = _digest(captured_transcript)
        self._observed_digest = _digest(captured_final_state)
        self._baseline_digest = _fingerprint(values)
        self._verifier_available = verifier_available
        self._capture_available = capture_available
        self._issuer_fresh = issuer_fresh
        self._counter = 0
        self._issued: dict[str, PublicChallenge] = {}
        self._used: set[str] = set()
        self._replay_store = replay_store

    @property
    def local_fixture_fingerprint(self) -> str:
        """Only a local digest, never an independently protected audit record."""
        return _fingerprint(
            (
                self._task_id,
                self._episode_id,
                self._issuer_id,
                self._reference,
                self._captured,
                self._transcript,
            )
        )

    def issue(self) -> PublicChallenge:
        if self._replay_store is not None:
            nonce = self._replay_store.issue(
                self._task_id, self._episode_id, self._issuer_id, self._baseline_digest
            )
            return PublicChallenge(self._task_id, self._episode_id, nonce, self._issuer_id)
        self._counter += 1
        nonce = _fingerprint(
            (
                "synthetic:local-nonce-v1",
                self._task_id,
                self._episode_id,
                self._issuer_id,
                str(self._counter),
            )
        )
        challenge = PublicChallenge(self._task_id, self._episode_id, nonce, self._issuer_id)
        self._issued[nonce] = challenge
        return challenge

    def candidate_submission(self, challenge: PublicChallenge) -> CandidateSubmission:
        """Make a synthetic happy-path input; this is not real model capture."""
        return CandidateSubmission(
            task_id=challenge.task_id,
            episode_id=challenge.episode_id,
            nonce=challenge.nonce,
            issuer_id=challenge.issuer_id,
            terminal_index=1,
            observed_final_state=self._captured,
            observed_digest=self._observed_digest,
            transcript_digest=self._transcript_digest,
        )

    @staticmethod
    def _result(
        disposition: Literal["local_match", "rejected", "blocked"], reason: str
    ) -> LocalAssessment:
        return LocalAssessment(disposition=disposition, reason=reason)

    def evaluate(self, submission: CandidateSubmission) -> LocalAssessment:
        if type(submission) is not CandidateSubmission:
            return self._result("rejected", "invalid_submission_type")
        if self.local_fixture_fingerprint != self._baseline_digest:
            return self._result("blocked", "local_fixture_integrity_conflict")
        if not self._verifier_available or not self._issuer_fresh:
            return self._result("blocked", "verifier_unavailable_or_stale")
        if not self._capture_available:
            return self._result("blocked", "capture_unavailable")
        if self._replay_store is not None:
            try:
                consumed = self._replay_store.consume(
                    submission.nonce,
                    self._task_id,
                    self._episode_id,
                    self._issuer_id,
                    self._baseline_digest,
                )
            except DurableReplayUnavailableError:
                return self._result("blocked", "replay_store_unavailable")
            if consumed == "unknown":
                return self._result("rejected", "unknown_nonce")
            if consumed == "replayed":
                return self._result("rejected", "nonce_replayed")
            if consumed == "binding_conflict":
                return self._result("rejected", "identity_binding_conflict")
            challenge = PublicChallenge(
                self._task_id, self._episode_id, submission.nonce, self._issuer_id
            )
        else:
            if submission.nonce not in self._issued:
                return self._result("rejected", "unknown_nonce")
            if submission.nonce in self._used:
                return self._result("rejected", "nonce_replayed")
            self._used.add(submission.nonce)
            challenge = self._issued[submission.nonce]
        if (
            type(submission.task_id) is not str
            or type(submission.episode_id) is not str
            or type(submission.issuer_id) is not str
            or (submission.task_id, submission.episode_id, submission.issuer_id)
            != (challenge.task_id, challenge.episode_id, challenge.issuer_id)
        ):
            return self._result("rejected", "identity_binding_conflict")
        if any(
            type(flag) is not bool
            for flag in (
                submission.claimed_success,
                submission.claimed_model_origin,
                submission.request_private_reference,
                submission.request_reference_write,
                submission.request_audit_write,
            )
        ):
            return self._result("rejected", "invalid_claim_type")
        if submission.candidate_supplied_expected_state is not None or submission.claimed_success:
            return self._result("rejected", "candidate_asserted_success_or_reference")
        if (
            submission.request_private_reference
            or submission.request_reference_write
            or submission.request_audit_write
        ):
            return self._result("rejected", "forbidden_candidate_operation")
        if type(submission.terminal_index) is not int or submission.terminal_index != 1:
            return self._result("rejected", "invalid_terminal_transition")
        if (
            type(submission.transcript_digest) is not str
            or submission.transcript_digest != self._transcript_digest
            or type(submission.observed_final_state) is not str
            or type(submission.observed_digest) is not str
            or submission.observed_digest != _digest(submission.observed_final_state)
            or submission.observed_digest != self._observed_digest
        ):
            return self._result("rejected", "capture_digest_conflict")
        if submission.claimed_model_origin:
            return self._result("blocked", "model_origin_unverified")
        if self._captured != self._reference:
            return self._result("rejected", "local_state_mismatch")
        return self._result("local_match", "synthetic_state_match_only")
