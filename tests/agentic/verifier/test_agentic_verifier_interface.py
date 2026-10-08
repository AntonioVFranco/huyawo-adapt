"""Pure M5 verifier contract, fidelity, and adversarial regression tests."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.contracts import (
    AgentEnvironmentIdentity,
    AgentTaskIdentity,
    AgentVerifierIdentity,
)
from huyawo_adapt.agentic.trajectory import (
    AgentFailureRecord,
    AgentStateTransitionRecord,
    AgentTrajectory,
    AgentTrajectoryPayload,
    AgentTrajectoryStep,
)
from huyawo_adapt.agentic.verifier import (
    AgentVerifier,
    AgentVerifierRequest,
    AgentVerifierResult,
    ExactStateReferenceVerifier,
    ExpectedStateReference,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload(text: str) -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(media_type="text/plain", content=text, sha256=digest(text))


def build_request(
    *,
    expected: str = "complete",
    observed: str = "complete",
    include_observation: bool = True,
    include_transition: bool = True,
    terminal_status: str = "success",
    transition_digest: str | None = None,
    verifier_kind: str = "deterministic",
) -> AgentVerifierRequest:
    environment = AgentEnvironmentIdentity(
        environment_id="fixture-env",
        definition_sha256=digest("environment-definition"),
        state_schema_sha256=digest("state-schema"),
        reset_semantics_sha256=digest("reset"),
        transition_semantics_sha256=digest("transition"),
        terminal_conditions_sha256=digest("terminal"),
    )
    kwargs: dict[str, Any] = {}
    if verifier_kind != "deterministic":
        kwargs = {
            "judge_model_fingerprint": digest("judge-model"),
            "judge_prompt_sha256": digest("judge-prompt"),
        }
    verifier = AgentVerifierIdentity(
        verifier_id="exact-fixture-verifier",
        verifier_kind=verifier_kind,
        implementation_sha256=digest("verifier-implementation"),
        configuration_sha256=digest("verifier-configuration"),
        output_schema_sha256=digest("verifier-output-schema"),
        **kwargs,
    )
    task = AgentTaskIdentity(
        task_id="fixture-task",
        environment_fingerprint=environment.fingerprint(),
        verifier_fingerprint=verifier.fingerprint(),
        task_definition_sha256=digest("task-definition"),
        initial_state_sha256=digest("initial-state"),
        allowed_actions_sha256=digest("allowed-actions"),
        success_criteria_sha256=digest("success-criteria"),
    )
    transition = (
        AgentStateTransitionRecord(
            before_state_sha256=digest("initial-state"),
            after_state_sha256=transition_digest or digest(observed),
            transition_evidence_sha256=digest("transition-evidence"),
        )
        if include_transition
        else None
    )
    failure = (
        AgentFailureRecord(failure_type="agent-error", retryable=False)
        if terminal_status in ("failure", "error")
        else None
    )
    step = AgentTrajectoryStep(
        step_index=0,
        observation=payload("agent observation"),
        model_input=payload("agent input"),
        generated_output=payload("agent output"),
        terminal_status=terminal_status,
        state_transition=transition,
        provenance_sha256=digest("step-provenance"),
        failure=failure,
    )
    trajectory = AgentTrajectory(
        trajectory_id="captured-fixture-trajectory",
        model_fingerprint=digest("model"),
        tokenizer_fingerprint=digest("tokenizer"),
        harness_fingerprint=digest("harness"),
        tool_set_fingerprint=digest("toolset"),
        environment_fingerprint=environment.fingerprint(),
        verifier_fingerprint=verifier.fingerprint(),
        task_fingerprint=task.fingerprint(),
        runtime_fingerprint=digest("runtime"),
        rollout_policy_fingerprint=digest("rollout"),
        steps=(step,),
        capture_provenance_sha256=digest("capture-provenance"),
    )
    reference = ExpectedStateReference(
        task_fingerprint=task.fingerprint(),
        success_criteria_sha256=task.success_criteria_sha256,
        expected_final_state=payload(expected),
        reference_provenance_sha256=digest("independent-reference-provenance"),
    )
    return AgentVerifierRequest(
        verifier_identity=verifier,
        task_identity=task,
        environment_identity=environment,
        trajectory=trajectory,
        reference=reference,
        observed_final_state=payload(observed) if include_observation else None,
        observation_provenance_sha256=(
            digest("observation-provenance") if include_observation else None
        ),
    )


@pytest.mark.parametrize("text", ["complete", "", " café\n", "line\nnext", "naïve", " "])
def test_exact_utf8_fixture_match_returns_pass(text: str) -> None:
    request = build_request(expected=text, observed=text)
    result = ExactStateReferenceVerifier().verify(request)
    assert (result.verdict, result.reason) == ("pass", "exact_state_match")
    assert result.authority_scope == "local_fixture_comparison_only"
    assert result.observed_final_state_sha256 == digest(text)
    assert result.observation_provenance_sha256 == request.observation_provenance_sha256


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        ("complete", "incomplete"),
        ("abc", "abc "),
        ("line\n", "line"),
        ("é", "e\u0301"),
        ('{"a":1,"b":2}', '{"b":2,"a":1}'),
    ],
)
def test_exact_utf8_mismatch_returns_fail(expected: str, observed: str) -> None:
    result = ExactStateReferenceVerifier().verify(
        build_request(expected=expected, observed=observed)
    )
    assert (result.verdict, result.reason) == ("fail", "exact_state_mismatch")


def test_claimed_agent_success_cannot_override_state_mismatch() -> None:
    result = ExactStateReferenceVerifier().verify(
        build_request(expected="done", observed="not-done", terminal_status="success")
    )
    assert result.verdict == "fail"


def test_missing_final_transition_is_blocked() -> None:
    result = ExactStateReferenceVerifier().verify(build_request(include_transition=False))
    assert (result.verdict, result.reason) == ("blocked", "missing_final_transition")


def test_missing_observation_is_blocked() -> None:
    result = ExactStateReferenceVerifier().verify(build_request(include_observation=False))
    assert (result.verdict, result.reason) == ("blocked", "missing_observation")
    assert result.observed_final_state_sha256 is None
    assert result.observation_provenance_sha256 is None


def test_missing_observation_and_transition_uses_stable_priority() -> None:
    result = ExactStateReferenceVerifier().verify(
        build_request(include_observation=False, include_transition=False)
    )
    assert (result.verdict, result.reason) == ("blocked", "missing_final_transition")


def test_transition_evidence_conflict_is_not_task_failure() -> None:
    result = ExactStateReferenceVerifier().verify(
        build_request(expected="match", observed="match", transition_digest=digest("forged"))
    )
    assert (result.verdict, result.reason) == ("blocked", "transition_evidence_conflict")


@pytest.mark.parametrize("terminal_status", ["failure", "error", "truncated"])
def test_terminal_non_success_equal_state_is_blocked(terminal_status: str) -> None:
    result = ExactStateReferenceVerifier().verify(build_request(terminal_status=terminal_status))
    assert (result.verdict, result.reason) == ("blocked", "terminal_evidence_conflict")


@pytest.mark.parametrize("terminal_status", ["failure", "error", "truncated"])
def test_terminal_non_success_and_mismatched_state_fails(terminal_status: str) -> None:
    result = ExactStateReferenceVerifier().verify(
        build_request(terminal_status=terminal_status, expected="done", observed="not done")
    )
    assert (result.verdict, result.reason) == ("fail", "exact_state_mismatch")


def test_verifier_implements_protocol_and_preserves_fingerprints() -> None:
    request = build_request()
    verifier: AgentVerifier = ExactStateReferenceVerifier()
    result = verifier.verify(request)
    assert result.verifier_fingerprint == request.verifier_identity.fingerprint()
    assert result.task_fingerprint == request.task_identity.fingerprint()
    assert result.environment_fingerprint == request.environment_identity.fingerprint()
    assert result.trajectory_fingerprint == request.trajectory.fingerprint()
    assert result.reference_fingerprint == request.reference.fingerprint()


def test_result_deterministic_bytes_fingerprint_and_roundtrip() -> None:
    request = build_request(expected="café\n", observed="café\n")
    verifier = ExactStateReferenceVerifier()
    first = verifier.verify(request)
    second = verifier.verify(request)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fingerprint() == second.fingerprint()
    assert AgentVerifierResult.from_canonical_json(first.canonical_bytes()) == first
    assert AgentVerifierRequest.from_canonical_json(request.canonical_bytes()) == request
    assert (
        ExpectedStateReference.from_canonical_json(request.reference.canonical_bytes())
        == request.reference
    )


def test_contract_models_frozen_and_forbid_unknown_fields() -> None:
    request = build_request()
    result = ExactStateReferenceVerifier().verify(request)
    for item in (request, result, request.reference):
        with pytest.raises(ValidationError):
            item.__class__.model_validate({**item.model_dump(mode="python"), "unexpected": 1})
        with pytest.raises(ValidationError):
            item.contract_type = "tampered"


def test_refuses_raw_dictionary_at_public_entrypoint() -> None:
    with pytest.raises(TypeError, match="AgentVerifierRequest"):
        bad_request = build_request().model_dump(mode="python")
        ExactStateReferenceVerifier().verify(bad_request)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "identity_name", "new_identity"),
    [
        ("task_identity", "environment_fingerprint", "foreign-env"),
        ("task_identity", "verifier_fingerprint", "foreign-verifier"),
        ("trajectory", "environment_fingerprint", "foreign-env"),
        ("trajectory", "verifier_fingerprint", "foreign-verifier"),
        ("trajectory", "task_fingerprint", "foreign-task"),
        ("reference", "task_fingerprint", "foreign-task"),
        ("reference", "success_criteria_sha256", "foreign-criteria"),
    ],
)
def test_identity_swaps_are_rejected(field: str, identity_name: str, new_identity: str) -> None:
    request = build_request()
    tampered_nested = getattr(request, field).model_copy(
        update={identity_name: digest(new_identity)}
    )
    with pytest.raises(ValidationError):
        AgentVerifierRequest.model_validate(
            {**request.model_dump(mode="python"), field: tampered_nested}
        )


@pytest.mark.parametrize("kind", ["llm_judge", "hybrid"])
def test_judge_or_hybrid_identity_rejected_by_reference_verifier(kind: str) -> None:
    with pytest.raises(ValidationError, match="deterministic"):
        build_request(verifier_kind=kind)


@pytest.mark.parametrize(
    ("observed", "provenance"),
    [(None, digest("orphan")), (payload("present"), None)],
)
def test_observation_and_provenance_are_atomic(
    observed: AgentTrajectoryPayload | None,
    provenance: str | None,
) -> None:
    request = build_request()
    with pytest.raises(ValidationError, match="together"):
        AgentVerifierRequest.model_validate(
            {
                **request.model_dump(mode="python"),
                "observed_final_state": observed,
                "observation_provenance_sha256": provenance,
            }
        )


def test_invalid_observation_payload_sha_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentTrajectoryPayload(media_type="text/plain", content="original", sha256=digest("fake"))


def test_invalid_reference_provenance_digest_is_rejected() -> None:
    request = build_request()
    with pytest.raises(ValidationError):
        ExpectedStateReference.model_validate(
            {
                **request.reference.model_dump(mode="python"),
                "reference_provenance_sha256": "garbage",
            }
        )


@pytest.mark.parametrize(
    ("verdict", "reason", "observed_digest"),
    [
        ("pass", "exact_state_mismatch", digest("state")),
        ("fail", "exact_state_match", digest("state")),
        ("blocked", "exact_state_match", digest("state")),
        ("pass", "exact_state_match", None),
        ("fail", "exact_state_mismatch", None),
        ("blocked", "missing_observation", digest("state")),
    ],
)
def test_result_invalid_verdict_combinations_are_rejected(
    verdict: str, reason: str, observed_digest: str | None
) -> None:
    base = ExactStateReferenceVerifier().verify(build_request())
    data = base.model_dump(mode="python")
    data.update(
        verdict=verdict,
        reason=reason,
        observed_final_state_sha256=observed_digest,
        observation_provenance_sha256=(digest("observation") if observed_digest else None),
    )
    with pytest.raises(ValidationError):
        AgentVerifierResult.model_validate(data)


def test_result_cannot_claim_reward_or_benchmark_authority() -> None:
    base = ExactStateReferenceVerifier().verify(build_request())
    for forbidden in ("reward", "benchmark_eligible", "training_label"):
        with pytest.raises(ValidationError):
            AgentVerifierResult.model_validate({**base.model_dump(mode="python"), forbidden: True})
    with pytest.raises(ValidationError):
        AgentVerifierResult.model_validate(
            {**base.model_dump(mode="python"), "authority_scope": "verified_training_reward"}
        )


def test_pure_verification_preserves_existing_immutable_objects() -> None:
    request = build_request()
    before = request.canonical_bytes()
    first = ExactStateReferenceVerifier().verify(request)
    assert request.canonical_bytes() == before
    assert first.verdict == "pass"
    assert request.trajectory.steps[-1].terminal_status == "success"
    assert request.reference.expected_final_state.content == "complete"


def test_verifier_revalidates_model_copy_mutations() -> None:
    request = build_request()
    forged = request.model_copy(update={"observation_provenance_sha256": None})
    with pytest.raises(ValidationError):
        ExactStateReferenceVerifier().verify(forged)
