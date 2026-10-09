"""M9-B2 caller-origin intake and negative-only eligibility boundary tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import ValidationError

from huyawo_adapt.agentic.training.provenance import (
    AgentDemonstrationIntakeAssessment,
    AgentDemonstrationOriginClaim,
    assess_demonstration_intake,
)
from huyawo_adapt.agentic.training.sft import (
    AgentSFTDecisionProjection,
    project_trajectory_decision,
)
from huyawo_adapt.agentic.trajectory.schema import (
    AgentTrajectory,
    AgentTrajectoryPayload,
    AgentTrajectoryStep,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload(text: str) -> AgentTrajectoryPayload:
    return AgentTrajectoryPayload(media_type="text/plain", content=text, sha256=_digest(text))


def _trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        trajectory_id="intake-trajectory",
        model_fingerprint=_digest("runtime-model"),
        tokenizer_fingerprint=_digest("tokenizer"),
        harness_fingerprint=_digest("harness"),
        tool_set_fingerprint=_digest("tools"),
        environment_fingerprint=_digest("environment"),
        verifier_fingerprint=_digest("verifier"),
        task_fingerprint=_digest("task"),
        runtime_fingerprint=_digest("runtime"),
        rollout_policy_fingerprint=_digest("policy"),
        capture_provenance_sha256=_digest("capture"),
        steps=(
            AgentTrajectoryStep(
                step_index=0,
                observation=_payload("public observation"),
                model_input=_payload("user input"),
                model_input_token_ids=(11, 12),
                generated_output=_payload("answer"),
                generated_token_ids=(13, 14),
                provenance_sha256=_digest("step"),
                terminal_status="success",
            ),
        ),
    )


def _projection(trajectory: AgentTrajectory | None = None) -> AgentSFTDecisionProjection:
    return project_trajectory_decision(trajectory or _trajectory(), 0, _digest("projection-policy"))


def _claim(kind: str = "model_runtime", **changes: Any) -> AgentDemonstrationOriginClaim:
    trajectory = _trajectory()
    fields: dict[str, Any] = {
        "source_kind": kind,
        "source_identity_sha256": trajectory.model_fingerprint,
        "source_revision_sha256": _digest("source-revision"),
        "source_content_sha256": trajectory.steps[0].generated_output.sha256,
        "capture_event_sha256": trajectory.capture_provenance_sha256,
        "usage_rights_evidence_sha256": _digest("caller-claimed-rights"),
        "task_fingerprint": trajectory.task_fingerprint,
        "source_family_fingerprint": _digest("family"),
        "root_source_fingerprint": _digest("root"),
        "trajectory_fingerprint": trajectory.fingerprint(),
    }
    if kind == "teacher_model":
        fields["teacher_model_fingerprint"] = trajectory.model_fingerprint
    elif kind == "expert_authored":
        fields["trajectory_fingerprint"] = None
        fields["expert_artifact_sha256"] = _digest("expert-artifact")
        fields["source_identity_sha256"] = _digest("expert-author")
    fields.update(changes)
    return AgentDemonstrationOriginClaim(**fields)


@pytest.mark.parametrize("kind", ["model_runtime", "teacher_model", "expert_authored"])
def test_valid_declared_origins_always_block_without_attestation(kind: str) -> None:
    claim = _claim(kind)
    result = assess_demonstration_intake(claim)
    assert result.disposition == "blocked"
    assert result.reason == "origin_not_attested"
    assert result.source_kind == kind
    assert result.authority_scope == "untrusted_intake_screen_only"
    assert claim.authority_scope == "caller_declared_intake_only"
    assert result.claim_fingerprint == claim.fingerprint()
    assert result.trajectory_fingerprint is None
    assert result.projection_fingerprint is None
    assert "training_eligible" not in AgentDemonstrationIntakeAssessment.model_fields
    assert "success" not in AgentDemonstrationIntakeAssessment.model_fields


@pytest.mark.parametrize("kind", ["model_runtime", "teacher_model"])
def test_valid_model_and_teacher_lineage_still_block(kind: str) -> None:
    trajectory = _trajectory()
    claim = _claim(kind)
    projection = _projection(trajectory)
    result = assess_demonstration_intake(claim, trajectory, projection)
    assert result.disposition == "blocked"
    assert result.reason == "origin_not_attested"
    assert result.trajectory_fingerprint == trajectory.fingerprint()
    assert result.projection_fingerprint == projection.fingerprint()
    assert result.claim_fingerprint == claim.fingerprint()
    assert (
        result.fingerprint()
        == assess_demonstration_intake(claim, trajectory, projection).fingerprint()
    )


def test_claim_and_assessment_canonical_round_trip() -> None:
    claim = _claim()
    assessment = assess_demonstration_intake(claim, _trajectory(), _projection())
    assert claim == AgentDemonstrationOriginClaim.from_canonical_json(claim.canonical_json())
    assert assessment == AgentDemonstrationIntakeAssessment.from_canonical_json(
        assessment.canonical_json()
    )
    assert assessment.fingerprint() == hashlib.sha256(assessment.canonical_bytes()).hexdigest()
    assert json.loads(claim.canonical_json())["source_kind"] == "model_runtime"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_identity_sha256", "UPPERCASE"),
        ("source_revision_sha256", "0" * 63),
        ("source_content_sha256", "G" * 64),
        ("capture_event_sha256", 7),
        ("usage_rights_evidence_sha256", None),
        ("task_fingerprint", True),
        ("source_family_fingerprint", " " + "1" * 64),
        ("root_source_fingerprint", "2" * 64 + " "),
        ("trajectory_fingerprint", False),
        ("authority_scope", "attested"),
        ("contract_type", "other"),
        ("source_kind", "model"),
        ("unknown_field", "unexpected"),
        ("training_eligible", True),
        ("verified", True),
        ("success", True),
        ("schema_version", "2.0"),
    ],
)
def test_invalid_claim_fields_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        AgentDemonstrationOriginClaim.model_validate({**_claim().model_dump(), field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"trajectory_fingerprint": None},
        {"expert_artifact_sha256": _digest("expert")},
        {"teacher_model_fingerprint": _digest("teacher")},
        {"source_kind": "expert_authored"},
        {"source_kind": "teacher_model"},
    ],
)
def test_model_origin_does_not_become_expert_or_teacher(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentDemonstrationOriginClaim.model_validate({**_claim().model_dump(), **change})


@pytest.mark.parametrize(
    "change",
    [
        {"trajectory_fingerprint": _digest("fabricated")},
        {"expert_artifact_sha256": None},
        {"teacher_model_fingerprint": _digest("teacher")},
        {"source_kind": "model_runtime"},
    ],
)
def test_expert_origin_disallows_fabricated_model_lineage(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentDemonstrationOriginClaim.model_validate(
            {**_claim("expert_authored").model_dump(), **change}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"teacher_model_fingerprint": None},
        {"teacher_model_fingerprint": _digest("not-the-source")},
        {"trajectory_fingerprint": None},
        {"expert_artifact_sha256": _digest("expert")},
        {"source_kind": "model_runtime"},
    ],
)
def test_teacher_origin_requires_explicit_teacher_identity(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AgentDemonstrationOriginClaim.model_validate(
            {**_claim("teacher_model").model_dump(), **change}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_identity_sha256", _digest("model-other")),
        ("task_fingerprint", _digest("task-other")),
        ("capture_event_sha256", _digest("capture-other")),
        ("trajectory_fingerprint", _digest("trajectory-other")),
        ("source_content_sha256", _digest("content-other")),
    ],
)
def test_actual_m3_link_conflicts_rejected(field: str, value: str) -> None:
    claim = _claim(**{field: value})
    result = assess_demonstration_intake(claim, _trajectory())
    assert result.disposition == "rejected"
    assert result.reason == "source_linkage_conflict"


def test_teacher_m3_model_identity_conflict_rejected() -> None:
    trajectory = _trajectory().model_copy(update={"model_fingerprint": _digest("other-teacher")})
    result = assess_demonstration_intake(_claim("teacher_model"), trajectory)
    assert result.disposition == "rejected"


def test_expert_cannot_claim_passed_model_trajectory() -> None:
    claim = _claim("expert_authored")
    result = assess_demonstration_intake(claim, _trajectory(), _projection())
    assert result.disposition == "rejected"
    assert result.reason == "model_vs_expert_claim_conflict"
    assert result.trajectory_fingerprint is None
    assert result.projection_fingerprint is None


def test_projection_requires_matching_canonical_trajectory() -> None:
    result = assess_demonstration_intake(_claim(), projection=_projection())
    assert result.disposition == "rejected"
    assert result.reason == "source_linkage_conflict"
    assert result.trajectory_fingerprint is None
    assert result.projection_fingerprint == _projection().fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_fingerprint", _digest("different-trajectory")),
        ("step_provenance_sha256", _digest("different-provenance")),
        ("model_input_sha256", _digest("different-input")),
        ("generated_output_sha256", _digest("different-output")),
        ("model_fingerprint", _digest("different-model")),
        ("tokenizer_fingerprint", _digest("different-tokenizer")),
        ("harness_fingerprint", _digest("different-harness")),
        ("tool_set_fingerprint", _digest("different-tools")),
        ("environment_fingerprint", _digest("different-environment")),
        ("verifier_fingerprint", _digest("different-verifier")),
        ("task_fingerprint", _digest("different-task")),
        ("runtime_fingerprint", _digest("different-runtime")),
        ("rollout_policy_fingerprint", _digest("different-policy")),
        ("capture_provenance_sha256", _digest("different-capture")),
    ],
)
def test_projection_tampered_identity_link_rejected(field: str, value: str) -> None:
    projection = _projection().model_copy(update={field: value})
    # The mismatched provenance is structurally plausible for a standalone record.
    result = assess_demonstration_intake(_claim(), _trajectory(), projection)
    assert result.disposition == "rejected"
    assert result.reason == "source_linkage_conflict"


def test_projection_parent_lineage_pair_conflict_rejected() -> None:
    projection = _projection().model_copy(
        update={
            "parent_trajectory_fingerprint": _digest("other-parent"),
            "transformation_fingerprint": _digest("other-transformation"),
        }
    )
    result = assess_demonstration_intake(_claim(), _trajectory(), projection)
    assert result.disposition == "rejected"
    assert result.reason == "source_linkage_conflict"


def test_projection_retokenization_without_digest_change_rejected() -> None:
    original = _projection()
    modified = original.model_copy(
        update={
            "input_ids": (15, 12, 13, 14),
            "labels": (-100, -100, 13, 14),
        }
    )
    result = assess_demonstration_intake(_claim(), _trajectory(), modified)
    assert result.disposition == "rejected"
    assert result.reason == "source_linkage_conflict"


def test_projection_unsupported_step_index_rejected() -> None:
    modified = _projection().model_copy(update={"step_index": 1})
    assert assess_demonstration_intake(_claim(), _trajectory(), modified).disposition == "rejected"


@pytest.mark.parametrize(
    "field",
    ["model_fingerprint", "task_fingerprint"],
)
def test_altered_m3_trajectory_cannot_reuse_intake_claim(field: str) -> None:
    altered = _trajectory().model_copy(update={field: _digest("different")})
    assert assess_demonstration_intake(_claim(), altered).disposition == "rejected"


def test_m3_declared_success_never_becomes_trusted() -> None:
    assert _trajectory().steps[-1].terminal_status == "success"
    assessment = assess_demonstration_intake(_claim(), _trajectory(), _projection())
    assert assessment.disposition == "blocked"
    with pytest.raises(ValidationError):
        AgentDemonstrationIntakeAssessment.model_validate(
            {**assessment.model_dump(), "training_eligible": True}
        )


def test_m7_terminated_m5_fixture_and_m8_scripted_are_not_authority() -> None:
    assessment = assess_demonstration_intake(_claim())
    for claim in (
        "terminated",
        "local_fixture_comparison_only",
        "scripted_local_replay_only",
    ):
        with pytest.raises(ValidationError):
            AgentDemonstrationIntakeAssessment.model_validate(
                {**assessment.model_dump(), "attestation": claim}
            )
    assert assessment.disposition == "blocked"


def test_declared_rights_or_capture_digests_are_not_validated_rights() -> None:
    claim = _claim(
        usage_rights_evidence_sha256=_digest("fictional-rights"),
        source_revision_sha256=_digest("fictional-revision"),
    )
    result = assess_demonstration_intake(claim, _trajectory(), _projection())
    assert result.disposition == "blocked"
    assert result.authority_scope == "untrusted_intake_screen_only"


@pytest.mark.parametrize(
    "change",
    [
        {"disposition": "eligible"},
        {"disposition": "approved"},
        {"reason": "trusted_outcome"},
        {"authority_scope": "trusted_origin"},
        {"training_eligible": True},
        {"independent_verifier_pass": True},
        {"claim_fingerprint": "A" * 64},
        {"source_kind": "human"},
        {"disposition": "rejected", "reason": "origin_not_attested"},
        {"disposition": "blocked", "reason": "source_linkage_conflict"},
        {"projection_fingerprint": _digest("projection-without-trajectory")},
        {"unknown": 1},
    ],
)
def test_forged_positive_or_incoherent_assessments_rejected(change: dict[str, Any]) -> None:
    base = assess_demonstration_intake(_claim()).model_dump()
    with pytest.raises(ValidationError):
        AgentDemonstrationIntakeAssessment.model_validate({**base, **change})


def test_claim_and_assessment_are_immutable() -> None:
    claim = _claim()
    assessment = assess_demonstration_intake(claim)
    with pytest.raises((ValidationError, FrozenInstanceError)):
        claim.source_kind = "expert_authored"  # type: ignore[misc]
    with pytest.raises((ValidationError, FrozenInstanceError)):
        assessment.disposition = "rejected"  # type: ignore[misc]


def test_exact_types_not_duck_typed() -> None:
    with pytest.raises(TypeError):
        assess_demonstration_intake({"source_kind": "model_runtime"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_demonstration_intake(_claim(), "trajectory")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        assess_demonstration_intake(_claim(), projection="projection")  # type: ignore[arg-type]


def test_invalid_model_copy_claim_cannot_bypass_revalidation() -> None:
    forged = _claim().model_copy(update={"source_kind": "expert_authored"})
    with pytest.raises(ValidationError):
        assess_demonstration_intake(forged)


def test_invalid_model_copy_projection_cannot_bypass_revalidation() -> None:
    forged = _projection().model_copy(update={"labels": (0, 0, 13, 14)})
    with pytest.raises(ValidationError):
        assess_demonstration_intake(_claim(), _trajectory(), forged)


def test_no_external_io_during_claim_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import socket

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("origin screening cannot access network or files")

    claim = _claim()
    trajectory = _trajectory()
    projection = _projection(trajectory)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(socket, "socket", forbidden)
        result = assess_demonstration_intake(claim, trajectory, projection)
        assert result.disposition == "blocked"
