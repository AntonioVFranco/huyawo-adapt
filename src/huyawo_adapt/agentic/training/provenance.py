"""Offline demonstration-origin claim intake without success or training authority.

Claims and even self-consistent trajectory/projection hashes are caller-supplied.
They must never be interpreted as authenticated model execution, expert rights,
independent task success, or training eligibility.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BeforeValidator, model_validator

from huyawo_adapt.agentic.training.sft import AgentSFTDecisionProjection
from huyawo_adapt.agentic.trajectory.schema import AgentTrajectory
from huyawo_adapt.contracts.base import ContractModel, Sha256Digest


def _verify_exact_digest(value: object) -> object:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("digest must be exactly 64 unmodified lowercase hexadecimal characters")
    return value


ExactDigest = Annotated[Sha256Digest, BeforeValidator(_verify_exact_digest)]
SourceKind = Literal["model_runtime", "teacher_model", "expert_authored"]
IntakeDisposition = Literal["blocked", "rejected"]
IntakeReason = Literal[
    "origin_not_attested",
    "outcome_not_attested",
    "source_rights_unverified",
    "tokenization_not_attested",
    "source_linkage_conflict",
    "model_vs_expert_claim_conflict",
    "private_reference_exposure_unassessed",
]


class AgentDemonstrationOriginClaim(ContractModel):
    """A typed caller declaration, never an attestation of demonstration origin."""

    contract_type: Literal["agent_demonstration_origin_claim"] = "agent_demonstration_origin_claim"
    authority_scope: Literal["caller_declared_intake_only"] = "caller_declared_intake_only"
    source_kind: SourceKind
    source_identity_sha256: ExactDigest
    source_revision_sha256: ExactDigest
    source_content_sha256: ExactDigest
    capture_event_sha256: ExactDigest
    usage_rights_evidence_sha256: ExactDigest
    task_fingerprint: ExactDigest
    source_family_fingerprint: ExactDigest
    root_source_fingerprint: ExactDigest
    trajectory_fingerprint: ExactDigest | None = None
    expert_artifact_sha256: ExactDigest | None = None
    teacher_model_fingerprint: ExactDigest | None = None

    @model_validator(mode="after")
    def validate_origin_discriminator(self) -> Self:
        if self.source_kind == "expert_authored":
            if (
                self.expert_artifact_sha256 is None
                or self.trajectory_fingerprint is not None
                or self.teacher_model_fingerprint is not None
            ):
                raise ValueError("expert claims require only expert artifact lineage")
        elif self.source_kind == "model_runtime":
            if (
                self.trajectory_fingerprint is None
                or self.expert_artifact_sha256 is not None
                or self.teacher_model_fingerprint is not None
            ):
                raise ValueError("runtime claims require only model trajectory lineage")
        elif (
            self.trajectory_fingerprint is None
            or self.expert_artifact_sha256 is not None
            or self.teacher_model_fingerprint is None
            or self.source_identity_sha256 != self.teacher_model_fingerprint
        ):
            raise ValueError("teacher claims require distinct teacher-model trajectory lineage")
        return self


class AgentDemonstrationIntakeAssessment(ContractModel):
    """A negative-only structural screen; no successful attestation is possible."""

    contract_type: Literal["agent_demonstration_intake_assessment"] = (
        "agent_demonstration_intake_assessment"
    )
    authority_scope: Literal["untrusted_intake_screen_only"] = "untrusted_intake_screen_only"
    claim_fingerprint: ExactDigest
    source_kind: SourceKind
    trajectory_fingerprint: ExactDigest | None = None
    projection_fingerprint: ExactDigest | None = None
    disposition: IntakeDisposition
    reason: IntakeReason

    @model_validator(mode="after")
    def validate_negative_only(self) -> Self:
        conflict_reasons = {"source_linkage_conflict", "model_vs_expert_claim_conflict"}
        if (self.disposition == "rejected") != (self.reason in conflict_reasons):
            raise ValueError("rejected disposition must identify a concrete linkage conflict")
        if (
            self.projection_fingerprint is not None
            and self.trajectory_fingerprint is None
            and not (self.disposition == "rejected" and self.reason == "source_linkage_conflict")
        ):
            raise ValueError("unmatched projection must be rejected as a linkage conflict")
        if self.source_kind == "expert_authored" and self.trajectory_fingerprint is not None:
            raise ValueError("expert intake cannot acquire model trajectory provenance")
        return self


def assess_demonstration_intake(
    claim: AgentDemonstrationOriginClaim,
    trajectory: AgentTrajectory | None = None,
    projection: AgentSFTDecisionProjection | None = None,
) -> AgentDemonstrationIntakeAssessment:
    """Revalidate claimed cross-links, then block unverifiable training provenance."""
    if type(claim) is not AgentDemonstrationOriginClaim:
        raise TypeError("claim must be an AgentDemonstrationOriginClaim")
    if trajectory is not None and type(trajectory) is not AgentTrajectory:
        raise TypeError("trajectory must be an AgentTrajectory")
    if projection is not None and type(projection) is not AgentSFTDecisionProjection:
        raise TypeError("projection must be an AgentSFTDecisionProjection")

    claim = AgentDemonstrationOriginClaim.model_validate(claim.model_dump(mode="python"))
    if trajectory is not None:
        trajectory = AgentTrajectory.model_validate(trajectory.model_dump(mode="python"))
    if projection is not None:
        projection = AgentSFTDecisionProjection.model_validate(projection.model_dump(mode="python"))

    observed_trajectory = trajectory.fingerprint() if trajectory is not None else None
    observed_projection = projection.fingerprint() if projection is not None else None
    disposition: IntakeDisposition = "blocked"
    reason: IntakeReason = "origin_not_attested"

    if claim.source_kind == "expert_authored":
        if trajectory is not None or projection is not None:
            disposition, reason = "rejected", "model_vs_expert_claim_conflict"
    elif trajectory is not None:
        if (
            claim.trajectory_fingerprint != observed_trajectory
            or claim.source_identity_sha256 != trajectory.model_fingerprint
            or claim.task_fingerprint != trajectory.task_fingerprint
            or claim.capture_event_sha256 != trajectory.capture_provenance_sha256
            or (
                claim.source_kind == "teacher_model"
                and claim.teacher_model_fingerprint != trajectory.model_fingerprint
            )
        ):
            disposition, reason = "rejected", "source_linkage_conflict"
        elif len(trajectory.steps) == 1 and (
            claim.source_content_sha256 != trajectory.steps[0].generated_output.sha256
        ):
            disposition, reason = "rejected", "source_linkage_conflict"

    if projection is not None and disposition != "rejected":
        if trajectory is None:
            disposition, reason = "rejected", "source_linkage_conflict"
        else:
            step_index = projection.step_index
            if step_index >= len(trajectory.steps):
                disposition, reason = "rejected", "source_linkage_conflict"
            else:
                step = trajectory.steps[step_index]
                identity_fields = (
                    "model_fingerprint",
                    "tokenizer_fingerprint",
                    "harness_fingerprint",
                    "tool_set_fingerprint",
                    "environment_fingerprint",
                    "verifier_fingerprint",
                    "task_fingerprint",
                    "runtime_fingerprint",
                    "rollout_policy_fingerprint",
                    "capture_provenance_sha256",
                    "parent_trajectory_fingerprint",
                    "transformation_fingerprint",
                )
                if (
                    projection.trajectory_fingerprint != observed_trajectory
                    or any(
                        getattr(projection, field) != getattr(trajectory, field)
                        for field in identity_fields
                    )
                    or projection.step_provenance_sha256 != step.provenance_sha256
                    or projection.model_input_sha256 != step.model_input.sha256
                    or projection.generated_output_sha256 != step.generated_output.sha256
                    or step.model_input_token_ids is None
                    or step.generated_token_ids is None
                    or projection.input_ids != step.model_input_token_ids + step.generated_token_ids
                ):
                    disposition, reason = "rejected", "source_linkage_conflict"

    return AgentDemonstrationIntakeAssessment(
        claim_fingerprint=claim.fingerprint(),
        source_kind=claim.source_kind,
        trajectory_fingerprint=(
            observed_trajectory if claim.source_kind != "expert_authored" else None
        ),
        projection_fingerprint=(
            observed_projection if claim.source_kind != "expert_authored" else None
        ),
        disposition=disposition,
        reason=reason,
    )
