"""Offline, structural-only projection of canonical agent trajectories into SFT tokens.

No claim about original generation, tokenizer fidelity, task success, or training
eligibility follows from a valid projection. This module never runs a model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from huyawo_adapt.agentic.trajectory.schema import AgentTrajectory
from huyawo_adapt.contracts.base import ContractModel, Sha256Digest

TokenId = Annotated[int, Field(ge=0, strict=True)]
InputIds = Annotated[tuple[TokenId, ...], Field(min_length=2)]
LabelIds = Annotated[tuple[Annotated[int, Field(strict=True)], ...], Field(min_length=2)]
SplitName = Literal["train", "validation", "test"]


def _token_digest(token_ids: tuple[int, ...]) -> str:
    """Hash the domain-separated canonical target token sequence."""
    canonical = json.dumps(
        {"domain": "haa-sft-target-token-ids-v1", "token_ids": token_ids},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AgentSFTDecisionProjection(ContractModel):
    """Masked token candidate with no implied training or success authority."""

    contract_type: Literal["agent_sft_decision_projection"] = "agent_sft_decision_projection"
    authority_scope: Literal["structural_projection_only"] = "structural_projection_only"
    trajectory_fingerprint: Sha256Digest
    step_index: int = Field(ge=0, strict=True)
    step_provenance_sha256: Sha256Digest
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    harness_fingerprint: Sha256Digest
    tool_set_fingerprint: Sha256Digest
    environment_fingerprint: Sha256Digest
    verifier_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    rollout_policy_fingerprint: Sha256Digest
    capture_provenance_sha256: Sha256Digest
    parent_trajectory_fingerprint: Sha256Digest | None = None
    transformation_fingerprint: Sha256Digest | None = None
    model_input_sha256: Sha256Digest
    generated_output_sha256: Sha256Digest
    projection_policy_sha256: Sha256Digest
    input_ids: InputIds
    labels: LabelIds
    target_start: int = Field(gt=0, strict=True)
    target_end: int = Field(gt=0, strict=True)
    target_token_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if (self.parent_trajectory_fingerprint is None) != (
            self.transformation_fingerprint is None
        ):
            raise ValueError("parent and transformation fingerprints must appear together")
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have equal lengths")
        if not 0 < self.target_start < self.target_end == len(self.input_ids):
            raise ValueError("target span must be nonempty, suffix-only and preceded by input")
        if any(value != -100 for value in self.labels[: self.target_start]):
            raise ValueError("all non-target prefix labels must be ignored")
        if self.labels[self.target_start :] != self.input_ids[self.target_start :]:
            raise ValueError("every generated token must be supervised without holes")
        if self.target_token_sha256 != _token_digest(self.input_ids[self.target_start :]):
            raise ValueError("target token digest mismatch")
        return self


def project_trajectory_decision(
    trajectory: AgentTrajectory,
    step_index: int,
    projection_policy_sha256: str,
) -> AgentSFTDecisionProjection:
    """Project one unambiguous text-only step without re-tokenizing or relabeling."""
    if type(trajectory) is not AgentTrajectory:
        raise TypeError("trajectory must be an AgentTrajectory")
    if type(step_index) is not int:
        raise TypeError("step_index must be an exact integer")
    if len(trajectory.steps) != 1 or step_index != 0:
        raise ValueError("v1 projection requires exactly one standalone step")
    checked = AgentTrajectory.model_validate(trajectory.model_dump(mode="python"))
    step = checked.steps[step_index]
    if (
        step.model_input_token_ids is None
        or step.generated_token_ids is None
        or not step.model_input_token_ids
        or not step.generated_token_ids
    ):
        raise ValueError("exact nonempty input and generated token IDs are required")
    if (
        step.model_input.media_type != "text/plain"
        or step.generated_output.media_type != "text/plain"
    ):
        raise ValueError("v1 only supports explicitly text/plain input and generation")
    if (
        step.tool_call is not None
        or step.tool_result is not None
        or step.state_transition is not None
        or step.reward_components
        or step.failure is not None
    ):
        raise ValueError("tool/control/reward/failure semantics are not projected in v1")
    if step.terminal_status != "success":
        raise ValueError("v1 rejects incomplete or failed terminal steps")
    inputs = step.model_input_token_ids + step.generated_token_ids
    labels = (-100,) * len(step.model_input_token_ids) + step.generated_token_ids
    return AgentSFTDecisionProjection(
        trajectory_fingerprint=checked.fingerprint(),
        step_index=step_index,
        step_provenance_sha256=step.provenance_sha256,
        model_fingerprint=checked.model_fingerprint,
        tokenizer_fingerprint=checked.tokenizer_fingerprint,
        harness_fingerprint=checked.harness_fingerprint,
        tool_set_fingerprint=checked.tool_set_fingerprint,
        environment_fingerprint=checked.environment_fingerprint,
        verifier_fingerprint=checked.verifier_fingerprint,
        task_fingerprint=checked.task_fingerprint,
        runtime_fingerprint=checked.runtime_fingerprint,
        rollout_policy_fingerprint=checked.rollout_policy_fingerprint,
        capture_provenance_sha256=checked.capture_provenance_sha256,
        parent_trajectory_fingerprint=checked.parent_trajectory_fingerprint,
        transformation_fingerprint=checked.transformation_fingerprint,
        model_input_sha256=step.model_input.sha256,
        generated_output_sha256=step.generated_output.sha256,
        projection_policy_sha256=projection_policy_sha256,
        input_ids=inputs,
        labels=labels,
        target_start=len(step.model_input_token_ids),
        target_end=len(inputs),
        target_token_sha256=_token_digest(step.generated_token_ids),
    )


class AgentSFTSplitEntry(ContractModel):
    """Caller-declared split and lineage, not an independently attested claim."""

    contract_type: Literal["agent_sft_split_entry"] = "agent_sft_split_entry"
    split: SplitName
    projection_fingerprint: Sha256Digest
    trajectory_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    family_fingerprint: Sha256Digest
    root_trajectory_fingerprint: Sha256Digest
    parent_trajectory_fingerprint: Sha256Digest | None = None
    source_content_sha256: Sha256Digest


def _validate_split_entries(entries: tuple[AgentSFTSplitEntry, ...]) -> None:
    projection_ids: set[str] = set()
    per_domain: dict[tuple[str, str], SplitName] = {}
    for entry in entries:
        if entry.projection_fingerprint in projection_ids:
            raise ValueError("duplicate projection fingerprint")
        projection_ids.add(entry.projection_fingerprint)
        bindings = [
            ("task", entry.task_fingerprint),
            ("family", entry.family_fingerprint),
            ("content", entry.source_content_sha256),
        ]
        lineage = {entry.trajectory_fingerprint, entry.root_trajectory_fingerprint}
        if entry.parent_trajectory_fingerprint is not None:
            lineage.add(entry.parent_trajectory_fingerprint)
        bindings.extend(("lineage", value) for value in sorted(lineage))
        for domain, value in bindings:
            key = (domain, value)
            prior = per_domain.setdefault(key, entry.split)
            if prior != entry.split:
                raise ValueError(f"cross-split {domain} leakage")


class AgentSFTSplitManifest(ContractModel):
    """No-leakage structural manifest; not evidence of held-out qualification."""

    contract_type: Literal["agent_sft_split_manifest"] = "agent_sft_split_manifest"
    split_policy_sha256: Sha256Digest
    entries: Annotated[tuple[AgentSFTSplitEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        _validate_split_entries(self.entries)
        return self


def validate_sft_split_manifest(manifest: AgentSFTSplitManifest) -> AgentSFTSplitManifest:
    """Revalidate a strictly typed split manifest and return the validated record."""
    if type(manifest) is not AgentSFTSplitManifest:
        raise TypeError("manifest must be an AgentSFTSplitManifest")
    return AgentSFTSplitManifest.model_validate(manifest.model_dump(mode="python"))
