from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from huyawo_adapt.agentic.contracts import AgentHarnessIdentity, AgentToolSetIdentity
from huyawo_adapt.agentic.trajectory import AgentTrajectory, AgentTrajectoryStep
from huyawo_adapt.contracts.base import NonEmptyString, Sha256Digest, StrictFrozenModel


class HarnessCaptureContext(StrictFrozenModel):
    """Frozen identities and provenance of one original capture."""

    trajectory_id: NonEmptyString
    model_fingerprint: Sha256Digest
    tokenizer_fingerprint: Sha256Digest
    harness_identity: AgentHarnessIdentity
    tool_set_identity: AgentToolSetIdentity
    environment_fingerprint: Sha256Digest
    verifier_fingerprint: Sha256Digest
    task_fingerprint: Sha256Digest
    runtime_fingerprint: Sha256Digest
    rollout_policy_fingerprint: Sha256Digest
    capture_provenance_sha256: Sha256Digest


class AgentHarnessAdapter(Protocol):
    """Passive boundary from captured canonical steps to one trajectory."""

    def capture(
        self,
        context: HarnessCaptureContext,
        steps: Iterable[AgentTrajectoryStep],
    ) -> AgentTrajectory: ...


class CanonicalHarnessCaptureAdapter:
    """Fail-closed reference assembler that never executes a harness or tool."""

    def capture(
        self,
        context: HarnessCaptureContext,
        steps: Iterable[AgentTrajectoryStep],
    ) -> AgentTrajectory:
        if type(context) is not HarnessCaptureContext:
            raise TypeError("capture context must be HarnessCaptureContext")

        validated_context = HarnessCaptureContext.model_validate(context.model_dump(mode="python"))
        membership = frozenset(validated_context.tool_set_identity.tool_fingerprints)
        captured: list[AgentTrajectoryStep] = []

        for index, step in enumerate(steps):
            if type(step) is not AgentTrajectoryStep:
                raise TypeError("capture requires canonical AgentTrajectoryStep records")
            AgentTrajectoryStep.model_validate(step.model_dump(mode="python"))

            if step.step_index != index:
                raise ValueError("capture step indexes must be contiguous and ordered from zero")
            if captured and captured[-1].terminal_status != "continue":
                raise ValueError("capture cannot contain steps after a terminal status")
            if step.tool_call is not None and step.tool_call.tool_fingerprint not in membership:
                raise ValueError("capture tool_call is not a member of the declared tool set")
            if step.tool_result is not None and step.tool_result.tool_fingerprint not in membership:
                raise ValueError("capture tool_result is not a member of the declared tool set")
            captured.append(step)

        if not captured:
            raise ValueError("capture requires at least one canonical step")
        if captured[-1].terminal_status == "continue":
            raise ValueError("capture requires an explicit terminal status")

        return AgentTrajectory(
            trajectory_id=validated_context.trajectory_id,
            model_fingerprint=validated_context.model_fingerprint,
            tokenizer_fingerprint=validated_context.tokenizer_fingerprint,
            harness_fingerprint=validated_context.harness_identity.fingerprint(),
            tool_set_fingerprint=validated_context.tool_set_identity.fingerprint(),
            environment_fingerprint=validated_context.environment_fingerprint,
            verifier_fingerprint=validated_context.verifier_fingerprint,
            task_fingerprint=validated_context.task_fingerprint,
            runtime_fingerprint=validated_context.runtime_fingerprint,
            rollout_policy_fingerprint=validated_context.rollout_policy_fingerprint,
            steps=tuple(captured),
            capture_provenance_sha256=validated_context.capture_provenance_sha256,
        )
