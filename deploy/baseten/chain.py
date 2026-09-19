"""Fail-closed Baseten Chains topology for a PLUMB rollout.

This is deployable Chain *wiring*, not a claim that any benchmark runtime is
available.  It purposely has no model download, no unpinned package install,
and no fabricated policy action, generated frame, validity label, or judge
vote.  With the supplied template contracts it returns a typed ``blocked``
response that names the unresolved stage; replacing a worker requires updating
the immutable compatibility profile and re-running the relevant gate.

Run only after resolving ``model-contracts.json`` and its lock inputs:

    truss chains push ./deploy/baseten/chain.py --environment <environment>

The command is documented for operators but is never run by this repository.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


try:  # Importing this repository must work without the remote-only Chains SDK.
    import truss_chains as chains
except ImportError:  # pragma: no cover - exercised by an import smoke check.
    chains = None  # type: ignore[assignment]


CHAINS_RUNTIME_AVAILABLE = chains is not None


class ChainRuntimeUnavailable(RuntimeError):
    """Raised if an operator tries to use a template without ``truss_chains``."""


class StageRequest(BaseModel):
    """Opaque, JSON-only stage input owned by a frozen PLUMB protocol."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1, max_length=256)
    protocol_hash: str = Field(min_length=1, max_length=256)
    payload: Dict[str, Any] = Field(default_factory=dict)


class RolloutRequest(BaseModel):
    """One logical episode.  The controller must not create a new identity."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=256)
    episode_id: str = Field(min_length=1, max_length=256)
    protocol_hash: str = Field(min_length=1, max_length=256)
    policy: StageRequest
    world: StageRequest
    validity: StageRequest
    judge: StageRequest

    def matching_episode_requests(self) -> bool:
        return all(
            stage.episode_id == self.episode_id and stage.protocol_hash == self.protocol_hash
            for stage in (self.policy, self.world, self.validity, self.judge)
        )


class StageResult(BaseModel):
    """A worker response that cannot silently turn unknown evidence into a score."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["policy", "world", "validity", "judge"]
    status: Literal["completed", "blocked", "failed"]
    output: Optional[Dict[str, Any]] = None
    unresolved_contracts: List[str] = Field(default_factory=list)


class RolloutResult(BaseModel):
    """Entrypoint result suitable for persisted Chain output and webhook delivery."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    episode_id: str
    protocol_hash: str
    status: Literal["completed", "blocked", "failed"]
    stages: List[StageResult] = Field(default_factory=list)
    missing_reason: Optional[str] = None


POLICY_CONTRACTS = [
    "A pinned policy variant, wrapper revision, action normalizer, and native feedback contract are required.",
    "Each incompatible policy dependency stack requires its own reviewed image and immutable digest.",
    "Gate B must certify the executed prefix and feedback/state mode before a named-policy cell is qualified.",
]
WORLD_CONTRACTS = [
    "The Cosmos/IRASim backend profile, request serializer, domain registry, and action compiler revision are unresolved.",
    "A Gate-A fixture must certify the actual deployed payload, frame mapping, timestamps, and action alignment.",
    "No model weights, safety configuration, or remote code revision is pinned in this template.",
]
VALIDITY_CONTRACTS = [
    "Deterministic validity parameters and calibration references must be frozen before primary scoring.",
    "The worker must persist artifact hashes and reason codes; it may not infer physical measurements from video.",
]
JUDGE_CONTRACTS = [
    "The rubric, model/processor/runtime revision, sampling seeds, and output schema must be frozen together.",
    "Gate D human calibration is required before this worker emits a primary-scoring label.",
    "Raw judge samples and bounded retry records must be persisted by the application control plane.",
]


def require_chains_runtime() -> None:
    """Make the local import guard explicit instead of failing mysteriously."""

    if not CHAINS_RUNTIME_AVAILABLE:
        raise ChainRuntimeUnavailable(
            "truss_chains is not installed locally; install the reviewed deployment environment before pushing this Chain"
        )


def _blocked(stage: Literal["policy", "world", "validity", "judge"], contracts: List[str]) -> StageResult:
    return StageResult(stage=stage, status="blocked", unresolved_contracts=list(contracts))


if CHAINS_RUNTIME_AVAILABLE:

    class PolicyWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Custom policy Chainlet placeholder; separate policy images are a hard gate."""

        # CPU-only blocking template. A concrete policy image/accelerator is not
        # guessed here; it belongs to a reviewed compatibility profile.
        remote_config = chains.RemoteConfig(
            name="plumb-policy-worker",
            docker_image=chains.DockerImage(base_image=chains.BasetenImage.PY311),
            compute=chains.Compute(cpu_count=2, memory="8Gi", predict_concurrency=1),
        )

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _blocked("policy", POLICY_CONTRACTS)


    class WorldWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Custom world-model Chainlet, initially resource-shaped for one H100."""

        # The build specification begins world benchmarking on one H100 with
        # in-container concurrency one. This is a capacity hypothesis, not a
        # performance claim or a pinned model runtime.
        remote_config = chains.RemoteConfig(
            name="plumb-world-worker",
            docker_image=chains.DockerImage(base_image=chains.BasetenImage.PY311),
            compute=chains.Compute(gpu="H100", gpu_count=1, predict_concurrency=1),
        )

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _blocked("world", WORLD_CONTRACTS)


    class ValidityWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Custom deterministic-validity Chainlet; it does not score success."""

        remote_config = chains.RemoteConfig(
            name="plumb-validity-worker",
            docker_image=chains.DockerImage(base_image=chains.BasetenImage.PY311),
            compute=chains.Compute(cpu_count=2, memory="8Gi", predict_concurrency=1),
        )

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _blocked("validity", VALIDITY_CONTRACTS)


    class JudgeWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Custom rubric-judge Chainlet; no model or sampling values are guessed."""

        remote_config = chains.RemoteConfig(
            name="plumb-judge-worker",
            docker_image=chains.DockerImage(base_image=chains.BasetenImage.PY311),
            compute=chains.Compute(cpu_count=2, memory="8Gi", predict_concurrency=1),
        )

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _blocked("judge", JUDGE_CONTRACTS)


    @chains.mark_entrypoint("PLUMB Rollout Controller")
    class RolloutController(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """CPU entrypoint coordinating in-order policy/world/validity/judge RPCs.

        A complete logical episode is submitted once to this entrypoint. The
        application—not this stateless worker—owns the durable ledger, outbox,
        artifact writes, retry-attempt records, and callback reconciliation.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-rollout-controller",
            docker_image=chains.DockerImage(base_image=chains.BasetenImage.PY311),
            compute=chains.Compute(cpu_count=2, memory="4Gi", predict_concurrency=1),
        )

        def __init__(
            self,
            policy: PolicyWorker = chains.depends(PolicyWorker, retries=0),
            world: WorldWorker = chains.depends(WorldWorker, retries=0),
            validity: ValidityWorker = chains.depends(ValidityWorker, retries=0),
            judge: JudgeWorker = chains.depends(JudgeWorker, retries=0),
        ) -> None:
            self._policy = policy
            self._world = world
            self._validity = validity
            self._judge = judge

        async def run_remote(self, request: RolloutRequest) -> RolloutResult:
            if not request.matching_episode_requests():
                return RolloutResult(
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    status="failed",
                    missing_reason="stage_request_identity_mismatch",
                )

            # This exact order mirrors an episode's sequential feedback path;
            # independent logical episodes are parallelized by external async
            # submissions. No internal Chainlet invokes an external queue.
            stages: List[StageResult] = []
            for worker, stage_request in (
                (self._policy, request.policy),
                (self._world, request.world),
                (self._validity, request.validity),
                (self._judge, request.judge),
            ):
                result = await worker.run_remote(stage_request)
                stages.append(result)
                if result.status != "completed":
                    return RolloutResult(
                        run_id=request.run_id,
                        episode_id=request.episode_id,
                        protocol_hash=request.protocol_hash,
                        status=result.status,
                        stages=stages,
                        missing_reason="{0}_stage_{1}".format(result.status, result.stage),
                    )
            return RolloutResult(
                run_id=request.run_id,
                episode_id=request.episode_id,
                protocol_hash=request.protocol_hash,
                status="completed",
                stages=stages,
            )


else:
    # These names are intentionally absent as deployable classes in a normal
    # local checkout. Callers can still import request/result schemas and check
    # CHAINS_RUNTIME_AVAILABLE without attempting to mock model execution.
    PolicyWorker = None
    WorldWorker = None
    ValidityWorker = None
    JudgeWorker = None
    RolloutController = None


__all__ = [
    "CHAINS_RUNTIME_AVAILABLE",
    "ChainRuntimeUnavailable",
    "StageRequest",
    "RolloutRequest",
    "StageResult",
    "RolloutResult",
    "require_chains_runtime",
    "PolicyWorker",
    "WorldWorker",
    "ValidityWorker",
    "JudgeWorker",
    "RolloutController",
]
