"""Fail-closed Baseten Chains topology for a PLUMB rollout.

This is deployable Chain *wiring plus real adapter calls*, not a claim that any
benchmark runtime is qualified.  Every worker either performs a genuine backend
call through a reviewed ``plumb`` adapter or returns a typed ``blocked`` result
that names the unresolved contract.  Nothing here fabricates a frame, an action,
a validity label, a judge vote, a container digest, or a price.

Topology (AGENT-BUILD-SPEC.md §7):

    RolloutController (CPU entrypoint, external /async_run_remote queue)
      -> PolicyRouter (entrypoint-side dispatch, not a Chainlet)
           -> OpenVLAWorker | OctoWorker | MiniVLAWorker
              | OpenPiZeroWorker | SusieWorker          (one image per stack)
      -> WorldWorker      (H100, micro-batched diffusion)
      -> ValidityWorker   (CPU, deterministic Stage A)
      -> JudgeWorker      (GPU, frozen five-sample Stage B)

Deploy with the documented CLI only after reading ``DEPLOY.md``:

    truss chains push ./deploy/baseten/chain.py --environment production

The command is documented for operators and is never run by this repository.

Two deliberate style deviations from the rest of the repository, both forced by
the Chains framework:

1. **No ``from __future__ import annotations``.**  ``truss_chains.framework``
   reads endpoint signatures with ``inspect.signature()`` and *without*
   ``eval_str=True``; PEP 563 would turn every annotation into a string and
   ``_validate_io_type`` rejects string annotations outright ("Use only actual
   types objects and avoid ``from __future__ import annotations``").  With the
   future import this module is unpushable, so it is omitted here on purpose.
2. **Chainlet endpoint containers must be builtin generics.**  The same
   validator rejects ``typing.List[...]`` for endpoint arguments.  Endpoint
   signatures therefore only use pydantic models; ``typing`` generics appear
   inside the models and in ordinary helper code, where they are fine.
"""

import asyncio
import base64
import hashlib
import inspect
import io
import json
import math
import os
import pathlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from plumb.result_store import (
    ResultStoreBinding,
    ResultStoreConfigurationError,
    ResultStoreConflictError,
    ResultStoreError,
    S3Credentials,
    S3ResultStore,
    S3ResultStoreConfig,
    request_payload_digest,
)


try:  # Importing this repository must work without the remote-only Chains SDK.
    import truss_chains as chains
    from truss.base import truss_config
except ImportError:  # pragma: no cover - exercised by an import smoke check.
    chains = None  # type: ignore[assignment]
    truss_config = None  # type: ignore[assignment]


CHAINS_RUNTIME_AVAILABLE = chains is not None

CHAIN_SOURCE_DIR = pathlib.Path(__file__).resolve().parent
REQUIREMENTS_DIR = CHAIN_SOURCE_DIR / "requirements"


# ---------------------------------------------------------------------------
# Environment knobs.
#
# These are read once at module import.  That happens twice with the same
# result: at ``truss chains push`` time, where the value chosen here is baked
# into ``ChainletOptions.env_variables``, and again inside the running container,
# where it is read back out of that baked environment.  So `BATCH_MAX=32 truss
# chains push ...` configures both the declared concurrency and the runtime
# collector, and the two can never disagree.
# ---------------------------------------------------------------------------

ENV_BATCH_MAX = "PLUMB_WORLD_BATCH_MAX"
ENV_BATCH_WINDOW_MS = "PLUMB_WORLD_BATCH_WINDOW_MS"
ENV_WORLD_MAX_REPLICA = "PLUMB_WORLD_MAX_REPLICA"
ENV_JUDGE_MAX_REPLICA = "PLUMB_JUDGE_MAX_REPLICA"
ENV_POLICY_MAX_REPLICA = "PLUMB_POLICY_MAX_REPLICA"
ENV_CONTROLLER_MAX_REPLICA = "PLUMB_CONTROLLER_MAX_REPLICA"
ENV_VALIDITY_MAX_REPLICA = "PLUMB_VALIDITY_MAX_REPLICA"
ENV_WORLD_VARIANT = "PLUMB_WORLD_VARIANT"
ENV_WORLD_EDGE_REVISION = "PLUMB_WORLD_EDGE_REVISION"
ENV_WORLD_RESOLUTION_TIER = "PLUMB_WORLD_RESOLUTION_TIER"
ENV_WORLD_FUSED_BATCH = "PLUMB_WORLD_FUSED_BATCH"
ENV_MODEL_CACHE_ROOT = "PLUMB_MODEL_CACHE_ROOT"
ENV_OPENVLA_REVIEWED_REMOTE_CODE = "PLUMB_OPENVLA_REVIEWED_REMOTE_CODE_ACK"
ENV_ASSET_MANIFEST_PATH = "PLUMB_ASSET_MANIFEST_PATH"
ENV_VALIDITY_CALIBRATION_CLASS = "PLUMB_VALIDITY_CALIBRATION_CLASS"
ENV_RESULT_STORE_BUCKET = "PLUMB_RESULT_STORE_BUCKET"
ENV_RESULT_STORE_REGION = "PLUMB_RESULT_STORE_REGION"
ENV_RESULT_STORE_ENDPOINT_URL = "PLUMB_RESULT_STORE_ENDPOINT_URL"
ENV_RESULT_STORE_PREFIX = "PLUMB_RESULT_STORE_PREFIX"
ENV_RESULT_STORE_EVIDENCE_URI = "PLUMB_RESULT_STORE_EVIDENCE_URI"
ENV_RESULT_STORE_VERIFIED_AT = "PLUMB_RESULT_STORE_VERIFIED_AT"
ENV_RESULT_STORE_EXPECTED_BUCKET_OWNER = "PLUMB_RESULT_STORE_EXPECTED_BUCKET_OWNER"
ENV_RESULT_STORE_MAX_BYTES = "PLUMB_RESULT_STORE_MAX_BYTES"
ENV_RESULT_STORE_CONNECT_TIMEOUT_SECONDS = "PLUMB_RESULT_STORE_CONNECT_TIMEOUT_SECONDS"
ENV_RESULT_STORE_READ_TIMEOUT_SECONDS = "PLUMB_RESULT_STORE_READ_TIMEOUT_SECONDS"
ENV_RESULT_STORE_MAX_ATTEMPTS = "PLUMB_RESULT_STORE_MAX_ATTEMPTS"
ENV_RESULT_STORE_ACCESS_KEY_SECRET_NAME = "PLUMB_RESULT_STORE_ACCESS_KEY_SECRET_NAME"
ENV_RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME = "PLUMB_RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME"
ENV_RESULT_STORE_SESSION_TOKEN_SECRET_NAME = "PLUMB_RESULT_STORE_SESSION_TOKEN_SECRET_NAME"

DEFAULT_BATCH_MAX = 16
DEFAULT_BATCH_WINDOW_MS = 15.0
# AGENT-BUILD-SPEC.md §7 keeps "100 world-model replicas as the initial capacity
# hypothesis" and notes that Baseten's own `max_replica` default is 1.  A
# configured maximum is neither an active replica count nor reserved capacity.
DEFAULT_WORLD_MAX_REPLICA = 100
DEFAULT_JUDGE_MAX_REPLICA = 100
DEFAULT_POLICY_MAX_REPLICA = 25
DEFAULT_VALIDITY_MAX_REPLICA = 10
DEFAULT_CONTROLLER_MAX_REPLICA = 10
DEFAULT_MODEL_CACHE_ROOT = "/app/model_cache"


def _env_text(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_flag(name: str) -> bool:
    return _env_text(name, "").lower() in ("1", "true", "yes", "on")


BATCH_MAX = _env_int(ENV_BATCH_MAX, DEFAULT_BATCH_MAX, 1, 256)
BATCH_WINDOW_MS = _env_float(ENV_BATCH_WINDOW_MS, DEFAULT_BATCH_WINDOW_MS, 0.0, 5000.0)
WORLD_MAX_REPLICA = _env_int(ENV_WORLD_MAX_REPLICA, DEFAULT_WORLD_MAX_REPLICA, 1, 1000)
JUDGE_MAX_REPLICA = _env_int(ENV_JUDGE_MAX_REPLICA, DEFAULT_JUDGE_MAX_REPLICA, 1, 1000)
POLICY_MAX_REPLICA = _env_int(ENV_POLICY_MAX_REPLICA, DEFAULT_POLICY_MAX_REPLICA, 1, 1000)
VALIDITY_MAX_REPLICA = _env_int(ENV_VALIDITY_MAX_REPLICA, DEFAULT_VALIDITY_MAX_REPLICA, 1, 1000)
CONTROLLER_MAX_REPLICA = _env_int(ENV_CONTROLLER_MAX_REPLICA, DEFAULT_CONTROLLER_MAX_REPLICA, 1, 1000)
MODEL_CACHE_ROOT = _env_text(ENV_MODEL_CACHE_ROOT, DEFAULT_MODEL_CACHE_ROOT)
# Defaults to the primary class. Selecting uncalibrated_development is safe to
# expose because plumb.validity stamps those reports with
# primary_scoring_eligible=False and an uncalibrated_development_mode reason
# code, so a development verdict cannot be mistaken for primary evidence.
VALIDITY_CALIBRATION_CLASS = _env_text(ENV_VALIDITY_CALIBRATION_CLASS, "calibrated_primary")
WORLD_VARIANT = _env_text(ENV_WORLD_VARIANT, "cosmos3_nano").lower()
WORLD_RESOLUTION_TIER = _env_int(ENV_WORLD_RESOLUTION_TIER, 256, 256, 720)
WORLD_EDGE_REVISION = _env_text(ENV_WORLD_EDGE_REVISION, "")


# ---------------------------------------------------------------------------
# Pinned identities.  Every 40-hex revision below appears in
# AGENT-BUILD-SPEC.md §12 or HANDOFF.md as an immutable cluster download.
# Anything PLUMB has not actually pinned stays empty and blocks its worker
# instead of being invented here.
# ---------------------------------------------------------------------------

COSMOS3_NANO_REPO = "nvidia/Cosmos3-Nano"
COSMOS3_NANO_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
COSMOS3_EDGE_REPO = "nvidia/Cosmos3-Edge"
OPENVLA_REPO = "openvla/openvla-7b"
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
QWEN_JUDGE_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_JUDGE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

COSMOS3_NANO_VOLUME = "cosmos3-nano"
COSMOS3_EDGE_VOLUME = "cosmos3-edge"
OPENVLA_VOLUME = "openvla-7b"
QWEN_JUDGE_VOLUME = "qwen2p5-vl-7b-instruct"

HF_SECRET_KEY = "hf_access_token"

# Policy checkpoints whose immutable Hugging Face commit PLUMB has NOT resolved.
# They are deliberately absent from every ``chains.Assets(cached=[...])`` list:
# ``truss_config.ModelRepo`` requires a revision for a cached volume, and
# inventing one would forge provenance.  The affected workers therefore start
# without weights and return ``blocked`` (their native loaders are unwired too).
UNPINNED_POLICY_ASSETS: Tuple[Tuple[str, str], ...] = (
    ("rail-berkeley/octo-small", "270000/default/checkpoint is not at the repo root; resolve/main/checkpoint 404s"),
    ("rail-berkeley/octo-base", "300000/default/checkpoint; Octo must be v1.0, never octo-small-1.5"),
    ("Stanford-ILIAD/minivla-vq-bridge-prismatic", "filename contains '=' and must be URL-encoded as %3D"),
    ("Stanford-ILIAD/pretrain_vq", "declares no license (cardData: null); redistribution rights unresolved"),
    ("allenzren/open-pi-zero", "legacy .pt checkpoint needing isolated offline conversion"),
    ("google/paligemma-3b-pt-224", "official access path under accepted terms; mirror provenance unresolved"),
    ("kvablack/susie", "subgoal diffusion component"),
    ("patreya/gcbc-bridge", "checkpoint_75000 low-level component"),
)


# ``PolicyWorker`` was the single combined policy Chainlet in the previous
# template.  It is deliberately retired to ``None``: AGENT-BUILD-SPEC.md §7
# requires one image per incompatible dependency stack, and one shared policy
# image cannot hold torch 2.2.0, torch 2.6.0, and JAX 0.4.20 at once.
PolicyWorker = None


class ChainRuntimeUnavailable(RuntimeError):
    """Raised if an operator tries to use a template without ``truss_chains``."""


class BatchContractError(RuntimeError):
    """A batch executor returned a result set that cannot be scattered safely."""


# ===========================================================================
# Wire contracts
# ===========================================================================


class FramePayload(BaseModel):
    """One RGB frame crossing a Chainlet RPC boundary.

    Both hashes are carried because HANDOFF.md records that the original
    schema-v1 bundle "labels pixel hashes ambiguously"; schema-v2 binds decoded
    pixels and PNG file bytes separately.  ``pixels_sha256`` is the digest of the
    decoded ``HxWx3`` uint8 buffer, ``png_sha256`` the digest of the PNG file
    bytes.  When a hash is supplied it is verified on decode; a mismatch is an
    explicit failure, never a silently accepted frame.
    """

    model_config = ConfigDict(extra="forbid")

    encoding: Literal["png_base64"] = "png_base64"
    data: str = Field(min_length=1)
    png_sha256: Optional[str] = None
    pixels_sha256: Optional[str] = None
    nominal_timestamp: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None


class StageRequest(BaseModel):
    """Opaque, JSON-only stage input owned by a frozen PLUMB protocol."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1, max_length=256)
    protocol_hash: str = Field(min_length=1, max_length=256)
    payload: Dict[str, Any] = Field(default_factory=dict)


class ResultStoreBindingPayload(BaseModel):
    """Secret-free object identity supplied by the durable application outbox."""

    model_config = ConfigDict(extra="forbid")

    result_key: str = Field(pattern=r"^[a-f0-9]{32}$")
    run_id: str = Field(min_length=1, max_length=256)
    episode_id: str = Field(min_length=1, max_length=256)
    protocol_hash: str = Field(min_length=1, max_length=256)
    request_payload_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")


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
    result_store: Optional[ResultStoreBindingPayload] = None

    def matching_episode_requests(self) -> bool:
        return all(
            stage.episode_id == self.episode_id and stage.protocol_hash == self.protocol_hash
            for stage in (self.policy, self.world, self.validity, self.judge)
        )

    def matching_result_store_binding(self) -> bool:
        binding = self.result_store
        if binding is None:
            return False
        if (
            binding.run_id != self.run_id
            or binding.episode_id != self.episode_id
            or binding.protocol_hash != self.protocol_hash
        ):
            return False
        unsigned = self.model_dump(mode="json", exclude={"result_store"})
        return binding.request_payload_sha256 == request_payload_digest(unsigned)


class StageTiming(BaseModel):
    """Instrumented timing for exactly one stage call.

    ``None`` means not observed.  It is never rewritten as zero, and none of
    these values is an allocation or billing measurement.
    """

    model_config = ConfigDict(extra="forbid")

    stage: Literal["policy", "compile", "world", "validity", "judge", "controller"]
    worker: Optional[str] = None
    wall_seconds: Optional[float] = None
    server_seconds: Optional[float] = None
    model_load_seconds: Optional[float] = None
    gpu_peak_memory_bytes: Optional[int] = None
    declared_gpu_count: Optional[int] = None
    batch_size: Optional[int] = None
    batch_execution_mode: Optional[str] = None
    cold_start: Optional[bool] = None


class StageResult(BaseModel):
    """A worker response that cannot silently turn unknown evidence into a score."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["policy", "world", "validity", "judge"]
    status: Literal["completed", "blocked", "failed"]
    output: Optional[Dict[str, Any]] = None
    unresolved_contracts: List[str] = Field(default_factory=list)
    timing: Optional[StageTiming] = None
    worker: Optional[str] = None


class SegmentResult(BaseModel):
    """One completed control segment, ready for a ``segment_completed`` event.

    The application control plane — not this stateless worker — owns the durable
    ledger and artifact writes.  This record supplies the stable, hash-bound
    content those writes need.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    action_offset: int
    action_count: int
    policy_arm: str
    native_actions: List[List[float]] = Field(default_factory=list)
    compiled_actions: List[List[float]] = Field(default_factory=list)
    policy_history_pixel_hashes: List[str] = Field(default_factory=list)
    generated_frame_pixel_hashes: List[str] = Field(default_factory=list)
    generated_frame_png_hashes: List[str] = Field(default_factory=list)
    nominal_frame_timestamps: List[float] = Field(default_factory=list)
    conditioning_frame_dropped: bool = False
    world_request_id: Optional[str] = None
    forecast_state_after: Optional[List[float]] = None
    feedback_mode: Optional[str] = None
    timings: List[StageTiming] = Field(default_factory=list)
    status: Literal["completed", "blocked", "failed"] = "completed"
    frames: List[FramePayload] = Field(default_factory=list)


class GpuSecondAccounting(BaseModel):
    """Instrumented per-stage GPU seconds.  NOT an allocation or billing ledger.

    AGENT-BUILD-SPEC.md §7 requires compute seconds and *allocated* GPU-seconds
    to be labelled separately.  This model only reports the first: measured stage
    wall time multiplied by the declared GPU count of the worker that ran it.
    ``allocated_gpu_seconds`` stays ``None`` because allocation spans prewarm,
    idle, and cooldown intervals that a single episode cannot observe, and
    ``estimated_usd`` is absent entirely because no verified price basis exists.
    """

    model_config = ConfigDict(extra="forbid")

    policy_gpu_seconds: Optional[float] = None
    world_gpu_seconds: Optional[float] = None
    judge_gpu_seconds: Optional[float] = None
    total_instrumented_gpu_seconds: Optional[float] = None
    allocated_gpu_seconds: Optional[float] = None
    accounting_basis: Literal[
        "instrumented_stage_wall_seconds_times_declared_gpu_count"
    ] = "instrumented_stage_wall_seconds_times_declared_gpu_count"
    limitations: List[str] = Field(default_factory=list)


class RolloutResult(BaseModel):
    """Entrypoint result suitable for persisted Chain output and webhook delivery."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    episode_id: str
    protocol_hash: str
    status: Literal["completed", "blocked", "failed"]
    # ``stages`` is the TERMINAL status per stage name, which is what a consumer
    # needs to answer "which stage blocked?". It is a mapping rather than a list
    # because plumb/backends/baseten.py indexes it by stage name.
    stages: Dict[str, StageResult] = Field(default_factory=dict)
    # ``stage_history`` is the full ordered record. A rollout makes MANY policy
    # and world calls (one pair per certified executed prefix), so the mapping
    # above necessarily keeps only the last of each; this keeps all of them.
    stage_history: List[StageResult] = Field(default_factory=list)
    missing_reason: Optional[str] = None
    policy_arm: Optional[str] = None
    horizon_actions: Optional[int] = None
    executed_actions: int = 0
    segments: List[SegmentResult] = Field(default_factory=list)
    timings: List[StageTiming] = Field(default_factory=list)
    gpu_seconds: Optional[GpuSecondAccounting] = None
    frame_pixel_hashes: List[str] = Field(default_factory=list)
    nominal_timestamps: List[float] = Field(default_factory=list)
    validity: Optional[Literal["valid", "invalid", "unknown"]] = None
    validity_reason_codes: List[str] = Field(default_factory=list)
    binary_success: Optional[bool] = None
    # Named progress_score, not progress, to match plumb.rollout.RolloutReport
    # and the reader in plumb/backends/baseten.py. The judge's own report field
    # stays `progress`; this is the rollout-level field.
    progress_score: Optional[int] = None
    judge_status: Optional[str] = None
    qualified: bool = False
    feedback_mode: Optional[str] = None
    physical_state_measured: bool = False
    world_calls: int = 0


# ---------------------------------------------------------------------------
# Stage payload contracts.  ``extra="forbid"`` is the structural blinding
# mechanism for the judge: a policy name, action row, command, reference rate,
# condition label, or Stage-A outcome cannot even be parsed into a
# ``JudgeStagePayload``, so it cannot reach the VLM by accident.
# ---------------------------------------------------------------------------


class PolicyObservationPayload(BaseModel):
    """Exactly one native policy feedback boundary."""

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4096)
    image_history: List[FramePayload] = Field(min_length=1, max_length=8)
    proprio: Optional[List[float]] = None
    goal_image: Optional[FramePayload] = None
    timestamp: Optional[float] = None
    remaining_actions: Optional[int] = Field(default=None, ge=1)
    policy_seed: Optional[int] = None
    local_model_path: Optional[str] = None
    checkpoint_revision: Optional[str] = None


class WorldStagePayload(BaseModel):
    """One backend-native world request; Cosmos rows are 10-D, IRASim rows 7-D."""

    model_config = ConfigDict(extra="forbid")

    compatibility_profile_id: str = Field(min_length=1, max_length=256)
    domain: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4096)
    conditioning_image: FramePayload
    compiled_actions: List[List[float]] = Field(min_length=1)
    nominal_control_timestamps: List[float] = Field(min_length=1)
    seed: int
    feedback_mode: str = "unqualified"
    source_state_lineage_id: Optional[str] = None
    request_id: Optional[str] = None
    return_frames: bool = True


class ValidityStagePayload(BaseModel):
    """Inputs for the deterministic Stage-A gate.  No success scoring here."""

    model_config = ConfigDict(extra="forbid")

    frames: List[FramePayload] = Field(min_length=1)
    nominal_timestamps: List[float] = Field(min_length=1)
    native_actions: List[List[float]] = Field(default_factory=list)
    conditioning_frame: Optional[FramePayload] = None
    forecast_states: Optional[List[List[float]]] = None
    # Whether the protocol expects this scene to be static. It comes from the
    # frozen run configuration, never from the video: a stationary gripper
    # pressing a drawer can be legitimately valid.
    expected_static: Optional[bool] = None
    episode_id: Optional[str] = None
    expected_frame_count: Optional[int] = Field(default=None, ge=1)
    task_id: Optional[str] = None
    parameters_hash: Optional[str] = None
    calibration_reference_hash: Optional[str] = None


class JudgeReferenceImagePayload(BaseModel):
    """A goal/reference image with the provenance the rubric protocol requires."""

    model_config = ConfigDict(extra="forbid")

    image: FramePayload
    source_uri: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class JudgeStagePayload(BaseModel):
    """Exactly the blinded information permitted to reach the rubric judge.

    There is no field for a policy name, an action, a command, a published
    reference rate, a source-condition label, or the Stage-A outcome, and
    ``extra="forbid"`` rejects any attempt to add one.
    """

    model_config = ConfigDict(extra="forbid")

    frames: List[FramePayload] = Field(min_length=16, max_length=16)
    frame_timestamps: List[float] = Field(min_length=16, max_length=16)
    reference_images: List[JudgeReferenceImagePayload] = Field(min_length=1)
    task_id: str = Field(min_length=1, max_length=128)
    seeds: List[int] = Field(min_length=5, max_length=5)
    clip_id: Optional[str] = None
    video_sha256: Optional[str] = None
    protocol_id: Optional[str] = None
    calibration_manifest_hash: Optional[str] = None


class EpisodeControlPayload(BaseModel):
    """The controller's own episode configuration (``request.policy.payload``).

    The controller never invents a horizon, a prefix, a seed, or a task prompt:
    each value here is supplied by the frozen run protocol.
    """

    model_config = ConfigDict(extra="forbid")

    policy: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4096)
    horizon_actions: int = Field(ge=1, le=1000)
    control_hz: float = Field(default=5.0, gt=0.0, le=1000.0)
    initial_frame: FramePayload
    initial_state: List[float] = Field(min_length=7, max_length=8)
    goal_image: Optional[FramePayload] = None
    policy_seed: Optional[int] = None
    bridge_control_profile_id: str = Field(min_length=1, max_length=256)
    normalization_boundary: Literal["external", "backend"] = "backend"
    normalizer_revision: Optional[str] = None
    feedback_mode: Literal["unqualified", "native_feedback", "forecast_state", "frozen_state"] = "forecast_state"
    return_frames: bool = False
    max_segments: Optional[int] = Field(default=None, ge=1, le=1000)
    # A declared reset bootstrap, matching plumb.rollout's documented behaviour:
    # a policy whose contract needs N images gets the ORIGINAL start frame
    # repeated N times at reset. It never manufactures a future feedback image
    # and never extends a control proposal. Absent means one frame, and a policy
    # needing more will reject it with an explicit contract error.
    initial_history_length: Optional[int] = Field(default=None, ge=1, le=8)
    local_model_path: Optional[str] = None
    checkpoint_revision: Optional[str] = None


# ===========================================================================
# Unresolved-contract text
# ===========================================================================

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


def _blocked(
    stage: Literal["policy", "world", "validity", "judge"],
    contracts: Sequence[str],
    *,
    worker: Optional[str] = None,
) -> StageResult:
    return StageResult(stage=stage, status="blocked", unresolved_contracts=list(contracts), worker=worker)


def _failed(
    stage: Literal["policy", "world", "validity", "judge"],
    reason: str,
    *,
    worker: Optional[str] = None,
) -> StageResult:
    """A caller/transport contract violation, distinct from an unresolved gate."""

    return StageResult(stage=stage, status="failed", unresolved_contracts=[reason], worker=worker)


# ===========================================================================
# Frame codec
# ===========================================================================


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _normalise_digest(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return "sha256:" + text


def _pillow_and_numpy() -> Tuple[Any, Any]:
    from PIL import Image  # type: ignore
    import numpy  # type: ignore

    return Image, numpy


def encode_frame(array: Any, *, nominal_timestamp: Optional[float] = None) -> FramePayload:
    """PNG-encode one RGB uint8 frame and bind both digests.

    The array is converted, never resampled, padded, cropped, or colour-shifted.
    """

    Image, numpy = _pillow_and_numpy()
    data = numpy.asarray(array)
    if data.dtype != numpy.uint8:
        if numpy.issubdtype(data.dtype, numpy.floating):
            # Diffusers ``output_type="np"`` returns float frames in [0, 1].
            # Scaling is the documented conversion, not a value invention.
            finite = numpy.isfinite(data)
            if not bool(finite.all()):
                raise ValueError("generated frame contains non-finite pixel values")
            data = numpy.clip(numpy.rint(data * 255.0), 0.0, 255.0).astype(numpy.uint8)
        else:
            data = data.astype(numpy.uint8)
    if data.ndim == 2:
        data = numpy.stack([data] * 3, axis=-1)
    if data.ndim != 3 or data.shape[2] not in (3, 4):
        raise ValueError("frame must be HxWx3 (or HxWx4) RGB; got shape %r" % (tuple(data.shape),))
    data = numpy.ascontiguousarray(data[:, :, :3])
    buffer = io.BytesIO()
    Image.fromarray(data, mode="RGB").save(buffer, format="PNG", optimize=False)
    png_bytes = buffer.getvalue()
    return FramePayload(
        encoding="png_base64",
        data=base64.b64encode(png_bytes).decode("ascii"),
        png_sha256=_sha256_hex(png_bytes),
        pixels_sha256=_sha256_hex(data.tobytes()),
        nominal_timestamp=nominal_timestamp,
        height=int(data.shape[0]),
        width=int(data.shape[1]),
    )


def decode_frame(payload: FramePayload) -> Any:
    """Decode a wire frame, verifying every digest the sender supplied."""

    Image, numpy = _pillow_and_numpy()
    try:
        png_bytes = base64.b64decode(payload.data, validate=True)
    except Exception as error:  # noqa: BLE001 - surfaced as an explicit failure
        raise ValueError("frame payload is not valid base64") from error
    expected_png = _normalise_digest(payload.png_sha256)
    if expected_png is not None and _sha256_hex(png_bytes) != expected_png:
        raise ValueError("frame png_sha256 does not match the transported bytes")
    with Image.open(io.BytesIO(png_bytes)) as image:
        data = numpy.asarray(image.convert("RGB"), dtype=numpy.uint8)
    data = numpy.ascontiguousarray(data)
    expected_pixels = _normalise_digest(payload.pixels_sha256)
    if expected_pixels is not None and _sha256_hex(data.tobytes()) != expected_pixels:
        raise ValueError("frame pixels_sha256 does not match the decoded buffer")
    if payload.height is not None and int(payload.height) != int(data.shape[0]):
        raise ValueError("frame height does not match the decoded buffer")
    if payload.width is not None and int(payload.width) != int(data.shape[1]):
        raise ValueError("frame width does not match the decoded buffer")
    return data


def segment_world_seed(seed: int, action_offset: int) -> int:
    """Derive a per-segment diffusion seed from the episode's protocol seed.

    This MUST stay byte-identical to ``plumb.rollout._segment_seed``: the local
    controller and this Chain have to produce the same video for the same
    protocol, and same-seed exact repeatability is the reproducibility evidence
    recorded in AGENT-BUILD-SPEC.md 12. tests/test_chain.py asserts the two
    implementations agree, so a drift in either one fails the suite rather than
    silently changing every generated frame.
    """

    payload = ("%d:%d" % (seed, action_offset)).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _file_sha256(path: pathlib.Path) -> Optional[str]:
    try:
        return _sha256_hex(path.read_bytes())
    except OSError:
        return None


# ===========================================================================
# Micro-batching
#
# The planner below is pure: no asyncio, no GPU, no adapter.  It is the piece
# AGENT-BUILD-SPEC.md §7 calls load-bearing ("throughput comes from batch
# packing and replica count"), so it is unit-tested directly in
# tests/test_chain.py without any model runtime.
# ===========================================================================


@dataclass
class MicroBatchTicket:
    """One awaiting caller inside the world-model collector."""

    ticket_id: int
    batch_key: str
    payload: Any
    submitted_at: float
    future: Any = None


@dataclass(frozen=True)
class BatchOutcome:
    """The scatter decision for exactly one ticket."""

    ticket: MicroBatchTicket
    value: Any = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class MicroBatchPlanner:
    """Pure grouping and scatter logic for a micro-batched GPU worker.

    Grouping is FIFO within a ``batch_key``.  The key must cover everything that
    changes a forward pass's tensor shapes or schedule (profile, domain, action
    row count and width, resolution tier, step count, guidance), so two tickets
    in one batch differ only in per-item data.  Requests with different keys are
    never mixed, because a "batch" whose members need different shapes is not one
    forward pass.
    """

    def __init__(self, batch_max: int, batch_window_ms: float) -> None:
        if batch_max < 1:
            raise ValueError("batch_max must be at least 1")
        if batch_window_ms < 0.0:
            raise ValueError("batch_window_ms cannot be negative")
        self.batch_max = int(batch_max)
        self.batch_window_seconds = float(batch_window_ms) / 1000.0
        self._pending: List[MicroBatchTicket] = []
        self._next_ticket_id = 0

    def next_ticket_id(self) -> int:
        self._next_ticket_id += 1
        return self._next_ticket_id

    def submit(self, ticket: MicroBatchTicket) -> None:
        self._pending.append(ticket)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def drop(self, predicate: Callable[[MicroBatchTicket], bool]) -> List[MicroBatchTicket]:
        """Remove tickets whose caller has gone away (e.g. a cancelled future)."""

        dropped = [ticket for ticket in self._pending if predicate(ticket)]
        if dropped:
            identifiers = {id(ticket) for ticket in dropped}
            self._pending = [ticket for ticket in self._pending if id(ticket) not in identifiers]
        return dropped

    def seconds_until_next_flush(self, now: float) -> Optional[float]:
        """Time until the oldest pending ticket's window expires, or ``None``."""

        if not self._pending:
            return None
        oldest = min(ticket.submitted_at for ticket in self._pending)
        return max(0.0, (oldest + self.batch_window_seconds) - now)

    def take_ready(self, now: float, *, force_flush: bool = False) -> List[List[MicroBatchTicket]]:
        """Return batches that may execute now, removing them from the queue.

        A group is emitted when it is full (``batch_max``) or when its oldest
        member has waited out the window.  Partial batches are legal; a partial
        batch is not padded with duplicated work to reach ``batch_max``.
        """

        grouped: Dict[str, List[MicroBatchTicket]] = {}
        order: List[str] = []
        for ticket in self._pending:
            if ticket.batch_key not in grouped:
                grouped[ticket.batch_key] = []
                order.append(ticket.batch_key)
            grouped[ticket.batch_key].append(ticket)

        batches: List[List[MicroBatchTicket]] = []
        remaining: List[MicroBatchTicket] = []
        for key in order:
            tickets = grouped[key]
            index = 0
            while len(tickets) - index >= self.batch_max:
                batches.append(tickets[index : index + self.batch_max])
                index += self.batch_max
            leftover = tickets[index:]
            if not leftover:
                continue
            oldest = min(ticket.submitted_at for ticket in leftover)
            expired = (now - oldest) >= self.batch_window_seconds
            if force_flush or expired:
                batches.append(leftover)
            else:
                remaining.extend(leftover)
        remaining.sort(key=lambda ticket: ticket.ticket_id)
        self._pending = remaining
        return batches

    @staticmethod
    def scatter(tickets: Sequence[MicroBatchTicket], outcomes: Any) -> List[BatchOutcome]:
        """Map executor output back onto individual tickets.

        Three cases, all fail-closed:

        * the executor raised for the whole batch -> every ticket gets that error;
        * it returned a different number of results than tickets -> every ticket
          gets an explicit :class:`BatchContractError`, because reusing another
          item's result would attribute one episode's frames to another;
        * otherwise results are paired positionally, and an exception in one
          slot is isolated to that one ticket.
        """

        if isinstance(outcomes, BaseException):
            return [BatchOutcome(ticket=ticket, error=outcomes) for ticket in tickets]
        try:
            materialised = list(outcomes)
        except TypeError:
            error = BatchContractError("batch executor returned a non-iterable result set")
            return [BatchOutcome(ticket=ticket, error=error) for ticket in tickets]
        if len(materialised) != len(tickets):
            error = BatchContractError(
                "batch executor returned %d results for %d requests; results are never reused across requests"
                % (len(materialised), len(tickets))
            )
            return [BatchOutcome(ticket=ticket, error=error) for ticket in tickets]
        scattered: List[BatchOutcome] = []
        for ticket, outcome in zip(tickets, materialised):
            if isinstance(outcome, BaseException):
                scattered.append(BatchOutcome(ticket=ticket, error=outcome))
            else:
                scattered.append(BatchOutcome(ticket=ticket, value=outcome))
        return scattered


class MicroBatchQueue:
    """asyncio collector around :class:`MicroBatchPlanner`.

    ``executor`` is a *blocking* callable: it runs on a worker thread through
    ``asyncio.to_thread`` so the event loop keeps accepting and grouping new
    requests while the GPU is busy.  Exactly one batch is in flight at a time,
    which is the correct shape for a single accelerator.
    """

    def __init__(
        self,
        executor: Callable[[List[Any]], Sequence[Any]],
        *,
        batch_max: int = BATCH_MAX,
        batch_window_ms: float = BATCH_WINDOW_MS,
        idle_tick_seconds: float = 0.002,
    ) -> None:
        self._executor = executor
        self._planner = MicroBatchPlanner(batch_max, batch_window_ms)
        self._idle_tick_seconds = max(0.0005, float(idle_tick_seconds))
        self._wakeup: Optional[Any] = None
        self._drain_task: Optional[Any] = None
        # Tickets handed to the executor. ``take_ready`` has already removed them
        # from the planner, so if the drain loop dies mid-execution they are in
        # neither place and their callers would await forever. Tracking them lets
        # the failure path resolve them.
        self._inflight: List[MicroBatchTicket] = []
        # Set when the collector dies, for diagnostics. Nothing reads it to make a
        # decision: the error itself has already been delivered to every caller.
        self.last_drain_error: Optional[BaseException] = None

    @property
    def planner(self) -> MicroBatchPlanner:
        return self._planner

    async def submit(self, batch_key: str, payload: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        ticket = MicroBatchTicket(
            ticket_id=self._planner.next_ticket_id(),
            batch_key=batch_key,
            payload=payload,
            submitted_at=time.monotonic(),
            future=future,
        )
        self._planner.submit(ticket)
        self._ensure_drain_task()
        if self._wakeup is not None:
            self._wakeup.set()
        return await future

    def _ensure_drain_task(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            return
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
        self._drain_task = asyncio.ensure_future(self._drain_forever())

    async def _drain_forever(self) -> None:
        assert self._wakeup is not None
        try:
            while True:
                if self._planner.pending_count == 0:
                    self._wakeup.clear()
                    try:
                        await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if self._planner.pending_count == 0:
                            return
                self._planner.drop(lambda ticket: ticket.future is not None and ticket.future.done())
                batches = self._planner.take_ready(time.monotonic())
                if not batches:
                    delay = self._planner.seconds_until_next_flush(time.monotonic())
                    await asyncio.sleep(self._idle_tick_seconds if delay is None else min(delay, 0.05))
                    continue
                for batch in batches:
                    await self._execute(batch)
        except asyncio.CancelledError as error:
            # Cancellation must propagate so the event loop can shut down, but the
            # waiting callers still get an explicit outcome first.
            self._fail_all_pending(error)
            raise
        except BaseException as error:  # noqa: BLE001 - delivered per caller below
            # A crashed collector must never leave callers awaiting forever: a
            # hung episode is worse than an explicit failure because it consumes
            # the run deadline while producing no record at all.
            #
            # The error is NOT re-raised. Every ticket this queue owned has just
            # received it, so re-raising would only add an "exception was never
            # retrieved" warning to the deployment logs for an error that has
            # already been reported to the only parties that can act on it. The
            # next submit starts a fresh collector.
            self.last_drain_error = error
            self._fail_all_pending(error)
            return

    def _fail_all_pending(self, error: BaseException) -> None:
        """Resolve every ticket this queue still owns, in flight or queued."""

        stranded = list(self._inflight) + self._planner.drop(lambda _ticket: True)
        self._inflight = []
        for ticket in stranded:
            future = ticket.future
            if future is None or future.done():
                continue
            try:
                future.set_exception(error)
            except Exception:  # noqa: BLE001 - a resolved/cancelled future is fine
                pass

    async def _execute(self, tickets: List[MicroBatchTicket]) -> None:
        payloads = [ticket.payload for ticket in tickets]
        self._inflight = list(tickets)
        try:
            try:
                raw = await asyncio.to_thread(self._executor, payloads)
            except BaseException as error:  # noqa: BLE001 - re-raised per ticket below
                raw = error
            for outcome in MicroBatchPlanner.scatter(tickets, raw):
                future = outcome.ticket.future
                if future is None or future.done():
                    continue
                if outcome.error is not None:
                    future.set_exception(outcome.error)
                else:
                    future.set_result(outcome.value)
        finally:
            # Cleared even if scatter or future resolution raised, so the drain
            # loop's failure handler sees an accurate in-flight set.
            self._inflight = [ticket for ticket in tickets if ticket.future is not None and not ticket.future.done()]


# ===========================================================================
# Policy arms
# ===========================================================================

POLICY_ARM_OPENVLA = "OpenVLA"
POLICY_ARM_OCTO_SMALL = "Octo-Small"
POLICY_ARM_OCTO_BASE = "Octo-Base"
POLICY_ARM_MINIVLA = "MiniVLA"
POLICY_ARM_OPEN_PI_ZERO = "OpenPiZero"
POLICY_ARM_SUSIE = "SuSIE"
POLICY_ARM_SUSIE_LL = "SuSIE_LL"

CANONICAL_POLICY_ARMS: Tuple[str, ...] = (
    POLICY_ARM_OPENVLA,
    POLICY_ARM_OCTO_SMALL,
    POLICY_ARM_OCTO_BASE,
    POLICY_ARM_MINIVLA,
    POLICY_ARM_OPEN_PI_ZERO,
    POLICY_ARM_SUSIE,
    POLICY_ARM_SUSIE_LL,
)

# Aliases are an explicit table, never a fuzzy match: silently mapping an unknown
# policy string onto a worker would attribute one policy's numbers to another.
POLICY_ARM_ALIASES: Mapping[str, str] = {
    "openvla": POLICY_ARM_OPENVLA,
    "openvla-7b": POLICY_ARM_OPENVLA,
    "octo": POLICY_ARM_OCTO_SMALL,
    "octo-small": POLICY_ARM_OCTO_SMALL,
    "octo_small": POLICY_ARM_OCTO_SMALL,
    "octo-small v1.0": POLICY_ARM_OCTO_SMALL,
    "octo-base": POLICY_ARM_OCTO_BASE,
    "octo_base": POLICY_ARM_OCTO_BASE,
    "octo-base v1.0 diagnostic": POLICY_ARM_OCTO_BASE,
    "minivla": POLICY_ARM_MINIVLA,
    "mini-vla": POLICY_ARM_MINIVLA,
    "openpizero": POLICY_ARM_OPEN_PI_ZERO,
    "open-pi-zero": POLICY_ARM_OPEN_PI_ZERO,
    "open_pi_zero": POLICY_ARM_OPEN_PI_ZERO,
    "susie": POLICY_ARM_SUSIE,
    "susie_ll": POLICY_ARM_SUSIE_LL,
    "susie-ll": POLICY_ARM_SUSIE_LL,
    "susie_lowlevel": POLICY_ARM_SUSIE_LL,
}

# Worker -> arms it serves.  SusieWorker serves both SuSIE arms from one image;
# the arm is selected by the request field, never inferred from the worker.
WORKER_ARMS: Mapping[str, Tuple[str, ...]] = {
    "plumb-openvla-worker": (POLICY_ARM_OPENVLA,),
    "plumb-octo-worker": (POLICY_ARM_OCTO_SMALL, POLICY_ARM_OCTO_BASE),
    "plumb-minivla-worker": (POLICY_ARM_MINIVLA,),
    "plumb-openpizero-worker": (POLICY_ARM_OPEN_PI_ZERO,),
    "plumb-susie-worker": (POLICY_ARM_SUSIE, POLICY_ARM_SUSIE_LL),
}

# ``plumb.policies`` class names for the arms whose native wrapper lives in
# plumb/policies/native.py.  OpenVLA is handled separately because it has a real
# bundled adapter with a different constructor.
NATIVE_POLICY_CLASSES: Mapping[str, str] = {
    POLICY_ARM_OCTO_SMALL: "OctoSmallV1Policy",
    POLICY_ARM_OCTO_BASE: "OctoBaseV1Policy",
    POLICY_ARM_MINIVLA: "MiniVLAPolicy",
    POLICY_ARM_OPEN_PI_ZERO: "OpenPiZeroPolicy",
    POLICY_ARM_SUSIE: "SuSIEPolicy",
    POLICY_ARM_SUSIE_LL: "SuSIELowLevelPolicy",
}

# plumb ships BOTH a non-fabricating ``ExternalPolicyHook`` per arm and, for the
# four incompatible stacks, a real evidence-bound ``*PolicyAdapter``.  The Chain
# constructs the hook, because every certified adapter requires provenance this
# Chain must not invent: converted-checkpoint digests, conversion-report hashes,
# PaliGemma support-file records, a pinned PolicyActionNormalizer.  Naming the
# adapter and its required profile fields in the block reason turns "no loader"
# into an actionable list.
NATIVE_POLICY_CERTIFIED_ADAPTERS: Mapping[str, Tuple[str, str]] = {
    POLICY_ARM_OCTO_SMALL: ("OctoSmallV1PolicyAdapter", "OctoPolicyProfile"),
    POLICY_ARM_OCTO_BASE: ("OctoBaseV1PolicyAdapter", "OctoPolicyProfile"),
    POLICY_ARM_MINIVLA: ("MiniVLAPolicyAdapter", "MiniVLAPolicyProfile"),
    POLICY_ARM_OPEN_PI_ZERO: ("OpenPiZeroPolicyAdapter", "OpenPiZeroPolicyProfile"),
    POLICY_ARM_SUSIE: ("SuSIEPolicyAdapter", "SuSIEPolicyProfile"),
    POLICY_ARM_SUSIE_LL: ("SuSIELowLevelPolicyAdapter", "SuSIEPolicyProfile"),
}

POLICY_ARM_CACHE_VOLUME: Mapping[str, str] = {
    POLICY_ARM_OPENVLA: OPENVLA_VOLUME,
    POLICY_ARM_OCTO_SMALL: "octo-small-v1",
    POLICY_ARM_OCTO_BASE: "octo-base-v1",
    POLICY_ARM_MINIVLA: "minivla-vq-bridge-prismatic",
    POLICY_ARM_OPEN_PI_ZERO: "open-pi-zero",
    POLICY_ARM_SUSIE: "susie",
    POLICY_ARM_SUSIE_LL: "gcbc-bridge",
}


def certified_adapter_requirements(policies: Any, arm: str) -> Optional[Dict[str, Any]]:
    """Describe the evidence-bound adapter for ``arm``, if plumb ships one.

    Discovered from the live classes rather than hard-coded, so the list of
    required profile fields cannot go stale when plumb adds one.
    """

    names = NATIVE_POLICY_CERTIFIED_ADAPTERS.get(arm)
    if names is None:
        return None
    adapter_name, profile_name = names
    if getattr(policies, adapter_name, None) is None:
        return None
    profile_cls = getattr(policies, profile_name, None)
    required: List[str] = []
    if profile_cls is not None:
        try:
            import dataclasses

            required = [
                item.name
                for item in dataclasses.fields(profile_cls)
                if item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING
            ]
        except (TypeError, AttributeError):
            required = []
    return {
        "adapter": "plumb.policies.%s" % adapter_name,
        "profile": "plumb.policies.%s" % profile_name,
        "required_profile_fields": required,
        "additional_required_arguments": ["normalizer (plumb.policies PolicyActionNormalizer)"],
        "why_the_chain_does_not_construct_it": (
            "Those fields are provenance: converted-artifact digests, conversion-report hashes, immutable "
            "checkpoint revisions, and a pinned action normalizer. The Chain has none of them and will not "
            "invent them, so it loads the non-fabricating hook and reports blocked instead."
        ),
    }


def normalise_policy_arm(value: str) -> Optional[str]:
    """Resolve a request's policy string to a canonical arm, or ``None``."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in CANONICAL_POLICY_ARMS:
        return text
    return POLICY_ARM_ALIASES.get(text.lower())


# ===========================================================================
# Autoscaling hypotheses
#
# truss 0.18.30's ``chains.RemoteConfig`` has NO autoscaling field: grep the
# installed SDK for ``min_replica`` and the only hit is a docstring.  Replica and
# concurrency-target settings are therefore applied through the Baseten
# management API *after* the push, which is exactly the side
# ``plumb/platform.py`` already reads back from
# ``chainlet.autoscaling_settings``.  The values are declared here as data so the
# deploy step is copy-pasteable and reviewable rather than typed by hand into a
# web form, and they are also attached to each Chainlet's
# ``ChainletOptions.metadata`` so the deployed artifact records the hypothesis it
# was launched with.
#
# THESE ARE INITIAL HYPOTHESES, NOT MEASUREMENTS.  AGENT-BUILD-SPEC.md:285
# states there is no assumed universal throughput preset and that hardware,
# batching, and concurrency must be chosen from measured memory and full-chain
# throughput.  Every number below must be re-tuned from Gate A measurements on
# Baseten hardware before it is quoted in a capacity or cost claim, and a
# configured ``max_replica`` is neither an active replica count nor reserved GPU
# capacity.
# ===========================================================================


@dataclass(frozen=True)
class AutoscalingHypothesis:
    """Post-push autoscaling settings for one Chainlet, with its rationale."""

    chainlet_name: str
    min_replica: int
    max_replica: int
    concurrency_target: int
    autoscaling_window_seconds: int
    rationale: str

    def as_settings(self) -> Dict[str, int]:
        return {
            "min_replica": self.min_replica,
            "max_replica": self.max_replica,
            "concurrency_target": self.concurrency_target,
            "autoscaling_window": self.autoscaling_window_seconds,
        }


AUTOSCALING_HYPOTHESES: Tuple[AutoscalingHypothesis, ...] = (
    AutoscalingHypothesis(
        chainlet_name="plumb-world-worker",
        min_replica=1,
        max_replica=WORLD_MAX_REPLICA,
        # §7: "Begin world-model benchmarking on one H100 with concurrency_target=1
        # and matching in-container concurrency."  Diffusion saturates the GPU, so
        # a queued second request would not overlap usefully; the throughput lever
        # is batch packing inside one forward plus replica count.
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale=(
            "concurrency_target=1 is the spec's starting point for a saturating diffusion forward. "
            "max_replica is the 100-replica capacity hypothesis, configurable via %s. Re-tune from Gate A." % ENV_WORLD_MAX_REPLICA
        ),
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-judge-worker",
        min_replica=1,
        max_replica=JUDGE_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale=(
            "Five sequential 512-token VLM samples per clip saturate one GPU. max_replica is configurable "
            "via %s; the recorded 29.29 s five-sample call was on a different cluster and does not size this." % ENV_JUDGE_MAX_REPLICA
        ),
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-openvla-worker",
        min_replica=1,
        max_replica=POLICY_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale="One native action per fresh image; the call is short but GPU-bound. Re-tune from Gate A.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-octo-worker",
        min_replica=0,
        max_replica=POLICY_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale="min_replica=0: no reviewed loader exists, so a warm replica would burn GPU-hours to return blocked.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-minivla-worker",
        min_replica=0,
        max_replica=POLICY_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale="min_replica=0: no reviewed loader exists, so a warm replica would burn GPU-hours to return blocked.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-openpizero-worker",
        min_replica=0,
        max_replica=POLICY_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale="min_replica=0: no reviewed loader exists, so a warm replica would burn GPU-hours to return blocked.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-susie-worker",
        min_replica=0,
        max_replica=POLICY_MAX_REPLICA,
        concurrency_target=1,
        autoscaling_window_seconds=60,
        rationale="min_replica=0: serves both SuSIE arms but neither loader is wired; a warm replica would only return blocked.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-validity-worker",
        min_replica=1,
        max_replica=VALIDITY_MAX_REPLICA,
        # CPU-bound deterministic checks overlap well inside one replica.
        concurrency_target=4,
        autoscaling_window_seconds=60,
        rationale="CPU-only deterministic gate; concurrency_target>1 is safe because no accelerator is contended.",
    ),
    AutoscalingHypothesis(
        chainlet_name="plumb-rollout-controller",
        min_replica=1,
        max_replica=CONTROLLER_MAX_REPLICA,
        # The entrypoint spends nearly all of its wall time awaiting Chainlet
        # RPCs, so one replica can hold many in-flight episodes.
        concurrency_target=32,
        autoscaling_window_seconds=60,
        rationale="Entrypoint is IO-bound on awaited Chainlet RPCs; it holds no model and must not be the burst bottleneck.",
    ),
)


def autoscaling_patch_payloads() -> List[Dict[str, Any]]:
    """Return the per-Chainlet autoscaling bodies to PATCH after a push.

    The exact management-API route and accepted body must be confirmed against
    the Baseten management-API reference at deploy time; see DEPLOY.md step 7.
    This function only supplies the *values*, each labelled as a hypothesis.
    """

    return [
        {
            "chainlet_name": hypothesis.chainlet_name,
            "autoscaling_settings": hypothesis.as_settings(),
            "status": "initial_hypothesis_not_measured",
            "rationale": hypothesis.rationale,
        }
        for hypothesis in AUTOSCALING_HYPOTHESES
    ]


def _chainlet_metadata(chainlet_name: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Metadata recorded on the deployed Chainlet.  Descriptive, not a claim."""

    hypothesis = next(
        (item for item in AUTOSCALING_HYPOTHESES if item.chainlet_name == chainlet_name),
        None,
    )
    metadata: Dict[str, Any] = {
        "project": "PLUMB",
        "chainlet": chainlet_name,
        "qualification": "unqualified",
        "evidence_note": (
            "Timings and GPU peaks in deploy/baseten/cluster-runtime-evidence.json were measured on "
            "ComputeCanada clusters with +computecanada wheel builds. They do not transfer to this image."
        ),
        "autoscaling_hypothesis": hypothesis.as_settings() if hypothesis is not None else None,
        "autoscaling_hypothesis_status": "initial_hypothesis_applied_out_of_band_via_management_api",
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


# Declared GPU counts, used only to label instrumented GPU-seconds.  They are
# the numbers this file requests, not an observation of what the account granted.
WORKER_DECLARED_GPU_COUNT: Mapping[str, int] = {
    "plumb-world-worker": 1,
    "plumb-judge-worker": 1,
    "plumb-openvla-worker": 1,
    "plumb-octo-worker": 1,
    "plumb-minivla-worker": 1,
    "plumb-openpizero-worker": 1,
    "plumb-susie-worker": 1,
    "plumb-validity-worker": 0,
    "plumb-rollout-controller": 0,
}


# ===========================================================================
# World-model variants
# ===========================================================================


@dataclass(frozen=True)
class WorldVariant:
    """One selectable world-model arm.

    ``metadata_model_id_is_profile_default`` records a real limitation:
    ``plumb.adapters.worlds.Cosmos3NanoDiffusersProfile.as_backend_profile()``
    hard-codes ``model_id="nvidia/Cosmos3-Nano"``.  Running the Edge weights
    through the same profile dataclass would therefore emit a Nano label in the
    adapter's own metadata.  This worker publishes its own authoritative
    ``model_id``/``revision`` alongside and flags the mismatch instead of letting
    a mislabelled record look clean.
    """

    variant_id: str
    repo_id: str
    revision: str
    volume_folder: str
    parameter_note: str
    metadata_model_id_is_profile_default: bool


WORLD_VARIANTS: Mapping[str, WorldVariant] = {
    "cosmos3_nano": WorldVariant(
        variant_id="cosmos3_nano",
        repo_id=COSMOS3_NANO_REPO,
        revision=COSMOS3_NANO_REVISION,
        volume_folder=COSMOS3_NANO_VOLUME,
        parameter_note="Primary arm. Historical download estimate ~33 GB; download size is not a memory fit guarantee.",
        metadata_model_id_is_profile_default=False,
    ),
    "cosmos3_edge": WorldVariant(
        variant_id="cosmos3_edge",
        repo_id=COSMOS3_EDGE_REPO,
        # No immutable Edge revision has been recorded by PLUMB. The operator
        # supplies one at push time; otherwise the arm stays unselectable.
        revision=WORLD_EDGE_REVISION,
        volume_folder=COSMOS3_EDGE_VOLUME,
        parameter_note="Speed arm; its model card lists 4B. Excluded Super (64B) from the single-H100 design.",
        metadata_model_id_is_profile_default=False,
    ),
}


def selected_world_variant() -> WorldVariant:
    return WORLD_VARIANTS.get(WORLD_VARIANT, WORLD_VARIANTS["cosmos3_nano"])


def _is_immutable_revision(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower())


# ===========================================================================
# Adapter cores.  These are plain classes, not Chainlets: keeping the model work
# out of the Chainlet subclasses means the same code can be reviewed and reasoned
# about without the framework, and no accidental second Chainlet gets registered.
# ===========================================================================


class WorldAdapterCore:
    """Owns exactly one Cosmos pipeline for the lifetime of a replica."""

    def __init__(self) -> None:
        self.variant = selected_world_variant()
        self.local_model_path = str(pathlib.PurePosixPath(MODEL_CACHE_ROOT) / self.variant.volume_folder)
        self.profile_id = "plumb-%s-fd-r%d" % (self.variant.variant_id, WORLD_RESOLUTION_TIER)
        self.runtime_lock_path = REQUIREMENTS_DIR / "world-cosmos.txt"
        self.runtime_lock_sha256 = _file_sha256(self.runtime_lock_path)
        self.fused_batch_enabled = _env_flag(ENV_WORLD_FUSED_BATCH)
        self.unresolved: List[str] = []
        self.adapter: Any = None
        self.profile: Any = None
        self._capability: Any = None
        self._load()

    def _load(self) -> None:
        if not _is_immutable_revision(self.variant.revision):
            self.unresolved.append(
                "World variant %r has no immutable 40-hex model revision (set %s for the Edge arm); "
                "a floating revision cannot be pushed under a frozen protocol."
                % (self.variant.variant_id, ENV_WORLD_EDGE_REVISION)
            )
            return
        try:
            from plumb.adapters.worlds import (  # type: ignore
                Cosmos3NanoDiffusersAdapter,
                Cosmos3NanoDiffusersProfile,
            )
        except ImportError as error:
            self.unresolved.append(
                "plumb.adapters.worlds is not importable inside this image (%s); stage the plumb package with "
                "deploy/baseten/stage_packages.py before pushing." % error
            )
            return
        try:
            # ``model_id`` is passed explicitly so the selected arm labels
            # itself. The profile used to hard-code the Nano repo id, which meant
            # selecting the Edge arm emitted a *Nano* label into the backend
            # profile hash and every episode record.
            self.profile = Cosmos3NanoDiffusersProfile(
                profile_id=self.profile_id,
                local_model_path=self.local_model_path,
                model_id=self.variant.repo_id,
                model_revision=self.variant.revision,
                resolution_tier=WORLD_RESOLUTION_TIER,
            )
            self.adapter = Cosmos3NanoDiffusersAdapter(self.profile)
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            self.unresolved.append("Cosmos adapter construction failed: %s" % error)
            return
        self._capability = self.adapter.capability()
        status = getattr(self._capability, "status", None)
        if status is not None and getattr(status, "value", str(status)) != "ready_unqualified":
            self.unresolved.append(
                "Cosmos adapter capability is %s: %s"
                % (getattr(status, "value", status), getattr(self._capability, "reason", "no reason supplied"))
            )

    @property
    def ready(self) -> bool:
        return self.adapter is not None and not self.unresolved

    def batch_key(
        self,
        action_rows: int,
        action_width: int,
        domain: str,
        compatibility_profile_id: Optional[str] = None,
    ) -> str:
        """Everything that changes the forward's shapes or schedule.

        ``compatibility_profile_id`` is the id the *request* declares, which the
        adapter's own key also covers.  Including it here keeps the two keys
        aligned: without it a payload naming a different profile would land in
        this group and then make the adapter reject the entire batch as mixed,
        costing every other episode in it their frames.
        """

        profile = self.profile
        return "|".join(
            str(part)
            for part in (
                self.profile_id,
                compatibility_profile_id or self.profile_id,
                domain,
                action_rows,
                action_width,
                getattr(profile, "resolution_tier", WORLD_RESOLUTION_TIER),
                getattr(profile, "num_inference_steps", None),
                getattr(profile, "guidance_scale", None),
                getattr(profile, "fps", None),
                getattr(profile, "view_point", None),
            )
        )

    def execute_batch(self, requests: List[Any]) -> List[Any]:
        """Run one collected batch.  Blocking; called on a worker thread.

        A genuinely *fused* diffusion forward requires the adapter to accept a
        batched ``CosmosActionCondition``.  ``Cosmos3NanoDiffusersAdapter``
        exposes only ``generate(request)``, so the honest default is to run the
        collected batch as consecutive adapter calls while still holding the GPU
        for the whole group, and to report ``batch_execution_mode`` accordingly.
        If a reviewed ``generate_batch`` is added to the adapter, this worker uses
        it automatically once PLUMB_WORLD_FUSED_BATCH is set.  It never claims one
        forward when it made several.
        """

        fused = getattr(self.adapter, "generate_batch", None)
        if len(requests) > 1 and self.fused_batch_enabled and callable(fused):
            try:
                results = list(fused(requests))
            except BaseException as error:  # noqa: BLE001 - scattered per ticket
                return [error for _ in requests]
            if len(results) != len(requests):
                return [
                    BatchContractError(
                        "adapter.generate_batch returned %d results for %d requests" % (len(results), len(requests))
                    )
                    for _ in requests
                ]
            # Per-item failures arrive in band as exception instances. Wrapping
            # one as a value would surface it downstream as a confusing
            # attribute error on a non-result; mirror the sequential path and
            # let it stay an exception for its own slot only.
            #
            # The fused size comes from the result, not from len(requests):
            # invalid members are excluded from the forward, so len(requests)
            # can overstate the batch and corrupt the GPU-second divisor.
            outcomes: List[Any] = []
            for result in results:
                if isinstance(result, BaseException):
                    outcomes.append(result)
                    continue
                timing = getattr(result, "timing", None)
                fused_size = getattr(timing, "batch_size", None)
                if not isinstance(fused_size, int) or fused_size < 1:
                    metadata = getattr(result, "metadata", None)
                    if isinstance(metadata, Mapping):
                        candidate = metadata.get("batch_size")
                        if isinstance(candidate, int) and candidate >= 1:
                            fused_size = candidate
                if not isinstance(fused_size, int) or fused_size < 1:
                    # Never guess the divisor: an unlabelled fused call is
                    # reported as a single-item call so GPU-seconds are not
                    # silently divided by a number nobody measured.
                    fused_size = 1
                outcomes.append((result, "fused_adapter_generate_batch", fused_size))
            return outcomes
        outcomes: List[Any] = []
        mode = "sequential_adapter_calls" if len(requests) > 1 else "single_request"
        for request in requests:
            try:
                outcomes.append((self.adapter.generate(request), mode, len(requests)))
            except BaseException as error:  # noqa: BLE001 - isolated to this item
                outcomes.append(error)
        return outcomes



class NativePolicyCore:
    """Loads one policy stack's adapters once per replica.

    Every failure path records a reason and leaves the arm blocked.  Nothing here
    substitutes another policy, repeats a stale action, or pads a proposal.
    """

    def __init__(self, worker_name: str, runtime_lock_file: str) -> None:
        self.worker_name = worker_name
        self.arms: Tuple[str, ...] = WORKER_ARMS[worker_name]
        self.runtime_lock_path = REQUIREMENTS_DIR / runtime_lock_file
        self.runtime_lock_sha256 = _file_sha256(self.runtime_lock_path)
        self.adapters: Dict[str, Any] = {}
        self.blocked_reasons: Dict[str, List[str]] = {}
        self.contracts: Dict[str, Any] = {}
        self.certified_adapter: Dict[str, Dict[str, Any]] = {}
        for arm in self.arms:
            self._load_arm(arm)

    def _cache_path(self, arm: str, override: Optional[str]) -> str:
        if override:
            return override
        return str(pathlib.PurePosixPath(MODEL_CACHE_ROOT) / POLICY_ARM_CACHE_VOLUME[arm])

    def _load_arm(self, arm: str) -> None:
        try:
            import plumb.policies as policies  # type: ignore
        except ImportError as error:
            self.blocked_reasons[arm] = [
                "plumb.policies is not importable inside this image (%s); stage the plumb package with "
                "deploy/baseten/stage_packages.py before pushing." % error
            ]
            return
        if arm == POLICY_ARM_OPENVLA:
            self._load_openvla(policies)
            return
        class_name = NATIVE_POLICY_CLASSES.get(arm)
        adapter_cls = getattr(policies, class_name, None) if class_name else None
        if adapter_cls is None:
            self.blocked_reasons[arm] = [
                "plumb.policies does not export %s; the native adapter for %s is not available in this image."
                % (class_name, arm)
            ]
            return
        profile_cls = getattr(policies, "ExternalPolicyProfile", None)
        if profile_cls is None:
            self.blocked_reasons[arm] = ["plumb.policies does not export ExternalPolicyProfile."]
            return
        requirements = certified_adapter_requirements(policies, arm)
        if requirements is not None:
            self.certified_adapter[arm] = requirements
        try:
            profile = profile_cls(
                profile_id="plumb-%s" % arm.lower().replace(" ", "-"),
                local_model_path=self._cache_path(arm, None),
            )
            adapter = adapter_cls(profile)
        except Exception as error:  # noqa: BLE001 - recorded as a block reason
            self.blocked_reasons[arm] = ["%s construction failed: %s" % (class_name, error)]
            return
        self.adapters[arm] = adapter
        contract = getattr(adapter, "contract", None)
        if contract is not None:
            self.contracts[arm] = contract
        capability = getattr(adapter, "capability", None)
        if callable(capability):
            result = capability()
            status = getattr(getattr(result, "status", None), "value", None)
            if status not in ("ready_unqualified", "qualified"):
                reasons = [
                    "%s capability is %s: %s"
                    % (arm, status, getattr(result, "reason", "no reason supplied")),
                    "AGENT-BUILD-SPEC.md §2 leaves this wrapper's executed prefix and feedback cadence "
                    "unresolved until its own fixtures pass; no action is emitted meanwhile.",
                ]
                requirements = self.certified_adapter.get(arm)
                if requirements is not None:
                    reasons.append(
                        "A certified loader exists as %s. Constructing it requires %s(%s) plus a pinned "
                        "action normalizer, all of which are provenance values this Chain must not invent."
                        % (
                            requirements["adapter"],
                            requirements["profile"],
                            ", ".join(requirements["required_profile_fields"]) or "reviewed fields",
                        )
                    )
                self.blocked_reasons[arm] = reasons

    def _load_openvla(self, policies: Any) -> None:
        adapter_cls = getattr(policies, "OpenVLAPolicyAdapter", None)
        profile_cls = getattr(policies, "OpenVLAPolicyProfile", None)
        if adapter_cls is None or profile_cls is None:
            self.blocked_reasons[POLICY_ARM_OPENVLA] = [
                "plumb.policies does not export OpenVLAPolicyAdapter/OpenVLAPolicyProfile."
            ]
            return
        acknowledged = _env_text(ENV_OPENVLA_REVIEWED_REMOTE_CODE, "").strip().lower()
        if acknowledged != OPENVLA_REVISION:
            self.blocked_reasons[POLICY_ARM_OPENVLA] = [
                "OpenVLA requires trust_remote_code. Loading stays blocked until the operator sets %s to the "
                "exact reviewed remote-code revision %s, which records that the review happened rather than "
                "inferring consent from a successful download."
                % (ENV_OPENVLA_REVIEWED_REMOTE_CODE, OPENVLA_REVISION)
            ]
            return
        manifest_id, manifest_sha = _asset_manifest_identity()
        try:
            profile = profile_cls(
                profile_id="plumb-openvla-bridge",
                local_model_path=self._cache_path(POLICY_ARM_OPENVLA, None),
                checkpoint_revision=OPENVLA_REVISION,
                remote_code_revision=OPENVLA_REVISION,
                reviewed_remote_code_revision=OPENVLA_REVISION,
                allow_trust_remote_code=True,
                asset_manifest_id=manifest_id,
                asset_manifest_sha256=manifest_sha,
                runtime_lock_id="requirements/policy-openvla.txt",
                runtime_lock_sha256=self.runtime_lock_sha256,
            )
            adapter = adapter_cls(profile)
        except Exception as error:  # noqa: BLE001 - recorded as a block reason
            self.blocked_reasons[POLICY_ARM_OPENVLA] = ["OpenVLAPolicyAdapter construction failed: %s" % error]
            return
        self.adapters[POLICY_ARM_OPENVLA] = adapter
        self.contracts[POLICY_ARM_OPENVLA] = getattr(adapter, "contract", None)
        result = adapter.capability()
        status = getattr(getattr(result, "status", None), "value", None)
        if status not in ("ready_unqualified", "qualified"):
            self.blocked_reasons[POLICY_ARM_OPENVLA] = [
                "OpenVLA capability is %s: %s" % (status, getattr(result, "reason", "no reason supplied"))
            ]

    def contract_summary(self, arm: str) -> Dict[str, Any]:
        contract = self.contracts.get(arm)
        if contract is None:
            return {"name": arm, "certified_execute_prefix": None}
        return {
            "name": getattr(contract, "name", arm),
            "required_observation_history": getattr(contract, "required_observation_history", None),
            "requires_proprio": getattr(contract, "requires_proprio", None),
            "native_proposal_horizon": getattr(contract, "native_proposal_horizon", None),
            "certified_execute_prefix": getattr(contract, "certified_execute_prefix", None),
            "temporal_ensembling": getattr(contract, "temporal_ensembling", None),
            "normalization": getattr(contract, "normalization", None),
            "implementation_status": getattr(getattr(contract, "implementation_status", None), "value", None),
            "limitation": getattr(contract, "limitation", None),
        }

    def predict(self, arm: str, observation: Any) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Return the native proposal exactly as the wrapper produced it."""

        adapter = self.adapters[arm]
        started = time.perf_counter()
        if hasattr(adapter, "predict_with_report"):
            report = adapter.predict_with_report(observation)
            actions = [list(float(value) for value in report.action)]
            details = {
                "wall_seconds": getattr(report, "wall_seconds", None),
                "gpu_peak_memory_bytes": getattr(report, "gpu_peak_memory_bytes", None),
                "backend_calls": getattr(report, "backend_calls", None),
                "unnorm_key": getattr(report, "unnorm_key", None),
            }
            return actions, details
        raw = adapter.predict(observation)
        elapsed = time.perf_counter() - started
        rows = list(raw)
        if rows and isinstance(rows[0], (int, float)):
            rows = [rows]
        actions = [[float(value) for value in row] for row in rows]
        return actions, {"wall_seconds": elapsed, "gpu_peak_memory_bytes": None, "backend_calls": 1}


def _asset_manifest_identity() -> Tuple[Optional[str], Optional[str]]:
    """Read ``assets.lock.json`` identity if the operator mounted one.

    AGENT-BUILD-SPEC.md §1 requires this lock before Gate A.  It does not exist
    in this repository, so both values stay ``None`` and every judge report is
    correctly reported as Gate-D unavailable rather than silently eligible.
    """

    raw_path = _env_text(ENV_ASSET_MANIFEST_PATH, "")
    if not raw_path:
        return None, None
    path = pathlib.Path(raw_path)
    digest = _file_sha256(path)
    if digest is None:
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    manifest_id = payload.get("manifest_id") if isinstance(payload, dict) else None
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        return None, None
    return manifest_id, digest[len("sha256:") :]


class ValidityGateCore:
    """Binds ``plumb.validity.StageAValidityGate`` without guessing its API."""

    # Mirrors plumb.validity.StageAValidityGate.evaluate's keyword-only
    # signature exactly.  The gate expects len(frames) == len(actions) + 1, i.e.
    # the original start frame plus one generated frame per executed action, with
    # the start frame ALSO supplied separately as ``conditioning_frame`` so the
    # gate can verify that frames[0] really is the conditioning image.
    CANONICAL_KEYWORDS: Tuple[str, ...] = (
        "frames",
        "actions",
        "nominal_timestamps",
        "conditioning_frame",
        "states",
        "expected_static",
        "episode_id",
    )

    def __init__(self, calibration_class: str = VALIDITY_CALIBRATION_CLASS) -> None:
        self.runtime_lock_path = REQUIREMENTS_DIR / "validity.txt"
        self.runtime_lock_sha256 = _file_sha256(self.runtime_lock_path)
        self.calibration_class = calibration_class
        self.gate: Any = None
        self.unresolved: List[str] = []
        self.accepted_keywords: Tuple[str, ...] = ()
        self.parameters_hash: Optional[str] = None
        self.motion_reference_missing: Optional[bool] = None
        self._load()

    def _load(self) -> None:
        try:
            from plumb.validity import StageAValidityGate  # type: ignore
        except ImportError as error:
            self.unresolved.append(
                "plumb.validity.StageAValidityGate is unavailable in this image (%s). The deterministic Stage-A "
                "gate is a hard prerequisite for primary scoring and is never approximated here." % error
            )
            return
        try:
            self.gate = StageAValidityGate(calibration_class=self.calibration_class)
        except Exception as error:  # noqa: BLE001 - recorded as a block reason
            self.unresolved.append(
                "StageAValidityGate construction failed for calibration_class=%r: %s"
                % (self.calibration_class, error)
            )
            return
        # A real, deployable parameter pin computed from the gate's own declared
        # parameters and resolved flow estimator.
        self.parameters_hash = getattr(self.gate, "parameters_hash", None)
        # No MotionReference is mounted, so the gate returns `unknown` with
        # `no_motion_reference` rather than equating low optical flow with
        # invalidity. That is a real outcome, not a failure of this worker, so it
        # is recorded here instead of blocking the stage.
        if getattr(self.gate, "motion_reference", None) is None:
            self.motion_reference_missing = True
        else:
            self.motion_reference_missing = False
        evaluate = getattr(self.gate, "evaluate", None)
        if not callable(evaluate):
            self.unresolved.append("StageAValidityGate has no callable .evaluate(...)")
            self.gate = None
            return
        try:
            signature = inspect.signature(evaluate)
        except (TypeError, ValueError):
            self.accepted_keywords = self.CANONICAL_KEYWORDS
            return
        accepts_var_keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        if accepts_var_keyword:
            self.accepted_keywords = self.CANONICAL_KEYWORDS
        else:
            self.accepted_keywords = tuple(
                name for name in self.CANONICAL_KEYWORDS if name in signature.parameters
            )
        required = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and name != "self"
        ]
        missing = [name for name in required if name not in self.CANONICAL_KEYWORDS]
        if missing:
            self.unresolved.append(
                "StageAValidityGate.evaluate requires parameters this Chainlet cannot supply: %s. Expected "
                "keyword contract: %s." % (", ".join(missing), ", ".join(self.CANONICAL_KEYWORDS))
            )
            self.gate = None

    @property
    def ready(self) -> bool:
        return self.gate is not None and not self.unresolved

    def evaluate(self, **kwargs: Any) -> Tuple[str, List[str], Dict[str, Any]]:
        filtered = {name: value for name, value in kwargs.items() if name in self.accepted_keywords}
        result = self.gate.evaluate(**filtered)
        return _normalise_validity_result(result)


def _normalise_validity_result(result: Any) -> Tuple[str, List[str], Dict[str, Any]]:
    """Accept a mapping or an object; reject anything outside the three labels."""

    if isinstance(result, Mapping):
        validity = result.get("validity")
        reasons = result.get("reason_codes", result.get("reasons", ()))
        details = {key: value for key, value in result.items() if key not in ("validity", "reason_codes", "reasons")}
    else:
        validity = getattr(result, "validity", None)
        reasons = getattr(result, "reason_codes", getattr(result, "reasons", ()))
        # plumb.validity.ValidityReport carries the whole evidence record
        # (parameters_hash, calibration_reference_hash, primary_scoring_eligible,
        # calibration_class, per-check details). Passing it through whole is
        # better than cherry-picking fields that may be added later.
        as_dict = getattr(result, "as_dict", None)
        if callable(as_dict):
            details = {
                key: value
                for key, value in dict(as_dict()).items()
                if key not in ("validity", "reason_codes")
            }
        else:
            details = {}
            for name in (
                "parameters_hash",
                "calibration_reference_hash",
                "primary_scoring_eligible",
                "calibration_class",
                "artifact_hashes",
                "measurements",
            ):
                value = getattr(result, name, None)
                if value is not None:
                    details[name] = value
    label = getattr(validity, "value", validity)
    if not isinstance(label, str) or label not in ("valid", "invalid", "unknown"):
        raise ValueError(
            "Stage-A gate returned validity %r; only 'valid', 'invalid', or 'unknown' is acceptable." % (label,)
        )
    if isinstance(reasons, (str, bytes)):
        reason_codes = [str(reasons)]
    else:
        reason_codes = [str(code) for code in (reasons or ())]
    return label, reason_codes, _json_safe(details)


class JudgeCore:
    """Owns one ``QwenRubricJudge`` and the frozen sampling contract."""

    def __init__(self) -> None:
        self.runtime_lock_path = REQUIREMENTS_DIR / "judge-qwen.txt"
        self.runtime_lock_sha256 = _file_sha256(self.runtime_lock_path)
        self.local_model_path = str(pathlib.PurePosixPath(MODEL_CACHE_ROOT) / QWEN_JUDGE_VOLUME)
        self.judge: Any = None
        self.request_cls: Any = None
        self.reference_cls: Any = None
        self.provenance_cls: Any = None
        self.unresolved: List[str] = []
        self._load()

    def _load(self) -> None:
        try:
            from plumb.policies import (  # type: ignore
                JudgeInputProvenance,
                JudgeSamplingConfig,
                JudgeRequest,
                QwenJudgeProfile,
                QwenRubricJudge,
                ReferenceImage,
            )
        except ImportError as error:
            self.unresolved.append(
                "plumb.policies judge symbols are unavailable in this image (%s); stage the plumb package with "
                "deploy/baseten/stage_packages.py before pushing." % error
            )
            return
        manifest_id, manifest_sha = _asset_manifest_identity()
        try:
            profile = QwenJudgeProfile(
                profile_id="plumb-qwen25vl-rubric",
                local_model_path=self.local_model_path,
                model_revision=QWEN_JUDGE_REVISION,
                processor_revision=QWEN_JUDGE_REVISION,
                transformers_version="4.49.0",
                asset_manifest_id=manifest_id,
                asset_manifest_sha256=manifest_sha,
                runtime_lock_id="requirements/judge-qwen.txt",
                runtime_lock_sha256=self.runtime_lock_sha256,
            )
            # Defaults ARE the frozen primary protocol: five independent samples,
            # temperature 0.7, top_p 1.0, max_new_tokens 512, quorum 3, exactly
            # one bounded retry per sample.  Constructing with anything else
            # raises inside QwenRubricJudge, so drift cannot pass silently.
            sampling = JudgeSamplingConfig()
            sampling.validate()
            self.judge = QwenRubricJudge(profile, sampling=sampling)
        except Exception as error:  # noqa: BLE001 - recorded as a block reason
            self.unresolved.append("QwenRubricJudge construction failed: %s" % error)
            return
        self.request_cls = JudgeRequest
        self.reference_cls = ReferenceImage
        self.provenance_cls = JudgeInputProvenance
        result = self.judge.capability()
        status = getattr(getattr(result, "status", None), "value", None)
        if status not in ("ready_unqualified", "qualified"):
            self.unresolved.append(
                "Qwen judge capability is %s: %s" % (status, getattr(result, "reason", "no reason supplied"))
            )

    @property
    def ready(self) -> bool:
        return self.judge is not None and not self.unresolved


def _policy_stage_result(core: NativePolicyCore, request: StageRequest) -> StageResult:
    """Shared policy body.  A plain function, so no extra Chainlet is registered."""

    worker = core.worker_name
    try:
        payload = PolicyObservationPayload.model_validate(request.payload)
    except Exception as error:  # noqa: BLE001 - explicit caller failure
        return _failed("policy", "policy_payload_invalid: %s" % error, worker=worker)
    arm = normalise_policy_arm(payload.policy)
    if arm is None:
        return _failed("policy", "unknown_policy_arm: %r" % payload.policy, worker=worker)
    if arm not in core.arms:
        return _failed(
            "policy",
            "policy_arm_not_served_here: %s serves %s" % (worker, ", ".join(core.arms)),
            worker=worker,
        )
    if arm in core.blocked_reasons:
        return _blocked("policy", core.blocked_reasons[arm] + POLICY_CONTRACTS, worker=worker)
    adapter = core.adapters.get(arm)
    if adapter is None:
        return _blocked(
            "policy",
            ["No loaded adapter for %s in %s." % (arm, worker)] + POLICY_CONTRACTS,
            worker=worker,
        )
    try:
        from plumb.adapters.contracts import PolicyObservation  # type: ignore
    except ImportError as error:
        return _blocked(
            "policy",
            ["plumb.adapters.contracts is unavailable in this image (%s)" % error] + POLICY_CONTRACTS,
            worker=worker,
        )
    try:
        history = tuple(decode_frame(frame) for frame in payload.image_history)
        goal_image = decode_frame(payload.goal_image) if payload.goal_image is not None else None
    except ValueError as error:
        return _failed("policy", "observation_frame_invalid: %s" % error, worker=worker)
    observation = PolicyObservation(
        image_history=history,
        prompt=payload.prompt,
        proprio=tuple(float(value) for value in payload.proprio) if payload.proprio is not None else None,
        goal_image=goal_image,
        timestamp=payload.timestamp,
    )
    started = time.perf_counter()
    try:
        actions, details = core.predict(arm, observation)
    except Exception as error:  # noqa: BLE001 - never a fabricated action
        return _blocked(
            "policy",
            [
                "%s native prediction did not produce an action: %s: %s" % (arm, type(error).__name__, error),
            ]
            + POLICY_CONTRACTS,
            worker=worker,
        )
    elapsed = time.perf_counter() - started
    contract = core.contract_summary(arm)
    output: Dict[str, Any] = {
        "policy_arm": arm,
        "native_actions": actions,
        "native_proposal_horizon": len(actions),
        "certified_execute_prefix": contract.get("certified_execute_prefix"),
        "history_length": len(history),
        "contract": contract,
        "runtime_lock": {"id": core.runtime_lock_path.name, "sha256": core.runtime_lock_sha256},
        "certified_adapter": core.certified_adapter.get(arm),
        "backend_detail": _json_safe(details),
        "limitations": [
            "A native proposal is not a certified executed prefix. When certified_execute_prefix is null the "
            "controller blocks instead of choosing one.",
            "Gate B must certify this arm's feedback cadence and state mode before a named-policy cell counts.",
        ],
    }
    return StageResult(
        stage="policy",
        status="completed",
        output=output,
        worker=worker,
        timing=StageTiming(
            stage="policy",
            worker=worker,
            wall_seconds=elapsed,
            server_seconds=details.get("wall_seconds"),
            gpu_peak_memory_bytes=details.get("gpu_peak_memory_bytes"),
            declared_gpu_count=WORKER_DECLARED_GPU_COUNT[worker],
        ),
    )


def _json_safe(value: Any) -> Any:
    """Coerce adapter output to JSON without inventing or dropping information."""

    if isinstance(value, Enum):
        # A str/int Enum also satisfies the scalar check below, which would leak
        # the Enum object into a JSON payload instead of its declared value.
        return _json_safe(value.value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_safe(value.as_dict())
    if hasattr(value, "value") and not callable(value.value):
        return _json_safe(value.value)
    return str(value)


# ===========================================================================
# Controller helpers.
#
# These are pure module-level functions rather than methods so they can be
# reviewed and unit-tested on a CPU with no truss_chains installed.  The
# entrypoint Chainlet below is then only sequencing and RPCs.
# ===========================================================================


def _build_initial_state(episode: EpisodeControlPayload) -> Any:
    try:
        from plumb.adapters.bridge import BridgeState  # type: ignore
    except ImportError as error:
        return "plumb.adapters.bridge is unavailable in this image (%s); stage the plumb package." % error
    try:
        return BridgeState.from_values(
            [float(value) for value in episode.initial_state], source="source_record"
        )
    except Exception as error:  # noqa: BLE001 - explicit block reason
        return "initial_bridge_state_rejected: %s" % error


def _build_compiler(episode: EpisodeControlPayload) -> Any:
    try:
        from plumb.adapters.bridge import (  # type: ignore
            BridgeControlIntegrator,
            BridgeControlProfile,
            BridgeToCosmosCompiler,
            NormalizationBoundary,
        )
    except ImportError as error:
        return "plumb.adapters.bridge is unavailable in this image (%s); stage the plumb package." % error
    if episode.normalization_boundary == "external":
        # External normalization needs a pinned ActionNormalizer built from
        # the checkpoint's own statistics file. None is mounted, and
        # guessing q01/q99 would corrupt every compiled action.
        return (
            "external normalization requires a pinned ActionNormalizer from the checkpoint's statistics "
            "file; none is mounted in this image, and PLUMB will not infer normalizer bounds."
        )
    try:
        profile = BridgeControlProfile(profile_id=episode.bridge_control_profile_id)
        integrator = BridgeControlIntegrator(profile)
        return BridgeToCosmosCompiler(integrator, None, NormalizationBoundary.BACKEND)
    except Exception as error:  # noqa: BLE001 - explicit block reason
        return "bridge_compiler_construction_failed: %s" % error


def _select_prefix(
    arm: str,
    proposal: Sequence[Sequence[float]],
    certified_prefix: Any,
    remaining: int,
) -> Any:
    if not proposal:
        return "policy_returned_empty_proposal: %s" % arm
    if not isinstance(certified_prefix, int) or isinstance(certified_prefix, bool) or certified_prefix < 1:
        return (
            "certified_execute_prefix_unresolved: %s declares no certified executed prefix, so the "
            "controller refuses to pick one. Gate B must certify it; the loop never assumes five chunks "
            "or a fixed cadence." % arm
        )
    if certified_prefix > len(proposal):
        return (
            "certified_prefix_%d_exceeds_proposal_%d: the proposal is never padded to reach it"
            % (certified_prefix, len(proposal))
        )
    if remaining >= certified_prefix:
        return certified_prefix
    return (
        "terminal_prefix_%d_not_demonstrated: %s certified prefix %d but only %d actions remain. A shorter "
        "terminal prefix requires demonstrated prefix invariance to post-horizon actions or an exact "
        "supported terminal length; truncation and padding are both forbidden."
        % (remaining, arm, certified_prefix, remaining)
    )


def _policy_payload(
    episode: EpisodeControlPayload,
    arm: str,
    history: Sequence[FramePayload],
    action_offset: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "policy": arm,
        "prompt": episode.prompt,
        "image_history": [frame.model_dump() for frame in history],
        "timestamp": action_offset / episode.control_hz,
        "remaining_actions": episode.horizon_actions - action_offset,
    }
    if episode.goal_image is not None:
        payload["goal_image"] = episode.goal_image.model_dump()
    if episode.policy_seed is not None:
        payload["policy_seed"] = episode.policy_seed
    if episode.local_model_path:
        payload["local_model_path"] = episode.local_model_path
    if episode.checkpoint_revision:
        payload["checkpoint_revision"] = episode.checkpoint_revision
    return payload


def _world_setup(request: RolloutRequest) -> Any:
    """Validate the protocol-owned world fields before any GPU work.

    The seed is mandatory. Defaulting it to 0 would silently replace the
    protocol's RNG lineage, and same-seed exact repeatability is the
    reproducibility evidence the whole study rests on.
    """

    base = dict(request.world.payload)
    profile_id = base.get("compatibility_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        return "world_payload_missing_compatibility_profile_id"
    domain = base.get("domain")
    if not isinstance(domain, str) or not domain:
        return "world_payload_missing_domain"
    seed = base.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        return (
            "world_payload_missing_seed: the episode's world seed comes from the frozen run protocol and "
            "is never defaulted, because same-seed exact repeatability is the reproducibility evidence"
        )
    lineage = base.get("source_state_lineage_id")
    return profile_id, domain, int(seed), lineage if isinstance(lineage, str) and lineage else None


def _world_payload(
    episode: EpisodeControlPayload,
    request: RolloutRequest,
    conditioning: FramePayload,
    compiled_rows: Sequence[Sequence[float]],
    segment_timestamps: Sequence[float],
    segment_index: int,
    *,
    profile_id: str,
    domain: str,
    episode_seed: int,
    action_offset: int,
    lineage: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "compatibility_profile_id": profile_id,
        "domain": domain,
        "prompt": episode.prompt,
        "conditioning_image": conditioning.model_dump(),
        "compiled_actions": [[float(value) for value in row] for row in compiled_rows],
        "nominal_control_timestamps": [float(value) for value in segment_timestamps],
        # Derived from the episode seed and the action offset, so a resumed
        # segment replays the same stream instead of drawing a fresh one.
        "seed": segment_world_seed(episode_seed, action_offset),
        "feedback_mode": episode.feedback_mode,
        "request_id": "%s:segment-%d" % (request.episode_id, segment_index),
        "return_frames": True,
    }
    if lineage is not None:
        payload["source_state_lineage_id"] = lineage
    return payload


def _accumulate_segment_frames(output: Mapping[str, Any], expected_actions: int) -> Any:
    raw_frames = output.get("frames") or []
    if not raw_frames:
        return (
            "world_returned_no_frame_payloads: the controller needs pixels to feed the next policy call "
            "and will not reuse the previous frame"
        )
    try:
        frames = [FramePayload.model_validate(item) for item in raw_frames]
    except Exception as error:  # noqa: BLE001 - explicit failure
        return "world_frame_payload_invalid: %s" % error
    conditioning_included = bool(output.get("conditioning_frame_included"))
    expected = expected_actions + (1 if conditioning_included else 0)
    if len(frames) != expected:
        return "world_returned_%d_frames_expected_%d" % (len(frames), expected)
    if conditioning_included:
        # Remove the repeated condition frame when stitching; the original
        # start frame is retained once, separately, by the caller.
        return frames[1:], True
    return frames, False


def _next_history(
    policy_output: Mapping[str, Any],
    history: Sequence[FramePayload],
    future_frames: Sequence[FramePayload],
) -> List[FramePayload]:
    contract = policy_output.get("contract") or {}
    required = contract.get("required_observation_history")
    window = required if isinstance(required, int) and required >= 1 else 1
    combined = list(history) + list(future_frames)
    return combined[-window:]


def _validity_payload(
    request: RolloutRequest,
    frames: Sequence[FramePayload],
    timestamps: Sequence[float],
    native_actions: Sequence[Sequence[float]],
    *,
    conditioning_frame: Optional[FramePayload] = None,
    forecast_states: Optional[Sequence[Sequence[float]]] = None,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the Stage-A request.

    The gate expects ``len(frames) == len(actions) + 1``: the original start
    frame plus exactly one generated frame per executed action.  The start frame
    is ALSO passed as ``conditioning_frame`` so the gate can verify frames[0] is
    really the conditioning image instead of taking it on trust.
    """

    base = dict(request.validity.payload)
    payload: Dict[str, Any] = {
        "frames": [frame.model_dump() for frame in frames],
        "nominal_timestamps": [float(value) for value in timestamps],
        "native_actions": [[float(value) for value in row] for row in native_actions],
        "expected_frame_count": len(frames),
    }
    if conditioning_frame is not None:
        payload["conditioning_frame"] = conditioning_frame.model_dump()
    if forecast_states is not None:
        payload["forecast_states"] = [[float(value) for value in row] for row in forecast_states]
    if episode_id:
        payload["episode_id"] = episode_id
    for name in ("task_id", "parameters_hash", "calibration_reference_hash"):
        value = base.get(name)
        if isinstance(value, str) and value:
            payload[name] = value
    # expected_static is a protocol declaration, never inferred from the video.
    expected_static = base.get("expected_static")
    if isinstance(expected_static, bool):
        payload["expected_static"] = expected_static
    return payload


def _judge_payload(
    request: RolloutRequest,
    frames: Sequence[FramePayload],
    timestamps: Sequence[float],
) -> Any:
    base = dict(request.judge.payload)
    task_id = base.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return "judge_task_id_missing: primary scoring requires one canonical TaskRegistry task_id"
    references = base.get("reference_images")
    if not isinstance(references, list) or not references:
        return (
            "judge_reference_images_missing: the rubric protocol requires at least one "
            "provenance-backed goal/reference image, and PLUMB has no matched five-task panels yet"
        )
    seeds = base.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
        or len(set(seeds)) != 5
    ):
        return (
            "judge_seeds_missing: five distinct explicitly logged integer seeds must come from the run "
            "protocol; the judge worker will not invent a sampling lineage"
        )
    if len(frames) < 16:
        return (
            "judge_clip_too_short: %d frames available but the rubric requires exactly 16 sampled from the "
            "original start through the final control tick, including both endpoints" % len(frames)
        )
    indices = _uniform_frame_indices(len(frames), 16)
    sampled = [frames[index] for index in indices]
    sampled_timestamps = [float(timestamps[index]) for index in indices]
    payload: Dict[str, Any] = {
        "frames": [frame.model_dump() for frame in sampled],
        "frame_timestamps": sampled_timestamps,
        "reference_images": references,
        "task_id": task_id,
        "seeds": [int(value) for value in seeds],
    }
    for name in ("clip_id", "video_sha256", "protocol_id", "calibration_manifest_hash"):
        value = base.get(name)
        if isinstance(value, str) and value:
            payload[name] = value
    return payload


def _gpu_seconds(timings: Sequence[StageTiming]) -> GpuSecondAccounting:
    buckets: Dict[str, float] = {"policy": 0.0, "world": 0.0, "judge": 0.0}
    observed: Dict[str, bool] = {"policy": False, "world": False, "judge": False}
    for timing in timings:
        if timing.stage not in buckets:
            continue
        seconds = timing.server_seconds if timing.server_seconds is not None else timing.wall_seconds
        gpus = timing.declared_gpu_count
        if seconds is None or gpus is None or gpus <= 0:
            continue
        buckets[timing.stage] += float(seconds) * int(gpus)
        observed[timing.stage] = True
    total = sum(value for key, value in buckets.items() if observed[key])
    any_observed = any(observed.values())
    return GpuSecondAccounting(
        policy_gpu_seconds=buckets["policy"] if observed["policy"] else None,
        world_gpu_seconds=buckets["world"] if observed["world"] else None,
        judge_gpu_seconds=buckets["judge"] if observed["judge"] else None,
        total_instrumented_gpu_seconds=total if any_observed else None,
        allocated_gpu_seconds=None,
        limitations=[
            "Instrumented stage time only. It excludes queueing, prewarm, idle, cooldown, and cold-start "
            "intervals, so it is a floor on allocated GPU-seconds, never the allocation itself.",
            "allocated_gpu_seconds is null on purpose: the allocation ledger is owned by the application "
            "control plane, and null means unknown, never zero.",
            "No USD figure is derived here. A verified price basis (currency, unit, applicability, "
            "retrieval time) does not exist yet.",
            "A micro-batched world call's GPU seconds cover the whole batch, so per-episode attribution "
            "needs the batch_size recorded alongside it.",
        ],
    )


# ===========================================================================
# Chainlets
# ===========================================================================

if CHAINS_RUNTIME_AVAILABLE:

    # ``make_abs_path_here`` resolves relative to THIS module and must be called
    # inline at module level (the SDK forbids wrapping it in a function).
    _WORLD_REQUIREMENTS = chains.make_abs_path_here("requirements/world-cosmos.txt")
    _OPENVLA_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-openvla.txt")
    _OCTO_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-octo.txt")
    _MINIVLA_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-minivla.txt")
    _OPENPIZERO_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-openpizero.txt")
    _SUSIE_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-susie.txt")
    _JUDGE_REQUIREMENTS = chains.make_abs_path_here("requirements/judge-qwen.txt")
    _VALIDITY_REQUIREMENTS = chains.make_abs_path_here("requirements/validity.txt")
    _CONTROLLER_REQUIREMENTS = chains.make_abs_path_here("requirements/controller.txt")

    # ``plumb`` lives outside the chain workspace dir (which is this file's
    # directory), so it is bundled as an external package.  The staging dir keeps
    # web/node_modules and .venv out of the image: `gather_chain` copies the
    # *contents* of every external dir into `packages/`, and it honours only
    # truss's built-in .truss_ignore, so pointing at the repository root would
    # ship ~117 MB of node_modules.  Run deploy/baseten/stage_packages.py first.
    _EXTERNAL_PACKAGES = chains.make_abs_path_here("../_chain_packages")

    # ``requirements_file`` is the current field name; ``pip_requirements_file``
    # is its deprecated alias, still present in 0.18.x and the only name older
    # releases accept.  Choosing by feature detection keeps the module pushable
    # on either, and silent on modern truss instead of emitting nine deprecation
    # warnings.  ``make_abs_path_here`` is still called inline at module level
    # above, as the SDK requires; only its already-resolved result is passed here.
    _REQUIREMENTS_FIELD = (
        "requirements_file" if "requirements_file" in chains.DockerImage.model_fields else "pip_requirements_file"
    )

    def _docker_image(base_image: Any, requirements_path: Any) -> Any:
        """One image definition per incompatible dependency stack."""

        return chains.DockerImage(
            base_image=base_image,
            external_package_dirs=[_EXTERNAL_PACKAGES],
            **{_REQUIREMENTS_FIELD: requirements_path},
        )

    _SHARED_ENVIRONMENT: Dict[str, str] = {
        ENV_MODEL_CACHE_ROOT: MODEL_CACHE_ROOT,
    }
    # The bucket destination is deployment context, never request data. Empty
    # values are intentionally omitted so a manually invoked Chain fails its
    # result-store configuration rather than using a made-up destination.
    for _result_store_env in (
        ENV_RESULT_STORE_BUCKET,
        ENV_RESULT_STORE_REGION,
        ENV_RESULT_STORE_ENDPOINT_URL,
        ENV_RESULT_STORE_PREFIX,
        ENV_RESULT_STORE_EVIDENCE_URI,
        ENV_RESULT_STORE_VERIFIED_AT,
        ENV_RESULT_STORE_EXPECTED_BUCKET_OWNER,
        ENV_RESULT_STORE_MAX_BYTES,
        ENV_RESULT_STORE_CONNECT_TIMEOUT_SECONDS,
        ENV_RESULT_STORE_READ_TIMEOUT_SECONDS,
        ENV_RESULT_STORE_MAX_ATTEMPTS,
    ):
        _value = _env_text(_result_store_env, "")
        if _value:
            _SHARED_ENVIRONMENT[_result_store_env] = _value
    _RESULT_STORE_ACCESS_KEY_SECRET_NAME = _env_text(ENV_RESULT_STORE_ACCESS_KEY_SECRET_NAME, "")
    _RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME = _env_text(ENV_RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME, "")
    _RESULT_STORE_SESSION_TOKEN_SECRET_NAME = _env_text(ENV_RESULT_STORE_SESSION_TOKEN_SECRET_NAME, "")
    _RESULT_STORE_SECRET_KEYS = tuple(
        item
        for item in (
            _RESULT_STORE_ACCESS_KEY_SECRET_NAME,
            _RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME,
            _RESULT_STORE_SESSION_TOKEN_SECRET_NAME,
        )
        if item
    )
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", item) for item in _RESULT_STORE_SECRET_KEYS):
        raise ValueError("PLUMB_RESULT_STORE_DEPLOY_SECRET_KEYS must be comma-separated deployment secret names")
    _VALIDITY_ENVIRONMENT: Dict[str, str] = dict(_SHARED_ENVIRONMENT)
    _VALIDITY_ENVIRONMENT[ENV_VALIDITY_CALIBRATION_CLASS] = VALIDITY_CALIBRATION_CLASS
    _WORLD_ENVIRONMENT: Dict[str, str] = dict(_SHARED_ENVIRONMENT)
    _WORLD_ENVIRONMENT.update(
        {
            ENV_BATCH_MAX: str(BATCH_MAX),
            ENV_BATCH_WINDOW_MS: str(BATCH_WINDOW_MS),
            ENV_WORLD_VARIANT: WORLD_VARIANT,
            ENV_WORLD_RESOLUTION_TIER: str(WORLD_RESOLUTION_TIER),
            ENV_WORLD_EDGE_REVISION: WORLD_EDGE_REVISION,
            ENV_WORLD_FUSED_BATCH: "1" if _env_flag(ENV_WORLD_FUSED_BATCH) else "0",
        }
    )
    _OPENVLA_ENVIRONMENT: Dict[str, str] = dict(_SHARED_ENVIRONMENT)
    _OPENVLA_ENVIRONMENT[ENV_OPENVLA_REVIEWED_REMOTE_CODE] = _env_text(ENV_OPENVLA_REVIEWED_REMOTE_CODE, "")

    # Cached weight volumes.  Only repositories with an immutable revision are
    # declared: truss_config.ModelRepo requires a revision for use_volume=True,
    # and inventing one would forge provenance.  Caching is what avoids paying
    # the recorded 167.68 s Cosmos cold load on every replica, which would be
    # fatal to a 60-second burst; the *residual* cold-start cost on this image is
    # still unmeasured and must be timed separately (spec §7).
    _COSMOS_NANO_CACHE = truss_config.ModelRepo(
        repo_id=COSMOS3_NANO_REPO,
        revision=COSMOS3_NANO_REVISION,
        use_volume=True,
        volume_folder=COSMOS3_NANO_VOLUME,
        runtime_secret_name=HF_SECRET_KEY,
    )
    _OPENVLA_CACHE = truss_config.ModelRepo(
        repo_id=OPENVLA_REPO,
        revision=OPENVLA_REVISION,
        use_volume=True,
        volume_folder=OPENVLA_VOLUME,
        runtime_secret_name=HF_SECRET_KEY,
    )
    _QWEN_JUDGE_CACHE = truss_config.ModelRepo(
        repo_id=QWEN_JUDGE_REPO,
        revision=QWEN_JUDGE_REVISION,
        use_volume=True,
        volume_folder=QWEN_JUDGE_VOLUME,
        runtime_secret_name=HF_SECRET_KEY,
    )
    # Cache ONLY the selected arm.  Mounting the unused sibling would add its
    # multi-gigabyte download to every container start for no benefit, and
    # container-start time is exactly what the 60-second burst cannot afford.
    # If the Edge arm is selected without an immutable revision, nothing is
    # cached and the worker blocks with that reason rather than quietly running
    # the Nano weights under an Edge label.
    if WORLD_VARIANT == "cosmos3_edge":
        _WORLD_CACHED = (
            [
                truss_config.ModelRepo(
                    repo_id=COSMOS3_EDGE_REPO,
                    revision=WORLD_EDGE_REVISION,
                    use_volume=True,
                    volume_folder=COSMOS3_EDGE_VOLUME,
                    runtime_secret_name=HF_SECRET_KEY,
                )
            ]
            if _is_immutable_revision(WORLD_EDGE_REVISION)
            else []
        )
    else:
        _WORLD_CACHED = [_COSMOS_NANO_CACHE]

    class WorldWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Micro-batched Cosmos3 forward-dynamics Chainlet on one H100.

        The pipeline is loaded exactly once per replica in ``__init__``.  Requests
        are collected by :class:`MicroBatchQueue` and executed as one group, which
        is the lever AGENT-BUILD-SPEC.md §7 identifies ("throughput comes from
        batch packing and replica count").  ``predict_concurrency`` matches
        ``BATCH_MAX`` so the collector can actually see concurrent callers.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-world-worker",
            docker_image=_docker_image(chains.BasetenImage.PY311, _WORLD_REQUIREMENTS),
            compute=chains.Compute(
                gpu="H100",
                gpu_count=1,
                cpu_count=8,
                memory="64Gi",
                # In-container concurrency must be at least the batch size or the
                # collector can never fill a batch.
                predict_concurrency=BATCH_MAX,
            ),
            assets=chains.Assets(cached=_WORLD_CACHED, secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_WORLD_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-world-worker",
                    {
                        "batch_max": BATCH_MAX,
                        "batch_window_ms": BATCH_WINDOW_MS,
                        "world_variant": WORLD_VARIANT,
                        "resolution_tier": WORLD_RESOLUTION_TIER,
                    },
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = WorldAdapterCore()
            self._queue = MicroBatchQueue(
                self._core.execute_batch,
                batch_max=BATCH_MAX,
                batch_window_ms=BATCH_WINDOW_MS,
            )

        async def run_remote(self, request: StageRequest) -> StageResult:
            worker = "plumb-world-worker"
            try:
                payload = WorldStagePayload.model_validate(request.payload)
            except Exception as error:  # noqa: BLE001 - explicit caller failure
                return _failed("world", "world_payload_invalid: %s" % error, worker=worker)
            if not self._core.ready:
                return _blocked("world", list(self._core.unresolved) + WORLD_CONTRACTS, worker=worker)
            if len(payload.compiled_actions) != len(payload.nominal_control_timestamps):
                return _failed(
                    "world",
                    "world_payload_invalid: one nominal control timestamp is required per compiled action row",
                    worker=worker,
                )
            widths = {len(row) for row in payload.compiled_actions}
            if len(widths) != 1:
                return _failed("world", "world_payload_invalid: compiled action rows have mixed widths", worker=worker)
            action_width = widths.pop()
            try:
                conditioning = decode_frame(payload.conditioning_image)
            except ValueError as error:
                return _failed("world", "conditioning_frame_invalid: %s" % error, worker=worker)
            try:
                from plumb.adapters.contracts import FeedbackMode, WorldRequest  # type: ignore
            except ImportError as error:
                return _blocked(
                    "world",
                    ["plumb.adapters.contracts is unavailable in this image (%s)" % error] + WORLD_CONTRACTS,
                    worker=worker,
                )
            try:
                feedback_mode = FeedbackMode(payload.feedback_mode)
            except ValueError:
                return _failed("world", "unknown feedback_mode %r" % payload.feedback_mode, worker=worker)
            try:
                world_request = WorldRequest(
                    conditioning_image=conditioning,
                    prompt=payload.prompt,
                    domain=payload.domain,
                    compiled_actions=tuple(tuple(float(value) for value in row) for row in payload.compiled_actions),
                    nominal_control_timestamps=tuple(float(value) for value in payload.nominal_control_timestamps),
                    seed=int(payload.seed),
                    compatibility_profile_id=payload.compatibility_profile_id,
                    feedback_mode=feedback_mode,
                    source_state_lineage_id=payload.source_state_lineage_id,
                    request_id=payload.request_id,
                )
                world_request.validate_basic()
            except Exception as error:  # noqa: BLE001 - explicit caller failure
                return _failed("world", "world_request_rejected: %s" % error, worker=worker)

            key = self._core.batch_key(
                len(payload.compiled_actions),
                action_width,
                payload.domain,
                payload.compatibility_profile_id,
            )
            started = time.perf_counter()
            try:
                bundle = await self._queue.submit(key, world_request)
            except Exception as error:  # noqa: BLE001 - never a fabricated frame
                return _blocked(
                    "world",
                    ["Cosmos backend call did not return a usable result: %s: %s" % (type(error).__name__, error)]
                    + WORLD_CONTRACTS,
                    worker=worker,
                )
            elapsed = time.perf_counter() - started
            result, batch_mode, batch_size = bundle
            try:
                frames = [
                    encode_frame(frame, nominal_timestamp=float(timestamp))
                    for frame, timestamp in zip(result.frames, result.nominal_frame_timestamps)
                ]
            except Exception as error:  # noqa: BLE001 - explicit failure, no repair
                return _failed("world", "generated_frame_encode_failed: %s" % error, worker=worker)
            timing = result.timing
            variant = self._core.variant
            output: Dict[str, Any] = {
                "backend": result.backend,
                "profile_id": result.profile_id,
                "frames": [frame.model_dump() for frame in frames] if payload.return_frames else [],
                "frame_pixel_hashes": [frame.pixels_sha256 for frame in frames],
                "frame_png_hashes": [frame.png_sha256 for frame in frames],
                "nominal_frame_timestamps": [float(value) for value in result.nominal_frame_timestamps],
                "conditioning_frame_included": bool(result.conditioning_frame_included),
                "action_count": len(payload.compiled_actions),
                "action_width": action_width,
                "request_id": result.request_id,
                "artifact_hashes": _json_safe(result.artifact_hashes),
                "adapter_metadata": _json_safe(result.metadata),
                "batch": {
                    "batch_key": key,
                    "batch_size": batch_size,
                    "batch_execution_mode": batch_mode,
                    "batch_max": BATCH_MAX,
                    "batch_window_ms": BATCH_WINDOW_MS,
                    "fused_forward": batch_mode == "fused_adapter_generate_batch",
                },
                "variant": {
                    "variant_id": variant.variant_id,
                    "authoritative_model_id": variant.repo_id,
                    "authoritative_model_revision": variant.revision,
                    "parameter_note": variant.parameter_note,
                    "adapter_metadata_model_id_is_profile_default": variant.metadata_model_id_is_profile_default,
                },
                "runtime_lock": {
                    "id": "requirements/world-cosmos.txt",
                    "sha256": self._core.runtime_lock_sha256,
                },
                "limitations": [
                    "Nominal frame timestamps come from the pinned control/frame mapping; they are not measured "
                    "physical control timestamps.",
                    "The conditioning frame is returned at index 0 and must be dropped when stitching; the caller "
                    "does that, this worker never trims frames.",
                    "Normalization boundary is BACKEND: this worker assumes the checkpoint applies its own "
                    "normalizer exactly once. That assumption is a Gate-A item, not a verified fact.",
                ],
            }
            if variant.metadata_model_id_is_profile_default:
                output["limitations"].append(
                    "adapter_metadata.model_id reports the profile default nvidia/Cosmos3-Nano even on the Edge "
                    "arm; use variant.authoritative_model_id. Fixing the label needs a plumb-side change."
                )
            return StageResult(
                stage="world",
                status="completed",
                output=output,
                worker=worker,
                timing=StageTiming(
                    stage="world",
                    worker=worker,
                    wall_seconds=elapsed,
                    server_seconds=getattr(timing, "wall_seconds", None),
                    model_load_seconds=getattr(timing, "model_load_seconds", None),
                    gpu_peak_memory_bytes=getattr(timing, "gpu_peak_memory_bytes", None),
                    declared_gpu_count=WORKER_DECLARED_GPU_COUNT[worker],
                    batch_size=batch_size,
                    batch_execution_mode=batch_mode,
                    cold_start=getattr(timing, "cold_start", None),
                ),
            )

    class OpenVLAWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """OpenVLA-7B, Transformers 4.40.1, one native 7-D action per fresh image.

        GPU-shaped from the recorded 15.5 GB peak for this model; that recording
        was on another cluster's build, so the size here is a starting hypothesis
        to be re-measured, not a transplanted measurement.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-openvla-worker",
            docker_image=_docker_image(chains.BasetenImage.PY311, _OPENVLA_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=2),
            assets=chains.Assets(cached=[_OPENVLA_CACHE], secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_OPENVLA_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-openvla-worker",
                    {"arms": list(WORKER_ARMS["plumb-openvla-worker"]), "model_revision": OPENVLA_REVISION},
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = NativePolicyCore("plumb-openvla-worker", "policy-openvla.txt")

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _policy_stage_result(self._core, request)

    class OctoWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Octo v1.0 JAX image (Small primary, Base diagnostic).

        No reviewed loader exists, so every call returns ``blocked``.  The image
        and GPU shape are declared anyway so the deployment graph, cold-start
        cost, and dependency isolation are real and measurable before a wrapper
        lands.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-octo-worker",
            docker_image=_docker_image(chains.BasetenImage.PY310, _OCTO_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=2),
            assets=chains.Assets(secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-octo-worker",
                    {
                        "arms": list(WORKER_ARMS["plumb-octo-worker"]),
                        "cached_weights": None,
                        "cached_weights_reason": "no immutable revision resolved for rail-berkeley/octo-small|octo-base",
                        "cuda_deviation": "jax[cuda12_pip] replaces the spec's cuda11_pip; see requirements/policy-octo.txt",
                    },
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = NativePolicyCore("plumb-octo-worker", "policy-octo.txt")

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _policy_stage_result(self._core, request)

    class MiniVLAWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """MiniVLA (openvla-mini + VQ head) on the pinned torch 2.2.0 stack."""

        remote_config = chains.RemoteConfig(
            name="plumb-minivla-worker",
            docker_image=_docker_image(chains.BasetenImage.PY310, _MINIVLA_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=2),
            assets=chains.Assets(secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-minivla-worker",
                    {
                        "arms": list(WORKER_ARMS["plumb-minivla-worker"]),
                        "cached_weights": None,
                        "cached_weights_reason": "no immutable revision resolved; pretrain_vq declares no license",
                    },
                ),
            ),
            # flash-attn 2.5.5 has no usable wheel-only install and needs torch
            # importable at build time with --no-build-isolation, which pip does
            # not accept inside a requirements file. truss runs build_commands
            # after the requirements install, so torch 2.2.0 is present here.
            build_commands=["pip install --no-cache-dir --no-build-isolation flash-attn==2.5.5"],
        )

        def __init__(self) -> None:
            self._core = NativePolicyCore("plumb-minivla-worker", "policy-minivla.txt")

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _policy_stage_result(self._core, request)

    class OpenPiZeroWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Open pi-zero with its PaliGemma backbone and 4-action proposal."""

        remote_config = chains.RemoteConfig(
            name="plumb-openpizero-worker",
            docker_image=_docker_image(chains.BasetenImage.PY310, _OPENPIZERO_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="48Gi", predict_concurrency=2),
            assets=chains.Assets(secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-openpizero-worker",
                    {
                        "arms": list(WORKER_ARMS["plumb-openpizero-worker"]),
                        "cached_weights": None,
                        "cached_weights_reason": "no immutable revision resolved; PaliGemma access terms unresolved",
                    },
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = NativePolicyCore("plumb-openpizero-worker", "policy-openpizero.txt")

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _policy_stage_result(self._core, request)

    class SusieWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """One JAX/Flax image serving both the SuSIE and SuSIE_LL arms.

        The arm comes from ``payload["policy"]`` and is resolved through the
        explicit alias table.  It is never inferred from the worker identity,
        because SuSIE and SuSIE_LL are different policies with different
        execution rules that happen to share a dependency stack.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-susie-worker",
            docker_image=_docker_image(chains.BasetenImage.PY310, _SUSIE_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="48Gi", predict_concurrency=2),
            assets=chains.Assets(secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-susie-worker",
                    {
                        "arms": list(WORKER_ARMS["plumb-susie-worker"]),
                        "cached_weights": None,
                        "cached_weights_reason": "no immutable revision resolved for kvablack/susie or patreya/gcbc-bridge",
                        "cuda_deviation": "jax[cuda12_pip] replaces the CUDA 11 era build; see requirements/policy-susie.txt",
                        "arm_disclosure": "AutoEval gc_bc replication arm differs from upstream SuSIE and stays separately named",
                    },
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = NativePolicyCore("plumb-susie-worker", "policy-susie.txt")

        async def run_remote(self, request: StageRequest) -> StageResult:
            return _policy_stage_result(self._core, request)

    class ValidityWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Deterministic Stage-A validity Chainlet.  It does not score success.

        CPU is correct here: Stage A is fixed-parameter classical analysis, and an
        accelerator would only invite a learned check into a gate that must stay
        deterministic.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-validity-worker",
            docker_image=_docker_image(chains.BasetenImage.PY311, _VALIDITY_REQUIREMENTS),
            compute=chains.Compute(cpu_count=4, memory="16Gi", predict_concurrency=4),
            assets=chains.Assets(),
            options=chains.ChainletOptions(
                env_variables=_VALIDITY_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-validity-worker",
                    {"accelerator": None, "calibration_class": VALIDITY_CALIBRATION_CLASS},
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = ValidityGateCore()

        async def run_remote(self, request: StageRequest) -> StageResult:
            worker = "plumb-validity-worker"
            try:
                payload = ValidityStagePayload.model_validate(request.payload)
            except Exception as error:  # noqa: BLE001 - explicit caller failure
                return _failed("validity", "validity_payload_invalid: %s" % error, worker=worker)
            if not self._core.ready:
                return _blocked("validity", list(self._core.unresolved) + VALIDITY_CONTRACTS, worker=worker)
            if len(payload.frames) != len(payload.nominal_timestamps):
                return _failed(
                    "validity",
                    "validity_payload_invalid: one nominal timestamp is required per frame",
                    worker=worker,
                )
            try:
                frames = [decode_frame(frame) for frame in payload.frames]
                conditioning = (
                    decode_frame(payload.conditioning_frame) if payload.conditioning_frame is not None else None
                )
            except ValueError as error:
                return _failed("validity", "clip_frame_invalid: %s" % error, worker=worker)
            started = time.perf_counter()
            try:
                label, reason_codes, details = self._core.evaluate(
                    frames=frames,
                    actions=[[float(value) for value in row] for row in payload.native_actions],
                    nominal_timestamps=[float(value) for value in payload.nominal_timestamps],
                    conditioning_frame=conditioning,
                    states=(
                        [[float(value) for value in row] for row in payload.forecast_states]
                        if payload.forecast_states is not None
                        else None
                    ),
                    expected_static=payload.expected_static,
                    episode_id=payload.episode_id or request.episode_id,
                )
            except Exception as error:  # noqa: BLE001 - never a fabricated label
                return _blocked(
                    "validity",
                    ["Stage-A gate did not return a usable label: %s: %s" % (type(error).__name__, error)]
                    + VALIDITY_CONTRACTS,
                    worker=worker,
                )
            elapsed = time.perf_counter() - started
            return StageResult(
                stage="validity",
                status="completed",
                output={
                    "validity": label,
                    "reason_codes": reason_codes,
                    "details": details,
                    "frame_count": len(frames),
                    "frame_pixel_hashes": [frame.pixels_sha256 for frame in payload.frames],
                    "frame_png_hashes": [frame.png_sha256 for frame in payload.frames],
                    "parameters_hash": payload.parameters_hash or self._core.parameters_hash,
                    "declared_parameters_hash": self._core.parameters_hash,
                    "calibration_class": self._core.calibration_class,
                    "motion_reference_missing": getattr(self._core, "motion_reference_missing", None),
                    "calibration_reference_hash": payload.calibration_reference_hash,
                    "runtime_lock": {
                        "id": "requirements/validity.txt",
                        "sha256": self._core.runtime_lock_sha256,
                    },
                    "limitations": [
                        "'unknown' is a real outcome; an ambiguous observation is never recorded as a known "
                        "physical failure and the attempt stays in coverage reporting.",
                        "No MotionReference is mounted, so the calibrated command-motion check cannot run and the "
                        "gate reports no_motion_reference rather than inferring a bound. Fit one from real Bridge "
                        "trajectories before primary scoring.",
                        "This gate reports validity only. It never emits binary success or a progress score.",
                        "Whole-image flow magnitude cannot identify arm motion, contact failure, or collision on "
                        "its own; a stationary gripper pressing a drawer can be valid.",
                    ],
                },
                worker=worker,
                timing=StageTiming(
                    stage="validity",
                    worker=worker,
                    wall_seconds=elapsed,
                    declared_gpu_count=WORKER_DECLARED_GPU_COUNT[worker],
                ),
            )

    class JudgeWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Blinded Qwen2.5-VL rubric Chainlet on the frozen five-sample protocol.

        GPU, not CPU: cluster-runtime-evidence.json records a 16.9 GB peak for
        this model.  The judge is constructed once per replica; the sampling
        contract (5 samples, temperature 0.7, top_p 1.0, max_new_tokens 512,
        quorum 3, one bounded retry per sample for schema/transport failures
        only) is enforced inside ``plumb.policies.judge``, and the episode label
        is computed in Python from the raw samples — never parsed out of a
        summary field.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-judge-worker",
            docker_image=_docker_image(chains.BasetenImage.PY311, _JUDGE_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="48Gi", predict_concurrency=2),
            assets=chains.Assets(cached=[_QWEN_JUDGE_CACHE], secret_keys=[HF_SECRET_KEY]),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-judge-worker",
                    {
                        "model_revision": QWEN_JUDGE_REVISION,
                        "sampling": "5 samples / t=0.7 / top_p=1.0 / 512 tokens / quorum 3 / 1 bounded retry",
                        "gate_d": "unsatisfied",
                    },
                ),
            ),
        )

        def __init__(self) -> None:
            self._core = JudgeCore()

        async def run_remote(self, request: StageRequest) -> StageResult:
            worker = "plumb-judge-worker"
            forbidden = sorted(set(request.payload) - set(JudgeStagePayload.model_fields))
            if forbidden:
                # A named rejection is clearer than a generic schema error, and it
                # makes an attempted blinding violation obvious in the ledger.
                return _failed(
                    "judge",
                    "judge_payload_forbidden_fields: %s (policy names, actions, commands, reference rates, "
                    "condition labels, and Stage-A outcomes must never reach the judge)" % ", ".join(forbidden),
                    worker=worker,
                )
            try:
                payload = JudgeStagePayload.model_validate(request.payload)
            except Exception as error:  # noqa: BLE001 - explicit caller failure
                return _failed("judge", "judge_payload_invalid: %s" % error, worker=worker)
            if not self._core.ready:
                return _blocked("judge", list(self._core.unresolved) + JUDGE_CONTRACTS, worker=worker)
            if len(set(payload.seeds)) != 5:
                return _failed(
                    "judge",
                    "judge_seeds_invalid: five distinct explicitly logged integer seeds are required",
                    worker=worker,
                )
            try:
                frames = tuple(decode_frame(frame) for frame in payload.frames)
                references = tuple(
                    self._core.reference_cls(
                        image=decode_frame(reference.image),
                        source_uri=reference.source_uri,
                        sha256=reference.sha256,
                    )
                    for reference in payload.reference_images
                )
            except ValueError as error:
                return _failed("judge", "judge_frame_invalid: %s" % error, worker=worker)
            provenance = self._core.provenance_cls(
                clip_id=payload.clip_id,
                video_sha256=payload.video_sha256,
                protocol_id=payload.protocol_id,
                calibration_manifest_hash=payload.calibration_manifest_hash,
            )
            try:
                judge_request = self._core.request_cls(
                    frames=frames,
                    frame_timestamps=tuple(float(value) for value in payload.frame_timestamps),
                    reference_images=references,
                    task_id=payload.task_id,
                    provenance=provenance,
                )
                judge_request.validate()
            except Exception as error:  # noqa: BLE001 - reject, never resample
                return _failed("judge", "judge_request_rejected: %s" % error, worker=worker)
            started = time.perf_counter()
            try:
                report = self._core.judge.evaluate(judge_request, seeds=list(payload.seeds))
            except Exception as error:  # noqa: BLE001 - never a fabricated vote
                return _blocked(
                    "judge",
                    ["Judge call did not return a report: %s: %s" % (type(error).__name__, error)] + JUDGE_CONTRACTS,
                    worker=worker,
                )
            elapsed = time.perf_counter() - started
            try:
                # Recompute the label from the raw samples; a forged summary field
                # must not survive transport.
                report.validate_aggregation()
            except Exception as error:  # noqa: BLE001 - explicit failure
                return _failed("judge", "judge_report_aggregation_mismatch: %s" % error, worker=worker)
            document = dict(report.as_dict())
            document["runtime_lock"] = {
                "id": "requirements/judge-qwen.txt",
                "sha256": self._core.runtime_lock_sha256,
            }
            document["limitations"] = [
                "Gate D blinded human calibration is unsatisfied, so this label is not primary-scoring evidence.",
                "Five samples are not five independent robot episodes.",
                "Refusals, exhausted retries, and disagreement are explicit missing outcomes, never forced labels.",
            ]
            return StageResult(
                stage="judge",
                status="completed",
                output=document,
                worker=worker,
                timing=StageTiming(
                    stage="judge",
                    worker=worker,
                    wall_seconds=elapsed,
                    server_seconds=getattr(report, "wall_seconds", None),
                    gpu_peak_memory_bytes=getattr(report, "gpu_peak_memory_bytes", None),
                    declared_gpu_count=WORKER_DECLARED_GPU_COUNT[worker],
                ),
            )

    class PolicyRouter:
        """Entrypoint-side dispatch from a policy arm to its Chainlet stub.

        This is deliberately NOT a Chainlet: adding a hop would put an extra
        network round trip inside a 60-second burst for a table lookup.  Unknown
        arms are rejected, never routed to a "closest" worker.
        """

        def __init__(self, workers: Mapping[str, Any]) -> None:
            self._by_arm: Dict[str, Tuple[str, Any]] = {}
            for worker_name, stub in workers.items():
                for arm in WORKER_ARMS[worker_name]:
                    self._by_arm[arm] = (worker_name, stub)

        @property
        def served_arms(self) -> Tuple[str, ...]:
            return tuple(sorted(self._by_arm))

        def route(self, policy: str) -> Tuple[Optional[str], Optional[str], Optional[Any]]:
            """Return ``(canonical_arm, worker_name, stub)``; ``None`` if unknown."""

            arm = normalise_policy_arm(policy)
            if arm is None or arm not in self._by_arm:
                return arm, None, None
            worker_name, stub = self._by_arm[arm]
            return arm, worker_name, stub

    def _result_store_credentials(context: chains.DeploymentContext) -> S3Credentials:  # type: ignore[name-defined]
        """Resolve only declared credential fields from deployment context."""

        if not _RESULT_STORE_ACCESS_KEY_SECRET_NAME or not _RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME:
            raise ResultStoreConfigurationError(
                "result-store deployment secret names for AWS access and secret keys are required"
            )
        try:
            secrets = context.secrets
            access_key = secrets[_RESULT_STORE_ACCESS_KEY_SECRET_NAME]
            secret_key = secrets[_RESULT_STORE_SECRET_ACCESS_KEY_SECRET_NAME]
            session_token = (
                secrets[_RESULT_STORE_SESSION_TOKEN_SECRET_NAME]
                if _RESULT_STORE_SESSION_TOKEN_SECRET_NAME
                else None
            )
        except Exception as error:  # do not expose the missing secret name/value to a request result
            raise ResultStoreConfigurationError("result-store deployment credentials are unavailable") from error
        return S3Credentials(str(access_key), str(secret_key), None if session_token is None else str(session_token))

    @chains.mark_entrypoint("PLUMB Rollout Controller")
    class RolloutController(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """CPU entrypoint running one episode's real sequential feedback loop.

        A complete logical episode is submitted once to this entrypoint through
        the deployment's ``/async_run_remote`` URL.  The application — not this
        stateless worker — owns the durable ledger, outbox, artifact writes,
        retry-attempt records, and callback reconciliation; the returned
        ``segments`` give that ledger stable, hash-bound content to persist and to
        emit ``segment_completed`` events from.

        The loop advances by the adapter's **certified executed prefix** and never
        assumes five chunks: when ``certified_execute_prefix`` is null the episode
        is blocked rather than run at a guessed cadence.
        """

        remote_config = chains.RemoteConfig(
            name="plumb-rollout-controller",
            docker_image=_docker_image(chains.BasetenImage.PY311, _CONTROLLER_REQUIREMENTS),
            compute=chains.Compute(cpu_count=4, memory="8Gi", predict_concurrency=32),
            assets=chains.Assets(secret_keys=list(_RESULT_STORE_SECRET_KEYS)),
            options=chains.ChainletOptions(
                env_variables=_SHARED_ENVIRONMENT,
                metadata=_chainlet_metadata(
                    "plumb-rollout-controller",
                    {"entrypoint": True, "async_route_suffix": "/async_run_remote"},
                ),
            ),
        )

        def __init__(
            self,
            world: WorldWorker = chains.depends(WorldWorker, retries=0),
            validity: ValidityWorker = chains.depends(ValidityWorker, retries=0),
            judge: JudgeWorker = chains.depends(JudgeWorker, retries=0),
            openvla: OpenVLAWorker = chains.depends(OpenVLAWorker, retries=0),
            octo: OctoWorker = chains.depends(OctoWorker, retries=0),
            minivla: MiniVLAWorker = chains.depends(MiniVLAWorker, retries=0),
            openpizero: OpenPiZeroWorker = chains.depends(OpenPiZeroWorker, retries=0),
            susie: SusieWorker = chains.depends(SusieWorker, retries=0),
            context: chains.DeploymentContext = chains.depends_context(),
        ) -> None:
            # retries=0 everywhere: a transport retry creates an attempt, not a
            # new statistical episode, and attempt accounting belongs to the
            # application ledger. Retrying here would hide it.
            self._world = world
            self._validity = validity
            self._judge = judge
            self._deployment_context = context
            self._router = PolicyRouter(
                {
                    "plumb-openvla-worker": openvla,
                    "plumb-octo-worker": octo,
                    "plumb-minivla-worker": minivla,
                    "plumb-openpizero-worker": openpizero,
                    "plumb-susie-worker": susie,
                }
            )

        async def run_remote(self, request: RolloutRequest) -> RolloutResult:
            """Run then conditionally persist every terminal result before return."""

            if not request.matching_result_store_binding():
                return RolloutResult(
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    status="failed",
                    missing_reason="result_store_binding_invalid",
                )
            try:
                assert request.result_store is not None
                binding = ResultStoreBinding(**request.result_store.model_dump(mode="json"))
                store = S3ResultStore(
                    S3ResultStoreConfig.from_env(), credentials=_result_store_credentials(self._deployment_context)
                )
                if not store.sdk_available():
                    raise ResultStoreConfigurationError("result-store boto3 dependency is unavailable")
            except (ResultStoreConfigurationError, ResultStoreError, ValueError, TypeError):
                # Validate storage and credential wiring before any policy/world
                # call. Do not spend a GPU second on a result that cannot be
                # committed durably.
                return RolloutResult(
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    status="failed",
                    missing_reason="result_store_unavailable_before_execution",
                )
            result = await self._run_remote(request)
            return await self._persist_terminal_result(request, result, store)

        async def _run_remote(self, request: RolloutRequest) -> RolloutResult:
            started = time.perf_counter()
            if not request.matching_episode_requests():
                return RolloutResult(
                    run_id=request.run_id,
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    status="failed",
                    missing_reason="stage_request_identity_mismatch",
                )
            try:
                episode = EpisodeControlPayload.model_validate(request.policy.payload)
            except Exception as error:  # noqa: BLE001 - explicit caller failure
                return self._terminal(
                    request,
                    "failed",
                    "episode_control_payload_invalid: %s" % error,
                    started=started,
                )
            arm, worker_name, policy_stub = self._router.route(episode.policy)
            if policy_stub is None or worker_name is None:
                return self._terminal(
                    request,
                    "failed",
                    "unroutable_policy_arm: %r (served: %s)" % (episode.policy, ", ".join(self._router.served_arms)),
                    started=started,
                    policy_arm=arm,
                )
            state = _build_initial_state(episode)
            if isinstance(state, str):
                return self._terminal(request, "blocked", state, started=started, policy_arm=arm)
            compiler = _build_compiler(episode)
            if isinstance(compiler, str):
                return self._terminal(request, "blocked", compiler, started=started, policy_arm=arm)
            world_setup = _world_setup(request)
            if isinstance(world_setup, str):
                # Checked before the first policy call so a missing protocol value
                # never costs a GPU round trip.
                return self._terminal(request, "failed", world_setup, started=started, policy_arm=arm)
            world_profile_id, world_domain, world_seed, world_lineage = world_setup

            stages: List[StageResult] = []
            timings: List[StageTiming] = []
            segments: List[SegmentResult] = []
            try:
                # Decode once to reject a corrupt or hash-mismatched start frame
                # before any GPU work is scheduled. The pixels themselves travel
                # as payloads; the workers decode their own inputs.
                decode_frame(episode.initial_frame)
            except ValueError as error:
                return self._terminal(
                    request, "failed", "initial_frame_invalid: %s" % error, started=started, policy_arm=arm
                )

            # The original start frame appears exactly once in the timeline.
            clip_frames: List[FramePayload] = [episode.initial_frame]
            timestamps: List[float] = [0.0]
            native_actions: List[List[float]] = []
            # The source image appears exactly once in the rollout timeline above;
            # these repeats only satisfy a policy's declared reset contract.
            bootstrap_repeats = episode.initial_history_length or 1
            history: List[FramePayload] = [episode.initial_frame for _ in range(bootstrap_repeats)]
            # One state row per frame, so the Stage-A gate can check command and
            # motion consistency. These are FORECAST rows from the pinned Bridge
            # integrator, never a physical state measurement.
            state_rows: List[List[float]] = [[float(value) for value in state.values8()]]
            action_offset = 0
            world_calls = 0
            feedback_mode: Optional[str] = None

            while action_offset < episode.horizon_actions:
                if episode.max_segments is not None and len(segments) >= episode.max_segments:
                    return self._terminal(
                        request,
                        "blocked",
                        "segment_budget_exhausted_before_horizon",
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                policy_stage = await policy_stub.run_remote(
                    StageRequest(
                        episode_id=request.episode_id,
                        protocol_hash=request.protocol_hash,
                        payload=_policy_payload(episode, arm or episode.policy, history, action_offset),
                    )
                )
                stages.append(policy_stage)
                if policy_stage.timing is not None:
                    timings.append(policy_stage.timing)
                if policy_stage.status != "completed" or not policy_stage.output:
                    return self._terminal(
                        request,
                        policy_stage.status if policy_stage.status != "completed" else "blocked",
                        "stage_policy_%s" % policy_stage.status,
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                proposal = [[float(value) for value in row] for row in policy_stage.output.get("native_actions", [])]
                prefix = policy_stage.output.get("certified_execute_prefix")
                remaining = episode.horizon_actions - action_offset
                selection = _select_prefix(arm or episode.policy, proposal, prefix, remaining)
                if isinstance(selection, str):
                    return self._terminal(
                        request,
                        "blocked",
                        selection,
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                count = selection
                segment_actions = [list(row) for row in proposal[:count]]

                compile_started = time.perf_counter()
                try:
                    compiled = compiler.compile(state, [tuple(row) for row in segment_actions])
                except Exception as error:  # noqa: BLE001 - no world call is made
                    return self._terminal(
                        request,
                        "failed",
                        "action_compilation_failed: %s: %s" % (type(error).__name__, error),
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                compile_timing = StageTiming(
                    stage="compile",
                    worker="plumb-rollout-controller",
                    wall_seconds=time.perf_counter() - compile_started,
                    declared_gpu_count=0,
                )
                timings.append(compile_timing)
                compiled_rows = [[float(value) for value in row] for row in compiled.backend_actions]
                segment_timestamps = [
                    (action_offset + index + 1) / episode.control_hz for index in range(count)
                ]

                world_stage = await self._world.run_remote(
                    StageRequest(
                        episode_id=request.episode_id,
                        protocol_hash=request.protocol_hash,
                        payload=_world_payload(
                            episode,
                            request,
                            history[-1],
                            compiled_rows,
                            segment_timestamps,
                            len(segments),
                            profile_id=world_profile_id,
                            domain=world_domain,
                            episode_seed=world_seed,
                            action_offset=action_offset,
                            lineage=world_lineage,
                        ),
                    )
                )
                stages.append(world_stage)
                if world_stage.timing is not None:
                    timings.append(world_stage.timing)
                if world_stage.status != "completed" or not world_stage.output:
                    return self._terminal(
                        request,
                        world_stage.status if world_stage.status != "completed" else "blocked",
                        "stage_world_%s" % world_stage.status,
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                world_calls += 1
                accumulated = _accumulate_segment_frames(world_stage.output, count)
                if isinstance(accumulated, str):
                    return self._terminal(
                        request,
                        "failed",
                        accumulated,
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                future_frames, dropped_conditioning = accumulated
                try:
                    # Verify the generated frames decode and match their declared
                    # digests before they enter the timeline. A frame that fails
                    # here is an explicit failure, never silently repaired.
                    for frame in future_frames:
                        decode_frame(frame)
                except ValueError as error:
                    return self._terminal(
                        request,
                        "failed",
                        "generated_frame_invalid: %s" % error,
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                clip_frames.extend(future_frames)
                timestamps.extend(segment_timestamps)
                native_actions.extend(segment_actions)

                forecast_states = list(getattr(compiled, "forecast_states", ()) or ())
                if len(forecast_states) != count + 1:
                    return self._terminal(
                        request,
                        "failed",
                        "compiler_returned_%d_forecast_states_for_%d_actions" % (len(forecast_states), count),
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )
                next_state = forecast_states[-1]
                compiled_feedback = getattr(compiled, "feedback_mode", None)
                compiled_feedback_value = getattr(compiled_feedback, "value", compiled_feedback)
                if feedback_mode is None:
                    feedback_mode = compiled_feedback_value
                elif compiled_feedback_value != feedback_mode:
                    return self._terminal(
                        request,
                        "failed",
                        "feedback_mode_changed_mid_episode: %s then %s" % (feedback_mode, compiled_feedback_value),
                        started=started,
                        policy_arm=arm,
                        stages=stages,
                        timings=timings,
                        segments=segments,
                        executed=action_offset,
                        horizon=episode.horizon_actions,
                        frames=clip_frames,
                        timestamps=timestamps,
                        world_calls=world_calls,
                        feedback_mode=feedback_mode,
                    )

                segments.append(
                    SegmentResult(
                        index=len(segments),
                        action_offset=action_offset,
                        action_count=count,
                        policy_arm=arm or episode.policy,
                        native_actions=segment_actions,
                        compiled_actions=compiled_rows,
                        policy_history_pixel_hashes=[
                            frame.pixels_sha256 for frame in history if frame.pixels_sha256 is not None
                        ],
                        generated_frame_pixel_hashes=[
                            frame.pixels_sha256 for frame in future_frames if frame.pixels_sha256 is not None
                        ],
                        generated_frame_png_hashes=[
                            frame.png_sha256 for frame in future_frames if frame.png_sha256 is not None
                        ],
                        nominal_frame_timestamps=segment_timestamps,
                        conditioning_frame_dropped=dropped_conditioning,
                        world_request_id=world_stage.output.get("request_id"),
                        forecast_state_after=[float(value) for value in next_state.values8()],
                        feedback_mode=compiled_feedback_value,
                        timings=[compile_timing]
                        + ([world_stage.timing] if world_stage.timing is not None else [])
                        + ([policy_stage.timing] if policy_stage.timing is not None else []),
                        status="completed",
                        frames=future_frames if episode.return_frames else [],
                    )
                )

                state_rows.extend(
                    [float(value) for value in row.values8()] for row in forecast_states[1:]
                )
                state = next_state
                action_offset += count
                history = _next_history(policy_stage.output, history, future_frames)

            validity_stage = await self._validity.run_remote(
                StageRequest(
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    payload=_validity_payload(
                        request,
                        clip_frames,
                        timestamps,
                        native_actions,
                        conditioning_frame=episode.initial_frame,
                        forecast_states=state_rows,
                        episode_id=request.episode_id,
                    ),
                )
            )
            stages.append(validity_stage)
            if validity_stage.timing is not None:
                timings.append(validity_stage.timing)
            validity_label: Optional[str] = None
            validity_reasons: List[str] = []
            if validity_stage.status == "completed" and validity_stage.output:
                raw_label = validity_stage.output.get("validity")
                if raw_label in ("valid", "invalid", "unknown"):
                    validity_label = raw_label
                validity_reasons = [str(code) for code in validity_stage.output.get("reason_codes", [])]
            if validity_stage.status != "completed":
                return self._terminal(
                    request,
                    validity_stage.status,
                    "stage_validity_%s" % validity_stage.status,
                    started=started,
                    policy_arm=arm,
                    stages=stages,
                    timings=timings,
                    segments=segments,
                    executed=action_offset,
                    horizon=episode.horizon_actions,
                    frames=clip_frames,
                    timestamps=timestamps,
                    world_calls=world_calls,
                    feedback_mode=feedback_mode,
                    validity=validity_label,
                    validity_reasons=validity_reasons,
                )

            judge_payload = _judge_payload(request, clip_frames, timestamps)
            if isinstance(judge_payload, str):
                return self._terminal(
                    request,
                    "blocked",
                    judge_payload,
                    started=started,
                    policy_arm=arm,
                    stages=stages,
                    timings=timings,
                    segments=segments,
                    executed=action_offset,
                    horizon=episode.horizon_actions,
                    frames=clip_frames,
                    timestamps=timestamps,
                    world_calls=world_calls,
                    feedback_mode=feedback_mode,
                    validity=validity_label,
                    validity_reasons=validity_reasons,
                )
            judge_stage = await self._judge.run_remote(
                StageRequest(
                    episode_id=request.episode_id,
                    protocol_hash=request.protocol_hash,
                    payload=judge_payload,
                )
            )
            stages.append(judge_stage)
            if judge_stage.timing is not None:
                timings.append(judge_stage.timing)
            status: Literal["completed", "blocked", "failed"] = (
                "completed" if judge_stage.status == "completed" else judge_stage.status
            )
            binary_success: Optional[bool] = None
            progress: Optional[int] = None
            judge_status: Optional[str] = None
            if judge_stage.status == "completed" and judge_stage.output:
                raw_success = judge_stage.output.get("binary_success")
                binary_success = raw_success if isinstance(raw_success, bool) else None
                raw_progress = judge_stage.output.get("progress")
                progress = int(raw_progress) if isinstance(raw_progress, int) and not isinstance(raw_progress, bool) else None
                raw_status = judge_stage.output.get("judge_status")
                judge_status = str(raw_status) if raw_status is not None else None
            return self._terminal(
                request,
                status,
                None if status == "completed" else "stage_judge_%s" % judge_stage.status,
                started=started,
                policy_arm=arm,
                stages=stages,
                timings=timings,
                segments=segments,
                executed=action_offset,
                horizon=episode.horizon_actions,
                frames=clip_frames,
                timestamps=timestamps,
                world_calls=world_calls,
                feedback_mode=feedback_mode,
                validity=validity_label,
                validity_reasons=validity_reasons,
                binary_success=binary_success,
                progress_score=progress,
                judge_status=judge_status,
            )

        async def _persist_terminal_result(
            self, request: RolloutRequest, result: RolloutResult, store: S3ResultStore
        ) -> RolloutResult:
            """Write the exact terminal object before returning it to Baseten.

            The configured bucket/prefix and SDK credentials come solely from
            Chain deployment context. The request carries only the pre-committed
            opaque key and identity/digest binding, so it cannot redirect a
            worker to an arbitrary object-store destination.
            """

            assert request.result_store is not None
            try:
                binding = ResultStoreBinding(**request.result_store.model_dump(mode="json"))
                await asyncio.to_thread(store.put_terminal_result, binding, result.model_dump(mode="json"))
                return result
            except (ResultStoreConfigurationError, ResultStoreConflictError, ResultStoreError, ValueError) as error:
                # Do not return a completed result when the durable write did
                # not happen. The failure result itself cannot be safely written
                # if the configured immutable key is unavailable/conflicting.
                return result.model_copy(
                    update={
                        "status": "failed",
                        "missing_reason": "result_store_write_failed:%s" % type(error).__name__,
                        "validity": None,
                        "binary_success": None,
                        "progress_score": None,
                    }
                )

        def _terminal(
            self,
            request: RolloutRequest,
            status: Literal["completed", "blocked", "failed"],
            missing_reason: Optional[str],
            *,
            started: float,
            policy_arm: Optional[str] = None,
            stages: Optional[List[StageResult]] = None,
            timings: Optional[List[StageTiming]] = None,
            segments: Optional[List[SegmentResult]] = None,
            executed: int = 0,
            horizon: Optional[int] = None,
            frames: Optional[Sequence[FramePayload]] = None,
            timestamps: Optional[Sequence[float]] = None,
            world_calls: int = 0,
            feedback_mode: Optional[str] = None,
            validity: Optional[str] = None,
            validity_reasons: Optional[List[str]] = None,
            binary_success: Optional[bool] = None,
            progress_score: Optional[int] = None,
            judge_status: Optional[str] = None,
        ) -> RolloutResult:
            all_timings = list(timings or [])
            all_timings.append(
                StageTiming(
                    stage="controller",
                    worker="plumb-rollout-controller",
                    wall_seconds=time.perf_counter() - started,
                    declared_gpu_count=0,
                )
            )
            validity_label = validity if validity in ("valid", "invalid", "unknown") else None
            history = list(stages or [])
            terminal: Dict[str, StageResult] = {}
            for entry in history:
                terminal[entry.stage] = entry
            return RolloutResult(
                run_id=request.run_id,
                episode_id=request.episode_id,
                protocol_hash=request.protocol_hash,
                status=status,
                stages=terminal,
                stage_history=history,
                missing_reason=missing_reason,
                policy_arm=policy_arm,
                horizon_actions=horizon,
                executed_actions=executed,
                segments=list(segments or []),
                timings=all_timings,
                gpu_seconds=_gpu_seconds(all_timings),
                frame_pixel_hashes=[
                    frame.pixels_sha256 for frame in (frames or []) if frame.pixels_sha256 is not None
                ],
                nominal_timestamps=[float(value) for value in (timestamps or [])],
                validity=validity_label,
                validity_reason_codes=list(validity_reasons or []),
                binary_success=binary_success,
                progress_score=progress_score,
                judge_status=judge_status,
                # No Chainlet may declare itself qualified: Gates A-D are external
                # reviews recorded in plumb.gates, never inferred from a call that
                # happened to return.
                qualified=False,
                feedback_mode=feedback_mode,
                physical_state_measured=False,
                world_calls=world_calls,
            )


else:
    # These names are intentionally absent as deployable classes in a normal
    # local checkout. Callers can still import request/result schemas, the pure
    # micro-batching logic, and check CHAINS_RUNTIME_AVAILABLE without
    # attempting to mock model execution.
    WorldWorker = None
    OpenVLAWorker = None
    OctoWorker = None
    MiniVLAWorker = None
    OpenPiZeroWorker = None
    SusieWorker = None
    ValidityWorker = None
    JudgeWorker = None
    PolicyRouter = None
    RolloutController = None


def _uniform_frame_indices(available: int, wanted: int) -> List[int]:
    """Uniform indices including both endpoints, with no duplicates.

    AGENT-BUILD-SPEC.md §5 requires exactly 16 frames sampled uniformly from the
    original start through the exact final control tick, including both
    endpoints.  A clip shorter than ``wanted`` is rejected by the caller rather
    than padded, so this never repeats a frame to reach the count.
    """

    if wanted < 2:
        raise ValueError("uniform sampling needs at least two frames")
    if available < wanted:
        raise ValueError("cannot sample %d frames from %d without repeating one" % (wanted, available))
    step = (available - 1) / float(wanted - 1)
    indices = [int(round(index * step)) for index in range(wanted)]
    if len(set(indices)) != wanted:
        raise ValueError("uniform sampling produced a repeated frame index; the clip is not sampled, it is rejected")
    return indices


POLICY_WORKER_CLASS_NAMES: Tuple[str, ...] = (
    "OpenVLAWorker",
    "OctoWorker",
    "MiniVLAWorker",
    "OpenPiZeroWorker",
    "SusieWorker",
)

WORKER_REQUIREMENTS_FILES: Mapping[str, str] = {
    "WorldWorker": "requirements/world-cosmos.txt",
    "OpenVLAWorker": "requirements/policy-openvla.txt",
    "OctoWorker": "requirements/policy-octo.txt",
    "MiniVLAWorker": "requirements/policy-minivla.txt",
    "OpenPiZeroWorker": "requirements/policy-openpizero.txt",
    "SusieWorker": "requirements/policy-susie.txt",
    "JudgeWorker": "requirements/judge-qwen.txt",
    "ValidityWorker": "requirements/validity.txt",
    "RolloutController": "requirements/controller.txt",
}


__all__ = [
    "CHAINS_RUNTIME_AVAILABLE",
    "ChainRuntimeUnavailable",
    "BatchContractError",
    "AUTOSCALING_HYPOTHESES",
    "AutoscalingHypothesis",
    "autoscaling_patch_payloads",
    "BATCH_MAX",
    "BATCH_WINDOW_MS",
    "BatchOutcome",
    "CANONICAL_POLICY_ARMS",
    "NATIVE_POLICY_CERTIFIED_ADAPTERS",
    "certified_adapter_requirements",
    "EpisodeControlPayload",
    "FramePayload",
    "GpuSecondAccounting",
    "JudgeCore",
    "JudgeReferenceImagePayload",
    "JudgeStagePayload",
    "MicroBatchPlanner",
    "MicroBatchQueue",
    "MicroBatchTicket",
    "NativePolicyCore",
    "POLICY_ARM_ALIASES",
    "POLICY_WORKER_CLASS_NAMES",
    "PolicyObservationPayload",
    "RolloutRequest",
    "RolloutResult",
    "SegmentResult",
    "StageRequest",
    "StageResult",
    "StageTiming",
    "UNPINNED_POLICY_ASSETS",
    "VALIDITY_CALIBRATION_CLASS",
    "ValidityGateCore",
    "ValidityStagePayload",
    "WORKER_ARMS",
    "WORKER_DECLARED_GPU_COUNT",
    "WORKER_REQUIREMENTS_FILES",
    "WORLD_VARIANTS",
    "WorldAdapterCore",
    "WorldStagePayload",
    "WorldVariant",
    "decode_frame",
    "encode_frame",
    "normalise_policy_arm",
    "require_chains_runtime",
    "segment_world_seed",
    "selected_world_variant",
    "PolicyWorker",
    "WorldWorker",
    "OpenVLAWorker",
    "OctoWorker",
    "MiniVLAWorker",
    "OpenPiZeroWorker",
    "SusieWorker",
    "ValidityWorker",
    "JudgeWorker",
    "PolicyRouter",
    "RolloutController",
]
