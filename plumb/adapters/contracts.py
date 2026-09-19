"""Dependency-light contracts shared by PLUMB runtime adapters.

The adapters deliberately do not import torch, diffusers, transformers, PIL, or
numpy at module import time.  A planning or gate-validation process must be
able to inspect this package on a machine that has none of the model stacks
installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


COSMOS3_FRAMEWORK_COMMIT = "c23e51f2f157ae3e51cfcd86ebfb5464850894f2"
COSMOS3_DIFFUSERS_SOURCE = (
    "https://github.com/huggingface/diffusers/blob/main/"
    "src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py"
)
IRASIM_COMMIT = "c72b6dade6fcd65971e0aa8ab49ea39b15108c90"

#: Attribution method for a backend call that several requests shared.  The
#: fused call's wall time is divided equally between its members.  Equal share
#: is a *convention*, not a per-item measurement: nothing in a single fused
#: diffusion forward isolates one batch member's share of the GPU.
FUSED_BATCH_EQUAL_SHARE = "fused_batch_equal_share"


class CapabilityStatus(str, Enum):
    """A capability is not a qualification result.

    ``READY_UNQUALIFIED`` means the local adapter can make a real backend call
    when its optional dependencies and assets are present.  It intentionally
    does *not* mean the call is scientifically qualified.
    """

    READY_UNQUALIFIED = "ready_unqualified"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    QUALIFIED = "qualified"


@dataclass(frozen=True)
class CapabilityResult:
    status: CapabilityStatus
    reason: str
    source_verified: bool = False
    evidence_uris: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allows_qualified_use(self) -> bool:
        """Only an external gate validator may return ``QUALIFIED``."""

        return self.status is CapabilityStatus.QUALIFIED


class FeedbackMode(str, Enum):
    UNQUALIFIED = "unqualified"
    NATIVE_FEEDBACK = "native_feedback"
    FORECAST_STATE = "forecast_state"
    FROZEN_STATE = "frozen_state"


@dataclass(frozen=True)
class BackendProfile:
    """Pinned configuration for exactly one backend request serializer."""

    profile_id: str
    backend: str
    model_id: str
    model_revision: Optional[str]
    code_revision: Optional[str]
    container_digest: Optional[str]
    normalizer_revision: Optional[str]
    source_uri: Optional[str] = None
    local_model_path: Optional[str] = None
    local_files_only: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Whether this can describe a reproducible profile (not performance)."""

        return bool(
            self.profile_id
            and self.model_id
            and self.model_revision
            and self.code_revision
            and self.container_digest
            and self.normalizer_revision
        )


@dataclass(frozen=True)
class ServerTiming:
    """Timing/memory captured around a real adapter call.

    Values are optional by design.  ``None`` means not observed, never zero.
    ``backend_calls`` counts actual Python backend invocations, not planned
    calls, and is therefore safe to put in an artifact report.

    Batch fields describe a *shared* backend call.  When ``batch_size`` is
    greater than one, ``wall_seconds``, ``gpu_peak_memory_bytes`` and
    ``model_load_seconds`` belong to the whole fused call rather than to this
    one result, and ``backend_calls`` counts the invocation that produced this
    result even though ``batch_size - 1`` other results came out of the same
    invocation.  Summing ``wall_seconds`` or ``backend_calls`` across the
    members of one batch therefore overcounts; use ``attributed_gpu_seconds``,
    which is explicitly an attribution named by ``attribution_method``.
    """

    backend_calls: int
    wall_seconds: Optional[float]
    cold_start: Optional[bool]
    gpu_peak_memory_bytes: Optional[int] = None
    model_load_seconds: Optional[float] = None
    queue_seconds: Optional[float] = None
    server_seconds: Optional[float] = None
    started_at_unix: Optional[float] = None
    finished_at_unix: Optional[float] = None
    #: How many requests shared the backend call that produced this result.
    #: ``None`` means the call was not batched or the batch size is unknown;
    #: it never means one.
    batch_size: Optional[int] = None
    #: ``wall_seconds`` divided between the batch members.  An attribution, not
    #: an independently measured per-item duration.
    attributed_gpu_seconds: Optional[float] = None
    #: Names the attribution rule, e.g. :data:`FUSED_BATCH_EQUAL_SHARE`.  It is
    #: ``None`` whenever ``attributed_gpu_seconds`` is ``None``, so an
    #: unlabelled attribution cannot exist.
    attribution_method: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.attributed_gpu_seconds is None) != (self.attribution_method is None):
            raise ValueError(
                "attributed_gpu_seconds and attribution_method must be set together; an attribution without a "
                "named method would be indistinguishable from a measurement."
            )
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("ServerTiming.batch_size must be at least 1 when it is recorded at all.")

    @property
    def is_shared_backend_call(self) -> bool:
        """Whether this timing was measured around a call shared with others."""

        return self.batch_size is not None and int(self.batch_size) > 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend_calls": self.backend_calls,
            "wall_seconds": self.wall_seconds,
            "cold_start": self.cold_start,
            "gpu_peak_memory_bytes": self.gpu_peak_memory_bytes,
            "model_load_seconds": self.model_load_seconds,
            "queue_seconds": self.queue_seconds,
            "server_seconds": self.server_seconds,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "batch_size": None if self.batch_size is None else int(self.batch_size),
            "attributed_gpu_seconds": (
                None if self.attributed_gpu_seconds is None else float(self.attributed_gpu_seconds)
            ),
            "attribution_method": self.attribution_method,
            "is_shared_backend_call": self.is_shared_backend_call,
            "wall_seconds_is_per_item_measurement": not self.is_shared_backend_call,
        }


@dataclass(frozen=True)
class WorldRequest:
    """Backend-neutral world-model request.

    ``conditioning_image`` remains opaque at this layer because a real image is
    fed directly to the lazily loaded backend.  ``compiled_actions`` must be
    backend-native: Cosmos receives 10-D raw action rows and IRASim receives
    7-D rows.  This makes accidental Cosmos-to-IRASim transport impossible to
    hide behind a generic ``actions`` field.
    """

    conditioning_image: Any
    prompt: str
    domain: str
    compiled_actions: Tuple[Tuple[float, ...], ...]
    nominal_control_timestamps: Tuple[float, ...]
    seed: int
    compatibility_profile_id: str
    feedback_mode: FeedbackMode = FeedbackMode.UNQUALIFIED
    source_state_lineage_id: Optional[str] = None
    request_id: Optional[str] = None

    def validate_basic(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("WorldRequest.prompt must be a nonempty exact task string.")
        if not self.domain:
            raise ValueError("WorldRequest.domain is required.")
        if not self.compatibility_profile_id:
            raise ValueError("WorldRequest.compatibility_profile_id is required.")
        if not self.compiled_actions:
            raise ValueError("WorldRequest requires at least one compiled action row.")
        if len(self.nominal_control_timestamps) != len(self.compiled_actions):
            raise ValueError(
                "nominal_control_timestamps must have one timestamp per action; "
                "they are wrapper-assigned nominal times, not physical measurements."
            )
        previous = None
        for timestamp in self.nominal_control_timestamps:
            if previous is not None and timestamp <= previous:
                raise ValueError("nominal_control_timestamps must be strictly increasing.")
            previous = timestamp
        for row in self.compiled_actions:
            if not row:
                raise ValueError("compiled action rows cannot be empty.")
            for value in row:
                if not isinstance(value, (int, float)):
                    raise ValueError("compiled action values must be finite numeric values.")
                # ``value != value`` catches NaN without importing math; reject infinities too.
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError("compiled action values must be finite.")


@dataclass(frozen=True)
class WorldResult:
    """A real backend result plus explicitly nominal frame timing metadata."""

    backend: str
    profile_id: str
    frames: Tuple[Any, ...]
    nominal_frame_timestamps: Tuple[float, ...]
    conditioning_frame_included: bool
    timing: ServerTiming
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def future_frames(self) -> Tuple[Any, ...]:
        if self.conditioning_frame_included:
            return self.frames[1:]
        return self.frames

    def validate(self, expected_action_count: int) -> None:
        expected = expected_action_count + (1 if self.conditioning_frame_included else 0)
        if len(self.frames) != expected:
            raise ValueError(
                "WorldResult has %d frames; expected %d for %d actions with "
                "conditioning_frame_included=%s."
                % (len(self.frames), expected, expected_action_count, self.conditioning_frame_included)
            )
        if len(self.nominal_frame_timestamps) != len(self.frames):
            raise ValueError("WorldResult frame timestamps must align one-to-one with frames.")
        if self.timing.backend_calls < 1:
            raise ValueError("WorldResult must report at least one actual backend call.")


@dataclass(frozen=True)
class PolicyContract:
    name: str
    required_observation_history: int
    requires_proprio: bool
    native_proposal_horizon: int
    certified_execute_prefix: Optional[int]
    temporal_ensembling: bool
    preprocessing: str
    normalization: str
    reset_rule: str
    rng_rule: str
    implementation_status: CapabilityStatus
    limitation: Optional[str] = None


@dataclass(frozen=True)
class PolicyObservation:
    """Canonical policy observation. Images are opaque lazy-backend values."""

    image_history: Tuple[Any, ...]
    prompt: str
    proprio: Optional[Tuple[float, ...]] = None
    goal_image: Any = None
    timestamp: Optional[float] = None
    proprio_convention: Optional[str] = None
