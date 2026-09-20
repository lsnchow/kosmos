"""KOSMOS experimental comparison Chain with low-level NDJSON streaming.

This module intentionally has a much smaller scope than ``chain.py``.  It is
the interactive/demo path: it streams *received* PNG frames to the local
receiver and never writes to S3, Cloudflare, an object store, or a callback.
The local application is the durable journal.  Consequently a broken stream
is terminally incomplete; this Chain never retries a rollout or pretends that
unreceived cloud output exists.

The Chain has one CPU controller, one selected world-model worker, and three
isolated policy workers.  The controller owns the 70-turn feedback loop.  A
world worker only turns one already-selected action (or one manual segment)
into frames.  This distinction prevents a localhost caller from accidentally
turning a native policy into a stale-image action chunk.

``DemoStreamCanary`` lives in ``demo_stream_canary.py`` so operators can prove
the streaming route on CPU before this Chain is pushed with any GPU workers.

The module deliberately omits ``from __future__ import annotations``.  The
Chains framework inspects endpoint annotations without resolving strings.
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
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


try:  # A regular checkout must remain importable without a deployment SDK.
    import truss_chains as chains
    from truss.base import truss_config
except ImportError:  # pragma: no cover - covered by import tests on laptops.
    chains = None  # type: ignore[assignment]
    truss_config = None  # type: ignore[assignment]


CHAINS_RUNTIME_AVAILABLE = chains is not None
STREAM_SCHEMA_VERSION = "kosmos-demo-ndjson-v1"
MAX_PNG_BYTES = 12 * 1024 * 1024
MAX_STREAM_RECORD_BYTES = 20 * 1024 * 1024
DEFAULT_HORIZON_ACTIONS = 70
DEFAULT_CONTROL_HZ = 5.0
TASK_TEXT = "Close the drawer"
COSMOS_MODEL_ID = "nvidia/Cosmos3-Nano"
COSMOS_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
OPENVLA_MODEL_ID = "openvla/openvla-7b"
OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
MODEL_CACHE_ROOT = "/app/model_cache"


class DemoContractError(ValueError):
    """A request cannot be executed without changing an experimental contract."""


class DemoRuntimeBlocked(RuntimeError):
    """A required local asset, license acknowledgement, or native runtime is absent."""


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    dumped = getattr(value, "as_dict", None)
    if callable(dumped):
        raw = dumped()
        return dict(raw) if isinstance(raw, Mapping) else {}
    return {}


def _bounded_text(value: Any, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DemoContractError("%s must be a nonempty bounded string" % field)
    return value.strip()


def _finite_row(value: Any, field: str, width: int = 7) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != width:
        raise DemoContractError("%s must be a %d-D numeric action row" % (field, width))
    row = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in row):
        raise DemoContractError("%s contains a non-finite value" % field)
    return row


def _finite_state(value: Any, field: str = "state") -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) not in (7, 8):
        raise DemoContractError("%s must be a source-layout 7-D or 8-D Bridge state" % field)
    state = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in state):
        raise DemoContractError("%s contains a non-finite value" % field)
    return state


def _validate_png_bytes(raw: bytes, field: str = "png") -> None:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise DemoContractError("%s must be PNG data" % field)
    if len(raw) > MAX_PNG_BYTES:
        raise DemoContractError("%s exceeds the %d byte limit" % (field, MAX_PNG_BYTES))


class PngFrame(BaseModel):
    """A real image transported once, with file and decoded-pixel digests."""

    model_config = ConfigDict(extra="forbid")

    encoding: Literal["png_base64"] = "png_base64"
    png_base64: str = Field(min_length=1, max_length=MAX_PNG_BYTES * 4 // 3 + 256)
    png_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    pixels_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)

    @classmethod
    def from_array(cls, frame: Any) -> "PngFrame":
        """Encode one backend-returned RGB frame without frame interpolation.

        World adapters may return uint8 RGB or float RGB in [0, 1].  The
        conversion is an image-file encoding boundary, not a synthesizer: no
        crop, resize, temporal repair, or image duplication is performed.
        """

        try:
            import numpy as numpy
            from PIL import Image
        except ImportError as error:  # pragma: no cover - image deploy requirement.
            raise DemoRuntimeBlocked("Pillow and NumPy are required to encode a received world frame") from error
        array = numpy.asarray(frame)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise DemoContractError("world frame must be HWC RGB/RGBA, got shape %r" % (tuple(array.shape),))
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != numpy.uint8:
            if not numpy.issubdtype(array.dtype, numpy.floating) or not numpy.isfinite(array).all():
                raise DemoContractError("world frame must be uint8 RGB or finite float RGB")
            lower, upper = float(array.min()), float(array.max())
            if lower < 0.0 or upper > 1.0:
                raise DemoContractError("floating world frame must be within [0,1]")
            array = numpy.rint(array * 255.0).astype(numpy.uint8)
        if not array.flags.c_contiguous:
            array = numpy.ascontiguousarray(array)
        image = Image.fromarray(array)
        payload = io.BytesIO()
        image.save(payload, format="PNG", optimize=False)
        raw = payload.getvalue()
        _validate_png_bytes(raw, "encoded world frame")
        return cls(
            png_base64=base64.b64encode(raw).decode("ascii"),
            png_sha256=_sha256(raw),
            pixels_sha256=_sha256(array.tobytes()),
            width=int(array.shape[1]),
            height=int(array.shape[0]),
        )

    def png_bytes(self) -> bytes:
        try:
            raw = base64.b64decode(self.png_base64, validate=True)
        except (ValueError, TypeError) as error:
            raise DemoContractError("png_base64 is not valid base64") from error
        _validate_png_bytes(raw)
        if _sha256(raw) != self.png_sha256:
            raise DemoContractError("png_sha256 does not bind png_base64")
        return raw

    def rgb_array(self) -> Any:
        try:
            import numpy as numpy
            from PIL import Image
        except ImportError as error:  # pragma: no cover
            raise DemoRuntimeBlocked("Pillow and NumPy are required to decode a source frame") from error
        raw = self.png_bytes()
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode != "RGB":
                    raise DemoContractError("source frames must already be RGB PNGs; implicit color conversion is forbidden")
                data = numpy.asarray(image)
        except DemoContractError:
            raise
        except Exception as error:  # noqa: BLE001 - malformed pixels are a request failure.
            raise DemoContractError("PNG decode failed") from error
        if data.shape != (self.height, self.width, 3):
            raise DemoContractError("PNG dimensions do not match its declared width/height")
        if _sha256(data.tobytes()) != self.pixels_sha256:
            raise DemoContractError("pixels_sha256 does not bind decoded PNG pixels")
        return data


class WorldProfile(BaseModel):
    """One chosen world family, not a hidden fallback list."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["cosmos", "irasim"]
    profile_id: str = Field(min_length=1, max_length=256)
    domain: str = Field(default="bridge_orig_lerobot", min_length=1, max_length=256)
    model_id: str = Field(min_length=1, max_length=256)
    model_revision: Optional[str] = Field(default=None, max_length=256)
    source_revision: Optional[str] = Field(default=None, max_length=256)
    resolution_tier: Optional[int] = Field(default=None, ge=1, le=2048)
    inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    asset_manifest_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def selected_backend_contract(self) -> "WorldProfile":
        if self.backend == "cosmos":
            if self.domain != "bridge_orig_lerobot":
                raise ValueError("Cosmos demo supports only bridge_orig_lerobot")
            if self.model_id != COSMOS_MODEL_ID or self.model_revision != COSMOS_REVISION:
                raise ValueError("Cosmos demo must retain the pinned Cosmos3-Nano identity")
            if self.resolution_tier not in (256, 480):
                raise ValueError("Cosmos demo profile must declare the 256 or 480 tier")
            if self.inference_steps != 30:
                raise ValueError("Cosmos demo profile must retain 30 denoising steps")
        else:
            if self.model_id != "bytedance/IRASim":
                raise ValueError("IRASim profile must retain its original model identity")
            if self.source_revision != "c72b6dade6fcd65971e0aa8ab49ea39b15108c90":
                raise ValueError("IRASim profile must retain the pinned original source revision")
            if self.resolution_tier != 256 or self.inference_steps not in (25, 50):
                raise ValueError("IRASim demo profile is fixed at 256 pixels and 25 or 50 inference steps")
        return self


class RolloutInput(BaseModel):
    """One cell, frozen by the local comparison service before remote work."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    comparison_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    cell_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    task: Literal["Close the drawer"] = TASK_TEXT
    policy_id: Literal["OpenVLA", "MiniVLA", "Octo-Small"]
    world_seed: int
    policy_seed: Optional[int] = None
    world: WorldProfile
    source_frame: PngFrame
    initial_source_state: List[float] = Field(min_length=7, max_length=8)
    bridge_control_profile_id: str = Field(min_length=1, max_length=256)
    held_gripper_action: float = Field(ge=0.0, le=1.0)
    horizon_actions: int = Field(default=DEFAULT_HORIZON_ACTIONS, ge=1, le=DEFAULT_HORIZON_ACTIONS)
    control_hz: float = Field(default=DEFAULT_CONTROL_HZ, gt=0.0, le=100.0)
    claim_tier: Literal["experimental_unscored"] = "experimental_unscored"
    source_bundle_id: str = Field(min_length=1, max_length=256)
    source_bundle_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    # The comparison service freezes these opaque, hash-bound values before it
    # opens the Chain stream.  The Chain echoes them verbatim; it never derives
    # an identity from an image URL or a deployment-local path.
    identity_hashes: Dict[str, str] = Field(min_length=1)
    policy_revision: str = Field(min_length=1, max_length=256)
    source_state_lineage_id: str = Field(min_length=1, max_length=256)
    proprio_lineage_id: str = Field(min_length=1, max_length=256)
    proprio_converter_revision: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def exactly_seventy_actions(self) -> "RolloutInput":
        if self.horizon_actions != DEFAULT_HORIZON_ACTIONS:
            raise ValueError("the controlled comparison is exactly 70 executed actions per cell")
        _finite_state(self.initial_source_state, "initial_source_state")
        return self


class ManualInput(BaseModel):
    """One stateless image-conditioned manual branch, never a latent restore."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    manual_session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    command_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    task: Literal["Close the drawer"] = TASK_TEXT
    direction: Literal["up", "down", "left", "right", "forward", "back"]
    world_seed: int
    world: WorldProfile
    condition_frame: PngFrame
    state_snapshot: List[float] = Field(min_length=7, max_length=8)
    bridge_control_profile_id: str = Field(min_length=1, max_length=256)
    held_gripper_action: float = Field(ge=0.0, le=1.0)
    translation_variant: Literal["baseline_0.0025", "development_0.005"] = "baseline_0.0025"
    source_branch_id: str = Field(min_length=1, max_length=256)
    source_branch_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    comparison_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    cell_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    identity_hashes: Dict[str, str] = Field(min_length=1)
    source_event_id: str = Field(min_length=1, max_length=256)
    source_png_sha256: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    claim_tier: Literal["experimental_unscored"] = "experimental_unscored"

    @model_validator(mode="after")
    def manual_state_is_explicit(self) -> "ManualInput":
        _finite_state(self.state_snapshot, "state_snapshot")
        if self.source_png_sha256 != self.condition_frame.png_sha256:
            raise ValueError("source_png_sha256 must bind the exact manual condition frame")
        return self


class DemoStreamRequest(BaseModel):
    """Single entrypoint shape; exactly one operation can be supplied."""

    model_config = ConfigDict(extra="forbid")

    rollout: Optional[RolloutInput] = None
    manual: Optional[ManualInput] = None

    @model_validator(mode="after")
    def one_operation(self) -> "DemoStreamRequest":
        if (self.rollout is None) == (self.manual is None):
            raise ValueError("exactly one of rollout or manual is required")
        return self


class ServiceDispatchRequest(BaseModel):
    """Flat local-receiver dispatch, carried under the Chain's ``request`` key.

    ``ComparisonService`` owns attempt creation and source-byte rehashing.  It
    submits its frozen flat record as ``{"request": <this object>}``; the
    Chain immediately converts it into its narrow typed execution request.
    The conversion is deliberately in this module so a field cannot disappear
    between the local journal and GPU controller unnoticed.
    """

    model_config = ConfigDict(extra="forbid")

    schema: Literal["kosmos-chain-stream-v1"]
    kind: Literal["comparison_rollout", "manual_world_segment"]
    comparison_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    cell_id: str = Field(min_length=1, max_length=128)
    task: Literal["close_drawer"]
    task_instruction: Literal["Close the drawer"]
    world: Dict[str, Any]
    identity_hashes: Dict[str, str] = Field(min_length=1)
    claim_tier: str = Field(min_length=1, max_length=128)
    scored: Literal[False]
    # Comparison-only frozen fields.
    policy: Optional[Literal["OpenVLA", "MiniVLA", "Octo-Small"]] = None
    world_seed: Optional[int] = None
    horizon: Optional[int] = None
    start: Optional[Dict[str, Any]] = None
    policy_profile: Optional[Dict[str, Any]] = None
    # Manual-only frozen fields. ``actions`` remains a service audit record;
    # it is never used as the world command. The Chain derives the approved
    # constant 0.0025-m direction row from ``direction``.
    mode: Optional[Literal["manual"]] = None
    session_id: Optional[str] = None
    command_id: Optional[str] = None
    direction: Optional[Literal["up", "down", "left", "right", "forward", "back"]] = None
    actions: Optional[List[List[float]]] = None
    source: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def dispatch_shape(self) -> "ServiceDispatchRequest":
        if self.kind == "comparison_rollout":
            if self.policy is None or self.world_seed is None or self.horizon != DEFAULT_HORIZON_ACTIONS:
                raise ValueError("comparison rollout must freeze policy, world_seed, and exactly a 70-action horizon")
            if self.start is None or self.policy_profile is None:
                raise ValueError("comparison rollout must include its frozen start and policy_profile")
            if any(value is not None for value in (self.command_id, self.direction, self.source)):
                raise ValueError("comparison rollout cannot carry manual command fields")
        else:
            if self.mode != "manual" or not self.session_id or not self.command_id or not self.direction or self.source is None:
                raise ValueError("manual dispatch must bind session, command, direction, and source")
            if self.actions is not None and len(self.actions) != 16:
                raise ValueError("manual audit action list must retain the frozen 16-row structural request")
        return self


def _png_frame_from_dispatch(value: Mapping[str, Any], *, field: str) -> PngFrame:
    """Bind the local service's original PNG bytes without a re-encode."""

    encoded = value.get("png_base64")
    expected = _normalised_sha(value.get("png_sha256"))
    if not isinstance(encoded, str) or expected is None:
        raise DemoContractError("%s requires hash-bound png_base64/png_sha256" % field)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as error:
        raise DemoContractError("%s png_base64 is invalid" % field) from error
    _validate_png_bytes(raw, field)
    if _sha256(raw) != expected:
        raise DemoContractError("%s png_sha256 does not bind supplied source bytes" % field)
    try:
        import numpy as numpy
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode != "RGB":
                raise DemoContractError("%s must be an RGB PNG; implicit source color conversion is forbidden" % field)
            array = numpy.asarray(image)
    except DemoContractError:
        raise
    except Exception as error:  # noqa: BLE001
        raise DemoContractError("%s cannot be decoded as RGB PNG" % field) from error
    return PngFrame(
        png_base64=encoded,
        png_sha256=expected,
        pixels_sha256=_sha256(array.tobytes()),
        width=int(array.shape[1]),
        height=int(array.shape[0]),
    )


def _dispatch_state(value: Any, *, field: str) -> Tuple[List[float], Mapping[str, Any], Literal["source_measured", "forecast_integrated"]]:
    if not isinstance(value, Mapping):
        raise DemoContractError("%s must be a hash-bound source state object" % field)
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise DemoContractError("%s.payload must be an object" % field)
    actual_digest = _sha256(_canonical_json(dict(payload)).encode("utf-8"))
    if _normalised_sha(value.get("sha256")) != actual_digest:
        raise DemoContractError("%s.sha256 does not bind its canonical payload" % field)
    conventions = _normalised_sha(value.get("conventions_sha256"))
    if conventions is None:
        raise DemoContractError("%s.conventions_sha256 is required" % field)
    values = payload.get("bridge_state", payload.get("values"))
    state = list(_finite_state(values, field + ".payload.bridge_state"))
    origin_raw = value.get("origin")
    if not isinstance(origin_raw, str):
        raise DemoContractError("%s.origin is required" % field)
    # Service source receipts name their exact source convention; only the two
    # categories cross into the native policy router.
    origin: Literal["source_measured", "forecast_integrated"] = (
        "forecast_integrated" if origin_raw == "forecast_integrated" else "source_measured"
    )
    return state, {"state_sha256": actual_digest, "conventions_sha256": conventions, "origin": origin_raw}, origin


def _dispatch_world(value: Mapping[str, Any], identity_hashes: Mapping[str, str]) -> Tuple[WorldProfile, str, float]:
    # A display identity/profile digest alone cannot tell a worker which model
    # serializer to run. The service must carry this immutable execution object
    # from its verified source manifest, and it must bind the recorded profile
    # digest in identity_hashes rather than looking up mutable configuration.
    runtime = value.get("runtime_profile")
    if not isinstance(runtime, Mapping):
        raise DemoContractError("world.runtime_profile is required for Chain execution; display identity is insufficient")
    profile = WorldProfile.model_validate(runtime)
    requested = _normalised_sha(value.get("profile_sha256"))
    identity = _normalised_sha(identity_hashes.get("world_profile"))
    if requested is None or identity is None or requested != identity:
        raise DemoContractError("world profile digest must match the frozen service identity hash")
    control_profile = value.get("bridge_control_profile_id")
    gripper = value.get("held_gripper_action")
    if not isinstance(control_profile, str) or not control_profile.strip():
        raise DemoContractError("world.bridge_control_profile_id is required")
    if isinstance(gripper, bool) or not isinstance(gripper, (int, float)) or not math.isfinite(float(gripper)) or not 0.0 <= float(gripper) <= 1.0:
        raise DemoContractError("world.held_gripper_action must be a source-bound finite [0,1] action value")
    return profile, control_profile, float(gripper)


def service_dispatch_to_stream_request(value: Any) -> DemoStreamRequest:
    """Strictly translate one local service payload to the Chain input model."""

    dispatch = value if isinstance(value, ServiceDispatchRequest) else ServiceDispatchRequest.model_validate(value)
    world, bridge_profile, held_gripper = _dispatch_world(dispatch.world, dispatch.identity_hashes)
    if dispatch.kind == "comparison_rollout":
        assert dispatch.start is not None and dispatch.policy is not None and dispatch.policy_profile is not None and dispatch.world_seed is not None
        frame = _png_frame_from_dispatch(dispatch.start, field="start")
        state, state_identity, state_kind = _dispatch_state(dispatch.start.get("state"), field="start.state")
        source_id = dispatch.start.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise DemoContractError("start.id is required")
        policy_revision = _normalised_sha(dispatch.policy_profile.get("profile"))
        if policy_revision is None:
            raise DemoContractError("policy_profile.profile digest is required")
        source_manifest = _normalised_sha(dispatch.identity_hashes.get("source_manifest"))
        if source_manifest is None:
            raise DemoContractError("identity_hashes.source_manifest is required")
        return DemoStreamRequest(
            rollout=RolloutInput(
                attempt_id=dispatch.attempt_id,
                comparison_id=dispatch.comparison_id,
                cell_id=dispatch.cell_id,
                policy_id=dispatch.policy,
                world_seed=dispatch.world_seed,
                world=world,
                source_frame=frame,
                initial_source_state=state,
                bridge_control_profile_id=bridge_profile,
                held_gripper_action=held_gripper,
                source_bundle_id=source_id,
                source_bundle_sha256=source_manifest,
                identity_hashes=dict(dispatch.identity_hashes),
                policy_revision=policy_revision,
                source_state_lineage_id=str(state_identity["state_sha256"]),
                proprio_lineage_id=str(state_identity["state_sha256"]),
                proprio_converter_revision=str(state_identity["conventions_sha256"]),
            )
        )
    assert dispatch.source is not None and dispatch.command_id is not None and dispatch.session_id is not None and dispatch.direction is not None
    frame = _png_frame_from_dispatch(dispatch.source, field="source")
    state, state_identity, _state_kind = _dispatch_state(dispatch.source.get("state"), field="source.state")
    event_id = dispatch.source.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise DemoContractError("manual source.event_id is required")
    manual_seed = dispatch.world.get("manual_seed")
    if isinstance(manual_seed, bool) or not isinstance(manual_seed, int):
        raise DemoContractError("world.manual_seed is required for a reproducible manual world call")
    return DemoStreamRequest(
        manual=ManualInput(
            attempt_id=dispatch.attempt_id,
            manual_session_id=dispatch.session_id,
            command_id=dispatch.command_id,
            direction=dispatch.direction,
            world_seed=int(manual_seed),
            world=world,
            condition_frame=frame,
            state_snapshot=state,
            bridge_control_profile_id=bridge_profile,
            held_gripper_action=held_gripper,
            source_branch_id=dispatch.session_id,
            source_branch_sha256=str(state_identity["state_sha256"]),
            comparison_id=dispatch.comparison_id,
            cell_id=dispatch.cell_id,
            identity_hashes=dict(dispatch.identity_hashes),
            source_event_id=event_id,
            source_png_sha256=frame.png_sha256,
        )
    )


class PolicyTurnRequest(BaseModel):
    """Boundary from controller to a single isolated policy image."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    cell_id: str
    policy_id: Literal["OpenVLA", "MiniVLA", "Octo-Small"]
    task: Literal["Close the drawer"] = TASK_TEXT
    observation: PngFrame
    prior_observation: Optional[PngFrame] = None
    proprio_state: List[float] = Field(min_length=7, max_length=8)
    policy_seed: Optional[int] = None
    prior_policy_snapshot: Optional[Dict[str, Any]] = None
    source_bundle_sha256: str
    action_index: int = Field(ge=0, le=DEFAULT_HORIZON_ACTIONS - 1)
    policy_revision: str = Field(min_length=1, max_length=256)
    state_kind: Literal["source_measured", "forecast_integrated"]
    state_lineage_id: str = Field(min_length=1, max_length=256)
    proprio_lineage_id: str = Field(min_length=1, max_length=256)
    proprio_converter_revision: str = Field(min_length=1, max_length=256)


class PolicyTurnResult(BaseModel):
    """Policy output retains native proposal *and* source-executed action."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "blocked", "failed"]
    raw_proposal_rows: List[List[float]] = Field(default_factory=list)
    source_executed_rows: List[List[float]] = Field(default_factory=list)
    policy_snapshot: Optional[Dict[str, Any]] = None
    identifiers: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


class WorldTurnRequest(BaseModel):
    """Boundary from controller to the selected world worker."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str
    operation: Literal["policy_step", "manual_segment"]
    task: Literal["Close the drawer"] = TASK_TEXT
    world: WorldProfile
    condition: PngFrame
    native_actions: List[List[float]] = Field(min_length=1, max_length=16)
    compiled_actions: List[List[float]] = Field(min_length=1, max_length=16)
    nominal_control_timestamps: List[float] = Field(min_length=1, max_length=16)
    world_seed: int
    state_before: List[float] = Field(min_length=7, max_length=8)
    state_after: Optional[List[float]] = Field(default=None, min_length=7, max_length=8)
    feedback_mode: Literal["forecast_state"] = "forecast_state"
    request_id: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def exact_backend_shapes(self) -> "WorldTurnRequest":
        if len(self.native_actions) != len(self.compiled_actions) or len(self.native_actions) != len(self.nominal_control_timestamps):
            raise ValueError("world turn needs one native row, compiled row, and timestamp per action")
        expected = 10 if self.world.backend == "cosmos" else 7
        for index, action in enumerate(self.native_actions):
            _finite_row(action, "native_actions[%d]" % index, 7)
        for index, action in enumerate(self.compiled_actions):
            _finite_row(action, "compiled_actions[%d]" % index, expected)
        if self.operation == "policy_step" and len(self.native_actions) != 1:
            raise ValueError("a policy feedback turn is exactly one native action")
        manual_count = 16 if self.world.backend == "cosmos" else 15
        if self.operation == "manual_segment" and len(self.native_actions) != manual_count:
            raise ValueError("manual %s needs exactly %d native rows" % (self.world.backend, manual_count))
        _finite_state(self.state_before, "state_before")
        if self.state_after is not None:
            _finite_state(self.state_after, "state_after")
        return self


class WorldTurnResult(BaseModel):
    """Only post-conditioning frames leave the world worker."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "blocked", "failed"]
    frames: List[PngFrame] = Field(default_factory=list)
    expected_post_conditioning_frames: int = Field(ge=0)
    conditioning_frame_removed: bool = False
    identifiers: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)


def _event(kind: Literal["header", "stage", "frame", "heartbeat", "terminal"], payload: Mapping[str, Any]) -> str:
    """Write one independently parseable bounded NDJSON record.

    The caller intentionally does not treat an HTTP chunk as a record.  A
    local receiver must buffer until ``\n`` and persist validated records before
    publishing browser SSE.  This helper only makes the producer's boundary
    unambiguous.
    """

    record = {"schema": "kosmos-chain-stream-v1", "kind": kind, **dict(payload)}
    event_id = record.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        # Event IDs bind semantic content, including frame digests. They are
        # deterministic across a reconnect/replay but distinct per stage/frame
        # record in one attempt because action/sequence/status are present.
        material = dict(record)
        material.pop("event_id", None)
        record["event_id"] = "evt-" + hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:32]
    encoded = _canonical_json(record).encode("utf-8")
    if len(encoded) > MAX_STREAM_RECORD_BYTES:
        raise DemoContractError("NDJSON %s record exceeds the bounded stream record size" % kind)
    return encoded.decode("utf-8") + "\n"


def parse_ndjson_records(chunks: Iterable[bytes]) -> List[Dict[str, Any]]:
    """Reference parser for local receiver tests; it makes no network calls."""

    buffer = b""
    records: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise DemoContractError("stream chunk must be bytes")
        buffer += chunk
        if len(buffer) > MAX_STREAM_RECORD_BYTES:
            raise DemoContractError("unterminated NDJSON record exceeds the bound")
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line:
                continue
            if len(line) > MAX_STREAM_RECORD_BYTES:
                raise DemoContractError("NDJSON record exceeds the bound")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DemoContractError("malformed NDJSON record") from error
            if not isinstance(value, Mapping) or value.get("schema") != "kosmos-chain-stream-v1":
                raise DemoContractError("NDJSON record does not carry the demo schema version")
            records.append(dict(value))
    if buffer:
        raise DemoContractError("truncated NDJSON record; result is incomplete")
    return records


def _state_evidence(values: Sequence[float], *, origin: Literal["source_measured", "forecast_integrated"], conventions: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the receiver's hash-bound state object without a hidden restore claim."""

    payload = {"values": list(_finite_state(values)), "mode": "forecast_state" if origin == "forecast_integrated" else "source_state"}
    convention_hash = _sha256(_canonical_json(dict(conventions)).encode("utf-8"))
    return {
        "payload": payload,
        "origin": origin,
        "conventions_sha256": convention_hash,
        "sha256": _sha256(_canonical_json(payload).encode("utf-8")),
    }


def _policy_result(value: Any) -> PolicyTurnResult:
    if isinstance(value, PolicyTurnResult):
        return value
    try:
        return PolicyTurnResult.model_validate(_json_mapping(value))
    except Exception as error:  # noqa: BLE001
        raise DemoContractError("isolated policy worker returned no valid policy turn result") from error


def _world_result(value: Any) -> WorldTurnResult:
    if isinstance(value, WorldTurnResult):
        return value
    try:
        return WorldTurnResult.model_validate(_json_mapping(value))
    except Exception as error:  # noqa: BLE001
        raise DemoContractError("world worker returned no valid world turn result") from error


async def _await_maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _forecast_compile(
    world: WorldProfile,
    bridge_control_profile_id: str,
    state_values: Sequence[float],
    actions: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[float], Dict[str, Any]]:
    """Use the pinned stateful Bridge/TCP/OpenCV compiler for Cosmos only."""

    try:
        from plumb.adapters.bridge import (
            BridgeControlIntegrator,
            BridgeControlProfile,
            BridgeState,
            BridgeToCosmosCompiler,
            NormalizationBoundary,
        )
    except ImportError as error:  # pragma: no cover - staging error in deployment.
        raise DemoRuntimeBlocked("the staged plumb Bridge compiler is unavailable") from error
    state = BridgeState.from_values(_finite_state(state_values), source="source_measured" if getattr(state_values, "source", None) else "forecast")
    profile = BridgeControlProfile(profile_id=bridge_control_profile_id)
    integrator = BridgeControlIntegrator(profile)
    canonical = [_finite_row(action, "native action", 7) for action in actions]
    if world.backend == "cosmos":
        # Backend normalization is deliberately zero applications here; a
        # checkpoint-specific normalizer may only be introduced as a separately
        # evidenced profile, never guessed by this interactive path.
        compiled = BridgeToCosmosCompiler(integrator, None, NormalizationBoundary.BACKEND).compile(state, canonical)
        return (
            [list(row) for row in compiled.backend_actions],
            list(compiled.forecast_states[-1].values8()),
            {
                "compiler": "plumb.adapters.bridge.BridgeToCosmosCompiler",
                "normalization_boundary": "backend",
                "normalization_applications": compiled.normalization_applications,
                "state_mode": "forecast_state",
                "state_source": compiled.forecast_states[-1].source,
            },
        )
    # IRASim consumes its native 7-D rows.  Integrating state is only a
    # controller-side forecast for the next policy observation, not an IRASim
    # state-restoration claim.
    forecast = integrator.integrate(state, canonical)
    return (
        [list(row) for row in canonical],
        list(forecast[-1].values8()),
        {
            "compiler": "IRASim native 7-D; Bridge integration only for explicit forecast proprioception",
            "normalization_boundary": "none",
            "normalization_applications": 0,
            "state_mode": "forecast_state",
            "state_source": forecast[-1].source,
        },
    )


def _manual_action(direction: str, magnitude: float, gripper: float) -> List[float]:
    """Frozen direction mapping; gripper is a source-bound *action* value."""

    mapping = {
        "right": (magnitude, 0.0, 0.0),
        "left": (-magnitude, 0.0, 0.0),
        "up": (0.0, 0.0, magnitude),
        "down": (0.0, 0.0, -magnitude),
        "forward": (0.0, magnitude, 0.0),
        "back": (0.0, -magnitude, 0.0),
    }
    if direction not in mapping:
        raise DemoContractError("unsupported manual direction")
    xyz = mapping[direction]
    return [xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, float(gripper)]


@dataclass
class DemoControllerCore:
    """Dependency-injected controller used both by Chainlets and CPU tests."""

    policy_workers: Mapping[str, Any]
    world_worker: Any

    async def _policy_call(self, policy_id: str, request: PolicyTurnRequest) -> PolicyTurnResult:
        worker = self.policy_workers.get(policy_id)
        if worker is None:
            return PolicyTurnResult(status="blocked", reasons=["no isolated worker is configured for %s" % policy_id])
        call = getattr(worker, "run_remote", None)
        if not callable(call):
            return PolicyTurnResult(status="blocked", reasons=["isolated policy worker has no run_remote endpoint"])
        return _policy_result(await _await_maybe(call(request)))

    async def _world_call(self, request: WorldTurnRequest) -> WorldTurnResult:
        call = getattr(self.world_worker, "run_remote", None)
        if not callable(call):
            return WorldTurnResult(status="blocked", expected_post_conditioning_frames=0, reasons=["selected world worker has no run_remote endpoint"])
        return _world_result(await _await_maybe(call(request)))

    async def rollout_events(self, request: RolloutInput) -> AsyncIterator[str]:
        """Run exactly one cell and stream only validated post-condition frames."""

        operation = "rollout"
        event_context = {
            "attempt_id": request.attempt_id,
            "comparison_id": request.comparison_id,
            "cell_id": request.cell_id,
            "mode": "comparison",
        }

        def emit(kind: Literal["header", "stage", "frame", "heartbeat", "terminal"], payload: Mapping[str, Any]) -> str:
            return _event(kind, {**event_context, **dict(payload)})

        start = time.time()
        emitted = 0
        condition = request.source_frame
        prior: Optional[PngFrame] = None
        state = list(_finite_state(request.initial_source_state))
        policy_snapshot: Optional[Dict[str, Any]] = None
        yield emit(
            "header",
            {
                "operation": operation,
                "task": "close_drawer",
                "task_instruction": request.task,
                "policy": request.policy_id,
                "policy_id": request.policy_id,
                "world": request.world.model_dump(mode="json"),
                "world_seed": request.world_seed,
                "policy_seed": request.policy_seed,
                "horizon_actions": request.horizon_actions,
                "source": {
                    "bundle_id": request.source_bundle_id,
                    "bundle_sha256": request.source_bundle_sha256,
                    "initial_png_sha256": request.source_frame.png_sha256,
                    "initial_pixels_sha256": request.source_frame.pixels_sha256,
                    "state_mode": "source_measured_initial_then_forecast",
                },
                "claim_tier": request.claim_tier,
                "scored": False,
                "persistence": "local_receiver_only_no_remote_result_store",
                "identity_hashes": request.identity_hashes,
            },
        )
        try:
            for action_index in range(request.horizon_actions):
                yield emit("stage", {"operation": operation, "action_index": action_index, "stage": "policy", "status": "started"})
                policy_request = PolicyTurnRequest(
                    attempt_id=request.attempt_id,
                    cell_id=request.cell_id,
                    policy_id=request.policy_id,
                    observation=condition,
                    prior_observation=prior,
                    proprio_state=state,
                    policy_seed=request.policy_seed,
                    prior_policy_snapshot=policy_snapshot,
                    source_bundle_sha256=request.source_bundle_sha256,
                    action_index=action_index,
                    policy_revision=request.policy_revision,
                    state_kind="source_measured" if action_index == 0 else "forecast_integrated",
                    state_lineage_id=request.source_state_lineage_id if action_index == 0 else "%s:forecast:%d" % (request.source_state_lineage_id, action_index),
                    proprio_lineage_id=request.proprio_lineage_id if action_index == 0 else "%s:forecast:%d" % (request.proprio_lineage_id, action_index),
                    proprio_converter_revision=request.proprio_converter_revision,
                )
                policy = await self._policy_call(request.policy_id, policy_request)
                yield emit(
                    "stage",
                    {
                        "operation": operation,
                        "action_index": action_index,
                        "stage": "policy",
                        "status": policy.status,
                        "identifiers": policy.identifiers,
                        "timing": policy.timing,
                        "reasons": policy.reasons,
                    },
                )
                if policy.status != "completed":
                    yield emit("terminal", _terminal_payload(request.attempt_id, operation, policy.status, emitted, request.horizon_actions, policy.reasons, start))
                    return
                raw_rows = [_finite_row(row, "raw_proposal_rows[%d]" % index) for index, row in enumerate(policy.raw_proposal_rows)]
                executed_rows = [_finite_row(row, "source_executed_rows[%d]" % index) for index, row in enumerate(policy.source_executed_rows)]
                if len(executed_rows) != 1:
                    yield emit("terminal", _terminal_payload(request.attempt_id, operation, "failed", emitted, request.horizon_actions, ["policy must return exactly one source-executed action per feedback turn"], start))
                    return
                native = list(executed_rows[0])
                yield emit("stage", {"operation": operation, "action_index": action_index, "stage": "compile", "status": "started"})
                compiled_rows, state_after, compiler_detail = _forecast_compile(
                    request.world, request.bridge_control_profile_id, state, [native]
                )
                yield emit(
                    "stage",
                    {
                        "operation": operation,
                        "action_index": action_index,
                        "stage": "compile",
                        "status": "completed",
                        "detail": compiler_detail,
                    },
                )
                timestamp = float(action_index) / request.control_hz
                world_request = WorldTurnRequest(
                    attempt_id=request.attempt_id,
                    operation="policy_step",
                    world=request.world,
                    condition=condition,
                    native_actions=[native],
                    compiled_actions=compiled_rows,
                    nominal_control_timestamps=[timestamp],
                    world_seed=request.world_seed,
                    state_before=state,
                    state_after=state_after,
                    request_id="%s:%s:%d" % (request.attempt_id, request.cell_id, action_index),
                )
                yield emit("stage", {"operation": operation, "action_index": action_index, "stage": "world", "status": "started"})
                world = await self._world_call(world_request)
                yield emit(
                    "stage",
                    {
                        "operation": operation,
                        "action_index": action_index,
                        "stage": "world",
                        "status": world.status,
                        "identifiers": world.identifiers,
                        "timing": world.timing,
                        "reasons": world.reasons,
                    },
                )
                if world.status != "completed":
                    yield emit("terminal", _terminal_payload(request.attempt_id, operation, world.status, emitted, request.horizon_actions, world.reasons, start))
                    return
                if len(world.frames) != 1 or world.expected_post_conditioning_frames != 1 or not world.conditioning_frame_removed:
                    yield emit("terminal", _terminal_payload(request.attempt_id, operation, "failed", emitted, request.horizon_actions, ["one policy action must yield exactly one received post-conditioning frame"], start))
                    return
                frame = world.frames[0]
                state_record = _state_evidence(
                    state_after,
                    origin="forecast_integrated",
                    conventions={"bridge_control_profile_id": request.bridge_control_profile_id, **compiler_detail},
                )
                yield emit(
                    "frame",
                    {
                        "operation": operation,
                        "sequence": emitted,
                        "action_index": action_index,
                        "segment_id": "%s:%s:%03d" % (request.attempt_id, request.cell_id, action_index),
                        "frame_index": emitted + 1,
                        "png_base64": frame.png_base64,
                        "png_sha256": frame.png_sha256,
                        "pixels_sha256": frame.pixels_sha256,
                        "state": state_record,
                        "action": {"raw_proposal": [list(row) for row in raw_rows], "executed": native},
                        "timing": {"policy": policy.timing, "world": world.timing, "nominal_control_timestamp": timestamp},
                        "identifiers": {"policy": policy.identifiers, "world": world.identifiers, "world_request_id": world_request.request_id},
                    },
                )
                emitted += 1
                prior, condition, state, policy_snapshot = condition, frame, state_after, policy.policy_snapshot
                if emitted % 5 == 0:
                    yield emit("heartbeat", {"operation": operation, "executed_actions": emitted, "horizon_actions": request.horizon_actions, "unix_time": time.time()})
        except Exception as error:  # noqa: BLE001 - an unexpected error is explicit incomplete evidence.
            yield emit("terminal", _terminal_payload(request.attempt_id, operation, "failed", emitted, request.horizon_actions, ["controller_exception:%s:%s" % (type(error).__name__, error)], start))
            return
        yield emit("terminal", _terminal_payload(request.attempt_id, operation, "completed", emitted, request.horizon_actions, [], start))

    async def manual_events(self, request: ManualInput) -> AsyncIterator[str]:
        """Generate one manual segment without invoking any policy worker."""

        operation = "manual_segment"
        event_context = {
            "attempt_id": request.attempt_id,
            "comparison_id": request.comparison_id,
            "cell_id": request.cell_id,
            "mode": "manual",
        }

        def emit(kind: Literal["header", "stage", "frame", "heartbeat", "terminal"], payload: Mapping[str, Any]) -> str:
            return _event(kind, {**event_context, **dict(payload)})

        start = time.time()
        magnitude = 0.0025 if request.translation_variant == "baseline_0.0025" else 0.005
        native = _manual_action(request.direction, magnitude, request.held_gripper_action)
        action_count = 16 if request.world.backend == "cosmos" else 15
        actions = [list(native) for _ in range(action_count)]
        yield emit(
            "header",
            {
                "operation": operation,
                "manual_session_id": request.manual_session_id,
                "command_id": request.command_id,
                "task": "close_drawer",
                "task_instruction": request.task,
                "world": request.world.model_dump(mode="json"),
                "world_seed": request.world_seed,
                "source_branch": {"id": request.source_branch_id, "sha256": request.source_branch_sha256, "condition_png_sha256": request.condition_frame.png_sha256},
                "source_event_id": request.source_event_id,
                "source_png_sha256": request.source_png_sha256,
                "identity_hashes": request.identity_hashes,
                "state_mode": "new_image_conditioned_branch_not_latent_restoration",
                "policy_calls": 0,
                "translation_variant": request.translation_variant,
                "translation_meters": magnitude,
                "claim_tier": request.claim_tier,
                "scored": False,
            },
        )
        if request.world.backend != "cosmos":
            # The admitted local steering receiver currently stores 16
            # post-condition frames. Original IRASim has only 15 post-condition
            # frames for its 15-action native operation; padding with the
            # condition frame would be fabricated, so it stays explicitly
            # unsupported until a separate manual protocol is frozen.
            yield emit("terminal", _terminal_payload(request.attempt_id, operation, "blocked", 0, 16, ["IRASim manual 15->16 structural profile is not admitted by the current 16-post-frame steering protocol"], start))
            return
        try:
            yield emit("stage", {"operation": operation, "stage": "compile", "status": "started"})
            compiled_rows, state_after, compiler_detail = _forecast_compile(
                request.world, request.bridge_control_profile_id, request.state_snapshot, actions
            )
            yield emit("stage", {"operation": operation, "stage": "compile", "status": "completed", "detail": compiler_detail})
            world_request = WorldTurnRequest(
                attempt_id=request.attempt_id,
                operation="manual_segment",
                world=request.world,
                condition=request.condition_frame,
                native_actions=actions,
                compiled_actions=compiled_rows,
                nominal_control_timestamps=[float(index) / DEFAULT_CONTROL_HZ for index in range(action_count)],
                world_seed=request.world_seed,
                state_before=list(_finite_state(request.state_snapshot)),
                state_after=state_after,
                request_id="%s:%s" % (request.attempt_id, request.command_id),
            )
            yield emit("stage", {"operation": operation, "stage": "world", "status": "started"})
            world = await self._world_call(world_request)
            yield emit("stage", {"operation": operation, "stage": "world", "status": world.status, "identifiers": world.identifiers, "timing": world.timing, "reasons": world.reasons})
            if world.status != "completed":
                yield emit("terminal", _terminal_payload(request.attempt_id, operation, world.status, 0, action_count, world.reasons, start))
                return
            if world.expected_post_conditioning_frames != action_count or len(world.frames) != action_count or not world.conditioning_frame_removed:
                yield emit("terminal", _terminal_payload(request.attempt_id, operation, "failed", 0, action_count, ["manual world result violated its post-conditioning frame contract"], start))
                return
            for sequence, frame in enumerate(world.frames):
                state_record = _state_evidence(
                    state_after,
                    origin="forecast_integrated",
                    conventions={"bridge_control_profile_id": request.bridge_control_profile_id, **compiler_detail},
                )
                yield emit(
                    "frame",
                    {
                        "operation": operation,
                        "manual_session_id": request.manual_session_id,
                        "command_id": request.command_id,
                        "sequence": sequence,
                        "segment_id": "%s:%s" % (request.attempt_id, request.command_id),
                        "frame_index": sequence + 1,
                        "png_base64": frame.png_base64,
                        "png_sha256": frame.png_sha256,
                        "pixels_sha256": frame.pixels_sha256,
                        "state": state_record,
                        "timing": world.timing,
                        "identifiers": {"world": world.identifiers, "world_request_id": world_request.request_id},
                    },
                )
        except Exception as error:  # noqa: BLE001
            yield emit("terminal", _terminal_payload(request.attempt_id, operation, "failed", 0, action_count, ["controller_exception:%s:%s" % (type(error).__name__, error)], start))
            return
        yield emit("terminal", _terminal_payload(request.attempt_id, operation, "completed", action_count, action_count, [], start))


def _terminal_payload(
    attempt_id: str,
    operation: str,
    status: str,
    executed: int,
    expected: int,
    reasons: Sequence[str],
    start: float,
) -> Dict[str, Any]:
    return {
        "operation": operation,
        "attempt_id": attempt_id,
        "status": status,
        "executed_actions": executed,
        "expected_actions": expected,
        "action_count": executed,
        "frame_count": executed,
        "complete": status == "completed" and executed == expected,
        "reasons": [str(reason)[:4096] for reason in reasons],
        "elapsed_seconds": time.time() - start,
        "retry": "none; a new explicit local attempt is required after interruption",
        "claim_tier": "experimental_unscored",
    }


# ---------------------------------------------------------------------------
# Actual world-model runtime.  It is used inside the selected GPU worker only.
# ---------------------------------------------------------------------------


def selected_world_backend() -> Literal["cosmos", "irasim"]:
    value = os.environ.get("KOSMOS_WORLD_BACKEND", "cosmos").strip().lower()
    if value not in ("cosmos", "irasim"):
        raise DemoRuntimeBlocked("KOSMOS_WORLD_BACKEND must be cosmos or irasim")
    return value  # type: ignore[return-value]


SELECTED_WORLD_BACKEND = selected_world_backend()


def required_asset_contracts() -> Dict[str, Any]:
    """Emit required records; it does not claim that any asset is available."""

    return {
        "schema_version": "kosmos-demo-assets-v1",
        "manifest_environment": "KOSMOS_ASSET_MANIFEST_PATH",
        "selected_world_backend": SELECTED_WORLD_BACKEND,
        "assets": {
            "OpenVLA": {
                "asset_id": "openvla-7b",
                "model_id": OPENVLA_MODEL_ID,
                "revision": OPENVLA_REVISION,
                "delivery": "chains.Assets cached ModelRepo volume",
                "required_fields": ["path", "sha256", "verified", "reviewed_remote_code_revision"],
            },
            "MiniVLA": {
                "asset_ids": ["minivla-vq-bridge", "minivla-pretrain-vq"],
                "delivery": "operator-provisioned Baseten-native asset/volume only after license acknowledgement and safe conversion",
                "required_fields": ["path", "sha256", "verified", "license_acknowledged", "safe_conversion_report_sha256"],
            },
            "Octo-Small": {
                "asset_id": "octo-small",
                "required_relative_path": "270000/default/checkpoint",
                "delivery": "operator-provisioned Baseten-native asset/volume only after immutable checkpoint provenance is recorded",
                "required_fields": ["path", "sha256", "verified", "checkpoint_revision", "normalizer_sha256"],
            },
            "Cosmos": {
                "asset_id": "cosmos3-nano",
                "model_id": COSMOS_MODEL_ID,
                "revision": COSMOS_REVISION,
                "delivery": "chains.Assets cached ModelRepo volume",
                "required_fields": ["path", "sha256", "verified"],
            },
            "IRASim": {
                "asset_ids": ["world.irasim_bridge", "irasim-bridge-safetensors", "irasim-vae", "irasim-scheduler"],
                "source_revision": "c72b6dade6fcd65971e0aa8ab49ea39b15108c90",
                "delivery": "operator-provisioned Baseten-native asset/volume; safetensors only",
                "required_fields": ["path", "sha256", "verified", "license_acknowledged"],
            },
        },
        "prohibited": ["R2", "Cloudflare", "S3", "remote callback receiver", "unsafe torch.load without weights_only"],
    }


_CONTENT_VALIDATION_CACHE: Dict[Tuple[str, str], bool] = {}


def _normalised_sha(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.lower().strip()
    if candidate.startswith("sha256:"):
        candidate = candidate[len("sha256:"):]
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        return None
    return "sha256:" + candidate


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _content_lock(asset: Mapping[str, Any]) -> Tuple[Mapping[str, Any], str]:
    """Read a packaged content lock; a boolean asset flag is never enough."""

    lock_path = asset.get("content_lock_path")
    expected_hash = _normalised_sha(asset.get("content_lock_sha256"))
    if not isinstance(lock_path, str) or not lock_path or expected_hash is None:
        raise DemoRuntimeBlocked("asset manifest requires content_lock_path and content_lock_sha256")
    path = pathlib.Path(lock_path)
    if not path.is_file():
        raise DemoRuntimeBlocked("packaged asset content lock is not mounted")
    actual_hash = _file_sha256(path)
    if actual_hash != expected_hash:
        raise DemoRuntimeBlocked("packaged asset content lock hash does not match the manifest")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DemoRuntimeBlocked("packaged asset content lock is not JSON") from error
    if not isinstance(lock, Mapping):
        raise DemoRuntimeBlocked("packaged asset content lock is not an object")
    return lock, actual_hash


def _verify_asset_tree(asset: Mapping[str, Any]) -> None:
    """Hash every lock-listed file under the mounted root once per lock digest."""

    root_value = asset.get("path")
    if not isinstance(root_value, str):
        raise DemoRuntimeBlocked("asset root path is missing")
    root = pathlib.Path(root_value)
    lock, lock_hash = _content_lock(asset)
    cache_key = (str(root.resolve()), lock_hash)
    if _CONTENT_VALIDATION_CACHE.get(cache_key):
        return
    records = lock.get("file_records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise DemoRuntimeBlocked("packaged asset content lock has no file_records")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DemoRuntimeBlocked("content-lock file record %d is invalid" % index)
        relative = record.get("path")
        expected = _normalised_sha(record.get("sha256"))
        byte_count = record.get("bytes")
        if not isinstance(relative, str) or not relative or pathlib.PurePosixPath(relative).is_absolute() or ".." in pathlib.PurePosixPath(relative).parts:
            raise DemoRuntimeBlocked("content-lock file record %d has an unsafe path" % index)
        if expected is None or not isinstance(byte_count, int) or byte_count < 0:
            raise DemoRuntimeBlocked("content-lock file record %d lacks exact digest/size" % index)
        candidate = root.joinpath(*pathlib.PurePosixPath(relative).parts)
        try:
            candidate.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise DemoRuntimeBlocked("content-lock file path escapes its asset root") from error
        if candidate.is_symlink():
            raise DemoRuntimeBlocked("mounted asset content lock does not permit symlinked runtime files")
        if not candidate.is_file() or candidate.stat().st_size != byte_count:
            raise DemoRuntimeBlocked("mounted asset differs from content lock at %s" % relative)
        if _file_sha256(candidate) != expected:
            raise DemoRuntimeBlocked("mounted asset digest differs from content lock at %s" % relative)
    _CONTENT_VALIDATION_CACHE[cache_key] = True


def _load_asset_manifest(expected_id: str) -> Mapping[str, Any]:
    location = os.environ.get("KOSMOS_ASSET_MANIFEST_PATH", "").strip()
    generated_at_runtime = False
    if location:
        path = pathlib.Path(location)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DemoRuntimeBlocked("asset manifest cannot be read as JSON") from error
        if not isinstance(raw, Mapping) or raw.get("schema_version") != "kosmos-demo-assets-v1":
            raise DemoRuntimeBlocked("asset manifest is not a kosmos-demo-assets-v1 manifest")
        assets = raw.get("assets")
        asset = assets.get(expected_id) if isinstance(assets, Mapping) else None
    elif expected_id == "cosmos3-nano":
        # The public publisher cache has no account secret. Before constructing
        # the actual adapter, mint an in-memory asset record by rehashing the
        # packaged 36-runtime-file lock against the mounted ModelRepo root.
        # This is not a pre-set ``verified`` flag and writes no remote output.
        try:
            from prepare_demo_assets import (
                create_verified_demo_asset_manifest,
                load_cosmos_content_lock,
            )
            packaged_lock = pathlib.Path(__file__).resolve().parent / "assets" / "cosmos3-nano-runtime-lock.json"
            lock, lock_sha256 = load_cosmos_content_lock(packaged_lock)
            raw = create_verified_demo_asset_manifest(
                lock,
                model_root=pathlib.Path(MODEL_CACHE_ROOT) / "cosmos3-nano",
                content_lock_path=packaged_lock,
                content_lock_sha256=lock_sha256,
            )
            assets = raw.get("assets") if isinstance(raw, Mapping) else None
            asset = assets.get(expected_id) if isinstance(assets, Mapping) else None
            generated_at_runtime = True
        except Exception as error:  # noqa: BLE001 - preparation is a hard asset gate.
            raise DemoRuntimeBlocked("public Cosmos runtime asset preflight failed: %s" % error) from error
    else:
        raise DemoRuntimeBlocked("KOSMOS_ASSET_MANIFEST_PATH is required before %s execution" % expected_id)
    if not isinstance(asset, Mapping):
        raise DemoRuntimeBlocked("asset manifest lacks %s" % expected_id)
    if asset.get("verified") is not True or _normalised_sha(asset.get("sha256")) is None:
        raise DemoRuntimeBlocked("asset manifest has not verified %s" % expected_id)
    path_value = asset.get("path")
    if not isinstance(path_value, str) or not pathlib.Path(path_value).exists():
        raise DemoRuntimeBlocked("verified asset path for %s is not mounted in this worker" % expected_id)
    # ``create_verified_demo_asset_manifest`` just rehashed every record using
    # the same packaged lock in this process. A second full 35-GB SHA pass
    # would only double the warm-up cost. External manifests still undergo the
    # independent in-worker rehash below; their boolean field is never trusted.
    if not generated_at_runtime:
        _verify_asset_tree(asset)
    return asset


class DemoWorldRuntime:
    """Lazy adapter host for exactly one build-selected world backend."""

    def __init__(self, backend: Literal["cosmos", "irasim"] = SELECTED_WORLD_BACKEND) -> None:
        self.backend = backend
        self._adapter: Any = None
        self._profile: Any = None

    def readiness(self) -> Dict[str, Any]:
        try:
            if self.backend == "cosmos":
                asset = _load_asset_manifest("cosmos3-nano")
                return {"status": "ready_unqualified", "asset_id": "cosmos3-nano", "path": asset.get("path"), "backend": self.backend}
            asset = _load_asset_manifest("world.irasim_bridge")
            for name in ("irasim-bridge-safetensors", "irasim-vae", "irasim-scheduler"):
                _load_asset_manifest(name)
            return {"status": "ready_unqualified", "asset_id": "world.irasim_bridge", "path": asset.get("path"), "backend": self.backend}
        except DemoRuntimeBlocked as error:
            return {"status": "blocked", "backend": self.backend, "reasons": [str(error)]}

    def _cosmos(self, request: WorldTurnRequest) -> Any:
        if self._adapter is not None:
            return self._adapter
        try:
            from plumb.adapters.worlds import Cosmos3NanoDiffusersAdapter, Cosmos3NanoDiffusersProfile
        except ImportError as error:  # pragma: no cover
            raise DemoRuntimeBlocked("Cosmos adapter package is not staged") from error
        asset = _load_asset_manifest("cosmos3-nano")
        self._profile = Cosmos3NanoDiffusersProfile(
            profile_id=request.world.profile_id,
            local_model_path=str(asset["path"]),
            model_id=COSMOS_MODEL_ID,
            model_revision=COSMOS_REVISION,
            resolution_tier=int(request.world.resolution_tier or 256),
            num_inference_steps=int(request.world.inference_steps or 30),
            normalizer_revision=None,
        )
        self._adapter = Cosmos3NanoDiffusersAdapter(self._profile)
        return self._adapter

    def _irasim(self, request: WorldTurnRequest) -> Any:
        if self._adapter is not None:
            return self._adapter
        try:
            from plumb.adapters.irasim_runtime import IRASimOneStepAdapter, IRASimOneStepProfile
        except ImportError as error:  # pragma: no cover
            raise DemoRuntimeBlocked("IRASim adapter package is not staged") from error
        runtime = _load_asset_manifest("world.irasim_bridge")
        weights = _load_asset_manifest("irasim-bridge-safetensors")
        vae = _load_asset_manifest("irasim-vae")
        scheduler = _load_asset_manifest("irasim-scheduler")
        checkpoint = pathlib.Path(str(weights["path"]))
        if checkpoint.suffix != ".safetensors":
            raise DemoRuntimeBlocked("IRASim demo accepts only an isolated safe-converted .safetensors checkpoint")
        self._profile = IRASimOneStepProfile(
            profile_id=request.world.profile_id,
            repo_path=str(runtime["path"]),
            checkpoint_path=str(checkpoint),
            vae_model_path=str(vae["path"]),
            scheduler_path=str(scheduler["path"]),
            config_path=str(pathlib.Path(str(runtime["path"])) / "configs/evaluation/bridge/frame_ada.yaml"),
            inference_steps=int(request.world.inference_steps or 50),
        )
        self._adapter = IRASimOneStepAdapter(self._profile)
        return self._adapter

    def generate(self, request: WorldTurnRequest) -> WorldTurnResult:
        if request.world.backend != self.backend:
            return WorldTurnResult(status="blocked", expected_post_conditioning_frames=0, reasons=["this deployment selected %s; it cannot silently substitute %s" % (self.backend, request.world.backend)])
        readiness = self.readiness()
        if readiness["status"] != "ready_unqualified":
            return WorldTurnResult(status="blocked", expected_post_conditioning_frames=0, reasons=list(readiness.get("reasons", [])))
        started = time.perf_counter()
        try:
            condition = request.condition.rgb_array()
            if self.backend == "cosmos":
                return self._generate_cosmos(request, condition, started)
            return self._generate_irasim(request, condition, started)
        except DemoRuntimeBlocked as error:
            return WorldTurnResult(status="blocked", expected_post_conditioning_frames=0, reasons=[str(error)])
        except Exception as error:  # noqa: BLE001 - never supply a fake image.
            return WorldTurnResult(status="failed", expected_post_conditioning_frames=0, reasons=["world_exception:%s:%s" % (type(error).__name__, error)])

    def _generate_cosmos(self, request: WorldTurnRequest, condition: Any, started: float) -> WorldTurnResult:
        from plumb.adapters.contracts import FeedbackMode, WorldRequest

        adapter = self._cosmos(request)
        result = adapter.generate(
            WorldRequest(
                conditioning_image=condition,
                prompt=request.task,
                domain=request.world.domain,
                compiled_actions=tuple(tuple(float(item) for item in row) for row in request.compiled_actions),
                nominal_control_timestamps=tuple(float(item) for item in request.nominal_control_timestamps),
                seed=int(request.world_seed),
                compatibility_profile_id=request.world.profile_id,
                feedback_mode=FeedbackMode.FORECAST_STATE,
                source_state_lineage_id=None,
                request_id=request.request_id,
            )
        )
        expected = len(request.native_actions)
        if not result.conditioning_frame_included or len(result.frames) != expected + 1:
            raise DemoContractError("Cosmos must return one structural condition frame plus one frame per action")
        frames = [PngFrame.from_array(frame) for frame in result.frames[1:]]
        return WorldTurnResult(
            status="completed",
            frames=frames,
            expected_post_conditioning_frames=expected,
            conditioning_frame_removed=True,
            identifiers={"backend": result.backend, "profile_id": result.profile_id, "request_id": result.request_id, "model_id": COSMOS_MODEL_ID, "model_revision": COSMOS_REVISION, "adapter_metadata": _json_mapping(result.metadata)},
            timing={"adapter": _json_mapping(result.timing), "controller_wall_seconds": time.perf_counter() - started},
        )

    def _generate_irasim(self, request: WorldTurnRequest, condition: Any, started: float) -> WorldTurnResult:
        adapter = self._irasim(request)
        if request.operation == "policy_step":
            result = adapter.generate_one_step(condition, request.native_actions[0], seed=int(request.world_seed))
            frames = tuple(result.frames)
            scaled = list(result.native_action_scaled)
            condition_detail = dict(result.condition_preprocessing)
        else:
            # This is the original 16-frame architecture with its released
            # 15-action shape.  It is implemented by the adapter rather than
            # looping 15 one-step calls, which would be a different operation.
            result = adapter.generate_manual_chunk(condition, request.native_actions, seed=int(request.world_seed))
            frames = tuple(result.frames)
            scaled = [list(row) for row in result.native_actions_scaled]
            condition_detail = dict(result.condition_preprocessing)
        expected = len(request.native_actions)
        if len(frames) != expected + 1:
            raise DemoContractError("IRASim must return the condition structure plus one transition per native action")
        encoded = [PngFrame.from_array(frame) for frame in frames[1:]]
        return WorldTurnResult(
            status="completed",
            frames=encoded,
            expected_post_conditioning_frames=expected,
            conditioning_frame_removed=True,
            identifiers={"backend": "irasim_original", "profile_id": request.world.profile_id, "request_id": request.request_id, "source_revision": "c72b6dade6fcd65971e0aa8ab49ea39b15108c90", "native_action_scaled": scaled, "condition_preprocessing": condition_detail},
            timing={"adapter": _json_mapping(result.timing), "controller_wall_seconds": time.perf_counter() - started},
        )


def _policy_adapter_factory(policy_id: str) -> Any:
    """Build a fresh source wrapper from one verified manifest profile.

    The manifest is an operator-produced immutable binding, not request data.
    It must carry every profile parameter that has not been safely fixed in
    source code.  This function never fills an absent normalizer, source
    checkout, VQ licence acknowledgement, or checkpoint revision.
    """

    asset_id = {"OpenVLA": "openvla-7b", "MiniVLA": "minivla-vq-bridge", "Octo-Small": "octo-small"}.get(policy_id)
    if asset_id is None:
        raise DemoRuntimeBlocked("unknown policy adapter requested")
    asset = _load_asset_manifest(asset_id)
    profile_values = asset.get("policy_profile")
    if not isinstance(profile_values, Mapping):
        raise DemoRuntimeBlocked("verified %s asset lacks its immutable policy_profile binding" % asset_id)
    profile = dict(profile_values)
    # Bind the displayed profile to the exact verified asset record rather than
    # accepting a path carried by a client request.
    manifest_location = os.environ.get("KOSMOS_ASSET_MANIFEST_PATH", "").strip()
    profile.setdefault("asset_manifest_id", _sha256(pathlib.Path(manifest_location).read_bytes()) if manifest_location else None)
    profile.setdefault("asset_manifest_sha256", _sha256(pathlib.Path(manifest_location).read_bytes()) if manifest_location else None)
    if policy_id == "OpenVLA":
        try:
            from plumb.policies.openvla import OpenVLAPolicyAdapter, OpenVLAPolicyProfile
        except ImportError as error:  # pragma: no cover
            raise DemoRuntimeBlocked("OpenVLA policy adapter is not staged") from error
        # These fields are fixed source facts, not configuration knobs.  The
        # remote-code acknowledgement remains explicit and must be in the
        # verified record; absent acknowledgement blocks profile validation.
        profile.update(
            {
                "local_model_path": str(asset["path"]),
                "checkpoint_revision": OPENVLA_REVISION,
                "transformers_version": "4.40.1",
                "unnorm_key": "bridge_orig",
                "local_files_only": True,
            }
        )
        try:
            frozen = OpenVLAPolicyProfile(**profile)
        except Exception as error:  # noqa: BLE001
            raise DemoRuntimeBlocked("OpenVLA immutable profile is invalid: %s" % error) from error
        return lambda: OpenVLAPolicyAdapter(frozen)
    if policy_id == "MiniVLA":
        # VQ is a second explicit asset because its publisher terms are an
        # independent gate. The profile class verifies both source checkouts,
        # conversion reports, artifact hashes, and terms before loading.
        _load_asset_manifest("minivla-pretrain-vq")
        try:
            from plumb.policies.diagnostics.minivla import MiniVLAPolicyAdapter, MiniVLAPolicyProfile
        except ImportError as error:  # pragma: no cover
            raise DemoRuntimeBlocked("MiniVLA policy adapter is not staged") from error
        try:
            frozen = MiniVLAPolicyProfile(**profile)
        except Exception as error:  # noqa: BLE001
            raise DemoRuntimeBlocked("MiniVLA immutable profile is invalid: %s" % error) from error
        return lambda: MiniVLAPolicyAdapter(frozen)
    try:
        from plumb.policies.diagnostics.octo_v0_1 import OctoSmallV1Policy, OctoV1PolicyProfile
    except ImportError as error:  # pragma: no cover
        raise DemoRuntimeBlocked("Octo policy adapter is not staged") from error
    try:
        frozen = OctoV1PolicyProfile(**profile)
    except Exception as error:  # noqa: BLE001
        raise DemoRuntimeBlocked("Octo immutable profile is invalid: %s" % error) from error
    return lambda: OctoSmallV1Policy(frozen)


class DemoPolicyRuntime:
    """Thin isolated-image bridge to the source-reviewed policy router.

    ``experimental.py`` owns model construction, normalizers, action execution,
    Octo ensemble state, and snapshots.  This module never selects a raw row.
    Keeping the bridge this small lets each worker have its incompatible native
    Torch/JAX environment without leaking a model into the controller image.
    """

    def __init__(self, policy_id: str, adapter_factory: Optional[Any] = None) -> None:
        self.policy_id = policy_id
        self._adapter_factory = adapter_factory
        self._router: Any = None

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        try:
            from plumb.policies.experimental import make_experimental_policy_router
        except ImportError as error:
            raise DemoRuntimeBlocked("plumb.policies.experimental policy router is not staged") from error
        if not callable(self._adapter_factory):
            raise DemoRuntimeBlocked(
                "%s has no immutable, source-reviewed adapter factory bound to this deployment's asset manifest"
                % self.policy_id
            )
        self._router = make_experimental_policy_router(self.policy_id, self._adapter_factory)
        return self._router

    def generate(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        if request.policy_id != self.policy_id:
            return PolicyTurnResult(status="failed", reasons=["%s worker received %s" % (self.policy_id, request.policy_id)])
        try:
            from plumb.policies.experimental import ExperimentalPolicyTurnRequest

            router = self._get_router()
            image = request.observation.rgb_array()
            identifiers = {
                "state_kind": request.state_kind,
                "state_lineage_id": request.state_lineage_id,
                "cell_id": request.cell_id,
                "policy_revision": request.policy_revision,
                "task_id": "close-drawer",
                "observation_id": "%s:%s:%d" % (request.attempt_id, request.cell_id, request.action_index),
                "proprio_lineage_id": request.proprio_lineage_id,
                "proprio_converter_revision": request.proprio_converter_revision,
                "timestamp": float(request.action_index) / DEFAULT_CONTROL_HZ,
                "source_bundle_sha256": request.source_bundle_sha256,
            }
            external_request = ExperimentalPolicyTurnRequest(
                policy_id=request.policy_id,
                prompt=request.task,
                image_history=(image,),
                proprio=tuple(float(value) for value in request.proprio_state),
                policy_seed=request.policy_seed,
                prior_snapshot=request.prior_policy_snapshot,
                identifiers=identifiers,
            )
            # One deliberate source call only. Router methods are synchronous
            # inside their isolated native runtime; an awaitable would be an
            # unsupported execution mode, not an excuse to retry.
            value = router.predict(external_request)
            if inspect.isawaitable(value):
                raise DemoRuntimeBlocked("policy router must complete inside its isolated synchronous GPU worker")
            result = PolicyTurnResult(
                status=str(value.status),
                raw_proposal_rows=[list(row) for row in value.raw_proposal_rows],
                source_executed_rows=[list(row) for row in value.source_executed_rows],
                policy_snapshot=None if value.snapshot is None else dict(value.snapshot),
                identifiers=dict(value.identifiers),
                timing=dict(value.timing),
                reasons=list(value.reasons),
            )
            if result.status == "completed":
                # The router must name the actual source-executed rows.  In
                # particular MiniVLA emits ret_action[:,0], and Octo emits the
                # temporal-ensemble action, not proposal row zero.
                if len(result.source_executed_rows) != 1:
                    return PolicyTurnResult(status="failed", reasons=["router did not return exactly one source-executed action"])
                _finite_row(result.source_executed_rows[0], "source_executed_rows[0]")
            return result
        except DemoRuntimeBlocked as error:
            return PolicyTurnResult(status="blocked", reasons=[str(error)])
        except Exception as error:  # noqa: BLE001
            return PolicyTurnResult(status="failed", reasons=["policy_exception:%s:%s" % (type(error).__name__, error)])


# ---------------------------------------------------------------------------
# Chainlet wiring.  Every internal dependency has retries=0: the local API
# creates an explicit new attempt if it elects to retry a failed/incomplete cell.
# ---------------------------------------------------------------------------


if CHAINS_RUNTIME_AVAILABLE:
    _CONTROLLER_REQUIREMENTS = chains.make_abs_path_here("requirements/demo-controller.txt")
    _COSMOS_REQUIREMENTS = chains.make_abs_path_here("requirements/world-cosmos.txt")
    _IRASIM_REQUIREMENTS = chains.make_abs_path_here("requirements/demo-world-irasim.txt")
    _OPENVLA_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-openvla.txt")
    _MINIVLA_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-minivla.txt")
    _OCTO_REQUIREMENTS = chains.make_abs_path_here("requirements/policy-octo.txt")
    _EXTERNAL_PACKAGES = chains.make_abs_path_here("../_chain_packages")
    _REQUIREMENTS_FIELD = "requirements_file" if "requirements_file" in chains.DockerImage.model_fields else "pip_requirements_file"

    def _image(base: Any, requirements: Any) -> Any:
        return chains.DockerImage(base_image=base, external_package_dirs=[_EXTERNAL_PACKAGES], **{_REQUIREMENTS_FIELD: requirements})

    _COMMON_ENV = {
        "KOSMOS_ASSET_MANIFEST_PATH": os.environ.get("KOSMOS_ASSET_MANIFEST_PATH", ""),
        "KOSMOS_WORLD_BACKEND": SELECTED_WORLD_BACKEND,
        "PLUMB_MODEL_CACHE_ROOT": MODEL_CACHE_ROOT,
    }
    # Cosmos3-Nano and OpenVLA are public pinned publisher repositories.  Do
    # not require an account secret just to exercise the public cache path.
    # Gated/private assets must arrive through a separately verified native
    # asset manifest; they are never smuggled through this public model cache.
    _OPENVLA_CACHE = truss_config.ModelRepo(
        repo_id=OPENVLA_MODEL_ID,
        revision=OPENVLA_REVISION,
        use_volume=True,
        volume_folder="openvla-7b",
    )
    _COSMOS_CACHE = truss_config.ModelRepo(
        repo_id=COSMOS_MODEL_ID,
        revision=COSMOS_REVISION,
        use_volume=True,
        volume_folder="cosmos3-nano",
    )

    class DemoOpenVLAWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        remote_config = chains.RemoteConfig(
            name="kosmos-demo-openvla",
            docker_image=_image(chains.BasetenImage.PY311, _OPENVLA_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=1),
            assets=chains.Assets(cached=[_OPENVLA_CACHE]),
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"policy": "OpenVLA", "retries": 0, "claim_tier": "experimental_unscored"}),
        )

        def __init__(self) -> None:
            self._runtime = DemoPolicyRuntime("OpenVLA", lambda: _policy_adapter_factory("OpenVLA")())

        async def run_remote(self, request: PolicyTurnRequest) -> PolicyTurnResult:
            return self._runtime.generate(request)

    class DemoMiniVLAWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        remote_config = chains.RemoteConfig(
            name="kosmos-demo-minivla",
            docker_image=_image(chains.BasetenImage.PY310, _MINIVLA_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=1),
            # MiniVLA assets are intentionally not invented as a ModelRepo:
            # provenance + VQ licence acknowledgement must be supplied through
            # the native asset manifest/mount before the router can run.
            assets=chains.Assets(),
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"policy": "MiniVLA", "retries": 0, "claim_tier": "experimental_unscored", "asset_gate": "license+safe_conversion"}),
        )

        def __init__(self) -> None:
            self._runtime = DemoPolicyRuntime("MiniVLA", lambda: _policy_adapter_factory("MiniVLA")())

        async def run_remote(self, request: PolicyTurnRequest) -> PolicyTurnResult:
            return self._runtime.generate(request)

    class DemoOctoWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        remote_config = chains.RemoteConfig(
            name="kosmos-demo-octo-small",
            docker_image=_image(chains.BasetenImage.PY310, _OCTO_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=4, memory="32Gi", predict_concurrency=1),
            assets=chains.Assets(),
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"policy": "Octo-Small", "retries": 0, "claim_tier": "experimental_unscored", "asset_gate": "checkpoint+normalizer"}),
        )

        def __init__(self) -> None:
            self._runtime = DemoPolicyRuntime("Octo-Small", lambda: _policy_adapter_factory("Octo-Small")())

        async def run_remote(self, request: PolicyTurnRequest) -> PolicyTurnResult:
            return self._runtime.generate(request)

    _SELECTED_WORLD_REQUIREMENTS = _COSMOS_REQUIREMENTS if SELECTED_WORLD_BACKEND == "cosmos" else _IRASIM_REQUIREMENTS
    _SELECTED_WORLD_ASSETS = chains.Assets(cached=[_COSMOS_CACHE]) if SELECTED_WORLD_BACKEND == "cosmos" else chains.Assets()

    class DemoWorldWorker(chains.ChainletBase):  # type: ignore[union-attr,misc]
        remote_config = chains.RemoteConfig(
            name="kosmos-demo-world-" + SELECTED_WORLD_BACKEND,
            docker_image=_image(chains.BasetenImage.PY311, _SELECTED_WORLD_REQUIREMENTS),
            compute=chains.Compute(gpu="H100", gpu_count=1, cpu_count=8, memory="64Gi", predict_concurrency=1),
            assets=_SELECTED_WORLD_ASSETS,
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"world_backend": SELECTED_WORLD_BACKEND, "retries": 0, "claim_tier": "experimental_unscored", "manual_shapes": {"cosmos": [16, 17], "irasim": [15, 16]}}),
        )

        def __init__(self) -> None:
            self._runtime = DemoWorldRuntime()

        async def run_remote(self, request: WorldTurnRequest) -> WorldTurnResult:
            return self._runtime.generate(request)

    class DemoWorldProbeController(chains.ChainletBase):  # type: ignore[union-attr,misc]
        """Explicit-entrypoint world smoke that deploys no policy workers.

        Push this class by name (``... demo_chain.py DemoWorldProbeController``)
        after the CPU stream canary.  It shares the exact manual compiler and
        world worker with the wall, but its dependency graph contains only the
        selected world worker.  It is deliberately not decorated as the default
        entrypoint; the full controller remains the normal named Chain.
        """

        remote_config = chains.RemoteConfig(
            name="kosmos-demo-world-probe-controller",
            docker_image=_image(chains.BasetenImage.PY311, _CONTROLLER_REQUIREMENTS),
            compute=chains.Compute(cpu_count=2, memory="4Gi", predict_concurrency=1),
            assets=chains.Assets(),
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"entrypoint": "explicit_world_probe", "streaming": "low_level_ndjson", "retries": 0, "policy_workers": 0, "no_remote_result_store": True}),
        )

        def __init__(self, world: DemoWorldWorker = chains.depends(DemoWorldWorker, retries=0)) -> None:
            self._core = DemoControllerCore({}, world)

        async def run_remote(self, request: ServiceDispatchRequest) -> AsyncIterator[str]:
            translated = service_dispatch_to_stream_request(request)
            if translated.manual is None:
                raise DemoContractError("DemoWorldProbeController accepts only manual_world_segment dispatches")
            async for record in self._core.manual_events(translated.manual):
                yield record

    @chains.mark_entrypoint("KOSMOS experimental streaming comparison")
    class DemoRolloutController(chains.ChainletBase):  # type: ignore[union-attr,misc]
        remote_config = chains.RemoteConfig(
            name="kosmos-demo-controller",
            docker_image=_image(chains.BasetenImage.PY311, _CONTROLLER_REQUIREMENTS),
            compute=chains.Compute(cpu_count=4, memory="8Gi", predict_concurrency=2),
            assets=chains.Assets(),
            options=chains.ChainletOptions(env_variables=_COMMON_ENV, metadata={"entrypoint": True, "streaming": "low_level_ndjson", "retries": 0, "no_remote_result_store": True}),
        )

        def __init__(
            self,
            openvla: DemoOpenVLAWorker = chains.depends(DemoOpenVLAWorker, retries=0),
            minivla: DemoMiniVLAWorker = chains.depends(DemoMiniVLAWorker, retries=0),
            octo: DemoOctoWorker = chains.depends(DemoOctoWorker, retries=0),
            world: DemoWorldWorker = chains.depends(DemoWorldWorker, retries=0),
        ) -> None:
            self._core = DemoControllerCore({"OpenVLA": openvla, "MiniVLA": minivla, "Octo-Small": octo}, world)

        async def run_remote(self, request: ServiceDispatchRequest) -> AsyncIterator[str]:
            translated = service_dispatch_to_stream_request(request)
            if translated.rollout is not None:
                async for record in self._core.rollout_events(translated.rollout):
                    yield record
                return
            if translated.manual is not None:
                async for record in self._core.manual_events(translated.manual):
                    yield record
                return
            # Pydantic validation makes this unreachable. It remains a valid
            # stream terminal in case a future SDK deserialiser bypasses it.
            yield _event("terminal", _terminal_payload("unknown", "unknown", "failed", 0, 0, ["missing operation"], time.time()))

else:
    DemoOpenVLAWorker = None
    DemoMiniVLAWorker = None
    DemoOctoWorker = None
    DemoWorldWorker = None
    DemoWorldProbeController = None
    DemoRolloutController = None
