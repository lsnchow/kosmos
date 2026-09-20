#!/usr/bin/env python3
"""Authenticated one-tick OpenVLA + original-IRASim live-demo worker.

This is intentionally a small, state-*less* HTTP worker.  The presentation
client owns the authoritative RGB PNG returned from each request and submits
it on the next request.  Consequently a worker restart never pretends to
continue a latent state, and the service cannot silently reuse a policy action.

It is an experimental visual steering probe only: each IRASim call re-encodes
the supplied RGB image.  It is neither an upstream latent-carry protocol nor
evidence of physical-state fidelity or a qualified evaluation result.

The worker deliberately has no web-framework dependency.  Start it only in
the allocated, pinned runtime via ``live_demo_server.sbatch``; the startup
path verifies local pins and warms both models before opening its port.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import cluster.closed_loop_smoke as closed_loop


DEFAULT_ROOT = closed_loop.DEFAULT_ROOT
DEFAULT_PORT = 8917
DEFAULT_HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_PNG_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
MAX_PROMPT_CHARS = 2_000
PROFILE_ID = "openvla-irasim-live-demo-experimental"
STATE_MODE = "experimental_reencoded_rgb_stateless"
EXPERIMENTAL_WARNING = (
    "Experimental, unqualified RGB re-encoding per tick. This is not native latent carry, "
    "a physics/state-fidelity claim, or a qualified evaluation."
)
MANUAL_STEP = 0.015
MANUAL_DIRECTIONS = ("up", "down", "left", "right", "forward", "back")
_DIRECTION_ACTIONS = {
    "right": (0, MANUAL_STEP),
    "left": (0, -MANUAL_STEP),
    "forward": (1, MANUAL_STEP),
    "back": (1, -MANUAL_STEP),
    "up": (2, MANUAL_STEP),
    "down": (2, -MANUAL_STEP),
}


class LiveDemoInputError(ValueError):
    """A request has no safe, well-defined one-step interpretation."""


@dataclass(frozen=True)
class StepRequest:
    prompt: str
    mode: str
    direction: Optional[str]
    seed: int
    png_base64: Optional[str]


@dataclass(frozen=True)
class LiveDemoConfig:
    loop: closed_loop.ClosedLoopConfig
    token_file: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_request_bytes: int = MAX_REQUEST_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise LiveDemoInputError("port must be an integer from 1 through 65535")
        if not isinstance(self.host, str) or not self.host.strip():
            raise LiveDemoInputError("host must be a nonempty address")
        if not isinstance(self.max_request_bytes, int) or self.max_request_bytes < 1024:
            raise LiveDemoInputError("max request bytes must be an integer of at least 1024")


def _jsonable(value: Any) -> Any:
    """Use the smoke runner's conservative JSON conversion in HTTP payloads."""

    return closed_loop._jsonable(value)


def _timing_ms(seconds: Optional[float]) -> Optional[float]:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 3)


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read_token(path: Path) -> str:
    """Read one nonempty bearer token without ever printing its contents."""

    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise LiveDemoInputError("live-demo token file is unreadable: %s" % path) from error
    if len(token) < 24:
        raise LiveDemoInputError("live-demo bearer token must contain at least 24 characters")
    if any(character.isspace() for character in token):
        raise LiveDemoInputError("live-demo bearer token must be a single non-whitespace line")
    return token


def _verify_irasim_source_revision(repo: Path) -> Dict[str, str]:
    """Require the exact checked-out original IRASim source used by the adapter."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LiveDemoInputError("could not inspect local IRASim source revision") from error
    observed = result.stdout.strip().lower()
    if result.returncode != 0 or observed != closed_loop.IRASIM_SOURCE_REVISION:
        raise LiveDemoInputError(
            "IRASim source revision must equal pinned %s" % closed_loop.IRASIM_SOURCE_REVISION
        )
    return {"expected": closed_loop.IRASIM_SOURCE_REVISION, "observed": observed}


def verify_runtime_inputs(config: closed_loop.ClosedLoopConfig) -> Dict[str, Any]:
    """Run the same strict asset checks as the closed-loop smoke plus git HEAD."""

    provenance = dict(closed_loop._verify_local_inputs(config))
    provenance["irasim_source_checkout"] = _verify_irasim_source_revision(config.irasim_repo)
    return provenance


def _warm_real_adapters(policy: Any, world: Any) -> Dict[str, Optional[float]]:
    """Load both pinned model stacks before accepting HTTP requests.

    This intentionally calls component loaders only -- no ``predict_action``
    and no IRASim generation occur at warm-up.  Thus ``/health`` cannot be an
    accidental inference endpoint after service readiness.
    """

    started = time.perf_counter()
    policy_adapter = getattr(policy, "adapter", policy)
    runtime_loader = getattr(policy_adapter, "_load_runtime", None)
    component_loader = getattr(policy_adapter, "_ensure_components", None)
    if not callable(runtime_loader) or not callable(component_loader):
        raise RuntimeError("live demo requires the pinned OpenVLA adapter's explicit component loader")
    runtime = runtime_loader()
    component_loader(runtime)
    policy_seconds = time.perf_counter() - started

    world_started = time.perf_counter()
    world_loader = getattr(world, "_load", None)
    if not callable(world_loader):
        raise RuntimeError("live demo requires the original IRASim one-step adapter loader")
    torch_module, _pipeline = world_loader()
    world_seconds = time.perf_counter() - world_started
    gpu_name: Optional[str] = None
    gpu_memory_bytes: Optional[float] = None
    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            cuda.synchronize()
            gpu_name = str(cuda.get_device_name(0))
            properties = cuda.get_device_properties(0)
            gpu_memory_bytes = float(getattr(properties, "total_memory", 0)) or None
    except (AttributeError, RuntimeError):
        # The model loaders themselves have already decided whether their
        # selected device is valid. Missing telemetry stays explicit, not zero.
        gpu_name = None
        gpu_memory_bytes = None
    return {
        "openvla_load_seconds": policy_seconds,
        "irasim_load_seconds": world_seconds,
        "total_load_seconds": time.perf_counter() - started,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
    }


def _decode_png_base64(value: str) -> Any:
    """Decode one bounded, single-frame PNG into canonical uint8 HWC RGB."""

    if not isinstance(value, str) or not value:
        raise LiveDemoInputError("png_base64 must be a nonempty base64 PNG string")
    # Base64 expands binary by 4/3. This precheck avoids allocating an
    # arbitrarily large decoded string before the exact byte cap below.
    if len(value) > ((MAX_PNG_BYTES + 2) // 3) * 4:
        raise LiveDemoInputError("png_base64 exceeds the image byte limit")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise LiveDemoInputError("png_base64 must be valid base64") from error
    if not raw or len(raw) > MAX_PNG_BYTES:
        raise LiveDemoInputError("PNG exceeds the image byte limit")
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG":
                raise LiveDemoInputError("png_base64 must encode a PNG image")
            if int(getattr(image, "n_frames", 1)) != 1:
                raise LiveDemoInputError("png_base64 must encode exactly one PNG frame")
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise LiveDemoInputError("PNG dimensions exceed the image pixel limit")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except LiveDemoInputError:
        raise
    except Exception as error:
        raise LiveDemoInputError("png_base64 could not be decoded as a PNG image") from error
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise LiveDemoInputError("PNG did not decode to RGB image data")
    return rgb


def _encode_png_base64(frame: Any) -> str:
    """Serialize canonical RGB with explicit re-encoding semantics."""

    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise LiveDemoInputError("response frame must be uint8 HWC RGB")
        output = BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")
    except LiveDemoInputError:
        raise
    except Exception as error:
        raise RuntimeError("could not serialize live-demo RGB PNG") from error


def _manual_action(direction: str) -> Tuple[float, float, float, float, float, float, float]:
    if direction not in _DIRECTION_ACTIONS:
        raise LiveDemoInputError("manual direction must be one of: %s" % ", ".join(MANUAL_DIRECTIONS))
    index, signed_step = _DIRECTION_ACTIONS[direction]
    action = [0.0] * 7
    action[index] = max(-MANUAL_STEP, min(MANUAL_STEP, float(signed_step)))
    return tuple(action)  # type: ignore[return-value]


def parse_step_request(payload: Any) -> StepRequest:
    """Validate the public request without accepting arbitrary trajectories."""

    if not isinstance(payload, Mapping):
        raise LiveDemoInputError("step body must be a JSON object")
    allowed = {"png_base64", "current_png_base64", "fixture_png_base64", "image_base64", "prompt", "mode", "direction", "seed"}
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise LiveDemoInputError("unsupported step fields: %s" % ", ".join(unknown))
    image_fields = [(key, payload.get(key)) for key in ("png_base64", "current_png_base64", "fixture_png_base64", "image_base64") if payload.get(key) is not None]
    if len(image_fields) > 1:
        raise LiveDemoInputError("supply at most one current PNG field (use png_base64)")
    png_base64: Optional[str] = None
    if image_fields:
        _name, png_base64 = image_fields[0]
        if not isinstance(png_base64, str):
            raise LiveDemoInputError("png_base64 must be a base64 string")

    prompt = payload.get("prompt", closed_loop.DEFAULT_TASK)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise LiveDemoInputError("prompt must be a nonempty string no longer than %d characters" % MAX_PROMPT_CHARS)
    mode = payload.get("mode", "policy")
    if mode not in ("policy", "manual"):
        raise LiveDemoInputError("mode must be policy or manual")
    direction = payload.get("direction")
    if mode == "manual":
        if not isinstance(direction, str):
            raise LiveDemoInputError("manual mode requires direction")
        _manual_action(direction)
    elif direction is not None:
        raise LiveDemoInputError("direction is only valid when mode is manual")
    seed = payload.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed <= 2**63 - 1:
        raise LiveDemoInputError("seed must be a signed 64-bit integer")
    return StepRequest(prompt=prompt.strip(), mode=mode, direction=direction if isinstance(direction, str) else None, seed=seed, png_base64=png_base64)


class LiveDemoService:
    """A pre-warmed real-adapter service with no cross-request image state."""

    def __init__(
        self,
        *,
        policy: Any,
        world: Any,
        vendor_frame: Any,
        model_info: Mapping[str, Any],
        load_timing: Optional[Mapping[str, Any]] = None,
        png_decoder: Callable[[str], Any] = _decode_png_base64,
        png_encoder: Callable[[Any], str] = _encode_png_base64,
        frame_converter: Callable[[Any], Any] = closed_loop._irasim_frame_to_rgb,
    ) -> None:
        self._policy = policy
        self._world = world
        self._vendor_frame = vendor_frame
        self._model_info = _jsonable(model_info)
        self._load_timing = _jsonable(load_timing or {})
        self._png_decoder = png_decoder
        self._png_encoder = png_encoder
        self._frame_converter = frame_converter
        self._inference_lock = threading.Lock()
        self._vendor_frame_hash = closed_loop._frame_hash(vendor_frame)

    @property
    def vendor_frame_hash(self) -> str:
        return self._vendor_frame_hash

    def health(self) -> Dict[str, Any]:
        """Return only cached readiness metadata; never call either model."""

        return {
            "status": "ready",
            "profile": PROFILE_ID,
            "model": self._model_info.get("model"),
            "revision": self._model_info.get("revision"),
            "integrity": self._model_info.get("integrity"),
            "source_release": self._model_info.get("source_release"),
            "state_mode": STATE_MODE,
            "warning": EXPERIMENTAL_WARNING,
            "manual_directions": list(MANUAL_DIRECTIONS),
            "manual_action7_contract": "one clamped native 7-D directional action; no arbitrary trajectory",
            "policy_action_contract": "exactly one fresh native OpenVLA 7-D call per policy step; no action cache",
            "step_contract": "one IRASim generate_one_step call returns condition plus one predicted frame; condition echo is dropped",
            "fixture": {"source": "vendor_fixture_frame_0_hash_bound", "decoded_rgb_sha256": self._vendor_frame_hash},
            "load_timing": self._load_timing,
        }

    def fixture(self) -> Dict[str, Any]:
        """Return the read-only hash-bound vendor condition without inference."""

        started = time.perf_counter()
        png = self._png_encoder(self._vendor_frame)
        return {
            "prompt": closed_loop.DEFAULT_TASK,
            "fixture_png_base64": png,
            "png_base64": png,
            "source": "vendor_fixture_frame_0_hash_bound",
            "decoded_rgb_sha256": self._vendor_frame_hash,
            "state_mode": STATE_MODE,
            "warning": EXPERIMENTAL_WARNING,
            "timings_ms": {"encode": _timing_ms(time.perf_counter() - started)},
        }

    def step(self, payload: Any) -> Dict[str, Any]:
        """Execute one real native policy/manual action and one IRASim tick."""

        total_started = time.perf_counter()
        request = parse_step_request(payload)
        decode_started = time.perf_counter()
        if request.png_base64 is None:
            condition = self._vendor_frame
            input_source = "vendor_fixture_frame_0_hash_bound"
        else:
            condition = self._png_decoder(request.png_base64)
            input_source = "provided_png_base64"
        decode_seconds = time.perf_counter() - decode_started
        conditioning_hash = closed_loop._frame_hash(condition)

        policy_seconds: Optional[float] = None
        world_seconds: Optional[float] = None
        policy_details: Mapping[str, Any] = {}
        with self._inference_lock:
            if request.mode == "policy":
                policy_started = time.perf_counter()
                # Explicitly one policy invocation.  The adapter itself checks
                # one current image and one native 7-D result.
                action, policy_details = closed_loop._call_action(self._policy, condition, request.prompt, 0.0)
                policy_seconds = time.perf_counter() - policy_started
                policy_calls = 1
                action_origin = "openvla_native_predict_action"
            else:
                action = _manual_action(str(request.direction))
                policy_calls = 0
                action_origin = "manual_clamped_direction"
            world_started = time.perf_counter()
            result = self._world.generate_one_step(condition, action, seed=request.seed)
            world_seconds = time.perf_counter() - world_started

        frames = tuple(getattr(result, "frames", ()))
        if len(frames) != 2:
            raise RuntimeError("IRASim one-step adapter returned %d frames, expected condition plus one prediction" % len(frames))
        predicted = self._frame_converter(frames[1])
        predicted_hash = closed_loop._frame_hash(predicted)
        encode_started = time.perf_counter()
        conditioning_png = self._png_encoder(condition)
        predicted_png = self._png_encoder(predicted)
        encode_seconds = time.perf_counter() - encode_started
        world_timing = closed_loop._timing_dict(getattr(result, "timing", None))
        return {
            "status": "completed_unqualified_experimental",
            "profile": PROFILE_ID,
            "model": self._model_info.get("model"),
            "revision": self._model_info.get("revision"),
            "state_mode": STATE_MODE,
            "warning": EXPERIMENTAL_WARNING,
            "prompt": request.prompt,
            "mode": request.mode,
            "direction": request.direction,
            "seed": request.seed,
            "source": input_source,
            "conditioning_png_base64": conditioning_png,
            "predicted_png_base64": predicted_png,
            "image_encoding": "PNG re-encoded from RGB arrays for transport",
            "conditioning_decoded_rgb_sha256": conditioning_hash,
            "predicted_decoded_rgb_sha256": predicted_hash,
            "action7": [float(value) for value in action],
            "action_origin": action_origin,
            "native_policy_calls": policy_calls,
            "world_calls": 1,
            "world_post_frame0_dropped": True,
            "world_returned_frame_count": 2,
            "policy_report": _jsonable(policy_details),
            "world_reported_timing": _jsonable(world_timing),
            "timings_ms": {
                "decode": _timing_ms(decode_seconds),
                "policy": _timing_ms(policy_seconds),
                "world": _timing_ms(world_seconds),
                "encode": _timing_ms(encode_seconds),
                "total": _timing_ms(time.perf_counter() - total_started),
            },
        }


def create_live_demo_service(config: LiveDemoConfig) -> LiveDemoService:
    """Verify pins, load the vendor frame, and warm real models before bind."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    release = closed_loop._release_binding(config.loop)
    provenance = verify_runtime_inputs(config.loop)
    vendor_frame = closed_loop._load_vendor_frame_zero(config.loop.fixture_video)
    policy, world = closed_loop._load_real_adapters(config.loop)
    load_timing = _warm_real_adapters(policy, world)
    integrity = {
        "openvla_manifest": provenance.get("openvla_manifest"),
        "irasim_checkpoint": provenance.get("irasim_checkpoint"),
        "irasim_config": provenance.get("irasim_config"),
        "irasim_source_checkout": provenance.get("irasim_source_checkout"),
        "fixture_video": provenance.get("fixture_video"),
    }
    model_info = {
        "model": {"policy": "openvla/openvla-7b", "world": "IRASim original Bridge frame_ada"},
        "revision": {
            "openvla": closed_loop.REVIEWED_OPENVLA_REVISION,
            "irasim_source": closed_loop.IRASIM_SOURCE_REVISION,
        },
        "integrity": integrity,
        "source_release": release,
    }
    return LiveDemoService(
        policy=policy,
        world=world,
        vendor_frame=vendor_frame,
        model_info=model_info,
        load_timing=load_timing,
    )


def make_handler(service: LiveDemoService, token: str, *, max_request_bytes: int) -> type[BaseHTTPRequestHandler]:
    """Build a token-protected handler bound to one loaded service."""

    class LiveDemoHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            # Never emit request headers (which contain the bearer token).
            return

        def _authorized(self) -> bool:
            value = self.headers.get("Authorization")
            prefix = "Bearer "
            return bool(value and value.startswith(prefix) and hmac.compare_digest(value[len(prefix):], token))

        def _send_json(self, status: int, payload: Mapping[str, Any], *, authenticate: bool = False) -> None:
            body = json.dumps(_jsonable(payload), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if authenticate:
                self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(body)

        def _require_auth(self) -> bool:
            if self._authorized():
                return True
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}, authenticate=True)
            return False

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._require_auth():
                return
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, service.health())
                return
            if self.path == "/fixture":
                try:
                    self._send_json(HTTPStatus.OK, service.fixture())
                except Exception as error:
                    print("live-demo fixture failure: %s" % type(error).__name__, file=sys.stderr, flush=True)
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "fixture_unavailable"})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._require_auth():
                return
            if self.path != "/step":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "content_type_must_be_application_json"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "content_length_required"})
                return
            if length < 1:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "request_body_required"})
                return
            if length > max_request_bytes:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "incomplete_request_body"})
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return
            try:
                response = service.step(payload)
            except LiveDemoInputError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_step", "message": str(error)})
                return
            except Exception as error:
                print("live-demo inference failure: %s: %s" % (type(error).__name__, str(error)), file=sys.stderr, flush=True)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "inference_failed"})
                return
            self._send_json(HTTPStatus.OK, response)

    return LiveDemoHandler


def create_http_server(config: LiveDemoConfig, service: LiveDemoService, token: str) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((config.host, config.port), make_handler(service, token, max_request_bytes=config.max_request_bytes))
    server.daemon_threads = True
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("PLUMB_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--token-file", type=Path, required=True, help="Single-line private Bearer token file; never logged.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-request-bytes", type=int, default=MAX_REQUEST_BYTES)
    parser.add_argument("--task", default=closed_loop.DEFAULT_TASK, help="Default prompt only; each policy request can supply prompt.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--openvla-manifest", type=Path)
    parser.add_argument("--irasim-repo", type=Path)
    parser.add_argument("--irasim-checkpoint", type=Path)
    parser.add_argument("--irasim-checkpoint-sha256")
    parser.add_argument("--irasim-conversion-report", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--irasim-config", type=Path)
    parser.add_argument("--fixture-video", type=Path)
    parser.add_argument("--source-release")
    parser.add_argument("--release-manifest", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> LiveDemoConfig:
    root = Path(args.root)
    loop = closed_loop.ClosedLoopConfig(
        root=root,
        model_path=args.model_path or root / "models" / "openvla--openvla-7b",
        openvla_manifest=args.openvla_manifest or root / "evidence" / "openvla--openvla-7b-download.json",
        irasim_repo=args.irasim_repo or root / "irasim-source",
        irasim_checkpoint=args.irasim_checkpoint or root / "models" / "irasim" / "frame_ada_0300000.safetensors",
        irasim_checkpoint_sha256=args.irasim_checkpoint_sha256,
        irasim_conversion_report=args.irasim_conversion_report,
        vae_path=args.vae or root / "models" / "stabilityai--stable-diffusion-xl-base-1.0",
        scheduler_path=args.scheduler or root / "irasim-source" / "pretrained_models" / "scheduler",
        irasim_config=args.irasim_config or root / "irasim-source" / "configs" / "evaluation" / "bridge" / "frame_ada.yaml",
        fixture_video=args.fixture_video or root / "fixtures" / "bridge_video.mp4",
        source_release=args.source_release or os.environ.get("PLUMB_SOURCE_RELEASE"),
        release_manifest=args.release_manifest,
        task=args.task,
        ticks=1,
        seed=0,
        output_dir=root / "evidence" / "live-demo-unused",
    )
    return LiveDemoConfig(loop=loop, token_file=Path(args.token_file), host=args.host, port=args.port, max_request_bytes=args.max_request_bytes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    token = _read_token(config.token_file)
    # All model work happens before the listener opens. Do not print token or
    # model paths: logs are commonly copied into an operations chat.
    service = create_live_demo_service(config)
    server = create_http_server(config, service, token)
    print(
        json.dumps(
            {
                "status": "ready",
                "host": config.host,
                "port": config.port,
                "profile": PROFILE_ID,
                "state_mode": STATE_MODE,
                "warning": EXPERIMENTAL_WARNING,
                "manual_directions": list(MANUAL_DIRECTIONS),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
