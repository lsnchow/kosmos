"""Local, fail-closed persistence for the KOSMOS matched comparison wall.

This module is deliberately independent from the legacy asynchronous outbox.
The comparison wall is a small, explicitly unscored presentation workflow:
one synchronous, streamed ``run_remote`` request per fixed cell.  Baseten is
the executor, while localhost owns the receipt journal and artifacts.  There
is no callback endpoint, object store, inferred remote continuation, or retry
loop in this module.

The stream protocol is newline-delimited JSON.  A producer must echo the
frozen identity hashes in its first ``header`` record.  Each generated PNG is
atomically written before its journal/event row is committed; browser SSE only
ever reads that committed journal.  An EOF without a valid terminal record is
therefore explicitly ambiguous rather than accidentally "complete".
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import select
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .records import canonical_json, utc_now


COMPARISON_SCHEMA = "kosmos-comparison-v1"
SOURCE_MANIFEST_SCHEMA = "kosmos-comparison-source-manifest-v1"
STREAM_SCHEMA = "kosmos-chain-stream-v1"
POLICIES: Tuple[str, ...] = ("OpenVLA", "MiniVLA", "Octo-Small")
SEEDS: Tuple[int, ...] = (101, 102, 103, 104)
TASK = "close_drawer"
TASK_INSTRUCTION = "Close the drawer"
HORIZON = 70
MAX_LINE_BYTES = 32 * 1024 * 1024
# Matches the authenticated stream helper's response/source byte ceiling.  The
# cap applies before Base64 expansion, and is comfortably above a 256/480 RGB
# robotics frame while keeping a request bounded end-to-end.
MAX_PNG_BYTES = 12 * 1024 * 1024
MAX_STATE_BYTES = 512 * 1024
QUOTE_TTL_SECONDS = 15 * 60
MAX_BUDGET_USD = 100.0
_PNG = b"\x89PNG\r\n\x1a\n"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CLI_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_RUN_REMOTE_PATH = re.compile(r"^/(production|development|deployment/[A-Za-z0-9_-]+|environments/[A-Za-z0-9_-]+)/run_remote$")
_DIRECTIONS = {"up", "down", "left", "right", "forward", "back"}


class ComparisonError(RuntimeError):
    """A comparison cannot be safely created, received, or presented."""


class ComparisonNotReady(ComparisonError):
    """A frozen source/deployment/pricing manifest is incomplete."""


class ComparisonConflict(ComparisonError):
    """An idempotency or state-transition rule was violated."""


class ComparisonTransportError(ComparisonError):
    """The remote stream did not produce a trustworthy terminal result."""


class StreamProtocolError(ComparisonError):
    """A streamed record is malformed, conflicting, or not identity-bound."""


class ChainStreamTransport(Protocol):
    """Injectable stream seam.

    Test transports may yield arbitrarily split byte chunks.  Production uses
    the Baseten CLI profile so no API credential is ever accepted from the
    browser or written to the local journal.
    """

    def stream(
        self,
        *,
        run_remote_url: str,
        cli_profile: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterable[bytes]:
        ...


class BasetenCliStreamTransport:
    """Run a documented-profile Chain stream without exposing credentials.

    ``baseten truss`` owns profile authentication in its local secret store.
    The approved wrapper receives its short-lived credential only through the
    CLI-injected ``BASETEN_TRUSS_AUTH_*`` environment, never from this process,
    a browser, or a scraped profile file.  The frozen manifest carries the
    exact Chain ``/run_remote`` URL.  stderr goes to a private temporary file
    so a noisy CLI cannot block stdout streaming, and neither stdout nor stderr
    is logged by this module.
    """

    def stream(
        self,
        *,
        run_remote_url: str,
        cli_profile: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Iterable[bytes]:
        if not _CLI_PROFILE.fullmatch(cli_profile):
            raise ComparisonTransportError("invalid Baseten CLI profile")
        _validate_run_remote_url(run_remote_url)
        encoded = canonical_json(dict(payload)).encode("utf-8")
        environment = dict(os.environ)
        # The CLI profile is the credential boundary.  Do not accidentally
        # bypass it with a process environment secret inherited by the server.
        environment.pop("BASETEN_API_KEY", None)
        environment.pop("BASETEN_AUTH_TOKEN", None)
        environment.pop("BASETEN_TRUSS_AUTH_API_KEY", None)
        environment.pop("BASETEN_TRUSS_AUTH_REMOTE_URL", None)
        environment["BASETEN_PROFILE"] = cli_profile
        truss_executable = os.environ.get("PLUMB_BASETEN_TRUSS_EXECUTABLE", ".venv-deploy/bin/python")
        helper = os.environ.get("PLUMB_BASETEN_STREAM_HELPER", "scripts/stream_baseten_chain.py")
        stderr_handle = tempfile.TemporaryFile(mode="w+b")
        process = subprocess.Popen(  # nosec B603: constant executable/arguments
            [
                "baseten",
                "truss",
                "--truss-executable",
                truss_executable,
                helper,
                "--url",
                run_remote_url,
                "--read-timeout",
                str(max(1, int(math.ceil(timeout_seconds)))),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            env=environment,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(encoded)
            process.stdin.close()
            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    raise ComparisonTransportError("Baseten Chain stream timed out")
                # ``BufferedReader.read(n)`` is allowed to wait for n bytes,
                # which would turn a healthy small receipt stream into a fake
                # buffered response.  Wait only for readability, then read the
                # currently available bytes with ``read1`` (or the equivalent
                # file-descriptor read).  That also lets the local deadline be
                # checked while the remote stays silent.
                try:
                    readable, _, _ = select.select([process.stdout], [], [], min(0.25, remaining))
                except (OSError, TypeError, ValueError):
                    # Small in-memory stdout fakes used by unit tests have no
                    # descriptor.  They are already nonblocking/finite.
                    readable = [process.stdout]
                if not readable:
                    if process.poll() is not None:
                        continue
                    continue
                reader = getattr(process.stdout, "read1", None)
                chunk = reader(64 * 1024) if callable(reader) else process.stdout.read(64 * 1024)
                if chunk:
                    yield chunk
                    continue
                returncode = process.poll()
                if returncode is not None:
                    if returncode != 0:
                        raise ComparisonTransportError("Baseten Chain CLI stream exited unsuccessfully")
                    return
                # stdout.read normally blocks until data/EOF.  This branch is
                # retained for alternate file objects supplied by tests.
                time.sleep(0.01)
        except BrokenPipeError as error:
            raise ComparisonTransportError("Baseten Chain stream closed before accepting the request") from error
        finally:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()
            stderr_handle.close()


@dataclass(frozen=True)
class ManifestReadiness:
    available: bool
    mechanical_reasons: Tuple[str, ...]
    demo_quality_reasons: Tuple[str, ...]
    manifest: Optional[Dict[str, Any]]
    manifest_sha256: Optional[str]
    configuration: Optional[Dict[str, Any]]
    budget: Dict[str, Any]

    def payload(self) -> Dict[str, Any]:
        start = None
        policies: List[Dict[str, Any]] = []
        world = None
        if self.configuration is not None:
            start = _public_start(self.configuration["start"])
            policies = [
                {
                    "id": policy,
                    "identity_hashes": self.configuration["policy_identity_hashes"][policy],
                }
                for policy in POLICIES
            ]
            world = dict(self.configuration["world"])
        return {
            "schema": COMPARISON_SCHEMA,
            "available": self.available,
            "mechanical": {
                "status": "ready" if not self.mechanical_reasons else "blocked",
                "reasons": list(self.mechanical_reasons),
            },
            "demo_quality": {
                "status": "approved" if not self.demo_quality_reasons else "blocked",
                "reasons": list(self.demo_quality_reasons),
            },
            "blockers": list(self.mechanical_reasons + self.demo_quality_reasons),
            "manifest_sha256": self.manifest_sha256,
            "start": start,
            "policies": policies,
            "seeds": list(SEEDS),
            "horizon": HORIZON,
            "world": world,
            "budget": dict(self.budget),
            "scored": False,
            "claim_tier": "preview",
        }


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _rgb_png_metadata(raw: bytes, field: str) -> Tuple[int, int, str]:
    """Verify the exact RGB pixels required by the Chain PngFrame contract.

    The local service must bind source and received frames to both PNG file
    bytes and decoded RGB pixels, but it intentionally has no image-library
    runtime dependency.  The selected Bridge stream profile accepts only
    non-interlaced, 8-bit RGB PNGs, so a compact standards-only decoder is
    sufficient and avoids any implicit color conversion.
    """
    if not raw.startswith(_PNG) or len(raw) > MAX_PNG_BYTES:
        raise ComparisonNotReady("%s must be a bounded PNG" % field)
    cursor = len(_PNG)
    width = height = None
    idat: List[bytes] = []
    while cursor < len(raw):
        if cursor + 12 > len(raw):
            raise ComparisonNotReady("%s is a truncated PNG" % field)
        size = int.from_bytes(raw[cursor:cursor + 4], "big")
        chunk_type = raw[cursor + 4:cursor + 8]
        end = cursor + 12 + size
        if end > len(raw):
            raise ComparisonNotReady("%s has an invalid PNG chunk length" % field)
        payload = raw[cursor + 8:cursor + 8 + size]
        if chunk_type == b"IHDR":
            if size != 13 or width is not None:
                raise ComparisonNotReady("%s has an invalid PNG header" % field)
            width = int.from_bytes(payload[0:4], "big")
            height = int.from_bytes(payload[4:8], "big")
            # bit depth 8, truecolour RGB, normal compression/filter, and no
            # Adam7 interlace: exactly the frame representation the Chain
            # runtime's PngFrame.rgb_array accepts.
            if not width or not height or payload[8:13] != b"\x08\x02\x00\x00\x00":
                raise ComparisonNotReady("%s must be a non-interlaced 8-bit RGB PNG" % field)
        elif chunk_type == b"IDAT":
            idat.append(payload)
        elif chunk_type == b"IEND":
            if size != 0 or end != len(raw):
                raise ComparisonNotReady("%s has an invalid PNG end marker" % field)
            break
        cursor = end
    if width is None or height is None or not idat:
        raise ComparisonNotReady("%s lacks RGB PNG image data" % field)
    stride = width * 3
    expected = height * (stride + 1)
    if expected > MAX_PNG_BYTES * 4:
        raise ComparisonNotReady("%s decoded pixels exceed the bound" % field)
    try:
        filtered = zlib.decompress(b"".join(idat))
    except zlib.error as error:
        raise ComparisonNotReady("%s PNG data cannot be decompressed" % field) from error
    if len(filtered) != expected:
        raise ComparisonNotReady("%s PNG scanline length is invalid" % field)
    rows: List[bytes] = []
    previous = bytearray(stride)
    position = 0
    for _ in range(height):
        filter_type = filtered[position]
        position += 1
        encoded = filtered[position:position + stride]
        position += stride
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - 3] if index >= 3 else 0
            above = previous[index]
            upper_left = previous[index - 3] if index >= 3 else 0
            if filter_type == 0:
                result = value
            elif filter_type == 1:
                result = (value + left) & 0xFF
            elif filter_type == 2:
                result = (value + above) & 0xFF
            elif filter_type == 3:
                result = (value + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = left + above - upper_left
                distances = (abs(predictor - left), abs(predictor - above), abs(predictor - upper_left))
                nearest = left if distances[0] <= distances[1] and distances[0] <= distances[2] else (above if distances[1] <= distances[2] else upper_left)
                result = (value + nearest) & 0xFF
            else:
                raise ComparisonNotReady("%s uses an unsupported PNG filter" % field)
            row[index] = result
        rows.append(bytes(row))
        previous = row
    return width, height, _sha256_bytes(b"".join(rows))


def _native_state_values(state: Mapping[str, Any], field: str) -> List[float]:
    payload = state.get("payload")
    values = payload.get("bridge_state", payload.get("values")) if isinstance(payload, Mapping) else None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) not in (7, 8):
        raise ComparisonNotReady("%s.payload.values must be a source-convention 7-D or 8-D vector" % field)
    result: List[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ComparisonNotReady("%s.payload.values[%d] is not finite" % (field, index))
        result.append(float(value))
    return result


def _bounded_string(value: Any, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ComparisonNotReady("%s must be a non-empty string up to %d bytes" % (field, maximum))
    return value


def _required_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ComparisonNotReady("%s must be a sha256: digest" % field)
    return value


def _finite_number(value: Any, field: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ComparisonNotReady("%s must be a finite number" % field)
    number = float(value)
    if number < minimum or (strict and number <= minimum):
        raise ComparisonNotReady("%s is below its required lower bound" % field)
    return number


def _validate_run_remote_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ComparisonNotReady("chain.run_remote_url must be an HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.lower().startswith("chain-")
        or not parsed.hostname.lower().endswith(".api.baseten.co")
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or not _RUN_REMOTE_PATH.fullmatch(parsed.path)
    ):
        raise ComparisonNotReady("chain.run_remote_url must be a documented HTTPS Baseten Chain /run_remote URL")
    return value


def _state_binding(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonNotReady("%s must be an object with payload, origin, and conventions hash" % field)
    payload = value.get("payload")
    if not isinstance(payload, Mapping) or not payload:
        raise ComparisonNotReady("%s.payload must be a non-empty object" % field)
    try:
        payload_encoded = canonical_json(dict(payload)).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ComparisonNotReady("%s.payload is not canonical JSON" % field) from error
    if len(payload_encoded) > MAX_STATE_BYTES:
        raise ComparisonNotReady("%s.payload exceeds the state size limit" % field)
    origin = _bounded_string(value.get("origin"), field + ".origin")
    conventions = _required_sha(value.get("conventions_sha256"), field + ".conventions_sha256")
    digest = _required_sha(value.get("sha256"), field + ".sha256")
    actual = _sha256_bytes(payload_encoded)
    if digest != actual:
        raise ComparisonNotReady("%s.sha256 does not bind its state payload" % field)
    return {
        "payload": dict(payload),
        "origin": origin,
        "conventions_sha256": conventions,
        "sha256": digest,
    }


def _evidence_status(value: Any, field: str, expected: str) -> Optional[str]:
    if not isinstance(value, Mapping):
        return "%s is missing" % field
    if value.get("status") != expected:
        return "%s.status must be %r" % (field, expected)
    try:
        _required_sha(value.get("evidence_sha256"), field + ".evidence_sha256")
    except ComparisonNotReady as error:
        return str(error)
    return None


def _read_manifest(path: Path, root: Path, cap_usd: float) -> ManifestReadiness:
    mechanical: List[str] = []
    quality: List[str] = []
    budget: Dict[str, Any] = {"cap_usd": cap_usd, "reserved_usd": 0.0, "available_usd": cap_usd, "breakdown": []}
    if not path.is_file():
        mechanical.append("No comparison-source manifest is present at %s" % path.name)
        return ManifestReadiness(False, tuple(mechanical), tuple(quality), None, None, None, budget)
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        mechanical.append("comparison-source manifest is unreadable: %s" % error)
        return ManifestReadiness(False, tuple(mechanical), tuple(quality), None, None, None, budget)
    manifest_sha = _sha256_bytes(raw_bytes)
    if not isinstance(raw, Mapping):
        mechanical.append("comparison-source manifest must be a JSON object")
        return ManifestReadiness(False, tuple(mechanical), tuple(quality), None, manifest_sha, None, budget)
    manifest = dict(raw)
    try:
        if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ComparisonNotReady("manifest.schema must be %r" % SOURCE_MANIFEST_SCHEMA)
        if manifest.get("task") != TASK or manifest.get("task_instruction") != TASK_INSTRUCTION:
            raise ComparisonNotReady("manifest must freeze the exact Close the drawer task")
        start_raw = manifest.get("start")
        if not isinstance(start_raw, Mapping):
            raise ComparisonNotReady("manifest.start is missing")
        start_id = _bounded_string(start_raw.get("id"), "start.id", 128)
        source_relative = _bounded_string(start_raw.get("png_path"), "start.png_path", 400)
        source_path = (root / source_relative).resolve()
        if root not in source_path.parents or not source_path.is_file():
            raise ComparisonNotReady("start.png_path must name an existing local file beneath the data directory")
        png = source_path.read_bytes()
        if not png.startswith(_PNG) or len(png) > MAX_PNG_BYTES:
            raise ComparisonNotReady("start.png_path is not a bounded PNG")
        start_sha = _required_sha(start_raw.get("sha256"), "start.sha256")
        if start_sha != _sha256_bytes(png):
            raise ComparisonNotReady("start.sha256 does not match the local PNG")
        start_state = _state_binding(start_raw.get("state"), "start.state")
        _native_state_values(start_state, "start.state")
        start = {
            "id": start_id,
            "png_url": "/api/artifacts/" + source_path.relative_to(root).as_posix(),
            "png_path": source_path.relative_to(root).as_posix(),
            "sha256": start_sha,
            "state": start_state,
        }
        policies_raw = manifest.get("policies")
        if not isinstance(policies_raw, Mapping) or set(policies_raw) != set(POLICIES):
            raise ComparisonNotReady("manifest.policies must contain exactly OpenVLA, MiniVLA, and Octo-Small")
        policy_identity_hashes: Dict[str, Dict[str, str]] = {}
        for policy in POLICIES:
            profile = policies_raw.get(policy)
            if not isinstance(profile, Mapping):
                raise ComparisonNotReady("policies.%s must be an object" % policy)
            _bounded_string(profile.get("identity"), "policies.%s.identity" % policy)
            for key in ("source_sha256", "normalizer_sha256", "runtime_sha256"):
                _required_sha(profile.get(key), "policies.%s.%s" % (policy, key))
            policy_identity_hashes[policy] = {
                "source": str(profile["source_sha256"]),
                "normalizer": str(profile["normalizer_sha256"]),
                "runtime": str(profile["runtime_sha256"]),
                "profile": _sha256_json(dict(profile)),
            }
            for quality_key, expected in (("readiness", "ready"), ("rights", "approved"), ("fidelity", "accepted")):
                reason = _evidence_status(profile.get(quality_key), "policies.%s.%s" % (policy, quality_key), expected)
                if reason:
                    quality.append(reason)
        world_raw = manifest.get("world")
        if not isinstance(world_raw, Mapping):
            raise ComparisonNotReady("manifest.world is missing")
        world_id = _bounded_string(world_raw.get("identity"), "world.identity")
        world_profile_sha = _required_sha(world_raw.get("profile_sha256"), "world.profile_sha256")
        world_bindings = {}
        for key in (
            "asset_sha256",
            "deployment_sha256",
            "bridge_control_sha256",
            "gripper_semantics_sha256",
            "seed_convention_sha256",
        ):
            world_bindings[key] = _required_sha(world_raw.get(key), "world.%s" % key)
        if world_raw.get("native_action_dim") != 7:
            raise ComparisonNotReady("world.native_action_dim must attest the native 7-D control contract")
        runtime_profile = world_raw.get("runtime_profile")
        if not isinstance(runtime_profile, Mapping):
            raise ComparisonNotReady("world.runtime_profile is required for the typed Chain execution boundary")
        expected_runtime = {
            "backend": "cosmos",
            "domain": "bridge_orig_lerobot",
            "model_id": "nvidia/Cosmos3-Nano",
            "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
            "source_revision": None,
            "inference_steps": 30,
        }
        for key, expected in expected_runtime.items():
            if runtime_profile.get(key) != expected:
                raise ComparisonNotReady("world.runtime_profile.%s is not the frozen Cosmos execution value" % key)
        if runtime_profile.get("resolution_tier") not in (256, 480):
            raise ComparisonNotReady("world.runtime_profile.resolution_tier must be 256 or 480")
        _bounded_string(runtime_profile.get("profile_id"), "world.runtime_profile.profile_id", 256)
        _bounded_string(runtime_profile.get("asset_manifest_id"), "world.runtime_profile.asset_manifest_id", 256)
        bridge_control_profile_id = _bounded_string(world_raw.get("bridge_control_profile_id"), "world.bridge_control_profile_id", 256)
        held_gripper_action = _finite_number(world_raw.get("held_gripper_action"), "world.held_gripper_action", minimum=0.0)
        if held_gripper_action > 1.0:
            raise ComparisonNotReady("world.held_gripper_action must be within the source-bound [0,1] range")
        manual_seed = world_raw.get("manual_seed")
        if isinstance(manual_seed, bool) or not isinstance(manual_seed, int):
            raise ComparisonNotReady("world.manual_seed must be an explicitly frozen integer")
        manual = world_raw.get("manual")
        if (
            not isinstance(manual, Mapping)
            or manual.get("backend") != "cosmos"
            or manual.get("action_rows") != 16
            or manual.get("structural_frames") != 17
        ):
            raise ComparisonNotReady("world.manual must attest the Cosmos 16-action / 17-structural-frame profile")
        world = {
            "id": world_id,
            "profile_sha256": world_profile_sha,
            **world_bindings,
            "native_action_dim": 7,
            "runtime_profile": dict(runtime_profile),
            "bridge_control_profile_id": bridge_control_profile_id,
            "held_gripper_action": held_gripper_action,
            "manual_seed": manual_seed,
            "manual": {
                "backend": "cosmos",
                "action_rows": 16,
                "structural_frames": 17,
                "post_conditioning_frames": 16,
            },
        }
        for quality_key, expected in (("readiness", "ready"), ("rights", "approved"), ("fidelity", "accepted")):
            reason = _evidence_status(world_raw.get(quality_key), "world.%s" % quality_key, expected)
            if reason:
                quality.append(reason)
        chain_raw = manifest.get("chain")
        if not isinstance(chain_raw, Mapping):
            raise ComparisonNotReady("manifest.chain is missing")
        run_remote_url = _validate_run_remote_url(chain_raw.get("run_remote_url"))
        cli_profile = _bounded_string(chain_raw.get("cli_profile"), "chain.cli_profile", 100)
        if not _CLI_PROFILE.fullmatch(cli_profile):
            raise ComparisonNotReady("chain.cli_profile has unsupported characters")
        timeout_seconds = _finite_number(chain_raw.get("timeout_seconds", 3600), "chain.timeout_seconds", minimum=1.0, strict=True)
        if timeout_seconds > 3600:
            raise ComparisonNotReady("chain.timeout_seconds must not exceed the bounded one-hour stream deadline")
        pricing = _pricing_quote(manifest.get("pricing"), cap_usd)
        budget = pricing
        configuration = {
            "task": TASK,
            "task_instruction": TASK_INSTRUCTION,
            "start": start,
            "policies": {policy: dict(policies_raw[policy]) for policy in POLICIES},
            "policy_identity_hashes": policy_identity_hashes,
            "world": world,
            "chain": {"run_remote_url": run_remote_url, "cli_profile": cli_profile, "timeout_seconds": timeout_seconds},
            "identity_base": {
                "source_manifest": manifest_sha,
                "start_png": start_sha,
                "start_state": start_state["sha256"],
                "state_conventions": start_state["conventions_sha256"],
                "world_profile": world_profile_sha,
                "world_asset": world_bindings["asset_sha256"],
                "world_deployment": world_bindings["deployment_sha256"],
                "bridge_control": world_bindings["bridge_control_sha256"],
                "gripper_semantics": world_bindings["gripper_semantics_sha256"],
                "seed_convention": world_bindings["seed_convention_sha256"],
            },
            "pricing": pricing,
        }
    except ComparisonNotReady as error:
        mechanical.append(str(error))
        configuration = None
    return ManifestReadiness(not mechanical and not quality, tuple(mechanical), tuple(quality), manifest, manifest_sha, configuration, budget)


def _pricing_quote(value: Any, cap_usd: float) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonNotReady("manifest.pricing is missing")
    _required_sha(value.get("source_sha256"), "pricing.source_sha256")
    workers = value.get("workers")
    if not isinstance(workers, Mapping):
        raise ComparisonNotReady("pricing.workers is missing")
    multiplicity = {"controller": 12, "world": 12, "OpenVLA": 4, "MiniVLA": 4, "Octo-Small": 4}
    if set(workers) != set(multiplicity):
        raise ComparisonNotReady("pricing.workers must account for controller, world, and each selected policy")
    breakdown: List[Dict[str, Any]] = []
    total = 0.0
    for worker, cell_count in multiplicity.items():
        row = workers[worker]
        if not isinstance(row, Mapping):
            raise ComparisonNotReady("pricing.workers.%s must be an object" % worker)
        rate = _finite_number(row.get("verified_usd_per_hour"), "pricing.workers.%s.verified_usd_per_hour" % worker, minimum=0.0)
        bound_seconds = _finite_number(row.get("bound_seconds"), "pricing.workers.%s.bound_seconds" % worker, minimum=0.0, strict=True)
        margin = _finite_number(row.get("residency_margin"), "pricing.workers.%s.residency_margin" % worker, minimum=1.5)
        components = row.get("bound_components")
        if not isinstance(components, Mapping):
            raise ComparisonNotReady("pricing.workers.%s.bound_components must record cold, execution, and scale-down residency" % worker)
        component_sum = 0.0
        for component in ("cold_load_seconds", "execution_seconds", "scale_down_tail_seconds"):
            component_sum += _finite_number(components.get(component), "pricing.workers.%s.bound_components.%s" % (worker, component), minimum=0.0)
        if component_sum > bound_seconds + 1e-9:
            raise ComparisonNotReady("pricing.workers.%s.bound_seconds is smaller than its residency components" % worker)
        _required_sha(row.get("evidence_sha256"), "pricing.workers.%s.evidence_sha256" % worker)
        reserved = rate * bound_seconds * margin / 3600.0
        total += reserved
        breakdown.append({
            "worker": worker,
            "cell_count": cell_count,
            "verified_usd_per_hour": rate,
            "bound_seconds": bound_seconds,
            "residency_margin": margin,
            "bound_components": {name: float(components[name]) for name in ("cold_load_seconds", "execution_seconds", "scale_down_tail_seconds")},
            "reserved_usd": reserved,
        })
    if total > cap_usd + 1e-9:
        raise ComparisonNotReady("the frozen full-wall reservation exceeds the $%.2f incremental cap" % cap_usd)
    return {"cap_usd": cap_usd, "reserved_usd": total, "available_usd": cap_usd - total, "breakdown": breakdown}


def _constant_chunk(direction: str, held_gripper_action: float) -> List[List[float]]:
    """The only public manual control mapping: 16 constant native 7-D rows."""
    vectors = {
        "up": (0.0, 0.0, 0.0025),
        "down": (0.0, 0.0, -0.0025),
        "left": (-0.0025, 0.0, 0.0),
        "right": (0.0025, 0.0, 0.0),
        "forward": (0.0, 0.0025, 0.0),
        "back": (0.0, -0.0025, 0.0),
    }
    if direction not in vectors:
        raise ComparisonConflict("unknown manual direction")
    if not math.isfinite(held_gripper_action) or not 0 <= held_gripper_action <= 1:
        raise ComparisonConflict("manual gripper action lacks a valid source binding")
    row = list(vectors[direction]) + [0.0, 0.0, 0.0, held_gripper_action]
    return [list(row) for _ in range(16)]


def _public_start(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return source provenance without leaking an internal local path."""
    return {key: value[key] for key in ("id", "png_url", "sha256", "state")}


class ComparisonService:
    """SQLite-backed local receipt journal and single-use stream dispatcher."""

    def __init__(
        self,
        data_root: Path,
        *,
        transport: Optional[ChainStreamTransport] = None,
        manifest_path: Optional[Path] = None,
        budget_cap_usd: Optional[float] = None,
        test_mode: bool = False,
    ) -> None:
        self.root = Path(data_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        selected_manifest = manifest_path or os.environ.get("PLUMB_COMPARISON_MANIFEST") or (self.root / "comparison-source-manifest.json")
        self.manifest_path = Path(selected_manifest).resolve()
        self.db_path = self.root / "comparisons.sqlite3"
        self.artifacts_root = self.root / "comparisons"
        self.artifacts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        configured_cap = budget_cap_usd
        if configured_cap is None:
            raw_cap = os.environ.get("PLUMB_COMPARISON_BUDGET_USD", str(MAX_BUDGET_USD))
            try:
                configured_cap = float(raw_cap)
            except ValueError as error:
                raise ValueError("PLUMB_COMPARISON_BUDGET_USD must be numeric") from error
        if not math.isfinite(float(configured_cap)) or not 0 < float(configured_cap) <= MAX_BUDGET_USD:
            raise ValueError("comparison budget cap must be more than zero and no more than $100")
        self.budget_cap_usd = float(configured_cap)
        self.transport = transport or BasetenCliStreamTransport()
        self.test_mode = test_mode
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, min(12, int(os.environ.get("PLUMB_COMPARISON_MAX_WORKERS", "1")))),
            thread_name_prefix="plumb-comparison",
        )
        self._futures: Dict[str, Future] = {}
        self._initialise()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialise(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS comparisons (
                  id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  quote_id TEXT NOT NULL,
                  config_json TEXT NOT NULL,
                  config_sha256 TEXT NOT NULL,
                  manifest_sha256 TEXT NOT NULL,
                  status TEXT NOT NULL,
                  reservation_usd REAL NOT NULL,
                  reservation_json TEXT NOT NULL,
                  origin TEXT NOT NULL DEFAULT 'fresh',
                  presentation INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  terminal_at TEXT,
                  error_json TEXT
                );
                CREATE TABLE IF NOT EXISTS comparison_quotes (
                  id TEXT PRIMARY KEY,
                  manifest_sha256 TEXT NOT NULL,
                  config_sha256 TEXT NOT NULL,
                  reservation_json TEXT NOT NULL,
                  expires_at REAL NOT NULL,
                  consumed_comparison_id TEXT UNIQUE,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS comparison_cells (
                  comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
                  cell_id TEXT NOT NULL,
                  policy TEXT NOT NULL,
                  seed INTEGER NOT NULL,
                  origin TEXT NOT NULL DEFAULT 'fresh',
                  status TEXT NOT NULL,
                  attempt_id TEXT,
                  frame_count INTEGER NOT NULL DEFAULT 0,
                  action_count INTEGER NOT NULL DEFAULT 0,
                  terminal_received INTEGER NOT NULL DEFAULT 0,
                  latest_event_id TEXT,
                  error_json TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(comparison_id, cell_id),
                  UNIQUE(comparison_id, policy, seed)
                );
                CREATE TABLE IF NOT EXISTS comparison_attempts (
                  id TEXT PRIMARY KEY,
                  comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
                  cell_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  status TEXT NOT NULL,
                  request_json TEXT NOT NULL,
                  request_sha256 TEXT NOT NULL,
                  expected_identities_json TEXT NOT NULL,
                  header_received INTEGER NOT NULL DEFAULT 0,
                  terminal_received INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  started_at TEXT,
                  finished_at TEXT,
                  error_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_comparison_attempts_cell ON comparison_attempts(comparison_id, cell_id, created_at);
                CREATE TABLE IF NOT EXISTS comparison_events (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                  comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
                  cell_id TEXT,
                  attempt_id TEXT NOT NULL,
                  event_id TEXT NOT NULL,
                  type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(attempt_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_comparison_events_replay ON comparison_events(comparison_id, sequence);
                CREATE TABLE IF NOT EXISTS comparison_frames (
                  attempt_id TEXT NOT NULL REFERENCES comparison_attempts(id) ON DELETE CASCADE,
                  event_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  frame_index INTEGER NOT NULL,
                  artifact_path TEXT NOT NULL,
                  png_sha256 TEXT NOT NULL,
                  action_json TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(attempt_id, event_id),
                  UNIQUE(attempt_id, segment_id, frame_index)
                );
                CREATE TABLE IF NOT EXISTS manual_sessions (
                  id TEXT PRIMARY KEY,
                  comparison_id TEXT NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
                  cell_id TEXT NOT NULL,
                  initial_source_event_id TEXT,
                  initial_source_artifact_path TEXT,
                  initial_source_png_sha256 TEXT,
                  initial_source_state_json TEXT,
                  source_event_id TEXT NOT NULL,
                  source_artifact_path TEXT NOT NULL,
                  source_png_sha256 TEXT NOT NULL,
                  source_state_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  active_command_id TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS manual_commands (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL REFERENCES manual_sessions(id) ON DELETE CASCADE,
                  idempotency_key TEXT NOT NULL,
                  direction TEXT NOT NULL,
                  action_json TEXT NOT NULL,
                  attempt_id TEXT NOT NULL REFERENCES comparison_attempts(id) ON DELETE CASCADE,
                  status TEXT NOT NULL,
                  frame_count INTEGER NOT NULL DEFAULT 0,
                  error_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(session_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS manual_frames (
                  command_id TEXT NOT NULL REFERENCES manual_commands(id) ON DELETE CASCADE,
                  event_id TEXT NOT NULL,
                  frame_index INTEGER NOT NULL,
                  artifact_path TEXT NOT NULL,
                  png_sha256 TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(command_id, event_id),
                  UNIQUE(command_id, frame_index)
                );
                """
            )
            for column, definition in (
                ("initial_source_event_id", "TEXT"),
                ("initial_source_artifact_path", "TEXT"),
                ("initial_source_png_sha256", "TEXT"),
                ("initial_source_state_json", "TEXT"),
            ):
                existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(manual_sessions)").fetchall()}
                if column not in existing_columns:
                    connection.execute("ALTER TABLE manual_sessions ADD COLUMN %s %s" % (column, definition))
            for table, column in (("comparisons", "origin"), ("comparison_cells", "origin")):
                existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(%s)" % table).fetchall()}
                if column not in existing_columns:
                    connection.execute("ALTER TABLE %s ADD COLUMN %s TEXT NOT NULL DEFAULT 'fresh'" % (table, column))
            command_columns = {row["name"] for row in connection.execute("PRAGMA table_info(manual_commands)").fetchall()}
            if "reservation_usd" not in command_columns:
                connection.execute("ALTER TABLE manual_commands ADD COLUMN reservation_usd REAL NOT NULL DEFAULT 0")
            # A local restart cannot prove whether a stream advanced remotely.
            # Preserve already received artifacts but force a fresh explicit
            # attempt rather than silently continuing/retrying anything.
            error = canonical_json({
                "kind": "InterruptedLocalReceiver",
                "message": "local receiver restarted before a terminal stream record",
                "automatic_retry_allowed": False,
                "manual_reconciliation_required": True,
            })
            now = utc_now()
            connection.execute(
                "UPDATE comparison_attempts SET status = 'ambiguous', finished_at = ?, error_json = ? WHERE status IN ('dispatching', 'running') AND terminal_received = 0",
                (now, error),
            )
            connection.execute(
                "UPDATE comparison_cells SET status = 'interrupted', error_json = ?, updated_at = ? WHERE status IN ('dispatching', 'running') AND terminal_received = 0",
                (error, now),
            )
            connection.execute(
                "UPDATE comparisons SET status = 'interrupted', error_json = ?, updated_at = ?, terminal_at = ? WHERE status IN ('queued', 'running') AND EXISTS (SELECT 1 FROM comparison_cells c WHERE c.comparison_id = comparisons.id AND c.status = 'interrupted')",
                (error, now, now),
            )
            connection.execute(
                "UPDATE manual_commands SET status = 'ambiguous', error_json = ?, updated_at = ? WHERE status IN ('queued', 'running')",
                (error, now),
            )
            connection.execute(
                "UPDATE manual_sessions SET status = 'ambiguous', active_command_id = NULL, updated_at = ? WHERE status = 'running'",
                (now,),
            )
        os.chmod(self.db_path, 0o600)

    def readiness(self) -> ManifestReadiness:
        value = _read_manifest(self.manifest_path, self.root, self.budget_cap_usd)
        active = self._active_reservations()
        budget = dict(value.budget)
        budget["reserved_usd"] = active
        budget["available_usd"] = max(0.0, self.budget_cap_usd - active)
        return ManifestReadiness(
            value.available and active + float(value.budget["reserved_usd"]) <= self.budget_cap_usd + 1e-9,
            value.mechanical_reasons + (() if active + float(value.budget["reserved_usd"]) <= self.budget_cap_usd + 1e-9 else ("active reservation would exceed the $100 incremental cap",)),
            value.demo_quality_reasons,
            value.manifest,
            value.manifest_sha256,
            value.configuration,
            budget,
        )

    def _active_reservations(self) -> float:
        with self._connect() as connection:
            row = connection.execute("SELECT COALESCE(SUM(reservation_usd), 0) AS total FROM comparisons").fetchone()
            manual = connection.execute("SELECT COALESCE(SUM(reservation_usd), 0) AS total FROM manual_commands").fetchone()
        return float(row["total"] if row is not None else 0.0) + float(manual["total"] if manual is not None else 0.0)

    def create_quote(self) -> Dict[str, Any]:
        readiness = self.readiness()
        if not readiness.available or readiness.configuration is None or readiness.manifest_sha256 is None:
            raise ComparisonNotReady("; ".join(readiness.mechanical_reasons + readiness.demo_quality_reasons))
        reservation = readiness.configuration["pricing"]
        quote_id = uuid.uuid4().hex
        config_sha = _sha256_json(self._frozen_config(readiness))
        expires_at = time.time() + QUOTE_TTL_SECONDS
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO comparison_quotes(id, manifest_sha256, config_sha256, reservation_json, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (quote_id, readiness.manifest_sha256, config_sha, canonical_json(reservation), expires_at, utc_now()),
            )
        return {
            "id": quote_id,
            "expires_at": expires_at,
            "manifest_sha256": readiness.manifest_sha256,
            "reservation_usd": reservation["reserved_usd"],
            "breakdown": reservation["breakdown"],
        }

    def _frozen_config(self, readiness: ManifestReadiness) -> Dict[str, Any]:
        assert readiness.configuration is not None and readiness.manifest_sha256 is not None
        config = readiness.configuration
        return {
            "schema": COMPARISON_SCHEMA,
            "manifest_sha256": readiness.manifest_sha256,
            "task": TASK,
            "task_instruction": TASK_INSTRUCTION,
            "horizon": HORIZON,
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "start": config["start"],
            "policy_identity_hashes": config["policy_identity_hashes"],
            "world": config["world"],
            "chain": config["chain"],
            "identity_base": config["identity_base"],
            "pricing": config["pricing"],
            "claim_tier": "preview",
            "scored": False,
        }

    def create_comparison(self, quote_id: str, idempotency_key: str) -> Tuple[Dict[str, Any], bool]:
        if not _IDENTIFIER.fullmatch(idempotency_key):
            raise ComparisonConflict("idempotency_key must be a bounded identifier")
        readiness = self.readiness()
        if not readiness.available or readiness.configuration is None or readiness.manifest_sha256 is None:
            raise ComparisonNotReady("; ".join(readiness.mechanical_reasons + readiness.demo_quality_reasons))
        config = self._frozen_config(readiness)
        current_config_sha = _sha256_json(config)
        with self._lock, self._connect() as connection:
            existing = connection.execute("SELECT id, config_sha256 FROM comparisons WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if existing is not None:
                if existing["config_sha256"] != current_config_sha:
                    raise ComparisonConflict("idempotency key is bound to a different frozen comparison configuration")
                return self._detail_in_connection(connection, str(existing["id"])), True
            quote = connection.execute("SELECT * FROM comparison_quotes WHERE id = ?", (quote_id,)).fetchone()
            if quote is None:
                raise ComparisonConflict("quote was not found")
            if quote["consumed_comparison_id"] is not None:
                raise ComparisonConflict("quote was already consumed")
            if float(quote["expires_at"]) < time.time():
                raise ComparisonConflict("quote has expired; obtain a fresh quote")
            if quote["manifest_sha256"] != readiness.manifest_sha256 or quote["config_sha256"] != current_config_sha:
                raise ComparisonConflict("quote is not bound to the current frozen manifest")
            reserved = json.loads(str(quote["reservation_json"]))
            total = self._active_reservations()
            if total + float(reserved["reserved_usd"]) > self.budget_cap_usd + 1e-9:
                raise ComparisonNotReady("admitting this comparison would exceed the $100 incremental budget reservation")
            if self._has_active_manual_command(connection):
                raise ComparisonConflict("manual world generation is active; comparison admission is mutually exclusive")
            comparison_id = uuid.uuid4().hex
            now = utc_now()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO comparisons(id, idempotency_key, quote_id, config_json, config_sha256, manifest_sha256, status, reservation_usd, reservation_json, origin, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, 'fresh', ?, ?)",
                (comparison_id, idempotency_key, quote_id, canonical_json(config), current_config_sha, readiness.manifest_sha256, float(reserved["reserved_usd"]), canonical_json(reserved), now, now),
            )
            for seed in SEEDS:
                for policy in POLICIES:
                    cell_id = self.cell_id(policy, seed)
                    connection.execute(
                        "INSERT INTO comparison_cells(comparison_id, cell_id, policy, seed, origin, status, updated_at) VALUES (?, ?, ?, ?, 'fresh', 'queued', ?)",
                        (comparison_id, cell_id, policy, seed, now),
                    )
            connection.execute("UPDATE comparison_quotes SET consumed_comparison_id = ? WHERE id = ?", (comparison_id, quote_id))
            self._append_event(connection, comparison_id, None, "local:%s" % comparison_id, "comparison-created", "comparison_created", {"status": "queued"})
            connection.commit()
            detail = self._detail_in_connection(connection, comparison_id)
        # Submit background work after the entire fixed wall and reservation are
        # durable.  A browser closing has no effect on this local receiver.
        for cell in detail["cells"]:
            future = self._pool.submit(self._dispatch_cell, comparison_id, str(cell["id"]))
            self._futures["%s:%s" % (comparison_id, cell["id"])] = future
        return detail, False

    @staticmethod
    def cell_id(policy: str, seed: int) -> str:
        return "%s-%s" % (policy.lower().replace("-", "_"), seed)

    def _has_active_manual_command(self, connection: sqlite3.Connection) -> bool:
        row = connection.execute("SELECT 1 FROM manual_commands WHERE status IN ('queued', 'running') LIMIT 1").fetchone()
        return row is not None

    def _has_active_comparison(self, connection: sqlite3.Connection) -> bool:
        row = connection.execute("SELECT 1 FROM comparisons WHERE status IN ('queued', 'running') LIMIT 1").fetchone()
        return row is not None

    def _dispatch_cell(self, comparison_id: str, cell_id: str) -> None:
        """One external request, exactly once.  Any failure is terminal locally."""
        try:
            with self._lock, self._connect() as connection:
                comparison = connection.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
                cell = connection.execute("SELECT * FROM comparison_cells WHERE comparison_id = ? AND cell_id = ?", (comparison_id, cell_id)).fetchone()
                if comparison is None or cell is None or cell["status"] != "queued":
                    return
                config = json.loads(str(comparison["config_json"]))
                attempt_id = uuid.uuid4().hex
                identities = self._cell_identities(config, str(cell["policy"]))
                dispatch = {
                    "schema": STREAM_SCHEMA,
                    "kind": "comparison_rollout",
                    "comparison_id": comparison_id,
                    "attempt_id": attempt_id,
                    "cell_id": cell_id,
                    "task": TASK,
                    "task_instruction": TASK_INSTRUCTION,
                    "policy": cell["policy"],
                    "world_seed": int(cell["seed"]),
                    "horizon": HORIZON,
                    "start": self._stream_start(config["start"]),
                    "world": config["world"],
                    "policy_profile": config["policy_identity_hashes"][str(cell["policy"])],
                    "identity_hashes": identities,
                    "claim_tier": "preview",
                    "scored": False,
                }
                # Chains serialise the entrypoint's named argument under
                # ``request``.  The flat inner record is intentionally owned
                # by the service/typed runtime translator and is what the
                # controller echoes in its identity-bound header.
                payload = {"request": dispatch}
                now = utc_now()
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO comparison_attempts(id, comparison_id, cell_id, kind, status, request_json, request_sha256, expected_identities_json, created_at, started_at) VALUES (?, ?, ?, 'comparison', 'running', ?, ?, ?, ?, ?)",
                (attempt_id, comparison_id, cell_id, canonical_json(payload), _sha256_json(payload), canonical_json(identities), now, now),
                )
                connection.execute("UPDATE comparison_cells SET status = 'running', attempt_id = ?, updated_at = ? WHERE comparison_id = ? AND cell_id = ?", (attempt_id, now, comparison_id, cell_id))
                connection.execute("UPDATE comparisons SET status = 'running', updated_at = ? WHERE id = ? AND status = 'queued'", (now, comparison_id))
                self._append_event(connection, comparison_id, cell_id, attempt_id, "local-dispatch", "cell_status", {"status": "running"})
                connection.commit()
            self._receive_stream(
                comparison_id=comparison_id,
                cell_id=cell_id,
                attempt_id=attempt_id,
                payload=payload,
                chain=config["chain"],
                manual_command_id=None,
            )
        except Exception as error:  # Never let a worker exception leave an in-flight cell.
            self._mark_attempt_ambiguous(comparison_id, cell_id, locals().get("attempt_id"), error)

    def _cell_identities(self, config: Mapping[str, Any], policy: str) -> Dict[str, str]:
        identities = dict(config["identity_base"])
        identities.update({"policy_source": config["policy_identity_hashes"][policy]["source"], "policy_normalizer": config["policy_identity_hashes"][policy]["normalizer"], "policy_runtime": config["policy_identity_hashes"][policy]["runtime"], "policy_profile": config["policy_identity_hashes"][policy]["profile"]})
        return identities

    def _stream_start(self, start: Mapping[str, Any]) -> Dict[str, Any]:
        """Read and rehash the actual local source just before dispatch.

        A Chain cannot dereference localhost's artifact URL.  The source PNG is
        supplied in the request itself after an on-dispatch hash check, so every
        policy/seed cell receives the same measured start bytes and state.
        """
        source_path = (self.root / str(start["png_path"])).resolve()
        if self.root not in source_path.parents or not source_path.is_file():
            raise ComparisonConflict("frozen shared source PNG is no longer available locally")
        png = source_path.read_bytes()
        if not png.startswith(_PNG) or _sha256_bytes(png) != start["sha256"]:
            raise ComparisonConflict("frozen shared source PNG no longer matches its manifest receipt")
        return {"id": start["id"], "png_base64": base64.b64encode(png).decode("ascii"), "png_sha256": start["sha256"], "state": start["state"]}

    def _receive_stream(
        self,
        *,
        comparison_id: str,
        cell_id: str,
        attempt_id: str,
        payload: Mapping[str, Any],
        chain: Mapping[str, Any],
        manual_command_id: Optional[str],
    ) -> None:
        terminal = False
        try:
            chunks = self.transport.stream(
                run_remote_url=str(chain["run_remote_url"]),
                cli_profile=str(chain["cli_profile"]),
                payload=payload,
                timeout_seconds=float(chain["timeout_seconds"]),
            )
            for record in self._decode_ndjson(chunks):
                terminal = self._ingest_record(comparison_id, cell_id, attempt_id, record, manual_command_id) or terminal
            if not terminal:
                raise ComparisonTransportError("stream ended without a terminal record")
        except Exception as error:
            self._mark_attempt_ambiguous(comparison_id, cell_id, attempt_id, error, manual_command_id=manual_command_id)

    @staticmethod
    def _decode_ndjson(chunks: Iterable[bytes]) -> Iterator[Dict[str, Any]]:
        remainder = b""
        for chunk in chunks:
            if not isinstance(chunk, (bytes, bytearray)):
                raise StreamProtocolError("stream transport yielded a non-byte chunk")
            remainder += bytes(chunk)
            if len(remainder) > MAX_LINE_BYTES and b"\n" not in remainder:
                raise StreamProtocolError("stream record exceeds the line byte limit")
            while b"\n" in remainder:
                line, remainder = remainder.split(b"\n", 1)
                if len(line) > MAX_LINE_BYTES:
                    raise StreamProtocolError("stream record exceeds the line byte limit")
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise StreamProtocolError("stream contained malformed NDJSON") from error
                if not isinstance(decoded, Mapping):
                    raise StreamProtocolError("stream record must be a JSON object")
                yield dict(decoded)
        if remainder.strip():
            raise StreamProtocolError("stream ended with a truncated NDJSON record")

    def _ingest_record(self, comparison_id: str, cell_id: str, attempt_id: str, record: Mapping[str, Any], manual_command_id: Optional[str]) -> bool:
        kind = record.get("kind")
        event_id = record.get("event_id")
        if record.get("schema") != STREAM_SCHEMA or kind not in {"header", "stage", "frame", "heartbeat", "terminal"}:
            raise StreamProtocolError("record has an unsupported stream schema or kind")
        if not isinstance(event_id, str) or not _IDENTIFIER.fullmatch(event_id):
            raise StreamProtocolError("record.event_id must be a bounded identifier")
        with self._lock, self._connect() as connection:
            attempt = connection.execute("SELECT * FROM comparison_attempts WHERE id = ?", (attempt_id,)).fetchone()
            if attempt is None or attempt["status"] not in {"running", "completed", "failed", "blocked"}:
                raise StreamProtocolError("record targets a non-receivable attempt")
            expected = json.loads(str(attempt["expected_identities_json"]))
            request_payload = json.loads(str(attempt["request_json"]))
            dispatch_payload = request_payload.get("request") if isinstance(request_payload, Mapping) else None
            if not isinstance(dispatch_payload, Mapping):
                raise StreamProtocolError("persisted Chain request lacks the required request envelope")
            self._validate_record_binding(record, comparison_id, cell_id, attempt_id, expected, manual_command_id, dispatch_payload)
            existing = connection.execute("SELECT payload_sha256 FROM comparison_events WHERE attempt_id = ? AND event_id = ?", (attempt_id, event_id)).fetchone()
            payload_sha = _sha256_json(record)
            if existing is not None:
                if existing["payload_sha256"] != payload_sha:
                    raise StreamProtocolError("duplicate stream event ID has conflicting content")
                return kind == "terminal"
            if bool(attempt["terminal_received"]):
                raise StreamProtocolError("stream emitted a new record after its terminal record")
            if kind != "header" and not bool(attempt["header_received"]):
                raise StreamProtocolError("stream record arrived before the identity header")
            if kind == "header":
                if bool(attempt["header_received"]):
                    raise StreamProtocolError("stream emitted more than one header")
                connection.execute("UPDATE comparison_attempts SET header_received = 1 WHERE id = ?", (attempt_id,))
                self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, "header", self._public_record(record))
                return False
            if kind == "frame":
                if manual_command_id is None:
                    self._ingest_comparison_frame(connection, comparison_id, cell_id, attempt_id, record)
                else:
                    self._ingest_manual_frame(connection, comparison_id, cell_id, attempt_id, manual_command_id, record)
                return False
            if kind == "terminal":
                self._ingest_terminal(connection, comparison_id, cell_id, attempt_id, record, manual_command_id)
                return True
            event_type = "heartbeat" if kind == "heartbeat" else "stage"
            self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, event_type, self._public_record(record))
        return False

    def _validate_record_binding(self, record: Mapping[str, Any], comparison_id: str, cell_id: str, attempt_id: str, identities: Mapping[str, Any], manual_command_id: Optional[str], request_payload: Mapping[str, Any]) -> None:
        for key, expected in (("comparison_id", comparison_id), ("attempt_id", attempt_id), ("cell_id", cell_id)):
            if record.get(key) != expected:
                raise StreamProtocolError("record %s does not bind the expected stream" % key)
        if manual_command_id is None:
            if record.get("mode", "comparison") != "comparison":
                raise StreamProtocolError("comparison stream record has the wrong mode")
        elif record.get("mode") != "manual" or record.get("command_id") != manual_command_id:
            raise StreamProtocolError("manual stream record does not bind its command")
        if record.get("kind") == "header":
            if record.get("task") != TASK or record.get("task_instruction") != TASK_INSTRUCTION:
                raise StreamProtocolError("header does not echo the fixed task")
            if record.get("identity_hashes") != identities:
                raise StreamProtocolError("header identity hashes do not match the frozen request")
            if manual_command_id is None:
                if record.get("policy") != request_payload.get("policy") or record.get("world_seed") != request_payload.get("world_seed"):
                    raise StreamProtocolError("header does not echo the frozen policy and world seed")
            else:
                requested_source = request_payload.get("source")
                if not isinstance(requested_source, Mapping) or record.get("source_event_id") != requested_source.get("event_id") or record.get("source_png_sha256") != requested_source.get("png_sha256"):
                    raise StreamProtocolError("manual header does not echo the frozen branch source")

    def _frame_fields(self, record: Mapping[str, Any], maximum: int) -> Tuple[str, int, bytes, str, Dict[str, Any]]:
        segment_id = record.get("segment_id")
        index = record.get("frame_index")
        if not isinstance(segment_id, str) or not _IDENTIFIER.fullmatch(segment_id):
            raise StreamProtocolError("frame.segment_id must be a bounded identifier")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= maximum:
            raise StreamProtocolError("frame.frame_index is outside the admitted segment horizon")
        encoded = record.get("png_base64")
        if not isinstance(encoded, str) or len(encoded) > MAX_PNG_BYTES * 2:
            raise StreamProtocolError("frame.png_base64 is missing or too large")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as error:
            raise StreamProtocolError("frame.png_base64 is invalid") from error
        if len(png) > MAX_PNG_BYTES or not png.startswith(_PNG):
            raise StreamProtocolError("frame is not a bounded PNG")
        digest = record.get("png_sha256")
        if not isinstance(digest, str) or digest != _sha256_bytes(png):
            raise StreamProtocolError("frame PNG digest mismatch")
        state = _state_binding(record.get("state"), "frame.state")
        return segment_id, index, png, digest, state

    @staticmethod
    def _action(value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise StreamProtocolError("frame.action must be the source-executed action object")
        executed = value.get("executed")
        raw = value.get("raw_proposal")
        if not isinstance(executed, list) or len(executed) != 7 or not isinstance(raw, (list, Mapping)):
            raise StreamProtocolError("frame.action must retain raw proposal and one executed native 7-D action")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in executed):
            raise StreamProtocolError("frame executed action has non-finite values")
        return dict(value)

    def _write_png(self, relative: str, data: bytes) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise StreamProtocolError("artifact path escapes the data root")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def _ingest_comparison_frame(self, connection: sqlite3.Connection, comparison_id: str, cell_id: str, attempt_id: str, record: Mapping[str, Any]) -> None:
        segment_id, index, png, digest, state = self._frame_fields(record, HORIZON)
        action = self._action(record.get("action"))
        event_id = str(record["event_id"])
        existing = connection.execute("SELECT event_id, png_sha256 FROM comparison_frames WHERE attempt_id = ? AND segment_id = ? AND frame_index = ?", (attempt_id, segment_id, index)).fetchone()
        if existing is not None:
            raise StreamProtocolError("duplicate segment/frame identity has different stream event identity")
        current = connection.execute("SELECT frame_count FROM comparison_cells WHERE comparison_id = ? AND cell_id = ?", (comparison_id, cell_id)).fetchone()
        expected_index = int(current["frame_count"]) + 1 if current is not None else 1
        if index != expected_index:
            raise StreamProtocolError("comparison frame order is not contiguous")
        relative = "comparisons/%s/cells/%s/%s/frame-%03d.png" % (comparison_id, cell_id, attempt_id, index)
        # Artifact first, then the SQLite transaction/event.  The SSE consumer
        # cannot observe a URL until the journal has committed below.
        self._write_png(relative, png)
        now = utc_now()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO comparison_frames(attempt_id, event_id, segment_id, frame_index, artifact_path, png_sha256, action_json, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (attempt_id, event_id, segment_id, index, relative, digest, canonical_json(action), canonical_json(state), now))
        next_count = expected_index
        connection.execute("UPDATE comparison_cells SET frame_count = ?, action_count = ?, latest_event_id = ?, updated_at = ? WHERE comparison_id = ? AND cell_id = ?", (next_count, next_count, event_id, now, comparison_id, cell_id))
        public = {"event_id": event_id, "segment_id": segment_id, "frame_index": index, "url": "/api/artifacts/" + relative, "sha256": digest, "action": action, "state": state, "created_at": now}
        self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, "frame", public)
        connection.commit()

    def _ingest_manual_frame(self, connection: sqlite3.Connection, comparison_id: str, cell_id: str, attempt_id: str, command_id: str, record: Mapping[str, Any]) -> None:
        segment_id, index, png, digest, state = self._frame_fields(record, 16)
        event_id = str(record["event_id"])
        existing = connection.execute("SELECT event_id FROM manual_frames WHERE command_id = ? AND frame_index = ?", (command_id, index)).fetchone()
        if existing is not None:
            raise StreamProtocolError("duplicate manual frame index has a different stream event identity")
        count_row = connection.execute("SELECT frame_count FROM manual_commands WHERE id = ?", (command_id,)).fetchone()
        expected_index = int(count_row["frame_count"]) + 1 if count_row is not None else 1
        if index != expected_index:
            raise StreamProtocolError("manual frame order is not contiguous")
        relative = "comparisons/%s/manual/%s/%s/frame-%03d.png" % (comparison_id, command_id, attempt_id, index)
        self._write_png(relative, png)
        now = utc_now()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO manual_frames(command_id, event_id, frame_index, artifact_path, png_sha256, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (command_id, event_id, index, relative, digest, canonical_json(state), now))
        count = expected_index
        connection.execute("UPDATE manual_commands SET frame_count = ?, updated_at = ? WHERE id = ?", (count, now, command_id))
        public = {"session_id": self._command_session_id(connection, command_id), "command_id": command_id, "event_id": event_id, "segment_id": segment_id, "frame_index": index, "url": "/api/artifacts/" + relative, "sha256": digest, "state": state, "created_at": now}
        self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, "manual_frame", public)
        connection.commit()

    @staticmethod
    def _command_session_id(connection: sqlite3.Connection, command_id: str) -> str:
        row = connection.execute("SELECT session_id FROM manual_commands WHERE id = ?", (command_id,)).fetchone()
        if row is None:
            raise StreamProtocolError("manual command no longer exists")
        return str(row["session_id"])

    def _ingest_terminal(self, connection: sqlite3.Connection, comparison_id: str, cell_id: str, attempt_id: str, record: Mapping[str, Any], manual_command_id: Optional[str]) -> None:
        status = record.get("status")
        if status not in {"completed", "failed", "blocked"}:
            raise StreamProtocolError("terminal status must be completed, failed, or blocked")
        event_id = str(record["event_id"])
        now = utc_now()
        if manual_command_id is None:
            count_row = connection.execute("SELECT frame_count FROM comparison_cells WHERE comparison_id = ? AND cell_id = ?", (comparison_id, cell_id)).fetchone()
            frame_count = int(count_row["frame_count"]) if count_row is not None else 0
            action_count = record.get("action_count")
            reported_frames = record.get("frame_count")
            if isinstance(reported_frames, bool) or not isinstance(reported_frames, int) or reported_frames != frame_count:
                raise StreamProtocolError("terminal frame count does not match committed comparison receipts")
            if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 0:
                raise StreamProtocolError("terminal action_count must be a non-negative integer")
            if status == "completed" and (frame_count != HORIZON or action_count != HORIZON):
                raise StreamProtocolError("completed cell must have exactly 70 committed generated frames and actions")
            public_status = "completed" if status == "completed" else "failed"
            error = record.get("error") if status != "completed" else None
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE comparison_attempts SET status = ?, terminal_received = 1, finished_at = ? WHERE id = ?", (status, now, attempt_id))
            connection.execute("UPDATE comparison_cells SET status = ?, terminal_received = 1, error_json = ?, updated_at = ? WHERE comparison_id = ? AND cell_id = ?", (public_status, canonical_json(error) if isinstance(error, Mapping) else None, now, comparison_id, cell_id))
            self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, "terminal", {"status": public_status, "frame_count": frame_count, "action_count": action_count, "error": error})
            self._refresh_comparison_terminal(connection, comparison_id)
            connection.commit()
            return
        count_row = connection.execute("SELECT frame_count, session_id FROM manual_commands WHERE id = ?", (manual_command_id,)).fetchone()
        if count_row is None:
            raise StreamProtocolError("manual command no longer exists")
        frame_count = int(count_row["frame_count"])
        reported_frames = record.get("frame_count")
        if isinstance(reported_frames, bool) or not isinstance(reported_frames, int) or reported_frames != frame_count:
            raise StreamProtocolError("terminal frame count does not match committed manual receipts")
        if status == "completed" and frame_count != 16:
            raise StreamProtocolError("completed manual command must have 16 post-conditioning frames")
        public_status = "completed" if status == "completed" else "failed"
        error = record.get("error") if status != "completed" else None
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE comparison_attempts SET status = ?, terminal_received = 1, finished_at = ? WHERE id = ?", (status, now, attempt_id))
        connection.execute("UPDATE manual_commands SET status = ?, error_json = ?, updated_at = ? WHERE id = ?", (public_status, canonical_json(error) if isinstance(error, Mapping) else None, now, manual_command_id))
        next_source = None
        if status == "completed":
            # The next manual request must condition on the final *committed*
            # generated frame, not the branch source selected when fullscreen
            # opened.  Updating this receipt in the same transaction makes the
            # browser display and the next Chain payload agree after a restart.
            last = connection.execute(
                "SELECT event_id, artifact_path, png_sha256, state_json FROM manual_frames WHERE command_id = ? ORDER BY frame_index DESC LIMIT 1",
                (manual_command_id,),
            ).fetchone()
            if last is None:
                raise StreamProtocolError("completed manual command lacks its final committed frame")
            connection.execute(
                "UPDATE manual_sessions SET source_event_id = ?, source_artifact_path = ?, source_png_sha256 = ?, source_state_json = ?, status = 'ready', active_command_id = NULL, updated_at = ? WHERE id = ?",
                (last["event_id"], last["artifact_path"], last["png_sha256"], last["state_json"], now, count_row["session_id"]),
            )
            next_source = {"event_id": last["event_id"], "url": "/api/artifacts/" + last["artifact_path"], "sha256": last["png_sha256"], "state": json.loads(last["state_json"])}
        else:
            connection.execute("UPDATE manual_sessions SET status = 'ready', active_command_id = NULL, updated_at = ? WHERE id = ?", (now, count_row["session_id"]))
        self._append_event(connection, comparison_id, cell_id, attempt_id, event_id, "manual_terminal", {"session_id": count_row["session_id"], "command_id": manual_command_id, "status": public_status, "frame_count": frame_count, "next_source": next_source, "error": error})
        connection.commit()

    def _refresh_comparison_terminal(self, connection: sqlite3.Connection, comparison_id: str) -> None:
        rows = connection.execute("SELECT status FROM comparison_cells WHERE comparison_id = ?", (comparison_id,)).fetchall()
        states = {str(row["status"]) for row in rows}
        if states & {"queued", "running"}:
            return
        status = "completed" if states == {"completed"} and len(rows) == 12 else ("interrupted" if "interrupted" in states else "failed")
        now = utc_now()
        connection.execute("UPDATE comparisons SET status = ?, terminal_at = ?, updated_at = ? WHERE id = ?", (status, now, now, comparison_id))
        self._append_event(connection, comparison_id, None, "local:%s" % comparison_id, "comparison-terminal", "comparison_terminal", {"status": status})

    def _mark_attempt_ambiguous(self, comparison_id: str, cell_id: str, attempt_id: Optional[str], error: Exception, *, manual_command_id: Optional[str] = None) -> None:
        if not attempt_id:
            return
        message = str(error)[:1000] or type(error).__name__
        payload = {"kind": type(error).__name__, "message": message, "automatic_retry_allowed": False, "manual_reconciliation_required": True}
        now = utc_now()
        with self._lock, self._connect() as connection:
            attempt = connection.execute("SELECT terminal_received FROM comparison_attempts WHERE id = ?", (attempt_id,)).fetchone()
            # A valid terminal receipt is authoritative.  Extra garbage after
            # it is rejected by the parser but cannot retroactively erase a
            # completed, already durable local result.
            if attempt is None or bool(attempt["terminal_received"]):
                return
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE comparison_attempts SET status = 'ambiguous', finished_at = ?, error_json = ? WHERE id = ? AND terminal_received = 0", (now, canonical_json(payload), attempt_id))
            if manual_command_id is None:
                connection.execute("UPDATE comparison_cells SET status = 'interrupted', error_json = ?, updated_at = ? WHERE comparison_id = ? AND cell_id = ? AND terminal_received = 0", (canonical_json(payload), now, comparison_id, cell_id))
                self._append_event(connection, comparison_id, cell_id, attempt_id, "local-interrupted", "cell_status", {"status": "interrupted", "error": payload})
                self._refresh_comparison_terminal(connection, comparison_id)
            else:
                row = connection.execute("SELECT session_id FROM manual_commands WHERE id = ?", (manual_command_id,)).fetchone()
                connection.execute("UPDATE manual_commands SET status = 'ambiguous', error_json = ?, updated_at = ? WHERE id = ?", (canonical_json(payload), now, manual_command_id))
                if row is not None:
                    connection.execute("UPDATE manual_sessions SET status = 'ambiguous', active_command_id = NULL, updated_at = ? WHERE id = ?", (now, row["session_id"]))
                    self._append_event(connection, comparison_id, cell_id, attempt_id, "local-manual-interrupted", "manual_terminal", {"session_id": row["session_id"], "command_id": manual_command_id, "status": "ambiguous", "frame_count": self._manual_frame_count(connection, manual_command_id), "error": payload})
            connection.commit()

    @staticmethod
    def _manual_frame_count(connection: sqlite3.Connection, command_id: str) -> int:
        row = connection.execute("SELECT frame_count FROM manual_commands WHERE id = ?", (command_id,)).fetchone()
        return int(row["frame_count"]) if row is not None else 0

    def _append_event(self, connection: sqlite3.Connection, comparison_id: str, cell_id: Optional[str], attempt_id: str, event_id: str, event_type: str, payload: Mapping[str, Any]) -> None:
        encoded = canonical_json(dict(payload))
        connection.execute("INSERT INTO comparison_events(comparison_id, cell_id, attempt_id, event_id, type, payload_json, payload_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (comparison_id, cell_id, attempt_id, event_id, event_type, encoded, _sha256_bytes(encoded.encode("utf-8")), utc_now()))

    @staticmethod
    def _public_record(record: Mapping[str, Any]) -> Dict[str, Any]:
        # Never journal Base64 again: it is already persisted in a PNG artifact
        # for frame records, and non-frame records must stay small enough for SSE.
        return {key: value for key, value in record.items() if key not in {"png_base64"}}

    def get_comparison(self, comparison_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            return self._detail_in_connection(connection, comparison_id)

    def _detail_in_connection(self, connection: sqlite3.Connection, comparison_id: str, *, presentation_view: bool = False) -> Dict[str, Any]:
        comparison = connection.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
        if comparison is None:
            raise KeyError(comparison_id)
        config = json.loads(str(comparison["config_json"]))
        cells = []
        for row in connection.execute("SELECT * FROM comparison_cells WHERE comparison_id = ? ORDER BY seed, CASE policy WHEN 'OpenVLA' THEN 0 WHEN 'MiniVLA' THEN 1 ELSE 2 END", (comparison_id,)).fetchall():
            frames = []
            if row["attempt_id"]:
                for frame in connection.execute("SELECT * FROM comparison_frames WHERE attempt_id = ? ORDER BY frame_index", (row["attempt_id"],)).fetchall():
                    frames.append({"event_id": frame["event_id"], "segment_id": frame["segment_id"], "frame_index": int(frame["frame_index"]), "url": "/api/artifacts/" + frame["artifact_path"], "sha256": frame["png_sha256"], "action": json.loads(frame["action_json"]), "state": json.loads(frame["state_json"]), "created_at": frame["created_at"]})
            execution_origin = str(row["origin"])
            cells.append({"id": row["cell_id"], "policy": row["policy"], "seed": int(row["seed"]), "origin": "precomputed" if presentation_view else execution_origin, "execution_origin": execution_origin, "status": row["status"], "attempt_id": row["attempt_id"], "action_count": int(row["action_count"]), "frame_count": int(row["frame_count"]), "terminal_received": bool(row["terminal_received"]), "frames": frames, "latest_frame": frames[-1] if frames else None, "error": json.loads(row["error_json"]) if row["error_json"] else None})
        reservation = json.loads(str(comparison["reservation_json"]))
        execution_origin = str(comparison["origin"])
        return {"schema": COMPARISON_SCHEMA, "id": comparison["id"], "origin": "precomputed" if presentation_view else execution_origin, "execution_origin": execution_origin, "status": comparison["status"], "created_at": comparison["created_at"], "updated_at": comparison["updated_at"], "terminal_at": comparison["terminal_at"], "task": config["task"], "task_instruction": config["task_instruction"], "horizon": HORIZON, "seeds": list(SEEDS), "policies": list(POLICIES), "start": _public_start(config["start"]), "world": config["world"], "manifest_sha256": comparison["manifest_sha256"], "reservation": {"reserved_usd": comparison["reservation_usd"], "breakdown": reservation["breakdown"], "cap_usd": reservation["cap_usd"]}, "cells": cells, "scores": None, "scored": False, "claim_tier": "preview", "presentation": bool(comparison["presentation"]), "error": json.loads(comparison["error_json"]) if comparison["error_json"] else None}

    def presentation(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT id FROM comparisons WHERE presentation = 1 AND status = 'completed' ORDER BY updated_at DESC LIMIT 1").fetchone()
            if row is None:
                return {"schema": COMPARISON_SCHEMA, "status": "unavailable", "presentation": None, "reason": "No explicitly promoted completed matched comparison is saved locally.", "task": TASK, "task_instruction": TASK_INSTRUCTION, "scored": False, "claim_tier": "preview"}
            return {"schema": COMPARISON_SCHEMA, "status": "available", "presentation": self._detail_in_connection(connection, str(row["id"]), presentation_view=True), "task": TASK, "task_instruction": TASK_INSTRUCTION, "scored": False, "claim_tier": "preview"}

    def promote(self, comparison_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            comparison = connection.execute("SELECT status FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
            if comparison is None:
                raise KeyError(comparison_id)
            if comparison["status"] != "completed":
                raise ComparisonConflict("only a completed 12-cell matched set can be promoted")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE comparisons SET presentation = 0 WHERE presentation = 1")
            connection.execute("UPDATE comparisons SET presentation = 1, updated_at = ? WHERE id = ?", (utc_now(), comparison_id))
            self._append_event(connection, comparison_id, None, "local:%s" % comparison_id, "presentation-promoted", "presentation_promoted", {"presentation": True})
            connection.commit()
            return self._detail_in_connection(connection, comparison_id, presentation_view=True)

    def events_after(self, comparison_id: str, sequence: int) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM comparisons WHERE id = ?", (comparison_id,)).fetchone() is None:
                raise KeyError(comparison_id)
            origin_row = connection.execute("SELECT origin FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
            rows = connection.execute("SELECT * FROM comparison_events WHERE comparison_id = ? AND sequence > ? ORDER BY sequence", (comparison_id, max(0, sequence))).fetchall()
        return [{"sequence": int(row["sequence"]), "type": row["type"], "origin": str(origin_row["origin"]), "comparison_id": comparison_id, "cell_id": row["cell_id"], "attempt_id": row["attempt_id"], "payload": json.loads(row["payload_json"])} for row in rows]

    def create_manual_session(self, comparison_id: str, cell_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            cell = connection.execute("SELECT * FROM comparison_cells WHERE comparison_id = ? AND cell_id = ?", (comparison_id, cell_id)).fetchone()
            if cell is None:
                raise KeyError(cell_id)
            if not cell["attempt_id"] or not cell["latest_event_id"]:
                raise ComparisonConflict("the selected cell has no committed generated frame to freeze")
            frame = connection.execute("SELECT * FROM comparison_frames WHERE attempt_id = ? AND event_id = ?", (cell["attempt_id"], cell["latest_event_id"])).fetchone()
            if frame is None:
                raise ComparisonConflict("the latest committed frame receipt is missing")
            session_id = uuid.uuid4().hex
            now = utc_now()
            connection.execute("INSERT INTO manual_sessions(id, comparison_id, cell_id, initial_source_event_id, initial_source_artifact_path, initial_source_png_sha256, initial_source_state_json, source_event_id, source_artifact_path, source_png_sha256, source_state_json, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)", (session_id, comparison_id, cell_id, frame["event_id"], frame["artifact_path"], frame["png_sha256"], frame["state_json"], frame["event_id"], frame["artifact_path"], frame["png_sha256"], frame["state_json"], now, now))
            self._append_event(connection, comparison_id, cell_id, "local:%s" % session_id, "manual-session-created", "manual_session_created", {"session_id": session_id, "source_event_id": frame["event_id"]})
            return self._manual_session_in_connection(connection, session_id)

    def get_manual_session(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            return self._manual_session_in_connection(connection, session_id)

    def _manual_session_in_connection(self, connection: sqlite3.Connection, session_id: str) -> Dict[str, Any]:
        session = connection.execute("SELECT * FROM manual_sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise KeyError(session_id)
        commands = []
        for command in connection.execute("SELECT * FROM manual_commands WHERE session_id = ? ORDER BY created_at", (session_id,)).fetchall():
            frames = [{"event_id": row["event_id"], "frame_index": int(row["frame_index"]), "url": "/api/artifacts/" + row["artifact_path"], "sha256": row["png_sha256"], "state": json.loads(row["state_json"]), "created_at": row["created_at"]} for row in connection.execute("SELECT * FROM manual_frames WHERE command_id = ? ORDER BY frame_index", (command["id"],)).fetchall()]
            commands.append({"id": command["id"], "status": command["status"], "direction": command["direction"], "action_rows": 16, "structural_frames": 17, "post_conditioning_frames": 16, "frame_count": int(command["frame_count"]), "frames": frames, "error": json.loads(command["error_json"]) if command["error_json"] else None})
        source = {"event_id": session["source_event_id"], "url": "/api/artifacts/" + session["source_artifact_path"], "sha256": session["source_png_sha256"], "state": json.loads(session["source_state_json"])}
        branch_origin = None
        if session["initial_source_event_id"]:
            branch_origin = {"event_id": session["initial_source_event_id"], "url": "/api/artifacts/" + session["initial_source_artifact_path"], "sha256": session["initial_source_png_sha256"], "state": json.loads(session["initial_source_state_json"])}
        return {"schema": COMPARISON_SCHEMA, "id": session["id"], "status": session["status"], "comparison_id": session["comparison_id"], "cell_id": session["cell_id"], "task": TASK, "task_instruction": TASK_INSTRUCTION, "mode": "image_conditioned_branch", "hidden_state_restored": False, "branch_origin": branch_origin, "source": source, "active_command": session["active_command_id"], "commands": commands, "scored": False, "claim_tier": "preview"}

    def create_manual_command(self, session_id: str, direction: str, idempotency_key: str) -> Tuple[Dict[str, Any], bool]:
        if direction not in _DIRECTIONS:
            raise ComparisonConflict("unknown manual direction")
        if not _IDENTIFIER.fullmatch(idempotency_key):
            raise ComparisonConflict("idempotency_key must be a bounded identifier")
        with self._lock, self._connect() as connection:
            session = connection.execute("SELECT * FROM manual_sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                raise KeyError(session_id)
            existing = connection.execute("SELECT id, direction FROM manual_commands WHERE session_id = ? AND idempotency_key = ?", (session_id, idempotency_key)).fetchone()
            if existing is not None:
                if existing["direction"] != direction:
                    raise ComparisonConflict("manual idempotency key is already bound to another direction")
                return self._manual_command_in_connection(connection, str(existing["id"])), True
            if session["status"] != "ready" or session["active_command_id"] is not None:
                raise ComparisonConflict("manual session already has a generation in flight or is ambiguous")
            if self._has_active_comparison(connection):
                raise ComparisonConflict("matched comparison collection is active; manual world generation is mutually exclusive")
            comparison = connection.execute("SELECT config_json FROM comparisons WHERE id = ?", (session["comparison_id"],)).fetchone()
            if comparison is None:
                raise ComparisonConflict("manual session comparison no longer exists")
            config = json.loads(str(comparison["config_json"]))
            # Until a smaller measured manual bound is frozen, reserve the
            # full controller+world residency bound per command. Conservative,
            # never a made-up per-action cost or an unbounded interactive loop.
            manual_reservation = sum(float(item["reserved_usd"]) for item in config["pricing"]["breakdown"] if item["worker"] in {"controller", "world"})
            if manual_reservation <= 0 or self._active_reservations() + manual_reservation > self.budget_cap_usd + 1e-9:
                raise ComparisonNotReady("manual command exceeds the remaining reserved budget")
            command_id = uuid.uuid4().hex
            attempt_id = uuid.uuid4().hex
            actions = _constant_chunk(direction, float(config["world"]["held_gripper_action"]))
            identities = self._cell_identities(config, str(connection.execute("SELECT policy FROM comparison_cells WHERE comparison_id = ? AND cell_id = ?", (session["comparison_id"], session["cell_id"])).fetchone()["policy"]))
            source_path = (self.root / str(session["source_artifact_path"])).resolve()
            if self.root not in source_path.parents or not source_path.is_file():
                raise ComparisonConflict("frozen manual source artifact is no longer available locally")
            source_png = source_path.read_bytes()
            if _sha256_bytes(source_png) != session["source_png_sha256"]:
                raise ComparisonConflict("frozen manual source artifact no longer matches its receipt")
            dispatch = {"schema": STREAM_SCHEMA, "kind": "manual_world_segment", "mode": "manual", "comparison_id": session["comparison_id"], "attempt_id": attempt_id, "cell_id": session["cell_id"], "session_id": session_id, "command_id": command_id, "direction": direction, "task": TASK, "task_instruction": TASK_INSTRUCTION, "world": config["world"], "actions": actions, "source": {"event_id": session["source_event_id"], "png_base64": base64.b64encode(source_png).decode("ascii"), "png_sha256": session["source_png_sha256"], "state": json.loads(session["source_state_json"])}, "identity_hashes": identities, "claim_tier": "preview", "scored": False}
            payload = {"request": dispatch}
            now = utc_now()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO comparison_attempts(id, comparison_id, cell_id, kind, status, request_json, request_sha256, expected_identities_json, created_at, started_at) VALUES (?, ?, ?, 'manual', 'running', ?, ?, ?, ?, ?)", (attempt_id, session["comparison_id"], session["cell_id"], canonical_json(payload), _sha256_json(payload), canonical_json(identities), now, now))
            connection.execute("INSERT INTO manual_commands(id, session_id, idempotency_key, direction, action_json, attempt_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)", (command_id, session_id, idempotency_key, direction, canonical_json(actions), attempt_id, now, now))
            connection.execute("UPDATE manual_commands SET reservation_usd = ? WHERE id = ?", (manual_reservation, command_id))
            connection.execute("UPDATE manual_sessions SET status = 'running', active_command_id = ?, updated_at = ? WHERE id = ?", (command_id, now, session_id))
            self._append_event(connection, session["comparison_id"], session["cell_id"], attempt_id, "manual-command-started", "manual_command_started", {"session_id": session_id, "command_id": command_id, "direction": direction, "action_rows": 16, "structural_frames": 17, "post_conditioning_frames": 16})
            connection.commit()
            result = self._manual_command_in_connection(connection, command_id)
            future = self._pool.submit(self._receive_stream, comparison_id=session["comparison_id"], cell_id=session["cell_id"], attempt_id=attempt_id, payload=payload, chain=config["chain"], manual_command_id=command_id)
            self._futures["manual:%s" % command_id] = future
            return result, False

    def _manual_command_in_connection(self, connection: sqlite3.Connection, command_id: str) -> Dict[str, Any]:
        row = connection.execute("SELECT * FROM manual_commands WHERE id = ?", (command_id,)).fetchone()
        if row is None:
            raise KeyError(command_id)
        return {"id": row["id"], "status": row["status"], "direction": row["direction"], "task": TASK, "task_instruction": TASK_INSTRUCTION, "action_rows": 16, "structural_frames": 17, "post_conditioning_frames": 16, "frame_count": int(row["frame_count"]), "segment": [], "error": json.loads(row["error_json"]) if row["error_json"] else None}

    def feed_ndjson_for_test(self, comparison_id: str, cell_id: str, attempt_id: str, chunks: Iterable[bytes], *, manual_command_id: Optional[str] = None) -> None:
        """Test-only fixture seam; it is intentionally unreachable from HTTP."""
        if not self.test_mode:
            raise RuntimeError("NDJSON fixture feeding is allowed only for tests")
        terminal = False
        for record in self._decode_ndjson(chunks):
            terminal = self._ingest_record(comparison_id, cell_id, attempt_id, record, manual_command_id) or terminal
        if not terminal:
            self._mark_attempt_ambiguous(comparison_id, cell_id, attempt_id, ComparisonTransportError("fixture stream had no terminal record"), manual_command_id=manual_command_id)


class QuoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateComparisonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class ManualSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cell_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class ManualCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str = Field(pattern="^(up|down|left|right|forward|back)$")
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


def register_comparison_routes(app: FastAPI, service: ComparisonService) -> None:
    """Register the comparison surface without modifying legacy run routes."""

    @app.get("/api/comparisons/readiness")
    def comparison_readiness() -> Dict[str, Any]:
        return service.readiness().payload()

    @app.get("/api/comparisons/presentation")
    def comparison_presentation() -> Dict[str, Any]:
        return service.presentation()

    @app.post("/api/comparisons/quote", status_code=201)
    def comparison_quote(_: QuoteInput) -> Dict[str, Any]:
        try:
            quote = service.create_quote()
        except ComparisonNotReady as error:
            raise HTTPException(409, {"reason": str(error), "readiness": service.readiness().payload()})
        return {"schema": COMPARISON_SCHEMA, "quote": quote, "readiness": service.readiness().payload(), "task": TASK, "task_instruction": TASK_INSTRUCTION, "scored": False, "claim_tier": "preview"}

    @app.post("/api/comparisons", status_code=202)
    def comparison_create(body: CreateComparisonInput) -> Dict[str, Any]:
        try:
            comparison, idempotent = service.create_comparison(body.quote_id, body.idempotency_key)
        except ComparisonNotReady as error:
            raise HTTPException(409, {"reason": str(error), "readiness": service.readiness().payload()})
        except ComparisonConflict as error:
            raise HTTPException(409, {"reason": str(error)})
        return {"schema": COMPARISON_SCHEMA, "comparison": comparison, "idempotent": idempotent, "task": TASK, "task_instruction": TASK_INSTRUCTION, "scored": False, "claim_tier": "preview"}

    @app.get("/api/comparisons/{comparison_id}")
    def comparison_detail(comparison_id: str) -> Dict[str, Any]:
        try:
            return service.get_comparison(comparison_id)
        except KeyError:
            raise HTTPException(404, "comparison not found")

    @app.get("/api/comparisons/{comparison_id}/events")
    async def comparison_events(comparison_id: str, request: Request, after: int = 0) -> StreamingResponse:
        try:
            service.get_comparison(comparison_id)
        except KeyError:
            raise HTTPException(404, "comparison not found")

        async def stream() -> Iterator[str]:
            cursor = max(0, after)
            while not await request.is_disconnected():
                events = service.events_after(comparison_id, cursor)
                for event in events:
                    cursor = max(cursor, int(event["sequence"]))
                    yield "event: comparison\ndata: " + json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n\n"
                detail = service.get_comparison(comparison_id)
                if detail["status"] in {"completed", "failed", "interrupted", "ambiguous"}:
                    yield "event: complete\ndata: " + json.dumps({"id": comparison_id, "status": detail["status"]}) + "\n\n"
                    return
                await _sleep_async(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/comparisons/{comparison_id}/manual-sessions", status_code=201)
    def manual_session(comparison_id: str, body: ManualSessionInput) -> Dict[str, Any]:
        try:
            return service.create_manual_session(comparison_id, body.cell_id)
        except KeyError:
            raise HTTPException(404, "comparison or cell not found")
        except ComparisonConflict as error:
            raise HTTPException(409, {"reason": str(error)})

    @app.get("/api/manual-sessions/{session_id}")
    def manual_session_detail(session_id: str) -> Dict[str, Any]:
        try:
            return service.get_manual_session(session_id)
        except KeyError:
            raise HTTPException(404, "manual session not found")

    @app.post("/api/manual-sessions/{session_id}/commands", status_code=202)
    def manual_command(session_id: str, body: ManualCommandInput) -> Dict[str, Any]:
        try:
            command, idempotent = service.create_manual_command(session_id, body.direction, body.idempotency_key)
        except KeyError:
            raise HTTPException(404, "manual session not found")
        except ComparisonConflict as error:
            raise HTTPException(409, {"reason": str(error)})
        return {"schema": COMPARISON_SCHEMA, "command": command, "idempotent": idempotent, "task": TASK, "task_instruction": TASK_INSTRUCTION, "scored": False, "claim_tier": "preview"}

    @app.post("/api/comparisons/{comparison_id}/promote")
    def comparison_promote(comparison_id: str) -> Dict[str, Any]:
        try:
            return service.promote(comparison_id)
        except KeyError:
            raise HTTPException(404, "comparison not found")
        except ComparisonConflict as error:
            raise HTTPException(409, {"reason": str(error)})


async def _sleep_async(seconds: float) -> None:
    # Avoid adding an asyncio import at module top solely for a one-line SSE
    # poll; FastAPI already runs this async generator on the event loop.
    import asyncio

    await asyncio.sleep(seconds)
