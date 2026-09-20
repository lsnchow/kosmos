"""Durable, local control plane for the one-step live demonstration.

This module deliberately has no policy or simulator implementation.  It talks
to the *local* HTTP tunnel operated by the demo worker, persists every input
and returned frame, and makes the experimental state representation explicit.
It is not part of the scored evaluation pipeline and it never manufactures a
frame if the worker is absent or a response is ambiguous.
"""

from __future__ import annotations

import base64
import asyncio
from io import BytesIO
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Request as ApiRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


STATE_MODE = "experimental_reencoded_rgb_stateless"
MAX_SESSIONS = 128
MAX_SESSION_STEPS = 32
MAX_PNG_BYTES = 12 * 1024 * 1024
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DIRECTIONS = {
    # The worker currently exposes only these native manual mappings.  ``backward``
    # remains an API spelling alias and is normalized to the worker's ``back``.
    "up", "down", "left", "right", "forward", "back", "backward",
}


class LiveDemoFailure(Exception):
    """A safe, public-facing local-demo error."""

    def __init__(self, message: str, *, status_code: int = 409, ambiguous: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.ambiguous = ambiguous


class WorkerTransport(Protocol):
    """Small seam for deterministic tests; production uses ``HttpWorkerTransport``."""

    def health(self) -> Mapping[str, Any]: ...

    def fixture(self) -> Mapping[str, Any]: ...

    def step(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class HttpWorkerTransport:
    """Dependency-free authenticated client for the SSH-tunneled worker.

    The endpoint is intentionally limited to loopback.  The worker runs on the
    allocated machine, but the browser/server only ever contacts its local SSH
    tunnel.  Tokens are read only at call time and are never included in an
    exception, response, or log record.
    """

    def __init__(self, url: Optional[str] = None, token_file: Optional[str] = None, timeout_seconds: float = 25.0):
        self.url = (url if url is not None else os.environ.get("PLUMB_LIVE_DEMO_URL", "")).strip().rstrip("/")
        self.token_file = token_file if token_file is not None else os.environ.get("PLUMB_LIVE_DEMO_TOKEN_FILE", "")
        self.timeout_seconds = timeout_seconds

    def configuration_error(self) -> Optional[str]:
        if not self.url:
            return "PLUMB_LIVE_DEMO_URL is not configured"
        parsed = urlparse(self.url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return "PLUMB_LIVE_DEMO_URL must be an http loopback SSH-tunnel URL"
        if not self.token_file:
            return "PLUMB_LIVE_DEMO_TOKEN_FILE is not configured"
        try:
            value = Path(self.token_file).read_text(encoding="utf-8")
        except OSError:
            return "PLUMB_LIVE_DEMO_TOKEN_FILE cannot be read"
        if not value.strip() or len(value) > 4096:
            return "PLUMB_LIVE_DEMO_TOKEN_FILE is empty or invalid"
        return None

    def _token(self) -> str:
        error = self.configuration_error()
        if error:
            raise LiveDemoFailure(error, status_code=503)
        # ``configuration_error`` only establishes that the short, local file is
        # readable.  Re-read it immediately so rotation is respected.
        try:
            return Path(self.token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LiveDemoFailure("PLUMB_LIVE_DEMO_TOKEN_FILE cannot be read", status_code=503) from exc

    def _request(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        token = self._token()
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.url + path,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer " + token,
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: URL is loopback-validated
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            # The worker definitively received and rejected this request.  It is
            # not retried, but it is not an unknown simulator state either.
            raise LiveDemoFailure("live demo worker rejected the request (%s)" % exc.code, status_code=502) from exc
        except (URLError, TimeoutError, OSError) as exc:
            # A timeout / broken tunnel can occur after the remote simulator has
            # advanced.  Callers turn this into ``blocked`` and never retry it.
            raise LiveDemoFailure("live demo worker connection outcome is ambiguous", status_code=503, ambiguous=True) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise LiveDemoFailure("live demo worker response exceeds the size limit", status_code=502, ambiguous=True)
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiveDemoFailure("live demo worker returned invalid JSON", status_code=502, ambiguous=True) from exc
        if not isinstance(decoded, Mapping):
            raise LiveDemoFailure("live demo worker returned a non-object response", status_code=502, ambiguous=True)
        return decoded

    def health(self) -> Mapping[str, Any]:
        return self._request("GET", "/health")

    def fixture(self) -> Mapping[str, Any]:
        return self._request("GET", "/fixture")

    def step(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request("POST", "/step", payload)

    def cosmos(self, payload: Mapping[str, Any]):
        request = Request(self.url + "/cosmos", data=_json(payload).encode(), method="POST", headers={"Authorization": "Bearer " + self._token(), "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=600) as response:
                while True:
                    line = response.readline(MAX_RESPONSE_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_RESPONSE_BYTES:
                        raise LiveDemoFailure("Cosmos event exceeds size limit", ambiguous=True)
                    yield json.loads(line)
        except (OSError, ValueError) as error:
            raise LiveDemoFailure("Cosmos generation stream failed; no automatic retry was issued", ambiguous=True) from error


class CreateSessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, max_length=160)
    prompt: Optional[str] = Field(default=None, min_length=1, max_length=1000)
    mode: str = Field(default="policy", pattern="^(policy|manual)$")
    steps: int = Field(default=4, ge=1, le=MAX_SESSION_STEPS)
    source_video_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    starting_scene: Optional[str] = Field(default=None, pattern="^(drawer|pot)$")
    seed: Optional[int] = Field(default=None, ge=0, le=2147483647)
    auto_assess: bool = False
    world_model: Optional[str] = Field(default=None, pattern="^cosmos$")

    @model_validator(mode="after")
    def source_requires_manual_mode(self) -> "CreateSessionInput":
        if self.source_video_id is not None and self.mode != "manual":
            raise ValueError("source_video_id is supported only for a manual fresh-image-reconditioned session")
        expected = {"drawer": "Close the drawer", "pot": "Put the pot to the left of the purple item."}
        if self.starting_scene and (self.source_video_id or self.mode != "policy" or self.prompt != expected[self.starting_scene]):
            raise ValueError("Starting scene must match its supported task without a video branch")
        if self.starting_scene == "pot" and self.world_model != "cosmos":
            raise ValueError("Pot fixture requires Cosmos")
        if self.auto_assess and (self.starting_scene not in expected or self.steps not in (16,32)):
            raise ValueError("automatic assessment requires a supported rollout")
        if self.world_model and (self.starting_scene not in expected or self.steps not in (16,32)):
            raise ValueError("Cosmos requires 16 or 32 actions")
        if self.steps == 32 and (self.world_model != "cosmos" or self.starting_scene != "pot"):
            raise ValueError("Two-chunk continuation currently supports the Cosmos pot fixture only")
        return self


class CommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str = Field(pattern="^(up|down|left|right|forward|back|backward)$")
    steps: int = Field(default=1, ge=1, le=4)
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


def _now() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(raw: Any) -> Dict[str, Any]:
    return dict(raw) if isinstance(raw, Mapping) else {}


def _loads(raw: Optional[str], default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _safe_string(value: Any, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True)
    return value.strip()


def _metadata(value: Any, field: str, maximum: int = 8192) -> Any:
    """Accept the worker's structured model/revision provenance verbatim.

    The worker identifies a composed OpenVLA→IRASim execution with small
    mappings, not invented display strings.  Restrict it to JSON-compatible,
    bounded data before persistence.
    """

    if isinstance(value, str):
        return _safe_string(value, field, maximum)
    if not isinstance(value, Mapping):
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True)
    try:
        encoded = _json(dict(value))
    except (TypeError, ValueError) as exc:
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True) from exc
    if len(encoded) > maximum:
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True)
    return dict(value)


def _stored_metadata(value: Optional[str]) -> Any:
    if value is None:
        return None
    decoded = _loads(value, None)
    return decoded if isinstance(decoded, (dict, list)) else value


def _png_from_base64(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_PNG_BYTES * 4 // 3 + 128):
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True)
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise LiveDemoFailure("live demo worker returned invalid %s" % field, status_code=502, ambiguous=True) from exc
    if not raw.startswith(_PNG_SIGNATURE) or len(raw) > MAX_PNG_BYTES:
        raise LiveDemoFailure("live demo worker returned a non-PNG %s" % field, status_code=502, ambiguous=True)
    return raw


def _decoded_rgb_sha256(raw: bytes) -> Optional[str]:
    """Match the worker's hash of RGB HWC bytes when Pillow is installed.

    PNG transport bytes are intentionally allowed to differ after re-encoding;
    the comparison is over decoded RGB pixels and shape.  Pillow is optional
    for the API package, so an unavailable decoder is reported in provenance
    rather than changing a live worker result into a fake byte comparison.
    """

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    try:
        with Image.open(BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            value = (repr((rgb.height, rgb.width, 3)) + ":").encode("utf-8") + rgb.tobytes()
        return "sha256:" + hashlib.sha256(value).hexdigest()
    except Exception as exc:
        raise LiveDemoFailure("could not decode a returned PNG for RGB hash verification", status_code=502, ambiguous=True) from exc


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite")
    return number


class LiveDemoService:
    """SQLite-backed sessions with one serial remote-work queue.

    Each remote call is represented by a durable job before it starts.  A job
    found ``running`` after a process restart is blocked rather than replayed:
    the remote worker may already have stepped IRASim, so a retry would lie
    about the returned trajectory.
    """

    def __init__(self, root: Path, transport: Optional[WorkerTransport] = None):
        self.root = Path(root).resolve()
        self.demo_root = self.root / "live-demo"
        self.frames_root = self.demo_root / "frames"
        self.demo_root.mkdir(parents=True, exist_ok=True)
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.demo_root / "live-demo.sqlite3"
        self.transport: WorkerTransport = transport or HttpWorkerTransport()
        self._lock = threading.RLock()
        self.judge_service = None
        # This single-worker executor is the remote worker lock/queue.  Local
        # SQLite work is brief, while every fixture/step request is serialized.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plumb-live-demo")
        self._scheduled: set[str] = set()
        self._init_schema()
        self._recover_jobs()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    requested_steps INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    current_frame_ordinal INTEGER,
                    source_json TEXT NOT NULL,
                    error TEXT,
                    warnings_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    latest_video_file TEXT
                );
                CREATE TABLE IF NOT EXISTS frames (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    input_sha256 TEXT,
                    conditioning_sha256 TEXT,
                    action_json TEXT,
                    timings_json TEXT,
                    model TEXT,
                    revision TEXT,
                    profile TEXT,
                    warning TEXT,
                    provenance_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS commands (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    request_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(session_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    command_id TEXT REFERENCES commands(id),
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS artifact_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS live_demo_frames_session ON frames(session_id, ordinal);
                CREATE INDEX IF NOT EXISTS live_demo_jobs_state ON jobs(state, created_at);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
            if "current_frame_ordinal" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN current_frame_ordinal INTEGER")

    def _recover_jobs(self) -> None:
        queued: list[str] = []
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _now()
            running = connection.execute("SELECT id, session_id, command_id FROM jobs WHERE state = 'running'").fetchall()
            for job in running:
                message = "local process ended while a remote worker result was outstanding; no retry was issued"
                connection.execute("UPDATE jobs SET state = 'blocked', error = ?, completed_at = ? WHERE id = ?", (message, now, job["id"]))
                if job["command_id"]:
                    connection.execute("UPDATE commands SET state = 'blocked', error = ?, updated_at = ? WHERE id = ?", (message, now, job["command_id"]))
                connection.execute("UPDATE sessions SET state = 'blocked', error = ?, updated_at = ? WHERE id = ?", (message, now, job["session_id"]))
            queued = [row["id"] for row in connection.execute("SELECT id FROM jobs WHERE state = 'queued' ORDER BY created_at").fetchall()]
            connection.commit()
        for job_id in queued:
            self._schedule(job_id)

    def _configured_error(self) -> Optional[str]:
        checker = getattr(self.transport, "configuration_error", None)
        return checker() if callable(checker) else None

    def status(self) -> Dict[str, Any]:
        configuration_error = self._configured_error()
        base = {
            "configured": configuration_error is None,
            "available": False,
            "state_mode": STATE_MODE,
            "manual_directions": ["up", "down", "left", "right", "forward", "back"],
            "qualified": False,
            "scored": False,
        }
        if configuration_error:
            return {**base, "health": "not_configured", "reason": configuration_error}
        try:
            result = _mapping(self.transport.health())
        except LiveDemoFailure as exc:
            return {**base, "health": "unavailable", "reason": str(exc)}
        except Exception:
            return {**base, "health": "unavailable", "reason": "live demo worker health check failed"}
        base["world_model"] = result.get("world_model", "irasim")
        if base["world_model"] == "cosmos":
            base["state_mode"] = "cosmos_full_chunk_supplied_actions"
        ready = result.get("ready") is True or result.get("status") in {"ok", "ready", "available"}
        if not ready:
            return {**base, "health": "unavailable", "reason": "live demo worker health check is not ready"}
        return {**base, "available": True, "health": "available", "reason": None}

    def _require_available(self) -> None:
        status = self.status()
        if not status["available"]:
            raise LiveDemoFailure(str(status.get("reason") or "live demo worker is unavailable"), status_code=503)

    def create(self, body: CreateSessionInput) -> Dict[str, Any]:
        self._require_available()
        if self.status().get("world_model") == "cosmos" and body.world_model != "cosmos":
            raise LiveDemoFailure("Use Generate to start a Cosmos experiment; the earlier interactive worker is offline.")
        # Resolve the catalog identity before queuing anything.  A client never
        # gets to supply a media URL/path, and an unknown selection is refused
        # synchronously rather than becoming an opaque background failure.
        if body.source_video_id is not None:
            self._catalog_source(body.source_video_id)
        with self._lock, self._connection() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
            if count >= MAX_SESSIONS:
                raise LiveDemoFailure("live demo session limit reached", status_code=429)
            session_id = uuid.uuid4().hex
            now = _now()
            prompt = body.prompt.strip() if body.prompt else ""
            title = body.title.strip() if body.title else ("OpenVLA live evaluation" if body.mode == "policy" else "Manual live steering")
            source = {"video_id": body.source_video_id} if body.source_video_id else {}
            if body.starting_scene:
                source = {"kind": "demo_pot_fixture" if body.starting_scene == "pot" else "demo_drawer_fixture", "auto_assess": body.auto_assess, "world_model": body.world_model or "irasim"}
            seed = body.seed if body.seed is not None else int(session_id[:8], 16) & 0x7FFFFFFF
            connection.execute(
                """INSERT INTO sessions (id, title, prompt, mode, requested_steps, seed, state, source_json, error, warnings_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, NULL, '[]', ?, ?)""",
                (session_id, title, prompt, body.mode, body.steps, seed, _json(source), now, now),
            )
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO jobs (id, session_id, command_id, kind, payload_json, state, error, created_at) VALUES (?, ?, NULL, 'initialize', ?, 'queued', NULL, ?)",
                (job_id, session_id, _json({}), now),
            )
        self._schedule(job_id)
        return self.get(session_id)

    def command(self, session_id: str, body: CommandInput) -> Tuple[Dict[str, Any], bool]:
        if body.direction not in _DIRECTIONS:
            raise LiveDemoFailure("unsupported direction", status_code=422)
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                connection.rollback()
                raise LiveDemoFailure("live demo session not found", status_code=404)
            request_id = body.request_id or uuid.uuid4().hex
            existing = connection.execute(
                "SELECT id, direction, steps FROM commands WHERE session_id = ? AND request_id = ?", (session_id, request_id)
            ).fetchone()
            if existing is not None:
                if existing["direction"] != body.direction or int(existing["steps"]) != body.steps:
                    connection.rollback()
                    raise LiveDemoFailure("request_id was already used with a different directional command")
                connection.commit()
                return self.get(session_id), True
            if session["mode"] != "manual":
                connection.rollback()
                raise LiveDemoFailure("directional commands are available only to manual sessions")
            if session["state"] not in {"ready", "completed"}:
                connection.rollback()
                raise LiveDemoFailure("session is not ready for a directional command")
            active = connection.execute(
                "SELECT 1 FROM commands WHERE session_id = ? AND state IN ('queued', 'running') LIMIT 1", (session_id,)
            ).fetchone()
            if active is not None:
                connection.rollback()
                raise LiveDemoFailure("session already has a directional command in flight")
            completed = int(connection.execute("SELECT COUNT(*) FROM frames WHERE session_id = ? AND role = 'predicted'", (session_id,)).fetchone()[0])
            if completed + body.steps > MAX_SESSION_STEPS:
                connection.rollback()
                raise LiveDemoFailure("manual session is bounded to %d live steps" % MAX_SESSION_STEPS, status_code=422)
            command_id, job_id, now = uuid.uuid4().hex, uuid.uuid4().hex, _now()
            connection.execute(
                "INSERT INTO commands (id, session_id, request_id, direction, steps, state, error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'queued', NULL, ?, ?)",
                (command_id, session_id, request_id, body.direction, body.steps, now, now),
            )
            connection.execute(
                "INSERT INTO jobs (id, session_id, command_id, kind, payload_json, state, error, created_at) VALUES (?, ?, ?, 'command', ?, 'queued', NULL, ?)",
                (job_id, session_id, command_id, _json({"direction": body.direction, "steps": body.steps}), now),
            )
            connection.execute("UPDATE sessions SET state = 'queued', error = NULL, updated_at = ? WHERE id = ?", (now, session_id))
            connection.commit()
        self._schedule(job_id)
        return self.get(session_id), False

    def _schedule(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._scheduled:
                return
            self._scheduled.add(job_id)
        future = self._pool.submit(self._run_job, job_id)

        def done(_: Any) -> None:
            with self._lock:
                self._scheduled.discard(job_id)

        future.add_done_callback(done)

    def _run_job(self, job_id: str) -> None:
        job: Optional[sqlite3.Row] = None
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None or job["state"] != "queued":
                connection.rollback()
                return
            now = _now()
            connection.execute("UPDATE jobs SET state = 'running', started_at = ? WHERE id = ?", (now, job_id))
            if job["command_id"]:
                connection.execute("UPDATE commands SET state = 'running', updated_at = ? WHERE id = ?", (now, job["command_id"]))
            connection.execute("UPDATE sessions SET state = ?, error = NULL, updated_at = ? WHERE id = ?", ("initializing" if job["kind"] == "initialize" else "running", now, job["session_id"]))
            connection.commit()
        assert job is not None
        try:
            if job["kind"] == "initialize":
                self._initialize(str(job["session_id"]))
            elif job["kind"] == "command":
                self._execute_command(str(job["session_id"]), _mapping(_loads(job["payload_json"], {})))
            else:
                raise LiveDemoFailure("unknown durable live-demo job", status_code=500)
        except LiveDemoFailure as exc:
            self._finish_job_error(job, str(exc), blocked=exc.ambiguous)
        except Exception:
            # This includes a malformed/unexpected worker response after the
            # request was made, so preserving the current frame is safer than
            # guessing that the remote state did not advance.
            self._finish_job_error(job, "live demo job ended with an unknown outcome", blocked=True)
        else:
            self._finish_job_success(job)

    def _finish_job_success(self, job: sqlite3.Row) -> None:
        with self._lock, self._connection() as connection:
            now = _now()
            connection.execute("UPDATE jobs SET state = 'completed', completed_at = ? WHERE id = ?", (now, job["id"]))
            if job["command_id"]:
                connection.execute("UPDATE commands SET state = 'completed', updated_at = ? WHERE id = ?", (now, job["command_id"]))
            session = connection.execute("SELECT mode FROM sessions WHERE id = ?", (job["session_id"],)).fetchone()
            state = "completed" if session and session["mode"] == "policy" else "ready"
            connection.execute("UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?", (state, now, job["session_id"]))

        session = self._session_row(job["session_id"])
        source = _mapping(_loads(session["source_json"], {}))
        if session["state"] == "completed":
            source["generation_completed_at"] = now
            self._update_session(job["session_id"], source=source)
        if source.get("auto_assess") and session["state"] == "completed":
            try:
                from plumb.demo_judge import DemoJudgmentInput
                if self.judge_service is None or not session["latest_video_file"]:
                    raise LiveDemoFailure("Automatic judge unavailable or generated video not committed")
                result = self.judge_service.submit(DemoJudgmentInput(
                    clip_id="live-demo:" + job["session_id"], profile="semantic_pilot_epoch_02",
                    idempotency_key="auto-judge:" + job["session_id"]))
                source["judgment_id"] = result["id"]
            except Exception as error:
                source["assessment_error"] = str(error)
            self._update_session(job["session_id"], source=source)

    def _finish_job_error(self, job: sqlite3.Row, message: str, *, blocked: bool) -> None:
        state = "blocked" if blocked else "error"
        with self._lock, self._connection() as connection:
            now = _now()
            connection.execute("UPDATE jobs SET state = ?, error = ?, completed_at = ? WHERE id = ?", (state, message, now, job["id"]))
            if job["command_id"]:
                connection.execute("UPDATE commands SET state = ?, error = ?, updated_at = ? WHERE id = ?", (state, message, now, job["command_id"]))
            connection.execute("UPDATE sessions SET state = ?, error = ?, updated_at = ? WHERE id = ?", (state, message, now, job["session_id"]))

    def _initialize(self, session_id: str) -> None:
        session = self._session_row(session_id)
        source = _mapping(_loads(session["source_json"], {}))
        if source.get("kind") == "demo_pot_fixture":
            raw = (self.root / "demo-judge-fixtures/pot-scene.png").read_bytes()
            if _sha(raw) != "sha256:65db577ed68303af3931781428da7d017337b3b7be8c2eaac4fa192b2cb9dd69":
                raise LiveDemoFailure("Pot starting scene hash mismatch")
            self._store_frame(session_id, "fixture", raw, provenance={"kind": "demo_pot_fixture", "source_video_sha256": "sha256:a86cfc81633b216891ca26dc58c72193a979c10ad72f123175fa8d61a67cdaec"})
            self._cosmos_chunk(session_id, raw, source)
            return
        if source.get("kind") == "demo_drawer_fixture":
            image = self.root / "demo-judge-fixtures/scene-reference.png"
            raw = image.read_bytes()
            if _sha(raw) != "sha256:42ad1fe7593823c0aa33888ffd10927c2dde7cce8b65544e48dd9a5754e7f24e":
                raise LiveDemoFailure("Drawer starting scene hash mismatch")
            self._store_frame(session_id, "fixture", raw, provenance={"kind": "demo_drawer_fixture", "sha256": _sha(raw)})
            if source.get("world_model") == "cosmos":
                self._cosmos_chunk(session_id, raw, source)
                return
            self._step_many(session_id, direction="forward", steps=int(session["requested_steps"]), mode="policy")
            return
        if source.get("video_id"):
            raw, binding = self._source_final_frame(str(source["video_id"]))
            self._store_frame(session_id, "source", raw, provenance=binding)
            self._update_session(session_id, prompt=session["prompt"] or "Interact with the scene.", source=binding)
            return
        fixture = _mapping(self.transport.fixture())
        png_value = fixture.get("png_base64", fixture.get("fixture_png_base64", fixture.get("image_base64")))
        raw = _png_from_base64(png_value, "fixture PNG")
        fixture_prompt = _safe_string(fixture.get("prompt"), "fixture prompt")
        prompt = str(session["prompt"]).strip() or fixture_prompt
        self._store_frame(
            session_id,
            "fixture",
            raw,
            provenance={"kind": "vendor_fixture", "fixture_prompt": fixture_prompt, "prompt_source": "request" if session["prompt"] else "fixture"},
        )
        self._update_session(session_id, prompt=prompt)
        if session["mode"] == "policy":
            self._step_many(session_id, direction="forward", steps=int(session["requested_steps"]), mode="policy")

    def _cosmos_chunk(self, session_id: str, image: bytes, source: dict) -> None:
        from deploy.baseten.demo_chain import _forecast_compile, service_dispatch_to_stream_request
        pot = source.get("kind") == "demo_pot_fixture"
        if pot:
            action_bytes = (self.root / "demo-judge-fixtures/pot-actions.json").read_bytes()
            if _sha(action_bytes) != "sha256:5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da":
                raise LiveDemoFailure("Pot action fixture hash mismatch")
            actions = json.loads(action_bytes)
            native = [None] * 16
            detail = {"source": "official_bridge_fixture", "action_sha256": _sha(action_bytes), "compiled_rows": actions}
        else:
            recorded = json.loads((self.root / "demo-judge-fixtures/cosmos-request.json").read_text())["request"]
            manual = service_dispatch_to_stream_request(recorded).manual
            if manual is None or manual.source_png_sha256 != _sha(image):
                raise LiveDemoFailure("Cosmos source/action fixture mismatch")
            native = [[0.0025, 0, 0, 0, 0, 0, manual.held_gripper_action] for _ in range(16)]
            actions, _, detail = _forecast_compile(manual.world, manual.bridge_control_profile_id, manual.state_snapshot, native)
        session = self._session_row(session_id)
        count = 0
        chunks = int(session["requested_steps"]) // 16
        timings = []
        for chunk in range(chunks):
            input_hash = _sha(image)
            payload = {"png_base64": base64.b64encode(image).decode(), "sha256": input_hash, "prompt": session["prompt"], "actions": actions, "seed": session["seed"] + chunk}
            terminal = None
            local_count = 0
            for event in self.transport.cosmos(payload):
                if event.get("kind") == "progress":
                    source["world_progress"] = {"stage": event["stage"], "step": chunk * 30 + event["step"], "total": chunks * 30, "chunk": chunk + 1}
                    self._update_session(session_id, source=source)
                elif event.get("kind") == "frame":
                    count += 1; local_count += 1
                    raw = _png_from_base64(event.get("png_base64"), "Cosmos frame")
                    if local_count > 16 or event.get("index") != local_count or event.get("sha256") != _sha(raw):
                        raise LiveDemoFailure("Cosmos frame order/hash mismatch", ambiguous=True)
                    self._store_frame(session_id, "predicted", raw, action=native[local_count - 1],
                        timings={"world": event["timing"]["world_seconds"] * 1000 if local_count == 1 else None},
                        model="nvidia/Cosmos3-Nano", revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
                        provenance={"kind": "cosmos_chunk", "chunk": chunk + 1, "chunk_input_sha256": input_hash, "policy_calls": 0, "action_source": "recorded_plan_repeated" if chunk else "recorded_fixture" if pot else "supplied_manual_right", "compiler": detail})
                    image = raw
                elif event.get("kind") == "terminal":
                    terminal = event
            if local_count != 16 or not terminal or terminal.get("status") != "completed":
                raise LiveDemoFailure("Cosmos did not complete: " + str((terminal or {}).get("reason", "incomplete stream")), ambiguous=True)
            timings.append(terminal["timing"])
        source["world_timing"] = {"world_seconds": sum(t["world_seconds"] for t in timings), "peak_gpu_bytes": max(t["peak_gpu_bytes"] for t in timings), "chunks": chunks}
        self._update_session(session_id, source=source)
        self._encode_latest_video(session_id)

    def _execute_command(self, session_id: str, payload: Dict[str, Any]) -> None:
        direction = payload.get("direction")
        steps = payload.get("steps")
        if direction not in _DIRECTIONS or not isinstance(steps, int) or not 1 <= steps <= 4:
            raise LiveDemoFailure("corrupt durable directional command", status_code=500)
        self._step_many(session_id, direction=direction, steps=steps, mode="manual")

    def _step_many(self, session_id: str, *, direction: str, steps: int, mode: str) -> None:
        for _ in range(steps):
            session = self._session_row(session_id)
            frame = self._current_frame(session)
            if frame is None:
                raise LiveDemoFailure("session has no verified conditioning frame", status_code=500)
            current = self._read_frame(frame)
            predicted_count = int(self._count_predicted(session_id))
            if predicted_count >= MAX_SESSION_STEPS:
                raise LiveDemoFailure("session live-step bound reached", status_code=422)
            seed = (int(session["seed"]) + predicted_count) & 0x7FFFFFFF
            request = {
                "current_png_base64": base64.b64encode(current).decode("ascii"),
                "prompt": str(session["prompt"]),
                "mode": mode,
                "seed": seed,
            }
            worker_direction = "back" if direction == "backward" else direction
            if mode == "manual":
                request["direction"] = worker_direction
            result = _mapping(self.transport.step(request))
            predicted = _png_from_base64(result.get("predicted_png_base64"), "predicted PNG")
            conditioning = _png_from_base64(result.get("conditioning_png_base64"), "conditioning PNG")
            action = result.get("action7")
            if not isinstance(action, list) or len(action) != 7:
                raise LiveDemoFailure("live demo worker returned invalid action7", status_code=502, ambiguous=True)
            try:
                action = [_number(value) for value in action]
            except (TypeError, ValueError):
                raise LiveDemoFailure("live demo worker returned invalid action7", status_code=502, ambiguous=True)
            timings_raw = result.get("timings_ms")
            if not isinstance(timings_raw, Mapping):
                raise LiveDemoFailure("live demo worker returned invalid timings_ms", status_code=502, ambiguous=True)
            try:
                timings = {
                    str(key): (None if value is None else _number(value))
                    for key, value in timings_raw.items()
                    if isinstance(key, str) and len(key) <= 100
                }
            except (TypeError, ValueError):
                raise LiveDemoFailure("live demo worker returned invalid timings_ms", status_code=502, ambiguous=True)
            if not timings:
                raise LiveDemoFailure("live demo worker returned empty timings_ms", status_code=502, ambiguous=True)
            state_mode = _safe_string(result.get("state_mode"), "state_mode", 100)
            if state_mode != STATE_MODE:
                raise LiveDemoFailure("live demo worker did not declare the expected experimental state mode", status_code=502, ambiguous=True)
            model = _metadata(result.get("model"), "model")
            revision = _metadata(result.get("revision"), "revision")
            profile = result.get("profile") if isinstance(result.get("profile"), str) and len(result["profile"]) <= 300 else None
            warning = result.get("warning") if isinstance(result.get("warning"), str) and len(result["warning"]) <= 1000 else None
            input_decoded_hash = _decoded_rgb_sha256(current)
            conditioning_decoded_hash = _decoded_rgb_sha256(conditioning)
            predicted_decoded_hash = _decoded_rgb_sha256(predicted)
            reported_conditioning_hash = result.get("conditioning_decoded_rgb_sha256")
            reported_predicted_hash = result.get("predicted_decoded_rgb_sha256")
            for field, expected, actual in (
                ("conditioning_decoded_rgb_sha256", reported_conditioning_hash, conditioning_decoded_hash),
                ("predicted_decoded_rgb_sha256", reported_predicted_hash, predicted_decoded_hash),
            ):
                if expected is not None and (not isinstance(expected, str) or (actual is not None and expected != actual)):
                    raise LiveDemoFailure("live demo worker %s did not match decoded RGB pixels" % field, status_code=502, ambiguous=True)
            if input_decoded_hash is not None and conditioning_decoded_hash is not None and input_decoded_hash != conditioning_decoded_hash:
                raise LiveDemoFailure("live demo worker conditioning pixels did not match the submitted frame", status_code=502, ambiguous=True)
            self._store_frame(
                session_id,
                "predicted",
                predicted,
                input_sha256=_sha(current),
                conditioning_sha256=_sha(conditioning),
                action=action,
                timings=timings,
                model=model,
                revision=revision,
                profile=profile,
                warning=warning,
                provenance={
                    "kind": "worker_step",
                    "mode": mode,
                    "direction": worker_direction if mode == "manual" else None,
                    "requested_direction": direction if mode == "manual" and direction != worker_direction else None,
                    "seed": seed,
                    "worker_state_mode": state_mode,
                    "conditioning_returned": True,
                    "exact_input_bytes_reencoded": False,
                    "input_decoded_rgb_sha256": input_decoded_hash,
                    "conditioning_decoded_rgb_sha256": conditioning_decoded_hash,
                    "predicted_decoded_rgb_sha256": predicted_decoded_hash,
                    "worker_conditioning_decoded_rgb_sha256": reported_conditioning_hash,
                    "worker_predicted_decoded_rgb_sha256": reported_predicted_hash,
                },
            )
            if warning:
                self._append_warning(session_id, warning)
        self._encode_latest_video(session_id)

    def _source_final_frame(self, video_id: str) -> Tuple[bytes, Dict[str, Any]]:
        video = self._catalog_source(video_id)
        expected = video.get("sha256")
        video_url = video.get("video_url")
        assert isinstance(expected, str) and isinstance(video_url, str)
        parsed = urlparse(video_url)
        prefix = "/api/artifacts/"
        relative = Path(unquote(parsed.path[len(prefix):]))
        candidate = (self.root / relative).resolve()
        try:
            if _sha(candidate.read_bytes()) != expected:
                raise LiveDemoFailure("selected catalog source video hash no longer matches", status_code=409)
        except OSError as exc:
            raise LiveDemoFailure("selected catalog source video cannot be read", status_code=409) from exc
        temporary = self.demo_root / ("source-" + uuid.uuid4().hex + ".tmp.png")
        try:
            result = subprocess.run(
                # Decode through the clip and retain the last actual frame.
                # Seeking 1ms before container EOF can land after the final
                # frame's presentation timestamp and emit nothing at 5 FPS.
                ["ffmpeg", "-v", "error", "-i", str(candidate), "-map", "0:v:0", "-an",
                 "-fps_mode", "passthrough", "-update", "1", "-y", str(temporary)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=40,
                check=False,
            )
            if result.returncode != 0 or not temporary.is_file():
                raise LiveDemoFailure("could not extract a final frame from the selected catalog video", status_code=409)
            raw = temporary.read_bytes()
            if _sha(candidate.read_bytes()) != expected:
                raise LiveDemoFailure("selected source changed during frame extraction", status_code=409)
        except (OSError, subprocess.SubprocessError) as exc:
            raise LiveDemoFailure("could not extract a final frame from the selected catalog video", status_code=409) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if not raw.startswith(_PNG_SIGNATURE) or len(raw) > MAX_PNG_BYTES:
            raise LiveDemoFailure("selected catalog video did not yield a bounded PNG frame", status_code=409)
        return raw, {
            "video_id": video_id,
            "source_sha256": expected,
            "mode": "fresh_image_reconditioned_not_exact_checkpoint_restore",
            "note": "The saved recording supplied pixels only; this is not restoration of its old simulator checkpoint.",
        }

    def _catalog_source(self, video_id: str) -> Dict[str, Any]:
        """Resolve only a single server-derived catalog object, never a URL."""
        from plumb.world_videos import world_videos_payload

        matches = [item for item in world_videos_payload(self.root).get("videos", []) if _mapping(item).get("id") == video_id]
        if len(matches) != 1:
            raise LiveDemoFailure("selected catalog source video is no longer available", status_code=409)
        video = _mapping(matches[0])
        expected = video.get("sha256")
        video_url = video.get("video_url")
        if not isinstance(expected, str) or not isinstance(video_url, str):
            raise LiveDemoFailure("selected catalog source video has no verified media binding", status_code=409)
        parsed = urlparse(video_url)
        prefix = "/api/artifacts/"
        if parsed.query or parsed.fragment or not parsed.path.startswith(prefix):
            raise LiveDemoFailure("selected catalog source video has an invalid local media binding", status_code=409)
        relative = Path(unquote(parsed.path[len(prefix):]))
        unresolved = self.root / relative
        candidate = unresolved.resolve()
        if self.root not in candidate.parents or candidate.suffix.lower() != ".mp4" or not candidate.is_file() or unresolved.is_symlink():
            raise LiveDemoFailure("selected catalog source video is not a regular local MP4", status_code=409)
        return video

    def _session_row(self, session_id: str) -> sqlite3.Row:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise LiveDemoFailure("live demo session not found", status_code=404)
        return row

    def _count_predicted(self, session_id: str) -> int:
        with self._lock, self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM frames WHERE session_id = ? AND role = 'predicted'", (session_id,)).fetchone()[0])

    def _latest_frame(self, session_id: str) -> Optional[sqlite3.Row]:
        with self._lock, self._connection() as connection:
            return connection.execute("SELECT * FROM frames WHERE session_id = ? ORDER BY ordinal DESC LIMIT 1", (session_id,)).fetchone()

    def _current_frame(self, session: sqlite3.Row) -> Optional[sqlite3.Row]:
        ordinal = session["current_frame_ordinal"]
        if ordinal is None:
            return None
        with self._lock, self._connection() as connection:
            return connection.execute(
                "SELECT * FROM frames WHERE session_id = ? AND ordinal = ?", (session["id"], int(ordinal))
            ).fetchone()

    def _read_frame(self, frame: sqlite3.Row) -> bytes:
        path = (self.demo_root / frame["file_name"]).resolve()
        if self.demo_root not in path.parents or not path.is_file():
            raise LiveDemoFailure("persisted conditioning frame is unavailable", status_code=500)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise LiveDemoFailure("persisted conditioning frame cannot be read", status_code=500) from exc
        if _sha(raw) != frame["sha256"] or not raw.startswith(_PNG_SIGNATURE):
            raise LiveDemoFailure("persisted conditioning frame hash does not match", status_code=500)
        return raw

    def _store_frame(
        self,
        session_id: str,
        role: str,
        raw: bytes,
        *,
        input_sha256: Optional[str] = None,
        conditioning_sha256: Optional[str] = None,
        action: Optional[Iterable[float]] = None,
        timings: Optional[Mapping[str, Any]] = None,
        model: Any = None,
        revision: Any = None,
        profile: Optional[str] = None,
        warning: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if role not in {"source", "fixture", "predicted"} or not raw.startswith(_PNG_SIGNATURE):
            raise LiveDemoFailure("refusing to persist a non-PNG live-demo artifact", status_code=500)
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            ordinal = int(connection.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM frames WHERE session_id = ?", (session_id,)).fetchone()[0])
            session_dir = self.frames_root / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            filename = "frames/%s/%04d-%s.png" % (session_id, ordinal, role)
            destination = self.demo_root / filename
            temporary = destination.with_name(destination.stem + "." + uuid.uuid4().hex + ".tmp.png")
            try:
                with temporary.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            except OSError as exc:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                connection.rollback()
                raise LiveDemoFailure("could not persist live-demo frame", status_code=500) from exc
            digest, now = _sha(raw), _now()
            connection.execute(
                """INSERT INTO frames (session_id, ordinal, role, file_name, sha256, input_sha256, conditioning_sha256, action_json, timings_json, model, revision, profile, warning, provenance_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, ordinal, role, filename, digest, input_sha256, conditioning_sha256,
                 _json(list(action)) if action is not None else None, _json(dict(timings)) if timings is not None else None,
                 _json(model) if isinstance(model, Mapping) else model,
                 _json(revision) if isinstance(revision, Mapping) else revision,
                 profile, warning, _json(dict(provenance or {})), now),
            )
            connection.execute(
                "INSERT INTO artifact_journal (session_id, ordinal, role, file_name, sha256, bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, ordinal, role, filename, digest, len(raw), now),
            )
            connection.execute("UPDATE sessions SET current_frame_ordinal = ?, updated_at = ? WHERE id = ?", (ordinal, now, session_id))
            connection.commit()

    def _update_session(self, session_id: str, *, prompt: Optional[str] = None, source: Optional[Mapping[str, Any]] = None) -> None:
        fields, values = ["updated_at = ?"], [_now()]
        if prompt is not None:
            fields.append("prompt = ?")
            values.append(prompt)
        if source is not None:
            fields.append("source_json = ?")
            values.append(_json(dict(source)))
        values.append(session_id)
        with self._lock, self._connection() as connection:
            connection.execute("UPDATE sessions SET %s WHERE id = ?" % ", ".join(fields), values)

    def _append_warning(self, session_id: str, warning: str) -> None:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT warnings_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
            values = _loads(row["warnings_json"] if row else None, [])
            values = values if isinstance(values, list) else []
            if warning not in values:
                values.append(warning)
                values = values[-16:]
                connection.execute("UPDATE sessions SET warnings_json = ?, updated_at = ? WHERE id = ?", (_json(values), _now(), session_id))

    def _encode_latest_video(self, session_id: str) -> None:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT file_name FROM frames WHERE session_id = ? ORDER BY ordinal", (session_id,)).fetchall()
        if len(rows) < 2:
            return
        paths = [(self.demo_root / row["file_name"]).resolve() for row in rows]
        if any(self.demo_root not in path.parents or not path.is_file() for path in paths):
            return
        concat = self.demo_root / ("video-" + session_id + ".txt")
        output = self.demo_root / ("video-" + session_id + ".mp4")
        temporary = self.demo_root / ("video-" + session_id + "." + uuid.uuid4().hex + ".tmp.mp4")
        # All filenames are service-generated, but quote the concat entries
        # correctly nonetheless.  This is not a shell command.
        try:
            width, height = 320, 256
            if json.loads(self._session_row(session_id)["source_json"]).get("world_model") == "cosmos":
                from PIL import Image
                with Image.open(paths[-1]) as image:
                    width, height = image.size
            concat.write_text("".join("file '%s'\nduration 0.2\n" % str(path).replace("'", "'\\''") for path in paths), encoding="utf-8")
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-r", "5", "-i", str(concat),
                 "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
                 "-fps_mode", "passthrough", "-movflags", "+faststart", str(temporary)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=40,
                check=False,
            )
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                os.replace(temporary, output)
                receipt = output.with_suffix(".manifest.json")
                receipt.write_text(_json({"sha256": _sha(output.read_bytes()), "frame_count": len(rows), "session_id": session_id}), encoding="utf-8")
                with self._lock, self._connection() as connection:
                    connection.execute("UPDATE sessions SET latest_video_file = ?, updated_at = ? WHERE id = ?", ("video-" + session_id + ".mp4", _now(), session_id))
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            try:
                concat.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _artifact_url(self, file_name: str) -> str:
        return "/api/artifacts/live-demo/" + file_name

    def _public(self, session: sqlite3.Row) -> Dict[str, Any]:
        session_id = str(session["id"])
        with self._lock, self._connection() as connection:
            frame_rows = connection.execute("SELECT * FROM frames WHERE session_id = ? ORDER BY ordinal", (session_id,)).fetchall()
            command_rows = connection.execute("SELECT request_id, direction, steps, state, error, created_at, updated_at FROM commands WHERE session_id = ? ORDER BY created_at", (session_id,)).fetchall()
        frames = []
        for frame in frame_rows:
            frames.append(
                {
                    "index": frame["ordinal"],
                    "role": frame["role"],
                    "url": self._artifact_url(frame["file_name"]),
                    "sha256": frame["sha256"],
                    "input_sha256": frame["input_sha256"],
                    "conditioning_sha256": frame["conditioning_sha256"],
                    "action7": _loads(frame["action_json"], None),
                    "timings_ms": _loads(frame["timings_json"], None),
                    "model": _stored_metadata(frame["model"]),
                    "revision": _stored_metadata(frame["revision"]),
                    "profile": frame["profile"],
                    "warning": frame["warning"],
                    "provenance": _loads(frame["provenance_json"], {}),
                    "created_at": frame["created_at"],
                }
            )
        latest_video = session["latest_video_file"]
        latest_video_url = None
        if isinstance(latest_video, str):
            candidate = (self.demo_root / latest_video).resolve()
            if self.demo_root in candidate.parents and candidate.is_file() and candidate.suffix == ".mp4":
                latest_video_url = self._artifact_url(latest_video)
        source = _mapping(_loads(session["source_json"], {}))
        judgment = None
        if source.get("judgment_id") and self.judge_service is not None:
            try:
                judgment = self.judge_service.get(source["judgment_id"])
            except KeyError:
                pass
        return {
            "id": session_id,
            "title": session["title"],
            "prompt": session["prompt"] or None,
            "mode": session["mode"],
            "policy_label": "Supplied manual actions" if source.get("world_model") == "cosmos" else "OpenVLA" if session["mode"] == "policy" else "Manual directional action",
            "world_model": source.get("world_model", "irasim"),
            "world_progress": source.get("world_progress"),
            "world_timing": source.get("world_timing"),
            "state": session["state"],
            "requested_steps": session["requested_steps"],
            "completed_steps": sum(1 for frame in frames if frame["role"] == "predicted"),
            "max_steps": MAX_SESSION_STEPS,
            "remaining_steps": max(0, MAX_SESSION_STEPS - sum(1 for frame in frames if frame["role"] == "predicted")),
            "seed": session["seed"],
            "current_frame_index": session["current_frame_ordinal"],
            "source": source or None,
            "auto_assess": bool(source.get("auto_assess")),
            "judgment": judgment,
            "assessment_error": source.get("assessment_error"),
            "generation_completed_at": source.get("generation_completed_at"),
            "frames": frames,
            "frame_urls": [frame["url"] for frame in frames],
            "latest_video_url": latest_video_url,
            "error": session["error"],
            "warnings": _loads(session["warnings_json"], []),
            "commands": [dict(row) for row in command_rows],
            "state_mode": "cosmos_full_chunk_supplied_actions" if source.get("world_model") == "cosmos" else STATE_MODE,
            "qualified": False,
            "scored": False,
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
        }

    def get(self, session_id: str) -> Dict[str, Any]:
        return self._public(self._session_row(session_id))

    def list(self) -> Dict[str, Any]:
        with self._lock, self._connection() as connection:
            rows = connection.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (MAX_SESSIONS,)).fetchall()
        return {"sessions": [self._public(row) for row in rows], "state_mode": STATE_MODE, "qualified": False}


def register_live_demo_routes(app: FastAPI, service: LiveDemoService) -> None:
    """Attach the small public API without coupling it to the run ledger."""

    def fail(error: LiveDemoFailure) -> None:
        raise HTTPException(error.status_code, {"reason": str(error), "state_mode": STATE_MODE, "qualified": False})

    @app.get("/api/demo/status")
    def status() -> dict:
        return service.status()

    @app.get("/api/demo/sessions")
    def list_sessions() -> dict:
        return service.list()

    @app.post("/api/demo/sessions", status_code=202)
    def create_session(body: CreateSessionInput) -> dict:
        try:
            return service.create(body)
        except LiveDemoFailure as error:
            fail(error)

    @app.get("/api/demo/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        try:
            return service.get(session_id)
        except LiveDemoFailure as error:
            fail(error)

    @app.get("/api/demo/sessions/{session_id}/events")
    async def session_events(session_id: str, request: ApiRequest):
        try:
            service.get(session_id)
        except LiveDemoFailure as error:
            fail(error)

        async def snapshots():
            last = None
            while not await request.is_disconnected():
                snapshot = service.get(session_id)
                encoded = _json(snapshot)
                if encoded != last:
                    yield "event: snapshot\ndata: " + encoded + "\n\n"
                    last = encoded
                else:
                    yield ": heartbeat\n\n"
                judgment = snapshot.get("judgment") or {}
                judging_done = not snapshot.get("auto_assess") or snapshot.get("assessment_error") or judgment.get("status") in {"completed", "abstained", "failed", "interrupted"}
                if snapshot["state"] in {"error", "blocked"} or (snapshot["state"] == "completed" and judging_done):
                    return
                await asyncio.sleep(0.25)
        return StreamingResponse(snapshots(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/demo/sessions/{session_id}/commands", status_code=202)
    def command(session_id: str, body: CommandInput) -> dict:
        try:
            session, duplicate = service.command(session_id, body)
        except LiveDemoFailure as error:
            fail(error)
        return {**session, "idempotent_replay": duplicate}
