"""Durable, explicitly unqualified cloud-model diagnostics.

This is intentionally separate from ``RunService`` and all Gates.  It sends one
fixed, hash-bound current/goal fixture through a server-side Baseten CLI profile,
persists the exact input and response artifacts locally, and never fabricates a
task label, retry, or qualification claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

from .records import canonical_json, utc_now


_REQUEST_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")
MAX_CLI_RESPONSE_BYTES = 1024 * 1024
AUTOEVAL_MEAN = (1.9296819e-04, 1.3667766e-04, -1.4583133e-04, -1.8390431e-04, -3.0808983e-04, 2.7425270e-04)
AUTOEVAL_STD = (0.00912848, 0.0127196, 0.01229497, 0.02606696, 0.02875283, 0.07807977)
EXPECTED_RESTORE_METHOD = "inference_params_only_source_checkpoint_no_optimizer_restore"


class CloudDiagnosticError(RuntimeError):
    """A cloud diagnostic is unavailable or its response cannot be trusted."""


class CloudDiagnosticValidationError(CloudDiagnosticError, ValueError):
    """Configuration or a model response violates the fixed diagnostic contract."""


@dataclass(frozen=True)
class CloudDiagnosticConfig:
    model_id: str
    deployment_id: str
    cli_profile: str
    current_png_base64: str
    goal_png_base64: str
    current_png_sha256: str
    goal_png_sha256: str
    prompt: str
    model_revision: str
    checkpoint_sha256: str
    soar_revision: str
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("model_id", "deployment_id", "cli_profile", "prompt", "model_revision", "checkpoint_sha256", "soar_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise CloudDiagnosticValidationError("%s must be a non-empty bounded string" % name)
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)) or not 1 <= float(self.timeout_seconds) <= 600:
            raise CloudDiagnosticValidationError("timeout_seconds must be between 1 and 600")
        for encoded, digest, label in (
            (self.current_png_base64, self.current_png_sha256, "current"),
            (self.goal_png_base64, self.goal_png_sha256, "goal"),
        ):
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise CloudDiagnosticValidationError("%s_png_base64 is invalid" % label) from error
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise CloudDiagnosticValidationError("%s fixture must be PNG bytes" % label)
            expected = "sha256:" + hashlib.sha256(raw).hexdigest()
            if digest != expected:
                raise CloudDiagnosticValidationError("%s fixture digest does not match bytes" % label)

    def request_payload(self, request_id: str) -> Dict[str, str]:
        return {
            "schema_version": 1,
            "current_png_base64": self.current_png_base64,
            "goal_png_base64": self.goal_png_base64,
            "prompt": self.prompt,
            "request_id": request_id,
        }


@dataclass(frozen=True)
class CliInvocationResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class CloudCliRunner(Protocol):
    def __call__(self, argv: Sequence[str], input_bytes: bytes, timeout_seconds: float) -> CliInvocationResult:
        """Run the explicitly profiled CLI without browser-delivered credentials."""


def subprocess_cli_runner(argv: Sequence[str], input_bytes: bytes, timeout_seconds: float) -> CliInvocationResult:
    environment = dict(os.environ)
    environment.pop("BASETEN_API_KEY", None)
    environment.pop("BASETEN_AUTH_TOKEN", None)
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            input=input_bytes,
            timeout=timeout_seconds,
            shell=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        raise CloudDiagnosticError("CLI timed out after %.1f seconds" % timeout_seconds) from error
    return CliInvocationResult(completed.returncode, bytes(completed.stdout), bytes(completed.stderr))


class CloudDiagnosticService:
    """One-at-a-time local journal around a fixed cloud diagnostic request."""

    def __init__(
        self,
        data_root: Path,
        config: CloudDiagnosticConfig,
        runner: CloudCliRunner = subprocess_cli_runner,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.config = config
        self.runner = runner
        self.artifacts_root = self.data_root / "cloud-diagnostics"
        self.db_path = self.data_root / "cloud-diagnostics.sqlite3"
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialise(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cloud_diagnostics (
                  request_id TEXT PRIMARY KEY,
                  input_sha256 TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  started_at TEXT,
                  finished_at TEXT,
                  error_json TEXT,
                  result_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_cloud_diagnostics_status ON cloud_diagnostics(status, created_at);
                """
            )
            failure = {"kind": "InterruptedDiagnostic", "message": "process restarted while diagnostic was running", "manual_reconciliation_required": True, "automatic_retry_allowed": False}
            connection.execute("UPDATE cloud_diagnostics SET status = 'ambiguous', error_json = ?, finished_at = ?, updated_at = ? WHERE status = 'running'", (canonical_json(failure), utc_now(), utc_now()))
            pending_failure = {"kind": "InterruptedBeforeSubmission", "message": "process restarted before the queued diagnostic was submitted", "manual_reconciliation_required": True, "automatic_retry_allowed": False}
            connection.execute("UPDATE cloud_diagnostics SET status = 'failed', error_json = ?, finished_at = ?, updated_at = ? WHERE status = 'pending'", (canonical_json(pending_failure), utc_now(), utc_now()))
        os.chmod(self.db_path, 0o600)

    def _target_binding(self) -> Dict[str, str]:
        return {"model_id": self.config.model_id, "deployment_id": self.config.deployment_id,
                "model_revision": self.config.model_revision, "checkpoint_sha256": self.config.checkpoint_sha256,
                "soar_revision": self.config.soar_revision}

    def _digest(self, value: Mapping[str, Any]) -> str:
        envelope = {"input": dict(value), "target": self._target_binding()}
        return "sha256:" + hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()

    def _artifact_dir(self, request_id: str) -> Path:
        path = (self.artifacts_root / request_id).resolve()
        if self.artifacts_root.resolve() not in path.parents:
            raise CloudDiagnosticValidationError("diagnostic artifact path escapes its root")
        return path

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(dict(value)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _argv(self, payload: Mapping[str, Any]) -> list[str]:
        # The profile is an explicit CLI argument. No browser, request payload,
        # local config read, or response artifact ever supplies credentials.
        return [
            "baseten",
            "model",
            "predict",
            "--profile",
            self.config.cli_profile,
            "--model-id",
            self.config.model_id,
            "--deployment-id",
            self.config.deployment_id,
            "--file",
            "-",
            "--output",
            "json",
        ]

    def _normalise_response(self, value: Any, request_id: str) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise CloudDiagnosticValidationError("cloud model response must be a JSON object")
        if type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("request_id") != request_id:
            raise CloudDiagnosticValidationError("cloud response does not bind the diagnostic request")
        if value.get("status") != "completed_unqualified" or value.get("qualified") is not False:
            raise CloudDiagnosticValidationError("cloud response must be explicitly completed_unqualified")
        actions = value.get("actions") if isinstance(value.get("actions"), Mapping) else {}
        if actions.get("shape") != [1, 7] or actions.get("finite") is not True:
            raise CloudDiagnosticValidationError("cloud response action shape/finite contract is invalid")
        action = value.get("physical_action", actions.get("transformed_physical"))
        if isinstance(action, list) and len(action) == 1 and isinstance(action[0], list):
            action = action[0]
        inputs = value.get("inputs") if isinstance(value.get("inputs"), Mapping) else {}
        model = value.get("model") if isinstance(value.get("model"), Mapping) else {}
        if model.get("repo_id") != "patreya/gcbc-bridge":
            raise CloudDiagnosticValidationError("cloud response model repository does not match diagnostic policy")
        hashes = value.get("hashes")
        if not isinstance(hashes, Mapping):
            hashes = {key: item for key, item in inputs.items() if key.endswith("_sha256") and isinstance(item, str)}
        native = value.get("nativeoutput", actions.get("native_model_normalized"))
        revision = value.get("modelrevision", model.get("revision"))
        if (
            not isinstance(action, list)
            or len(action) != 7
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in action)
        ):
            raise CloudDiagnosticValidationError("cloud response physical_action must be one finite 7-D row")
        if not isinstance(hashes, Mapping) or not hashes or any(not isinstance(key, str) or not isinstance(item, str) for key, item in hashes.items()):
            raise CloudDiagnosticValidationError("cloud response hashes must be a non-empty string mapping")
        if not isinstance(native, (Mapping, list, str)):
            raise CloudDiagnosticValidationError("cloud response nativeoutput must be persisted")
        if not isinstance(revision, str) or not revision:
            raise CloudDiagnosticValidationError("cloud response modelrevision is required")
        if revision != self.config.model_revision or model.get("checkpoint_sha256") != self.config.checkpoint_sha256 or model.get("soar_revision") != self.config.soar_revision:
            raise CloudDiagnosticValidationError("cloud response model binding does not match pinned diagnostic configuration")
        if not isinstance(native, list) or len(native) != 7 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in native):
            raise CloudDiagnosticValidationError("cloud response native action must be one finite 7-D row")
        expected = [float(native[i]) * AUTOEVAL_STD[i] + AUTOEVAL_MEAN[i] for i in range(6)] + [1.0 if float(native[6]) > 0.0 else 0.0]
        if any(abs(float(actual) - target) > 1e-7 for actual, target in zip(action, expected)):
            raise CloudDiagnosticValidationError("cloud response physical action does not match pinned AutoEval normalization")
        def digest_matches(actual: Any, expected: str) -> bool:
            return isinstance(actual, str) and actual.removeprefix("sha256:") == expected.removeprefix("sha256:")
        if not digest_matches(inputs.get("current_png_sha256"), self.config.current_png_sha256) or not digest_matches(inputs.get("goal_png_sha256"), self.config.goal_png_sha256):
            raise CloudDiagnosticValidationError("cloud response fixture hashes do not match request")
        if not digest_matches(inputs.get("prompt_sha256"), hashlib.sha256(self.config.prompt.encode("utf-8")).hexdigest()):
            raise CloudDiagnosticValidationError("cloud response prompt hash does not match request")
        restore = model.get("restore") if isinstance(model.get("restore"), Mapping) else {}
        if restore.get("changed") is not True or restore.get("optimizer_state_excluded") is not True or restore.get("method") != EXPECTED_RESTORE_METHOD:
            raise CloudDiagnosticValidationError("cloud response restore contract is not inference-only pinned restore")
        return {
            "physical_action": [float(item) for item in action],
            "hashes": dict(hashes),
            "nativeoutput": native,
            "modelrevision": revision,
            "timing": value.get("timing") if isinstance(value.get("timing"), Mapping) else {},
            "qualified": False,
            "diagnostic_only": True,
            "target": self._target_binding(),
        }

    def submit(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        request_id = request_id or ("cloud-" + uuid.uuid4().hex)
        if not _REQUEST_ID_RE.fullmatch(request_id):
            raise CloudDiagnosticValidationError("request_id is invalid")
        payload = self.config.request_payload(request_id)
        digest = self._digest(payload)
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM cloud_diagnostics WHERE request_id = ?", (request_id,)).fetchone()
            if existing is not None:
                if existing["input_sha256"] != digest:
                    connection.rollback()
                    raise CloudDiagnosticValidationError("request_id already binds a different diagnostic input")
                connection.commit()
                return self._row(existing)
            connection.execute(
                """INSERT INTO cloud_diagnostics(request_id, input_sha256, status, created_at, updated_at)
                   VALUES (?, ?, 'pending', ?, ?)""",
                (request_id, digest, now, now),
            )
            connection.commit()
        self._write_json(
            self._artifact_dir(request_id) / "input.json",
            {"input": payload, "target": self._target_binding(), "input_sha256": digest, "qualified": False, "diagnostic_only": True},
        )
        return self.get(request_id)

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "request_id": row["request_id"],
            "input_sha256": row["input_sha256"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "qualified": False,
            "diagnostic_only": True,
        }

    def get(self, request_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cloud_diagnostics WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            raise KeyError("cloud diagnostic not found")
        return self._row(row)

    def list_requests(self, limit: int = 100) -> list[Dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise CloudDiagnosticValidationError("limit must be from 1 through 1000")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cloud_diagnostics ORDER BY created_at DESC, request_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def execute(self, request_id: str) -> Dict[str, Any]:
        """Run exactly once. Any uncertainty is terminal/manual, never retried."""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cloud_diagnostics WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("cloud diagnostic not found")
            if row["status"] != "pending":
                connection.commit()
                return self._row(row)
            now = utc_now()
            connection.execute(
                "UPDATE cloud_diagnostics SET status = 'running', started_at = ?, updated_at = ? WHERE request_id = ? AND status = 'pending'",
                (now, now, request_id),
            )
            connection.commit()
        payload = self.config.request_payload(request_id)
        try:
            invocation = self.runner(
                self._argv(payload), canonical_json(payload).encode("utf-8"), float(self.config.timeout_seconds)
            )
            raw = {
                "returncode": invocation.returncode,
                "stdout": invocation.stdout.decode("utf-8", "replace"),
                "stderr": invocation.stderr.decode("utf-8", "replace"),
            }
            if len(invocation.stdout) + len(invocation.stderr) > MAX_CLI_RESPONSE_BYTES:
                raise CloudDiagnosticError("CLI response exceeds diagnostic size limit")
            self._write_json(self._artifact_dir(request_id) / "raw-response.json", raw)
            if invocation.returncode != 0:
                raise CloudDiagnosticError("CLI returned nonzero status %d" % invocation.returncode)
            try:
                parsed = json.loads(raw["stdout"])
            except ValueError as error:
                raise CloudDiagnosticError("CLI returned unreadable JSON") from error
            result = self._normalise_response(parsed, request_id)
            self._write_json(self._artifact_dir(request_id) / "result.json", result)
        except Exception as error:
            failure = {
                "kind": type(error).__name__,
                "message": str(error),
                "manual_reconciliation_required": True,
                "automatic_retry_allowed": False,
            }
            self._write_json(self._artifact_dir(request_id) / "failure.json", failure)
            with self._lock, self._connect() as connection:
                now = utc_now()
                connection.execute(
                    """UPDATE cloud_diagnostics SET status = 'failed', error_json = ?, finished_at = ?, updated_at = ?
                       WHERE request_id = ? AND status = 'running'""",
                    (canonical_json(failure), now, now, request_id),
                )
            return self.get(request_id)
        with self._lock, self._connect() as connection:
            now = utc_now()
            connection.execute(
                """UPDATE cloud_diagnostics SET status = 'completed', result_json = ?, finished_at = ?, updated_at = ?
                   WHERE request_id = ? AND status = 'running'""",
                (canonical_json(result), now, now, request_id),
            )
        return self.get(request_id)


__all__ = [
    "CliInvocationResult",
    "CloudCliRunner",
    "CloudDiagnosticConfig",
    "CloudDiagnosticError",
    "CloudDiagnosticService",
    "CloudDiagnosticValidationError",
    "subprocess_cli_runner",
]
