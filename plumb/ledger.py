"""SQLite persistence for PLUMB logical runs, episodes, attempts, and events.

The Baseten tables near the bottom of this module are deliberately part of the
same SQLite database as the logical episode ledger.  A remote submission is
therefore an outbox record *before* bytes can be sent over HTTP, and a webhook
is a durable receipt before any caller is allowed to score it.  Keeping those
facts together is what makes crash recovery and callback de-duplication
possible; the platform is not an output store.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .records import ConfigurationError, canonical_json, lease_deadline, normalise_backend_result, utc_now


TERMINAL_EPISODE_STATUSES = ("completed", "failed", "cancelled")

#: Spec section 8 per-episode record fields, added to ``episodes`` as a
#: backwards-compatible in-place migration.  ``plumb/measurement.py`` requires
#: ``cohort``, ``protocol_hash`` and ``policy_variant`` on every row before it
#: will accept a run as a primary analysis rather than a diagnostic, so these
#: are load-bearing rather than cosmetic.
EPISODE_RECORD_COLUMNS = (
    ("schema_version", "INTEGER NOT NULL DEFAULT 1"),
    ("cohort", "TEXT NOT NULL DEFAULT 'primary'"),
    ("protocol_hash", "TEXT"),
    ("scenario_manifest_hash", "TEXT"),
    ("policy_variant", "TEXT"),
    ("policy_identity_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("world_identity_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("judge_identity_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("seeds_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("feedback_mode", "TEXT NOT NULL DEFAULT 'unqualified'"),
    ("parity_status", "TEXT NOT NULL DEFAULT 'unqualified'"),
    ("executed_actions", "INTEGER NOT NULL DEFAULT 0"),
    ("n_segments", "INTEGER NOT NULL DEFAULT 0"),
    ("raw_judge_samples_ref", "TEXT"),
    ("segments_manifest_ref", "TEXT"),
    ("video_ref", "TEXT"),
    ("timing_ref", "TEXT"),
    ("platform_request_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("attempt_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("compute_gpu_seconds", "REAL"),
    ("allocated_gpu_seconds", "REAL"),
    ("estimated_usd", "REAL"),
    ("cost_basis_ref", "TEXT"),
    ("exclusion_reason", "TEXT"),
    # Presentation fields the console reads off the row, so a client that
    # connects late or reloads rebuilds its wall without replaying every event.
    ("segments_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("presentation_track", "TEXT"),
    ("resolution", "INTEGER"),
)

#: Identity fields supplied when a run is planned.  ``feedback_mode`` and
#: ``parity_status`` default to ``unqualified`` precisely so that an unlabelled
#: run can never be mistaken for a qualified one.
_PLANNED_IDENTITY_FIELDS = (
    ("schema_version", 1),
    ("cohort", "primary"),
    ("protocol_hash", None),
    ("scenario_manifest_hash", None),
    ("policy_variant", None),
    ("feedback_mode", "unqualified"),
    ("parity_status", "unqualified"),
)

#: Fields a terminal result may set.  Measurement/cost values stay ``None``
#: unless the backend actually measured them.
_RESULT_MEASUREMENT_FIELDS = (
    "executed_actions",
    "n_segments",
    "raw_judge_samples_ref",
    "segments_manifest_ref",
    "video_ref",
    "timing_ref",
    "compute_gpu_seconds",
    "allocated_gpu_seconds",
    "estimated_usd",
    "cost_basis_ref",
    "exclusion_reason",
    "presentation_track",
    "resolution",
)

class LeaseActiveError(RuntimeError):
    """A safe recovery was requested while another owner still has a lease."""


class CallbackConflictError(RuntimeError):
    """The same platform request ID was delivered with different immutable data."""


def _validate_platform_artifact_refs(refs: Mapping[str, Any]) -> None:
    """Validate committed artifact identities; the storage writer verifies bytes."""
    if not isinstance(refs, Mapping) or not refs:
        raise ValueError("a typed final artifact reference is required before terminalization")
    for name, ref in refs.items():
        if not isinstance(name, str) or not name or not isinstance(ref, Mapping):
            raise ValueError("platform artifact references must be named typed objects")
        uri, digest, media = ref.get("uri"), ref.get("sha256"), ref.get("media_type")
        if not isinstance(uri, str) or not uri.strip() or ":" not in uri:
            raise ValueError("platform artifact reference requires an absolute URI")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("platform artifact reference requires a prefixed SHA-256")
        value = digest[7:]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError("platform artifact reference has an invalid SHA-256")
        if not isinstance(media, str) or "/" not in media or not media.strip():
            raise ValueError("platform artifact reference requires media_type")
    json.dumps(dict(refs), allow_nan=False)


class Ledger:
    """Small transactional ledger designed to be safely used by worker threads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY,
                  idempotency_key TEXT UNIQUE,
                  mode TEXT NOT NULL,
                  config_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  cancellation_requested INTEGER NOT NULL DEFAULT 0,
                  owner_token TEXT,
                  owner_heartbeat_at TEXT,
                  owner_lease_expires_at TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS episodes (
                  episode_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  logical_key TEXT NOT NULL,
                  policy TEXT NOT NULL,
                  task TEXT NOT NULL,
                  start_id TEXT NOT NULL,
                  start_lineage_id TEXT NOT NULL,
                  world_seed INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  validity TEXT NOT NULL DEFAULT 'unknown',
                  binary_success INTEGER,
                  progress_score INTEGER,
                  missing_reason TEXT,
                  horizon_actions INTEGER NOT NULL,
                  artifact_refs_json TEXT NOT NULL DEFAULT '{}',
                  timing_json TEXT NOT NULL DEFAULT '{}',
                  error_json TEXT,
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  attempt_token TEXT,
                  lease_expires_at TEXT,
                  created_at TEXT NOT NULL,
                  submitted_at TEXT,
                  started_at TEXT,
                  completed_at TEXT,
                  updated_at TEXT NOT NULL,
                  -- Spec section 8 record contract.  These are nullable because
                  -- ``null`` means unknown, never zero, and a planned episode
                  -- legitimately knows none of its measurements yet.
                  schema_version INTEGER NOT NULL DEFAULT 1,
                  cohort TEXT NOT NULL DEFAULT 'primary',
                  protocol_hash TEXT,
                  scenario_manifest_hash TEXT,
                  policy_variant TEXT,
                  policy_identity_json TEXT NOT NULL DEFAULT '{}',
                  world_identity_json TEXT NOT NULL DEFAULT '{}',
                  judge_identity_json TEXT NOT NULL DEFAULT '{}',
                  seeds_json TEXT NOT NULL DEFAULT '{}',
                  feedback_mode TEXT NOT NULL DEFAULT 'unqualified',
                  parity_status TEXT NOT NULL DEFAULT 'unqualified',
                  executed_actions INTEGER NOT NULL DEFAULT 0,
                  n_segments INTEGER NOT NULL DEFAULT 0,
                  raw_judge_samples_ref TEXT,
                  segments_manifest_ref TEXT,
                  video_ref TEXT,
                  timing_ref TEXT,
                  platform_request_ids_json TEXT NOT NULL DEFAULT '[]',
                  attempt_ids_json TEXT NOT NULL DEFAULT '[]',
                  compute_gpu_seconds REAL,
                  allocated_gpu_seconds REAL,
                  estimated_usd REAL,
                  cost_basis_ref TEXT,
                  exclusion_reason TEXT,
                  UNIQUE(run_id, logical_key)
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_run_status
                  ON episodes(run_id, status, policy, task, start_id);

                CREATE TABLE IF NOT EXISTS attempts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                  attempt_no INTEGER NOT NULL,
                  status TEXT NOT NULL,
                  attempt_token TEXT,
                  started_at TEXT NOT NULL,
                  completed_at TEXT,
                  error_json TEXT,
                  result_json TEXT,
                  UNIQUE(run_id, episode_id, attempt_no)
                );

                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  sequence INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  timestamp TEXT NOT NULL,
                  UNIQUE(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_sequence
                  ON events(run_id, sequence);

                CREATE TABLE IF NOT EXISTS baseten_outbox (
                  outbox_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                  episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
                  payload_json TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  options_json TEXT NOT NULL,
                  deadline_at TEXT NOT NULL,
                  state TEXT NOT NULL,
                  dispatch_token TEXT,
                  dispatch_lease_expires_at TEXT,
                  post_attempt_count INTEGER NOT NULL DEFAULT 0,
                  platform_request_id TEXT UNIQUE,
                  remote_status TEXT,
                  remote_status_observed_at TEXT,
                  cancellation_requested INTEGER NOT NULL DEFAULT 0,
                  cancellation_requested_at TEXT,
                  cancel_dispatch_started_at TEXT,
                  cancel_token TEXT,
                  cancel_lease_expires_at TEXT,
                  last_error_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  accepted_at TEXT,
                  UNIQUE(run_id, episode_id)
                );
                CREATE INDEX IF NOT EXISTS idx_baseten_outbox_state
                  ON baseten_outbox(state, dispatch_lease_expires_at, deadline_at);
                CREATE INDEX IF NOT EXISTS idx_baseten_outbox_request
                  ON baseten_outbox(platform_request_id);

                CREATE TABLE IF NOT EXISTS baseten_callbacks (
                  request_id TEXT PRIMARY KEY,
                  outbox_id TEXT REFERENCES baseten_outbox(outbox_id) ON DELETE SET NULL,
                  raw_body BLOB NOT NULL,
                  body_sha256 TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  signature_header TEXT,
                  request_id_header TEXT,
                  received_at TEXT NOT NULL,
                  state TEXT NOT NULL,
                  processor_token TEXT,
                  processor_lease_expires_at TEXT,
                  finalization_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_baseten_callbacks_outbox
                  ON baseten_callbacks(outbox_id, state, processor_lease_expires_at);

                CREATE TABLE IF NOT EXISTS baseten_outbox_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  outbox_id TEXT NOT NULL REFERENCES baseten_outbox(outbox_id) ON DELETE CASCADE,
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_baseten_outbox_events
                  ON baseten_outbox_events(outbox_id, id);
                """
            )
            # Existing local fixture databases are upgraded in place. SQLite
            # intentionally has a limited ALTER TABLE syntax, so keep the
            # additions explicit and backwards-compatible.
            self._ensure_column(connection, "runs", "owner_token", "TEXT")
            self._ensure_column(connection, "runs", "owner_heartbeat_at", "TEXT")
            self._ensure_column(connection, "runs", "owner_lease_expires_at", "TEXT")
            self._ensure_column(connection, "episodes", "attempt_token", "TEXT")
            self._ensure_column(connection, "episodes", "lease_expires_at", "TEXT")
            self._ensure_column(connection, "attempts", "attempt_token", "TEXT")
            for column, definition in EPISODE_RECORD_COLUMNS:
                self._ensure_column(connection, "episodes", column, definition)
            # Created after the migration so an upgraded database has the column.
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_cohort ON episodes(run_id, cohort, task, policy)"
            )
            self._ensure_column(connection, "baseten_outbox", "cancel_token", "TEXT")
            self._ensure_column(connection, "baseten_outbox", "cancel_lease_expires_at", "TEXT")
            self._ensure_column(connection, "baseten_callbacks", "raw_body", "BLOB")
            self._ensure_column(connection, "baseten_callbacks", "signature_header", "TEXT")
            self._ensure_column(connection, "baseten_callbacks", "request_id_header", "TEXT")

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(%s)" % table).fetchall()}
        if column not in existing:
            connection.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, definition))

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    @staticmethod
    def _positive_finite(value: float, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError("%s must be a finite positive value" % label)
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("%s must be a finite positive value" % label) from error
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("%s must be a finite positive value" % label)
        return normalized

    @staticmethod
    def _utc_timestamp(value: str, label: str) -> str:
        """Parse an offset-bearing UTC timestamp into one comparable wire form."""

        if not isinstance(value, str) or not value:
            raise ValueError("%s must be an ISO-8601 UTC timestamp" % label)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("%s must be an ISO-8601 UTC timestamp" % label) from error
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("%s must be expressed in UTC" % label)
        return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _append_event(self, connection: sqlite3.Connection, run_id: str, event_type: str, payload: Mapping[str, Any]) -> int:
        next_sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO events(run_id, sequence, event_type, payload_json, timestamp) VALUES (?, ?, ?, ?, ?)",
            (run_id, next_sequence, event_type, canonical_json(dict(payload)), utc_now()),
        )
        return int(next_sequence)

    def create_run(
        self,
        run_id: str,
        config: Mapping[str, Any],
        episodes: Iterable[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Commit the run and every planned logical episode in one transaction."""

        idempotency_key = config.get("idempotency_key")
        now = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT id, config_json FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    if existing["config_json"] != canonical_json(dict(config)):
                        connection.rollback()
                        raise ConfigurationError(
                            "idempotency_key is already associated with a different run configuration"
                        )
                    connection.rollback()
                    return self.get_run(str(existing["id"]))
            connection.execute(
                """INSERT INTO runs(id, idempotency_key, mode, config_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'planned', ?, ?)""",
                (run_id, idempotency_key, config["mode"], canonical_json(dict(config)), now, now),
            )
            episode_rows = []
            for episode in episodes:
                identity = [episode.get(name, default) for name, default in _PLANNED_IDENTITY_FIELDS]
                episode_rows.append(
                    (
                        episode["episode_id"],
                        run_id,
                        episode["logical_key"],
                        episode["policy"],
                        episode["task"],
                        episode["start_id"],
                        episode["start_lineage_id"],
                        episode["world_seed"],
                        "planned",
                        episode["mode"],
                        episode["horizon_actions"],
                        now,
                        now,
                        *identity,
                        canonical_json(dict(episode.get("policy_identity", {}))),
                        canonical_json(dict(episode.get("world_identity", {}))),
                        canonical_json(dict(episode.get("judge_identity", {}))),
                        canonical_json(dict(episode.get("seeds", {}))),
                    )
                )
            identity_columns = ", ".join(name for name, _ in _PLANNED_IDENTITY_FIELDS)
            identity_placeholders = ", ".join("?" for _ in _PLANNED_IDENTITY_FIELDS)
            connection.executemany(
                """INSERT INTO episodes(
                    episode_id, run_id, logical_key, policy, task, start_id, start_lineage_id, world_seed,
                    status, mode, horizon_actions, created_at, updated_at,
                    %s,
                    policy_identity_json, world_identity_json, judge_identity_json, seeds_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, %s, ?, ?, ?, ?)"""
                % (identity_columns, identity_placeholders),
                episode_rows,
            )
            self._append_event(
                connection,
                run_id,
                "run_created",
                {"mode": config["mode"], "planned": len(episode_rows), "qualification": "unqualified"},
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def _stats(self, connection: sqlite3.Connection, run_id: str) -> Dict[str, int]:
        row = connection.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
              SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
              SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
              SUM(CASE WHEN binary_success IS NOT NULL THEN 1 ELSE 0 END) AS evaluable,
              SUM(CASE WHEN binary_success = 1 THEN 1 ELSE 0 END) AS successes,
              SUM(CASE WHEN status IN ('completed', 'failed', 'cancelled') THEN 1 ELSE 0 END) AS terminal
            FROM episodes WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}

    def _run_from_row(self, row: sqlite3.Row, stats: Mapping[str, int]) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "status": row["status"],
            "mode": row["mode"],
            "config": json.loads(row["config_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "completed": int(stats["completed"]),
            "failed": int(stats["failed"]),
            "cancelled": int(stats["cancelled"]),
            "total": int(stats["total"]),
            "evaluable": int(stats["evaluable"]),
            "successes": int(stats["successes"]),
        }

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("run not found: %s" % run_id)
            return self._run_from_row(row, self._stats(connection, run_id))

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC, id DESC").fetchall()
            return [self._run_from_row(row, self._stats(connection, row["id"])) for row in rows]

    @staticmethod
    def _json_column(row: sqlite3.Row, column: str, fallback: Any) -> Any:
        """Decode a JSON column, degrading to *fallback* rather than raising.

        An upgraded database can hold a row written before the column existed.
        """

        try:
            raw = row[column]
        except (IndexError, KeyError):
            return fallback
        if raw in (None, ""):
            return fallback
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _column(row: sqlite3.Row, column: str, fallback: Any = None) -> Any:
        try:
            value = row[column]
        except (IndexError, KeyError):
            return fallback
        return fallback if value is None else value

    def _episode_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        binary = row["binary_success"]
        artifact_refs = json.loads(row["artifact_refs_json"])
        frame_ref = artifact_refs.get("fixture_frame") if isinstance(artifact_refs, dict) else None
        frame_url = artifact_refs.get("frame_url") if isinstance(artifact_refs, dict) else None
        if frame_url is None and isinstance(frame_ref, dict):
            frame_url = frame_ref.get("url")
        record = {
            "run_id": row["run_id"],
            "episode_id": row["episode_id"],
            "logical_key": row["logical_key"],
            "policy": row["policy"],
            "task": row["task"],
            "start_id": row["start_id"],
            "start_lineage_id": row["start_lineage_id"],
            "world_seed": row["world_seed"],
            "status": row["status"],
            "binary_success": None if binary is None else bool(binary),
            "validity": row["validity"],
            "progress_score": row["progress_score"],
            "missing_reason": row["missing_reason"],
            "horizon_actions": row["horizon_actions"],
            "mode": row["mode"],
            "artifact_refs": artifact_refs,
            # Convenient direct fields for the frontend. They remain null for
            # a failed/cancelled cell that never produced a fixture frame.
            "frame_url": frame_url,
            "video_url": artifact_refs.get("video_url") if isinstance(artifact_refs, dict) else None,
            "timing": json.loads(row["timing_json"]),
            "attempt_count": row["attempt_count"],
            # Internal lease identity is included for the executor/recovery
            # path, never in public events. It is unguessable UUID material.
            "attempt_token": row["attempt_token"],
            "lease_expires_at": row["lease_expires_at"],
            "created_at": row["created_at"],
            "submitted_at": row["submitted_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "updated_at": row["updated_at"],
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }
        # Spec section 8 identity and measurement fields.  Nulls are preserved as
        # nulls: an unmeasured GPU-second is unknown, not zero.
        record.update(
            {
                "schema_version": int(self._column(row, "schema_version", 1)),
                "cohort": self._column(row, "cohort", "primary"),
                "protocol_hash": self._column(row, "protocol_hash"),
                "scenario_manifest_hash": self._column(row, "scenario_manifest_hash"),
                "policy_variant": self._column(row, "policy_variant"),
                "policy_identity": self._json_column(row, "policy_identity_json", {}),
                "world_identity": self._json_column(row, "world_identity_json", {}),
                "judge_identity": self._json_column(row, "judge_identity_json", {}),
                "seeds": self._json_column(row, "seeds_json", {}),
                "feedback_mode": self._column(row, "feedback_mode", "unqualified"),
                "parity_status": self._column(row, "parity_status", "unqualified"),
                "executed_actions": int(self._column(row, "executed_actions", 0)),
                "n_segments": int(self._column(row, "n_segments", 0)),
                "raw_judge_samples_ref": self._column(row, "raw_judge_samples_ref"),
                "segments_manifest_ref": self._column(row, "segments_manifest_ref"),
                "video_ref": self._column(row, "video_ref"),
                "timing_ref": self._column(row, "timing_ref"),
                "platform_request_ids": self._json_column(row, "platform_request_ids_json", []),
                "attempt_ids": self._json_column(row, "attempt_ids_json", []),
                "compute_gpu_seconds": self._column(row, "compute_gpu_seconds"),
                "allocated_gpu_seconds": self._column(row, "allocated_gpu_seconds"),
                "estimated_usd": self._column(row, "estimated_usd"),
                "cost_basis_ref": self._column(row, "cost_basis_ref"),
                "exclusion_reason": self._column(row, "exclusion_reason"),
                "presentation_track": self._column(row, "presentation_track"),
                "resolution": self._column(row, "resolution"),
            }
        )
        # Flatten the persisted segments into the shape the console's wall reads.
        segments = self._json_column(row, "segments_json", [])
        frame_urls: List[str] = []
        certified: Optional[int] = None
        provenance: Optional[str] = None
        for segment in segments if isinstance(segments, list) else []:
            if not isinstance(segment, Mapping):
                continue
            frame_urls.extend(str(url) for url in (segment.get("frame_urls") or []))
            count = segment.get("certified_frame_count")
            if isinstance(count, int) and not isinstance(count, bool):
                certified = (certified or 0) + count
            if provenance is None and segment.get("provenance"):
                provenance = str(segment["provenance"])
        record["segments"] = segments if isinstance(segments, list) else []
        record["frame_urls"] = frame_urls
        # None, not 0: a segment that did not report a certified count must show
        # as unreported rather than as a measured zero.
        record["certified_frame_count"] = certified
        record["provenance"] = provenance
        # ``video_ref`` is the durable spec-section-8 reference; ``video_url`` is
        # the presentation convenience the console reads.  Prefer an explicit
        # artifact URL and fall back to the durable reference.
        if record.get("video_url") is None and record["video_ref"]:
            record["video_url"] = record["video_ref"]
        return record

    def list_episodes(self, run_id: str, limit: int = 1500) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM episodes WHERE run_id = ?
                   ORDER BY policy, task, start_id, episode_id LIMIT ?""",
                (run_id, limit),
            ).fetchall()
            return [self._episode_from_row(row) for row in rows]

    def list_events(self, run_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event_type, payload_json, timestamp FROM events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
            return [
                {
                    "sequence": int(row["sequence"]),
                    "type": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Baseten durable outbox and webhook receipts
    # ------------------------------------------------------------------
    # The platform integration intentionally does not reuse ``claim_next_episode``:
    # it needs to make a durable "about to POST" transition before the HTTP call,
    # and a crash in that small window is necessarily ambiguous.  The ordinary
    # synthetic worker has different failure semantics and remains isolated.

    @staticmethod
    def _baseten_payload_digest(payload_json: str) -> str:
        return "sha256:" + hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _baseten_body_digest(body: bytes) -> str:
        return "sha256:" + hashlib.sha256(body).hexdigest()

    def _append_baseten_event(
        self,
        connection: sqlite3.Connection,
        outbox_id: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> int:
        """Append immutable outbox history and a corresponding application event."""

        now = utc_now()
        cursor = connection.execute(
            """INSERT INTO baseten_outbox_events(outbox_id, event_type, payload_json, timestamp)
               VALUES (?, ?, ?, ?)""",
            (outbox_id, event_type, canonical_json(dict(payload)), now),
        )
        event_id = int(cursor.lastrowid)
        row = connection.execute("SELECT run_id FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
        if row is not None:
            self._append_event(
                connection,
                str(row["run_id"]),
                "baseten_" + event_type,
                {"outbox_id": outbox_id, "outbox_event_id": event_id, **dict(payload)},
            )
        return event_id

    def _baseten_outbox_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "outbox_id": row["outbox_id"],
            "run_id": row["run_id"],
            "episode_id": row["episode_id"],
            "payload": json.loads(row["payload_json"]),
            "payload_sha256": row["payload_sha256"],
            "options": json.loads(row["options_json"]),
            "deadline_at": row["deadline_at"],
            "state": row["state"],
            "dispatch_token": row["dispatch_token"],
            "dispatch_lease_expires_at": row["dispatch_lease_expires_at"],
            "post_attempt_count": int(row["post_attempt_count"]),
            "platform_request_id": row["platform_request_id"],
            "remote_status": row["remote_status"],
            "remote_status_observed_at": row["remote_status_observed_at"],
            "cancellation_requested": bool(row["cancellation_requested"]),
            "cancellation_requested_at": row["cancellation_requested_at"],
            "cancel_dispatch_started_at": row["cancel_dispatch_started_at"],
            "cancel_token": row["cancel_token"],
            "cancel_lease_expires_at": row["cancel_lease_expires_at"],
            "last_error": json.loads(row["last_error_json"]) if row["last_error_json"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "accepted_at": row["accepted_at"],
        }

    def _baseten_callback_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "request_id": row["request_id"],
            "outbox_id": row["outbox_id"],
            # Raw signed evidence is internal-only. API read models must not
            # expose this potentially large vendor payload.
            "raw_body": bytes(row["raw_body"]) if row["raw_body"] is not None else None,
            "body_sha256": row["body_sha256"],
            "payload": json.loads(row["payload_json"]),
            "event_type": row["event_type"],
            "signature_header": row["signature_header"],
            "request_id_header": row["request_id_header"],
            "received_at": row["received_at"],
            "state": row["state"],
            "processor_token": row["processor_token"],
            "processor_lease_expires_at": row["processor_lease_expires_at"],
            "finalization": json.loads(row["finalization_json"]) if row["finalization_json"] else None,
        }

    @staticmethod
    def _validate_baseten_entrypoint_identity(
        run_id: str, episode_id: str, config_json: str, payload: Mapping[str, Any]
    ) -> None:
        """Bind an async Chain body to the authoritative logical row.

        The deployed entrypoint takes a single ``request: RolloutRequest``
        argument.  Accepting arbitrary JSON here would let an operator attach a
        different episode's work to this ledger row, which defeats the
        outbox's idempotence guarantee.
        """

        config = json.loads(config_json)
        protocol_hash = config.get("protocol_hash")
        if not isinstance(protocol_hash, str) or not protocol_hash:
            raise ConfigurationError("Baseten submission requires a frozen run config.protocol_hash")
        request = payload.get("request")
        if not isinstance(request, Mapping):
            raise ConfigurationError("Baseten Chain entrypoint input must contain request object")
        if request.get("run_id") != run_id or request.get("episode_id") != episode_id:
            raise ConfigurationError("Baseten request run_id/episode_id must equal the authoritative logical row")
        if request.get("protocol_hash") != protocol_hash:
            raise ConfigurationError("Baseten request protocol_hash must equal frozen run config.protocol_hash")
        for stage_name in ("policy", "world", "validity", "judge"):
            stage = request.get(stage_name)
            if not isinstance(stage, Mapping):
                raise ConfigurationError("Baseten request.%s must be a stage object" % stage_name)
            if stage.get("episode_id") != episode_id or stage.get("protocol_hash") != protocol_hash:
                raise ConfigurationError("Baseten stage identity must match the authoritative logical episode")

    def enqueue_baseten_submission(
        self,
        outbox_id: str,
        run_id: str,
        episode_id: str,
        payload: Mapping[str, Any],
        options: Mapping[str, Any],
        deadline_at: str,
    ) -> Dict[str, Any]:
        """Persist one platform delivery intent before any HTTP submission.

        The unique logical episode association deliberately permits no second
        outbox row.  A known pre-acceptance rejection may be manually requeued
        on this same record; an ambiguous POST remains locked for
        reconciliation.
        """

        if not outbox_id or not deadline_at:
            raise ValueError("outbox_id and deadline_at are required")
        deadline_at = self._utc_timestamp(deadline_at, "deadline_at")
        if not isinstance(payload, Mapping) or not isinstance(options, Mapping):
            raise ConfigurationError("Baseten payload and options must be mappings")
        payload_json = canonical_json(dict(payload))
        options_json = canonical_json(dict(options))
        now = utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            episode = connection.execute(
                "SELECT status FROM episodes WHERE run_id = ? AND episode_id = ?", (run_id, episode_id)
            ).fetchone()
            if episode is None:
                raise KeyError("episode not found for Baseten outbox")
            run = connection.execute(
                "SELECT config_json, cancellation_requested, status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if run["cancellation_requested"] or run["status"] in ("completed", "cancelled", "failed"):
                raise ConfigurationError("cannot enqueue a Baseten submission for a cancelled or terminal run")
            self._validate_baseten_entrypoint_identity(run_id, episode_id, str(run["config_json"]), payload)
            existing = connection.execute(
                "SELECT * FROM baseten_outbox WHERE run_id = ? AND episode_id = ?", (run_id, episode_id)
            ).fetchone()
            if existing is not None:
                if (
                    existing["outbox_id"] != outbox_id
                    or existing["payload_json"] != payload_json
                    or existing["options_json"] != options_json
                    or existing["deadline_at"] != deadline_at
                ):
                    raise ConfigurationError(
                        "logical episode already has a different durable Baseten submission"
                    )
                connection.commit()
                return self._baseten_outbox_from_row(existing)
            if episode["status"] != "planned":
                raise ConfigurationError("a Baseten outbox can only be created for a planned logical episode")
            connection.execute(
                """INSERT INTO baseten_outbox(
                     outbox_id, run_id, episode_id, payload_json, payload_sha256, options_json,
                     deadline_at, state, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)""",
                (
                    outbox_id,
                    run_id,
                    episode_id,
                    payload_json,
                    self._baseten_payload_digest(payload_json),
                    options_json,
                    deadline_at,
                    now,
                    now,
                ),
            )
            self._append_baseten_event(connection, outbox_id, "submission_enqueued", {"episode_id": episode_id})
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            connection.commit()
            return self._baseten_outbox_from_row(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_baseten_submission(self, outbox_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            return self._baseten_outbox_from_row(row)

    def list_baseten_submissions(
        self, run_id: Optional[str] = None, states: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        values: List[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        state_values = list(states or ())
        if state_values:
            clauses.append("state IN (%s)" % ",".join("?" for _ in state_values))
            values.extend(state_values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM baseten_outbox" + where + " ORDER BY created_at, outbox_id", values
            ).fetchall()
            return [self._baseten_outbox_from_row(row) for row in rows]

    def list_baseten_outbox_events(self, outbox_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, event_type, payload_json, timestamp FROM baseten_outbox_events
                   WHERE outbox_id = ? ORDER BY id""",
                (outbox_id,),
            ).fetchall()
            return [
                {
                    "id": int(row["id"]),
                    "type": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    def claim_baseten_dispatch(
        self, outbox_id: str, dispatch_token: str, lease_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Record the potentially irreversible POST boundary before sending it."""

        lease_seconds = self._positive_finite(lease_seconds, "lease_seconds")
        if not dispatch_token:
            raise ValueError("dispatch_token and positive lease_seconds are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] != "prepared" or row["cancellation_requested"]:
                connection.commit()
                return None
            run = connection.execute(
                "SELECT cancellation_requested FROM runs WHERE id = ?", (row["run_id"],)
            ).fetchone()
            if run is None or run["cancellation_requested"]:
                connection.commit()
                return None
            episode = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND episode_id = ?", (row["run_id"], row["episode_id"])
            ).fetchone()
            if episode is None or episode["status"] != "planned":
                raise ConfigurationError("Baseten dispatch requires its logical episode to remain planned")
            now = utc_now()
            attempt_no = int(episode["attempt_count"]) + 1
            # ``dispatching`` is deliberately committed *before* the POST. A
            # process crash at any later point has an ambiguous remote outcome.
            connection.execute(
                """UPDATE baseten_outbox SET state = 'dispatching', dispatch_token = ?,
                   dispatch_lease_expires_at = ?, post_attempt_count = post_attempt_count + 1,
                   updated_at = ? WHERE outbox_id = ? AND state = 'prepared' AND cancellation_requested = 0""",
                (dispatch_token, lease_deadline(lease_seconds), now, outbox_id),
            )
            connection.execute(
                """UPDATE episodes SET status = 'running', attempt_count = ?, attempt_token = ?,
                   submitted_at = COALESCE(submitted_at, ?), started_at = ?, lease_expires_at = NULL,
                   updated_at = ? WHERE episode_id = ? AND status = 'planned'""",
                (attempt_no, outbox_id, now, now, now, row["episode_id"]),
            )
            connection.execute(
                """INSERT INTO attempts(run_id, episode_id, attempt_no, status, attempt_token, started_at)
                   VALUES (?, ?, ?, 'platform_dispatching', ?, ?)""",
                (row["run_id"], row["episode_id"], attempt_no, outbox_id, now),
            )
            self._append_baseten_event(
                connection,
                outbox_id,
                "post_started",
                {"post_attempt": int(row["post_attempt_count"]) + 1},
            )
            claimed = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            connection.commit()
            return self._baseten_outbox_from_row(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_acceptance(
        self, outbox_id: str, dispatch_token: str, request_id: str, response: Mapping[str, Any]
    ) -> bool:
        """Persist the returned request ID and atomically associate an early callback."""

        if not dispatch_token or not request_id:
            raise ValueError("dispatch_token and request_id are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["platform_request_id"] == request_id:
                connection.commit()
                return False
            if row["state"] != "dispatching" or row["dispatch_token"] != dispatch_token:
                connection.commit()
                return False
            duplicate = connection.execute(
                "SELECT outbox_id FROM baseten_outbox WHERE platform_request_id = ?", (request_id,)
            ).fetchone()
            if duplicate is not None and duplicate["outbox_id"] != outbox_id:
                raise CallbackConflictError("Baseten request ID is already associated with another logical episode")
            callback = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE request_id = ?", (request_id,)
            ).fetchone()
            now = utc_now()
            callback_ready = callback is not None
            next_state = "callback_received" if callback_ready else (
                "cancellation_requested" if row["cancellation_requested"] else "accepted"
            )
            updated = connection.execute(
                """UPDATE baseten_outbox SET state = ?, platform_request_id = ?, dispatch_token = NULL,
                   dispatch_lease_expires_at = NULL, accepted_at = ?, updated_at = ?, last_error_json = NULL
                   WHERE outbox_id = ? AND state = 'dispatching' AND dispatch_token = ?""",
                (next_state, request_id, now, now, outbox_id, dispatch_token),
            )
            if updated.rowcount != 1:
                connection.commit()
                return False
            if callback is not None:
                if callback["outbox_id"] not in (None, outbox_id):
                    raise CallbackConflictError("early callback is associated with another logical episode")
                connection.execute(
                    "UPDATE baseten_callbacks SET outbox_id = ?, state = 'ready' WHERE request_id = ?",
                    (outbox_id, request_id),
                )
            connection.execute(
                """UPDATE attempts SET status = 'submitted', result_json = ?
                   WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
                (
                    canonical_json(dict(response)),
                    row["run_id"],
                    row["episode_id"],
                    int(row["post_attempt_count"]) + 1,
                    outbox_id,
                ),
            )
            self._append_baseten_event(
                connection,
                outbox_id,
                "request_accepted",
                {"request_id": request_id, "early_callback": callback_ready},
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _record_baseten_dispatch_outcome(
        self,
        outbox_id: str,
        dispatch_token: str,
        state: str,
        error: Mapping[str, Any],
        event_type: str,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] != "dispatching" or row["dispatch_token"] != dispatch_token:
                connection.commit()
                return False
            now = utc_now()
            next_state = "cancellation_requested" if row["cancellation_requested"] else state
            updated = connection.execute(
                """UPDATE baseten_outbox SET state = ?, dispatch_token = NULL, dispatch_lease_expires_at = NULL,
                   last_error_json = ?, updated_at = ?
                   WHERE outbox_id = ? AND state = 'dispatching' AND dispatch_token = ?""",
                (next_state, canonical_json(dict(error)), now, outbox_id, dispatch_token),
            )
            if updated.rowcount != 1:
                connection.commit()
                return False
            connection.execute(
                """UPDATE attempts SET status = ?, completed_at = ?, error_json = ?
                   WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
                (
                    "ambiguous" if state == "ambiguous" else "rejected",
                    now,
                    canonical_json(dict(error)),
                    row["run_id"],
                    row["episode_id"],
                    int(row["post_attempt_count"]) + 1,
                    outbox_id,
                ),
            )
            self._append_baseten_event(connection, outbox_id, event_type, {"error": dict(error)})
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_ambiguity(
        self, outbox_id: str, dispatch_token: str, error: Mapping[str, Any]
    ) -> bool:
        """Lock a possibly accepted POST; no automatic requeue is available."""

        return self._record_baseten_dispatch_outcome(
            outbox_id, dispatch_token, "ambiguous", error, "post_ambiguous"
        )

    def record_baseten_known_rejection(
        self, outbox_id: str, dispatch_token: str, error: Mapping[str, Any]
    ) -> bool:
        """Record a response that proved this POST was not accepted."""

        return self._record_baseten_dispatch_outcome(
            outbox_id, dispatch_token, "rejected", error, "post_rejected"
        )

    def recover_expired_baseten_dispatches(self, now: Optional[str] = None) -> int:
        """Turn crashed pre-ack POSTs into durable ambiguity, never a retry."""

        current = now or utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM baseten_outbox WHERE state = 'dispatching'
                   AND dispatch_lease_expires_at IS NOT NULL AND dispatch_lease_expires_at <= ?""",
                (current,),
            ).fetchall()
            for row in rows:
                error = {"kind": "dispatch_lease_expired", "message": "process may have crashed after POST began"}
                next_state = "cancellation_requested" if row["cancellation_requested"] else "ambiguous"
                connection.execute(
                    """UPDATE baseten_outbox SET state = ?, dispatch_token = NULL, dispatch_lease_expires_at = NULL,
                       last_error_json = ?, updated_at = ? WHERE outbox_id = ? AND state = 'dispatching'""",
                    (next_state, canonical_json(error), current, row["outbox_id"]),
                )
                connection.execute(
                    """UPDATE attempts SET status = 'ambiguous', completed_at = ?, error_json = ?
                       WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
                    (
                        current,
                        canonical_json(error),
                        row["run_id"],
                        row["episode_id"],
                        int(row["post_attempt_count"]),
                        row["outbox_id"],
                    ),
                )
                self._append_baseten_event(connection, row["outbox_id"], "dispatch_recovered_ambiguous", error)
            connection.commit()
            return len(rows)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def requeue_baseten_known_rejection(self, outbox_id: str, operator_reason: str) -> bool:
        """Explicitly requeue only a proved rejection; ambiguous POSTs stay locked."""

        if not operator_reason.strip():
            raise ValueError("operator_reason is required for a manual requeue")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] != "rejected" or row["platform_request_id"] is not None:
                connection.commit()
                return False
            episode = connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ? AND run_id = ?", (row["episode_id"], row["run_id"])
            ).fetchone()
            if episode is None or episode["status"] != "running" or episode["attempt_token"] != outbox_id:
                raise ConfigurationError("known-rejection requeue lost its logical-episode association")
            current = utc_now()
            connection.execute(
                """UPDATE baseten_outbox SET state = 'prepared', last_error_json = NULL, updated_at = ?
                   WHERE outbox_id = ? AND state = 'rejected'""",
                (current, outbox_id),
            )
            connection.execute(
                """UPDATE episodes SET status = 'planned', attempt_token = NULL, lease_expires_at = NULL,
                   updated_at = ? WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                (current, row["episode_id"], outbox_id),
            )
            self._append_baseten_event(
                connection, outbox_id, "known_rejection_requeued", {"operator_reason": operator_reason.strip()}
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_callback(
        self,
        request_id: str,
        raw_body: bytes,
        payload: Mapping[str, Any],
        event_type: str,
        signature_header: Optional[str] = None,
        request_id_header: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Durably accept an authenticated webhook, including callback-before-ack.

        Both raw signed bytes and canonical authenticated JSON are persisted.
        The former preserves HMAC evidence exactly; the latter supports indexed
        control-plane reads without re-parsing untrusted bytes.
        """

        if not request_id or not event_type:
            raise ValueError("request_id and event_type are required")
        body_digest = self._baseten_body_digest(raw_body)
        payload_json = canonical_json(dict(payload))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["body_sha256"] != body_digest
                    or existing["payload_json"] != payload_json
                    or existing["raw_body"] != raw_body
                ):
                    raise CallbackConflictError("same Baseten request ID arrived with different callback data")
                connection.commit()
                callback = self._baseten_callback_from_row(existing)
                callback["duplicate"] = True
                return callback
            outbox = connection.execute(
                "SELECT * FROM baseten_outbox WHERE platform_request_id = ?", (request_id,)
            ).fetchone()
            outbox_id = outbox["outbox_id"] if outbox is not None else None
            state = "ready" if outbox_id is not None else "unassociated"
            connection.execute(
                """INSERT INTO baseten_callbacks(
                     request_id, outbox_id, raw_body, body_sha256, payload_json, event_type,
                     signature_header, request_id_header, received_at, state
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    outbox_id,
                    raw_body,
                    body_digest,
                    payload_json,
                    event_type,
                    signature_header,
                    request_id_header,
                    utc_now(),
                    state,
                ),
            )
            if outbox is not None:
                if outbox["state"] not in ("finalized",):
                    connection.execute(
                        "UPDATE baseten_outbox SET state = 'callback_received', updated_at = ? WHERE outbox_id = ?",
                        (utc_now(), outbox_id),
                    )
                self._append_baseten_event(
                    connection, outbox_id, "callback_received", {"request_id": request_id}
                )
            callback = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE request_id = ?", (request_id,)
            ).fetchone()
            connection.commit()
            result = self._baseten_callback_from_row(callback)
            result["duplicate"] = False
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_baseten_callback(self, request_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM baseten_callbacks WHERE request_id = ?", (request_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten callback not found: %s" % request_id)
            return self._baseten_callback_from_row(row)

    def request_baseten_cancellation(self, outbox_id: str) -> Dict[str, Any]:
        """Persist cancellation intent before any optional remote DELETE attempt.

        A callback that arrives after this point remains durable and must still
        be artifact-finalized.  Local cancellation must not erase allocated
        remote work or make its cost invisible.
        """

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] == "finalized":
                connection.commit()
                return self._baseten_outbox_from_row(row)
            next_state = row["state"]
            if row["state"] in ("prepared", "rejected"):
                next_state = "cancelled_pre_dispatch"
            elif row["state"] in (
                "dispatching",
                "callback_received",
                "callback_finalizing",
                "cancellation_dispatching",
                "cancellation_ambiguous",
                "remote_terminal_without_callback",
            ):
                # Do not overwrite an in-flight POST/DELETE or callback CAS.
                # The owner still has to record its request-ID/outcome before
                # the cancellation intent can be reconciled.
                next_state = row["state"]
            else:
                next_state = "cancellation_requested"
            current = utc_now()
            if row["state"] == "rejected":
                # The HTTP response proved this request never entered the
                # platform queue. Return its logical row to the pre-dispatch
                # shape so the required cancellation-artifact finalizer can
                # close it without manufacturing remote work.
                connection.execute(
                    """UPDATE episodes SET status = 'planned', attempt_token = NULL, lease_expires_at = NULL,
                       updated_at = ? WHERE episode_id = ? AND run_id = ? AND status = 'running'
                         AND attempt_token = ?""",
                    (current, row["episode_id"], row["run_id"], outbox_id),
                )
            connection.execute(
                """UPDATE baseten_outbox SET cancellation_requested = 1,
                   cancellation_requested_at = COALESCE(cancellation_requested_at, ?), state = ?, updated_at = ?
                   WHERE outbox_id = ?""",
                (current, next_state, current, outbox_id),
            )
            self._append_baseten_event(connection, outbox_id, "cancellation_requested", {"state": next_state})
            result = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            connection.commit()
            return self._baseten_outbox_from_row(result)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_baseten_remote_cancellation(
        self, outbox_id: str, cancel_token: str, lease_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Persist a potential remote DELETE boundary before issuing it."""

        lease_seconds = self._positive_finite(lease_seconds, "lease_seconds")
        if not cancel_token:
            raise ValueError("cancel_token and positive lease_seconds are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if (
                row["state"] != "cancellation_requested"
                or not row["cancellation_requested"]
                or not row["platform_request_id"]
            ):
                connection.commit()
                return None
            current = utc_now()
            connection.execute(
                """UPDATE baseten_outbox SET state = 'cancellation_dispatching', cancel_token = ?,
                   cancel_lease_expires_at = ?, cancel_dispatch_started_at = ?, updated_at = ?
                   WHERE outbox_id = ? AND state = 'cancellation_requested'""",
                (cancel_token, lease_deadline(lease_seconds), current, current, outbox_id),
            )
            self._append_baseten_event(connection, outbox_id, "remote_cancel_started", {})
            claimed = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            connection.commit()
            return self._baseten_outbox_from_row(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_remote_cancellation_status(
        self, outbox_id: str, cancel_token: str, remote_status: str
    ) -> bool:
        """Record a non-ambiguous remote cancellation observation.

        A cancelled request still needs an explicit service-failure/cancellation
        artifact before its logical episode is terminalized, so this method does
        not score it.
        """

        if not cancel_token or not remote_status:
            raise ValueError("cancel_token and remote_status are required")
        normalized = remote_status.upper()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] != "cancellation_dispatching" or row["cancel_token"] != cancel_token:
                connection.commit()
                return False
            if normalized in ("CANCELED", "FAILED", "EXPIRED", "SUCCEEDED", "WEBHOOK_FAILED"):
                next_state = "remote_terminal_without_callback"
            else:
                next_state = "cancellation_requested"
            current = utc_now()
            connection.execute(
                """UPDATE baseten_outbox SET state = ?, remote_status = ?, remote_status_observed_at = ?,
                   cancel_token = NULL, cancel_lease_expires_at = NULL, updated_at = ?
                   WHERE outbox_id = ? AND state = 'cancellation_dispatching' AND cancel_token = ?""",
                (next_state, normalized, current, current, outbox_id, cancel_token),
            )
            self._append_baseten_event(
                connection,
                outbox_id,
                "remote_cancel_observed",
                {"remote_status": normalized, "state": next_state},
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_remote_cancellation_ambiguity(
        self, outbox_id: str, cancel_token: str, error: Mapping[str, Any]
    ) -> bool:
        """Lock an uncertain DELETE until a verified lifecycle probe resolves it."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] != "cancellation_dispatching" or row["cancel_token"] != cancel_token:
                connection.commit()
                return False
            current = utc_now()
            connection.execute(
                """UPDATE baseten_outbox SET state = 'cancellation_ambiguous', cancel_token = NULL,
                   cancel_lease_expires_at = NULL, last_error_json = ?, updated_at = ?
                   WHERE outbox_id = ? AND state = 'cancellation_dispatching' AND cancel_token = ?""",
                (canonical_json(dict(error)), current, outbox_id, cancel_token),
            )
            self._append_baseten_event(connection, outbox_id, "remote_cancel_ambiguous", {"error": dict(error)})
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def recover_expired_baseten_remote_cancellations(self, now: Optional[str] = None) -> int:
        """Treat a crash around DELETE as ambiguous instead of reissuing it."""

        current = now or utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM baseten_outbox WHERE state = 'cancellation_dispatching'
                   AND cancel_lease_expires_at IS NOT NULL AND cancel_lease_expires_at <= ?""",
                (current,),
            ).fetchall()
            for row in rows:
                error = {"kind": "cancel_lease_expired", "message": "process may have crashed after DELETE began"}
                connection.execute(
                    """UPDATE baseten_outbox SET state = 'cancellation_ambiguous', cancel_token = NULL,
                       cancel_lease_expires_at = NULL, last_error_json = ?, updated_at = ?
                       WHERE outbox_id = ? AND state = 'cancellation_dispatching'""",
                    (canonical_json(error), current, row["outbox_id"]),
                )
                self._append_baseten_event(connection, row["outbox_id"], "remote_cancel_recovered_ambiguous", error)
            connection.commit()
            return len(rows)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_baseten_callback_finalization(
        self, outbox_id: str, processor_token: str, lease_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Lease one already-associated callback for idempotent artifact writing."""

        lease_seconds = self._positive_finite(lease_seconds, "lease_seconds")
        if not processor_token:
            raise ValueError("processor_token and positive lease_seconds are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outbox = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if outbox is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            callback = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE outbox_id = ?", (outbox_id,)
            ).fetchone()
            if callback is None or callback["state"] not in ("ready",):
                connection.commit()
                return None
            if outbox["state"] != "callback_received":
                connection.commit()
                return None
            current = utc_now()
            connection.execute(
                """UPDATE baseten_callbacks SET state = 'processing', processor_token = ?,
                   processor_lease_expires_at = ? WHERE request_id = ? AND state = 'ready'""",
                (processor_token, lease_deadline(lease_seconds), callback["request_id"]),
            )
            connection.execute(
                """UPDATE baseten_outbox SET state = 'callback_finalizing', updated_at = ?
                   WHERE outbox_id = ? AND state = 'callback_received'""",
                (current, outbox_id),
            )
            self._append_baseten_event(connection, outbox_id, "callback_finalization_claimed", {})
            row = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE request_id = ?", (callback["request_id"],)
            ).fetchone()
            connection.commit()
            return self._baseten_callback_from_row(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def recover_expired_baseten_callback_claims(self, now: Optional[str] = None) -> int:
        """Make interrupted callback artifact processing available again."""

        current = now or utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT * FROM baseten_callbacks WHERE state = 'processing'
                   AND processor_lease_expires_at IS NOT NULL AND processor_lease_expires_at <= ?""",
                (current,),
            ).fetchall()
            for callback in rows:
                connection.execute(
                    """UPDATE baseten_callbacks SET state = 'ready', processor_token = NULL,
                       processor_lease_expires_at = NULL WHERE request_id = ? AND state = 'processing'""",
                    (callback["request_id"],),
                )
                if callback["outbox_id"]:
                    connection.execute(
                        """UPDATE baseten_outbox SET state = 'callback_received', updated_at = ?
                           WHERE outbox_id = ? AND state = 'callback_finalizing'""",
                        (current, callback["outbox_id"]),
                    )
                    self._append_baseten_event(
                        connection, callback["outbox_id"], "callback_finalization_recovered", {}
                    )
            connection.commit()
            return len(rows)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finalise_baseten_callback(
        self,
        outbox_id: str,
        processor_token: str,
        terminal_status: str,
        artifact_refs: Mapping[str, Any],
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
        cancellation_reason: Optional[str] = None,
    ) -> bool:
        """Commit a callback's final artifact and logical terminal state once.

        The artifact writer must call this only after it has durably written the
        final artifact/object manifest referenced by ``artifact_refs``.  The
        callback claim token makes a repeated webhook or restarted finalizer a
        no-op rather than a second score.
        """

        if terminal_status not in TERMINAL_EPISODE_STATUSES:
            raise ValueError("terminal_status must be completed, failed, or cancelled")
        if not processor_token:
            raise ValueError("processor_token is required")
        _validate_platform_artifact_refs(artifact_refs)
        if terminal_status == "completed":
            required_result = {
                "validity",
                "binary_success",
                "progress_score",
                "missing_reason",
                "horizon_actions",
                "executed_actions",
                "timing",
                "run_id",
                "episode_id",
                "protocol_hash",
            }
            if result is None or not required_result.issubset(result.keys()):
                raise ValueError("completed platform callbacks require the normalized logical result fields")
        if terminal_status == "failed" and not error:
            raise ValueError("failed platform callbacks require an explicit error record")
        if terminal_status == "cancelled" and not cancellation_reason:
            raise ValueError("cancelled platform callbacks require a cancellation reason")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outbox = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if outbox is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            callback = connection.execute(
                "SELECT * FROM baseten_callbacks WHERE outbox_id = ?", (outbox_id,)
            ).fetchone()
            if callback is None:
                raise ConfigurationError("a platform callback must be associated before finalization")
            if callback["state"] == "finalized":
                connection.commit()
                return False
            if (
                callback["state"] != "processing"
                or callback["processor_token"] != processor_token
                or outbox["state"] != "callback_finalizing"
            ):
                connection.commit()
                return False
            episode = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND episode_id = ?", (outbox["run_id"], outbox["episode_id"])
            ).fetchone()
            if (
                episode is None
                or episode["status"] != "running"
                or episode["attempt_token"] != outbox_id
            ):
                raise ConfigurationError("platform callback finalization lost its logical-episode CAS association")
            if terminal_status == "completed":
                assert result is not None
                expected_request = json.loads(outbox["payload_json"])["request"]
                for field in ("run_id", "episode_id", "protocol_hash"):
                    if result[field] != expected_request[field]:
                        raise ConfigurationError("completed platform result has mismatched %s" % field)
                raw_data = json.loads(callback["payload_json"]).get("data")
                raw_output = raw_data.get("output") if isinstance(raw_data, Mapping) else None
                for declared in (raw_data, raw_output):
                    if not isinstance(declared, Mapping):
                        continue
                    for field in ("run_id", "episode_id", "protocol_hash"):
                        if field in declared and declared[field] != expected_request[field]:
                            raise ConfigurationError("authenticated callback result has mismatched %s" % field)
                success = result["binary_success"]
                if success is not None and not isinstance(success, bool):
                    raise ValueError("platform binary_success must be a boolean or null")
                horizon = result["horizon_actions"]
                if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon != episode["horizon_actions"]:
                    raise ValueError("platform result must preserve the planned exact task horizon")
                executed = result["executed_actions"]
                if isinstance(executed, bool) or not isinstance(executed, int) or executed != horizon:
                    raise ValueError("completed platform result must execute the entire planned horizon")
                missing = result["missing_reason"]
                if success is None and (not isinstance(missing, str) or not missing.strip()):
                    raise ValueError("unevaluable platform result requires an explicit missing reason")
                if success is not None and missing is not None:
                    raise ValueError("evaluable platform result cannot also assert a missing outcome")
                # The existing logical-result validator checks validity/progress
                # consistency. Cloud refs have their own typed storage contract,
                # so do not send them through the synthetic SVG artifact path.
                normalized = normalise_backend_result({**result, "artifact_refs": {}}, dict(episode))
                result = {**result, **normalized}
                json.dumps(dict(result), allow_nan=False)
            current = utc_now()
            artifact_json = canonical_json(dict(artifact_refs))
            if terminal_status == "completed":
                assert result is not None
                # Platform completions are logical completions too.  Preserve
                # the same measured fields and persisted segment projection as
                # the generic backend path; otherwise the durable callback path
                # would silently erase production telemetry and leave the
                # console wall without its committed segment events.
                extra_columns, extra_values = self._measurement_update(result)
                connection.execute(
                    """UPDATE episodes SET status = 'completed', validity = ?, binary_success = ?, progress_score = ?,
                       missing_reason = ?, horizon_actions = ?, artifact_refs_json = ?, timing_json = ?,
                       error_json = NULL, completed_at = ?, lease_expires_at = NULL, updated_at = ?%s
                       WHERE episode_id = ? AND status = 'running' AND attempt_token = ?"""
                    % extra_columns,
                    (
                        result["validity"],
                        None if result["binary_success"] is None else int(bool(result["binary_success"])),
                        result["progress_score"],
                        result["missing_reason"],
                        result["horizon_actions"],
                        artifact_json,
                        canonical_json(result["timing"]),
                        current,
                        current,
                        *extra_values,
                        outbox["episode_id"],
                        outbox_id,
                    ),
                )
                attempt_result = dict(result)
                attempt_result["artifact_refs"] = dict(artifact_refs)
                attempt_error = None
                event_name = "episode_completed"
                mode = str(episode["mode"] or "")
                event_payload = {
                    "episode_id": outbox["episode_id"],
                    "evaluable": result["binary_success"] is not None,
                    "mode": mode,
                    "synthetic": "synthetic" in mode,
                    "unqualified": "qualified" not in mode or "unqualified" in mode,
                }
            elif terminal_status == "failed":
                assert error is not None
                connection.execute(
                    """UPDATE episodes SET status = 'failed', validity = 'unknown', binary_success = NULL,
                       progress_score = NULL, missing_reason = 'service_failure', artifact_refs_json = ?,
                       error_json = ?, completed_at = ?, lease_expires_at = NULL, updated_at = ?
                       WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                    (
                        artifact_json,
                        canonical_json(dict(error)),
                        current,
                        current,
                        outbox["episode_id"],
                        outbox_id,
                    ),
                )
                attempt_result = None
                attempt_error = dict(error)
                event_name = "episode_failed"
                event_payload = {"episode_id": outbox["episode_id"], "error": dict(error)}
            else:
                assert cancellation_reason is not None
                cancelled_error = {"reason": cancellation_reason, "callback_received": True}
                connection.execute(
                    """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
                       progress_score = NULL, missing_reason = ?, artifact_refs_json = ?, error_json = ?,
                       completed_at = ?, lease_expires_at = NULL, updated_at = ?
                       WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                    (
                        cancellation_reason,
                        artifact_json,
                        canonical_json(cancelled_error),
                        current,
                        current,
                        outbox["episode_id"],
                        outbox_id,
                    ),
                )
                attempt_result = None
                attempt_error = cancelled_error
                event_name = "episode_cancelled"
                event_payload = {"episode_id": outbox["episode_id"], "reason": cancellation_reason}
            connection.execute(
                """UPDATE attempts SET status = ?, completed_at = ?, result_json = ?, error_json = ?
                   WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
                (
                    terminal_status,
                    current,
                    canonical_json(attempt_result) if attempt_result is not None else None,
                    canonical_json(attempt_error) if attempt_error is not None else None,
                    outbox["run_id"],
                    outbox["episode_id"],
                    episode["attempt_count"],
                    outbox_id,
                ),
            )
            finalization = {
                "terminal_status": terminal_status,
                "artifact_refs": dict(artifact_refs),
                "result": dict(result) if result is not None else None,
                "error": dict(error) if error is not None else attempt_error,
                "cancellation_reason": cancellation_reason,
            }
            connection.execute(
                """UPDATE baseten_callbacks SET state = 'finalized', processor_lease_expires_at = NULL,
                   finalization_json = ? WHERE request_id = ? AND state = 'processing' AND processor_token = ?""",
                (canonical_json(finalization), callback["request_id"], processor_token),
            )
            connection.execute(
                """UPDATE baseten_outbox SET state = 'finalized', dispatch_token = NULL,
                   dispatch_lease_expires_at = NULL, updated_at = ? WHERE outbox_id = ?
                     AND state = 'callback_finalizing'""",
                (current, outbox_id),
            )
            if terminal_status == "completed":
                assert result is not None
                # The callback receipt, episode projection, and every segment
                # event commit in this one transaction.  A client can therefore
                # never render frames from a callback whose logical outcome did
                # not make it into the ledger.
                for segment in result.get("segments") or ():
                    if not isinstance(segment, Mapping):
                        continue
                    self._append_event(
                        connection,
                        outbox["run_id"],
                        "segment_completed",
                        {
                            "episode_id": outbox["episode_id"],
                            "policy": episode["policy"],
                            "task": episode["task"],
                            "run_id": outbox["run_id"],
                            "segment_index": segment.get("segment_index"),
                            "frame_urls": list(segment.get("frame_urls") or []),
                            "certified_frame_count": segment.get("certified_frame_count"),
                            "provenance": segment.get("provenance"),
                            "status": segment.get("status"),
                        },
                    )
            self._append_event(connection, outbox["run_id"], event_name, event_payload)
            self._append_baseten_event(
                connection, outbox_id, "callback_finalized", {"terminal_status": terminal_status}
            )
            self._finalise_run_state(connection, outbox["run_id"])
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finalise_baseten_pre_dispatch_cancellation(
        self, outbox_id: str, artifact_refs: Mapping[str, Any], reason: str = "cancelled_before_submission"
    ) -> bool:
        """Write the cancellation artifact before terminalizing a never-posted episode."""

        _validate_platform_artifact_refs(artifact_refs)
        if not reason:
            raise ValueError("a cancellation reason is required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outbox = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if outbox is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if outbox["state"] == "finalized":
                connection.commit()
                return False
            if outbox["state"] != "cancelled_pre_dispatch":
                connection.commit()
                return False
            episode = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND episode_id = ?", (outbox["run_id"], outbox["episode_id"])
            ).fetchone()
            if episode is None or episode["status"] != "planned":
                raise ConfigurationError("pre-dispatch cancellation lost its planned logical episode")
            current = utc_now()
            error = {"reason": reason, "platform_post_started": False}
            connection.execute(
                """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
                   progress_score = NULL, missing_reason = ?, artifact_refs_json = ?, error_json = ?,
                   completed_at = ?, updated_at = ? WHERE episode_id = ? AND status = 'planned'""",
                (
                    reason,
                    canonical_json(dict(artifact_refs)),
                    canonical_json(error),
                    current,
                    current,
                    outbox["episode_id"],
                ),
            )
            connection.execute(
                "UPDATE baseten_outbox SET state = 'finalized', updated_at = ? WHERE outbox_id = ?",
                (current, outbox_id),
            )
            self._append_event(
                connection, outbox["run_id"], "episode_cancelled", {"episode_id": outbox["episode_id"], "reason": reason}
            )
            self._append_baseten_event(
                connection, outbox_id, "pre_dispatch_cancellation_finalized", {"reason": reason}
            )
            self._finalise_run_state(connection, outbox["run_id"])
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finalise_baseten_remote_terminal(
        self,
        outbox_id: str,
        terminal_status: str,
        artifact_refs: Mapping[str, Any],
        error: Optional[Mapping[str, Any]] = None,
        cancellation_reason: Optional[str] = None,
    ) -> bool:
        """Close a remotely terminal request whose callback will not arrive.

        This path is deliberately unavailable for a remote ``SUCCEEDED`` or
        ``WEBHOOK_FAILED`` observation: those states have no result output, so
        an external Chain/object-store recovery must first provide one.  Failed,
        expired, or cancelled work can instead receive an explicit service
        record, preserving coverage and cost without inventing a score.
        """

        if terminal_status not in ("failed", "cancelled"):
            raise ValueError("remote callback-less finalization must be failed or cancelled")
        _validate_platform_artifact_refs(artifact_refs)
        if terminal_status == "failed" and not error:
            raise ValueError("failed remote terminal finalization requires an error record")
        if terminal_status == "cancelled" and not cancellation_reason:
            raise ValueError("cancelled remote terminal finalization requires a reason")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outbox = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if outbox is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if outbox["state"] == "finalized":
                connection.commit()
                return False
            if outbox["state"] != "remote_terminal_without_callback":
                connection.commit()
                return False
            if str(outbox["remote_status"] or "").upper() not in ("FAILED", "EXPIRED", "CANCELED"):
                raise ConfigurationError("remote terminal finalization requires failed, expired, or cancelled status")
            callback = connection.execute(
                "SELECT request_id FROM baseten_callbacks WHERE outbox_id = ?", (outbox_id,)
            ).fetchone()
            if callback is not None:
                raise ConfigurationError("associated callback must use callback finalization")
            episode = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND episode_id = ?", (outbox["run_id"], outbox["episode_id"])
            ).fetchone()
            if episode is None or episode["status"] != "running" or episode["attempt_token"] != outbox_id:
                raise ConfigurationError("remote terminal finalization lost its logical-episode CAS association")
            current = utc_now()
            if terminal_status == "failed":
                assert error is not None
                attempt_error = dict(error)
                connection.execute(
                    """UPDATE episodes SET status = 'failed', validity = 'unknown', binary_success = NULL,
                       progress_score = NULL, missing_reason = 'service_failure', artifact_refs_json = ?,
                       error_json = ?, completed_at = ?, lease_expires_at = NULL, updated_at = ?
                       WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                    (
                        canonical_json(dict(artifact_refs)),
                        canonical_json(attempt_error),
                        current,
                        current,
                        outbox["episode_id"],
                        outbox_id,
                    ),
                )
                event_type = "episode_failed"
                event_payload = {"episode_id": outbox["episode_id"], "error": attempt_error}
            else:
                assert cancellation_reason is not None
                attempt_error = {"reason": cancellation_reason, "remote_status": outbox["remote_status"]}
                connection.execute(
                    """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
                       progress_score = NULL, missing_reason = ?, artifact_refs_json = ?, error_json = ?,
                       completed_at = ?, lease_expires_at = NULL, updated_at = ?
                       WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                    (
                        cancellation_reason,
                        canonical_json(dict(artifact_refs)),
                        canonical_json(attempt_error),
                        current,
                        current,
                        outbox["episode_id"],
                        outbox_id,
                    ),
                )
                event_type = "episode_cancelled"
                event_payload = {"episode_id": outbox["episode_id"], "reason": cancellation_reason}
            connection.execute(
                """UPDATE attempts SET status = ?, completed_at = ?, error_json = ?
                   WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
                (
                    terminal_status,
                    current,
                    canonical_json(attempt_error),
                    outbox["run_id"],
                    outbox["episode_id"],
                    episode["attempt_count"],
                    outbox_id,
                ),
            )
            connection.execute(
                "UPDATE baseten_outbox SET state = 'finalized', updated_at = ? WHERE outbox_id = ?",
                (current, outbox_id),
            )
            self._append_event(connection, outbox["run_id"], event_type, event_payload)
            self._append_baseten_event(
                connection,
                outbox_id,
                "remote_terminal_finalized",
                {"terminal_status": terminal_status, "remote_status": outbox["remote_status"]},
            )
            self._finalise_run_state(connection, outbox["run_id"])
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_baseten_remote_status(self, outbox_id: str, remote_status: str) -> bool:
        """Record a reconciler observation without pretending it includes output data."""

        if not remote_status:
            raise ValueError("remote_status is required")
        normalized = remote_status.upper()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM baseten_outbox WHERE outbox_id = ?", (outbox_id,)).fetchone()
            if row is None:
                raise KeyError("Baseten outbox record not found: %s" % outbox_id)
            if row["state"] == "finalized":
                connection.commit()
                return False
            next_state = row["state"]
            # Status polling is lifecycle-only per Baseten's documented API;
            # terminal observations cannot score an episode without its stored
            # result artifact/callback receipt.
            if normalized in ("SUCCEEDED", "WEBHOOK_FAILED"):
                next_state = "callback_missing"
            elif normalized in ("FAILED", "EXPIRED", "CANCELED"):
                next_state = "remote_terminal_without_callback"
            elif row["cancellation_requested"]:
                next_state = "cancellation_requested"
            elif row["state"] in ("accepted", "callback_missing", "remote_terminal_without_callback"):
                next_state = "accepted"
            current = utc_now()
            connection.execute(
                """UPDATE baseten_outbox SET state = ?, remote_status = ?, remote_status_observed_at = ?,
                   updated_at = ? WHERE outbox_id = ?""",
                (next_state, normalized, current, current, outbox_id),
            )
            self._append_baseten_event(
                connection, outbox_id, "remote_status_observed", {"remote_status": normalized, "state": next_state}
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def acquire_run_lease(
        self,
        run_id: str,
        owner_token: str,
        lease_seconds: float,
        force: bool = False,
        recovery: bool = False,
    ) -> bool:
        """Acquire a single-owner lease, without silently stealing live work."""

        if not owner_token or lease_seconds <= 0:
            raise ValueError("owner_token and a positive lease duration are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if run["status"] in ("completed", "cancelled", "failed"):
                connection.commit()
                return False
            now = utc_now()
            has_live_owner = bool(
                run["owner_token"]
                and run["owner_token"] != owner_token
                and run["owner_lease_expires_at"]
                and str(run["owner_lease_expires_at"]) > now
            )
            if has_live_owner and not force:
                connection.commit()
                raise LeaseActiveError("run has an active owner lease; use force only after deliberate review")
            connection.execute(
                """UPDATE runs SET owner_token = ?, owner_heartbeat_at = ?, owner_lease_expires_at = ?,
                   updated_at = ? WHERE id = ?""",
                (owner_token, now, lease_deadline(lease_seconds), now, run_id),
            )
            self._append_event(
                connection,
                run_id,
                "run_lease_acquired",
                {"recovery": recovery, "forced": bool(force and has_live_owner)},
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat_run_lease(self, run_id: str, owner_token: str, lease_seconds: float) -> bool:
        """Renew the owner and every current attempt lease using token CAS."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            deadline = lease_deadline(lease_seconds)
            updated = connection.execute(
                """UPDATE runs SET owner_heartbeat_at = ?, owner_lease_expires_at = ?, updated_at = ?
                   WHERE id = ? AND owner_token = ? AND status NOT IN ('completed', 'cancelled', 'failed')""",
                (now, deadline, now, run_id, owner_token),
            )
            if updated.rowcount == 1:
                connection.execute(
                    """UPDATE episodes SET lease_expires_at = ?, updated_at = ?
                       WHERE run_id = ? AND status = 'running'
                         AND NOT EXISTS (
                           SELECT 1 FROM baseten_outbox AS outbox
                           WHERE outbox.run_id = episodes.run_id AND outbox.episode_id = episodes.episode_id
                         )""",
                    (deadline, now, run_id),
                )
            connection.commit()
            return updated.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release_run_lease(self, run_id: str, owner_token: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE runs SET owner_token = NULL, owner_heartbeat_at = NULL, owner_lease_expires_at = NULL,
                   updated_at = ? WHERE id = ? AND owner_token = ?""",
                (utc_now(), run_id, owner_token),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin_run(self, run_id: str, owner_token: str) -> bool:
        """Transition a startable run to running.  Returns false if cancelled."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if run["owner_token"] != owner_token:
                connection.commit()
                return False
            if run["cancellation_requested"]:
                connection.commit()
                return False
            if run["status"] in ("completed", "cancelled", "failed"):
                connection.commit()
                return False
            if run["status"] != "running":
                now = utc_now()
                connection.execute("UPDATE runs SET status = 'running', updated_at = ? WHERE id = ?", (now, run_id))
                self._append_event(connection, run_id, "run_started", {})
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim_next_episode(
        self, run_id: str, owner_token: str, attempt_token: str, lease_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Atomically lease one planned logical episode to a worker."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT cancellation_requested, owner_token, owner_lease_expires_at FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if run["cancellation_requested"]:
                connection.commit()
                return None
            if (
                run["owner_token"] != owner_token
                or not run["owner_lease_expires_at"]
                or str(run["owner_lease_expires_at"]) <= utc_now()
            ):
                connection.commit()
                return None
            row = connection.execute(
                """SELECT * FROM episodes AS episode WHERE run_id = ? AND status = 'planned'
                   AND NOT EXISTS (
                     SELECT 1 FROM baseten_outbox AS outbox
                     WHERE outbox.run_id = episode.run_id AND outbox.episode_id = episode.episode_id
                   ) ORDER BY policy, task, start_id, episode_id LIMIT 1""",
                (run_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now = utc_now()
            attempt_no = int(row["attempt_count"]) + 1
            lease_expires_at = lease_deadline(lease_seconds)
            updated = connection.execute(
                """UPDATE episodes SET status = 'running', attempt_count = ?, submitted_at = COALESCE(submitted_at, ?),
                   started_at = ?, attempt_token = ?, lease_expires_at = ?, updated_at = ?
                   WHERE episode_id = ? AND status = 'planned'""",
                (attempt_no, now, now, attempt_token, lease_expires_at, now, row["episode_id"]),
            )
            if updated.rowcount != 1:
                connection.commit()
                return None
            connection.execute(
                """INSERT INTO attempts(run_id, episode_id, attempt_no, status, attempt_token, started_at)
                   VALUES (?, ?, ?, 'running', ?, ?)""",
                (run_id, row["episode_id"], attempt_no, attempt_token, now),
            )
            self._append_event(
                connection,
                run_id,
                "episode_started",
                {"episode_id": row["episode_id"], "attempt": attempt_no, "synthetic": True},
            )
            final_row = connection.execute("SELECT * FROM episodes WHERE episode_id = ?", (row["episode_id"],)).fetchone()
            connection.commit()
            return self._episode_from_row(final_row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancellation_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT cancellation_requested FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError("run not found: %s" % run_id)
            return bool(row["cancellation_requested"])

    def _set_attempt_terminal(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        episode_id: str,
        status: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row = connection.execute(
            "SELECT attempt_count, attempt_token FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None or not row["attempt_count"]:
            return
        connection.execute(
            """UPDATE attempts SET status = ?, completed_at = ?, result_json = ?, error_json = ?
               WHERE run_id = ? AND episode_id = ? AND attempt_no = ? AND attempt_token = ?""",
            (
                status,
                utc_now(),
                canonical_json(dict(result)) if result is not None else None,
                canonical_json(dict(error)) if error is not None else None,
                run_id,
                episode_id,
                row["attempt_count"],
                row["attempt_token"],
            ),
        )

    def _current_owned_attempt(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        episode_id: str,
        owner_token: Optional[str],
        attempt_token: Optional[str],
    ) -> Optional[sqlite3.Row]:
        """Return a running row only when both owner and attempt tokens match."""

        if not owner_token or not attempt_token:
            return None
        run = connection.execute("SELECT owner_token FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None or run["owner_token"] != owner_token:
            return None
        row = connection.execute(
            "SELECT * FROM episodes WHERE episode_id = ? AND run_id = ?", (episode_id, run_id)
        ).fetchone()
        if row is None or row["status"] != "running" or row["attempt_token"] != attempt_token:
            return None
        return row

    @staticmethod
    def _measurement_update(result: Mapping[str, Any]) -> tuple:
        """Build the optional spec-section-8 tail of a terminal UPDATE.

        Only keys the backend actually supplied are written, so an absent
        measurement stays ``NULL`` (unknown) instead of being overwritten with a
        zero.  ``platform_request_ids`` and ``attempt_ids`` are appended rather
        than replaced by the caller, which is why they are JSON columns.
        """

        fragments: List[str] = []
        values: List[Any] = []
        for name in _RESULT_MEASUREMENT_FIELDS:
            if name in result:
                fragments.append(name + " = ?")
                values.append(result[name])
        for name, column in (
            ("platform_request_ids", "platform_request_ids_json"),
            ("attempt_ids", "attempt_ids_json"),
            ("segments", "segments_json"),
        ):
            if name in result:
                fragments.append(column + " = ?")
                values.append(canonical_json(list(result[name])))
        for name, column in (
            ("policy_identity", "policy_identity_json"),
            ("world_identity", "world_identity_json"),
            ("judge_identity", "judge_identity_json"),
            ("seeds", "seeds_json"),
        ):
            if name in result:
                fragments.append(column + " = ?")
                values.append(canonical_json(dict(result[name])))
        for name in ("feedback_mode", "parity_status"):
            if name in result:
                fragments.append(name + " = ?")
                values.append(result[name])
        if not fragments:
            return "", ()
        return ", " + ", ".join(fragments), tuple(values)

    def complete_episode(
        self,
        run_id: str,
        episode_id: str,
        result: Mapping[str, Any],
        owner_token: Optional[str] = None,
        attempt_token: Optional[str] = None,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._current_owned_attempt(connection, run_id, episode_id, owner_token, attempt_token)
            if row is None:
                connection.commit()
                return False
            now = utc_now()
            extra_columns, extra_values = self._measurement_update(result)
            connection.execute(
                """UPDATE episodes SET status = 'completed', validity = ?, binary_success = ?, progress_score = ?,
                   missing_reason = ?, horizon_actions = ?, artifact_refs_json = ?, timing_json = ?,
                   completed_at = ?, lease_expires_at = NULL, updated_at = ?%s
                   WHERE episode_id = ? AND status = 'running' AND attempt_token = ?"""
                % extra_columns,
                (
                    result["validity"],
                    None if result["binary_success"] is None else int(bool(result["binary_success"])),
                    result["progress_score"],
                    result["missing_reason"],
                    result["horizon_actions"],
                    canonical_json(result["artifact_refs"]),
                    canonical_json(result["timing"]),
                    now,
                    now,
                    *extra_values,
                    episode_id,
                    attempt_token,
                ),
            )
            self._set_attempt_terminal(connection, run_id, episode_id, "completed", result=result)
            mode = str(row["mode"] or "")
            # One segment_completed event per persisted segment. The console's
            # rollout wall accumulates frames from these; emitting them from the
            # same transaction that finalises the episode means a tile can never
            # show frames for an episode the ledger did not commit.
            for segment in result.get("segments") or ():
                if not isinstance(segment, Mapping):
                    continue
                self._append_event(
                    connection,
                    run_id,
                    "segment_completed",
                    {
                        "episode_id": episode_id,
                        "policy": row["policy"],
                        "task": row["task"],
                        "run_id": run_id,
                        "segment_index": segment.get("segment_index"),
                        "frame_urls": list(segment.get("frame_urls") or []),
                        # Null rather than a default: an unreported certified
                        # count must show as unreported, never as an assumed 16.
                        "certified_frame_count": segment.get("certified_frame_count"),
                        "provenance": segment.get("provenance"),
                        "status": segment.get("status"),
                    },
                )
            self._append_event(
                connection,
                run_id,
                "episode_completed",
                {
                    "episode_id": episode_id,
                    "evaluable": result["binary_success"] is not None,
                    # Derived from the episode's own recorded mode rather than
                    # asserted, so a real backend is never labelled synthetic
                    # and a fixture is never labelled qualified.
                    "mode": mode,
                    "synthetic": "synthetic" in mode,
                    "unqualified": "qualified" not in mode or "unqualified" in mode,
                },
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def fail_episode(
        self,
        run_id: str,
        episode_id: str,
        error: Mapping[str, Any],
        artifact_refs: Mapping[str, Any],
        owner_token: Optional[str] = None,
        attempt_token: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._current_owned_attempt(connection, run_id, episode_id, owner_token, attempt_token)
            if row is None:
                connection.commit()
                return False
            now = utc_now()
            extra_columns, extra_values = self._measurement_update(metadata or {})
            connection.execute(
                """UPDATE episodes SET status = 'failed', validity = 'unknown', binary_success = NULL,
                   progress_score = NULL, missing_reason = 'service_failure', artifact_refs_json = ?,
                   error_json = ?, completed_at = ?, lease_expires_at = NULL, updated_at = ?%s
                   WHERE episode_id = ? AND status = 'running' AND attempt_token = ?"""
                % extra_columns,
                (
                    canonical_json(dict(artifact_refs)),
                    canonical_json(dict(error)),
                    now,
                    now,
                    *extra_values,
                    episode_id,
                    attempt_token,
                ),
            )
            self._set_attempt_terminal(connection, run_id, episode_id, "failed", error=error)
            self._append_event(connection, run_id, "episode_failed", {"episode_id": episode_id, "error": dict(error)})
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancel_episode(
        self,
        run_id: str,
        episode_id: str,
        reason: str,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        owner_token: Optional[str] = None,
        attempt_token: Optional[str] = None,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM episodes WHERE episode_id = ? AND run_id = ?", (episode_id, run_id)).fetchone()
            if row is None or row["status"] not in ("planned", "running"):
                connection.commit()
                return False
            if row["status"] == "running" and self._current_owned_attempt(
                connection, run_id, episode_id, owner_token, attempt_token
            ) is None:
                connection.commit()
                return False
            now = utc_now()
            connection.execute(
                """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
                   progress_score = NULL, missing_reason = ?, artifact_refs_json = ?, completed_at = ?, updated_at = ?
                   WHERE episode_id = ? AND status = ? AND (attempt_token IS ? OR attempt_token = ?)""",
                (
                    reason,
                    canonical_json(dict(artifact_refs or {})),
                    now,
                    now,
                    episode_id,
                    row["status"],
                    attempt_token,
                    attempt_token,
                ),
            )
            if row["status"] == "running":
                self._set_attempt_terminal(connection, run_id, episode_id, "cancelled")
            self._append_event(connection, run_id, "episode_cancelled", {"episode_id": episode_id, "reason": reason})
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _cancel_planned(self, connection: sqlite3.Connection, run_id: str, reason: str) -> int:
        rows = connection.execute(
            """SELECT episode_id FROM episodes AS episode
               WHERE run_id = ? AND status = 'planned'
                 AND NOT EXISTS (
                   SELECT 1 FROM baseten_outbox AS outbox
                   WHERE outbox.run_id = episode.run_id AND outbox.episode_id = episode.episode_id
                 )""",
            (run_id,),
        ).fetchall()
        if not rows:
            return 0
        now = utc_now()
        connection.execute(
            """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
               progress_score = NULL, missing_reason = ?, completed_at = ?, updated_at = ?
               WHERE run_id = ? AND status = 'planned'
                 AND NOT EXISTS (
                   SELECT 1 FROM baseten_outbox AS outbox
                   WHERE outbox.run_id = episodes.run_id AND outbox.episode_id = episodes.episode_id
                 )""",
            (reason, now, now, run_id),
        )
        for row in rows:
            self._append_event(connection, run_id, "episode_cancelled", {"episode_id": row["episode_id"], "reason": reason})
        return len(rows)

    def _request_baseten_cancellation_for_run(self, connection: sqlite3.Connection, run_id: str) -> int:
        """Persist cancellation intent for every cloud-owned logical episode.

        Generic fixture cancellation has no authority to manufacture a terminal
        artifact for remote work.  It marks intent here; the outbox owns any
        later request-ID association, remote cancel reconciliation, and final
        artifact-before-terminal transition.
        """

        rows = connection.execute(
            "SELECT * FROM baseten_outbox WHERE run_id = ? AND state != 'finalized'", (run_id,)
        ).fetchall()
        current = utc_now()
        changed = 0
        for row in rows:
            next_state = row["state"]
            if row["state"] in ("prepared", "rejected"):
                next_state = "cancelled_pre_dispatch"
            elif row["state"] in (
                "dispatching",
                "callback_received",
                "callback_finalizing",
                "cancellation_dispatching",
                "cancellation_ambiguous",
                "remote_terminal_without_callback",
            ):
                next_state = row["state"]
            else:
                next_state = "cancellation_requested"
            if row["state"] == "rejected":
                connection.execute(
                    """UPDATE episodes SET status = 'planned', attempt_token = NULL, lease_expires_at = NULL,
                       updated_at = ? WHERE run_id = ? AND episode_id = ? AND status = 'running'
                         AND attempt_token = ?""",
                    (current, run_id, row["episode_id"], row["outbox_id"]),
                )
            connection.execute(
                """UPDATE baseten_outbox SET cancellation_requested = 1,
                   cancellation_requested_at = COALESCE(cancellation_requested_at, ?), state = ?, updated_at = ?
                   WHERE outbox_id = ?""",
                (current, next_state, current, row["outbox_id"]),
            )
            self._append_baseten_event(
                connection, row["outbox_id"], "cancellation_requested", {"state": next_state, "scope": "run"}
            )
            changed += 1
        return changed

    def request_cancellation(self, run_id: str) -> Dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if not run["cancellation_requested"]:
                connection.execute(
                    "UPDATE runs SET cancellation_requested = 1, updated_at = ? WHERE id = ?", (utc_now(), run_id)
                )
                cancelled = self._cancel_planned(connection, run_id, "cancelled_before_execution")
                outbox_requested = self._request_baseten_cancellation_for_run(connection, run_id)
                self._append_event(
                    connection,
                    run_id,
                    "run_cancellation_requested",
                    {"cancelled_planned": cancelled, "baseten_cancellation_requested": outbox_requested},
                )
            self._finalise_run_state(connection, run_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def start_cancellation(self, run_id: str) -> Dict[str, Any]:
        """Stop further claims without terminalizing planned rows.

        The service uses this short transaction before writing a per-episode
        cancellation JSON artifact, preserving the final-artifact-before-final
        status ordering even for work that never reached a worker.
        """

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if not run["cancellation_requested"]:
                connection.execute(
                    "UPDATE runs SET cancellation_requested = 1, updated_at = ? WHERE id = ?", (utc_now(), run_id)
                )
                outbox_requested = self._request_baseten_cancellation_for_run(connection, run_id)
                self._append_event(
                    connection,
                    run_id,
                    "run_cancellation_requested",
                    {"pending_artifacts": True, "baseten_cancellation_requested": outbox_requested},
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def planned_episodes(self, run_id: str) -> List[Dict[str, Any]]:
        """Return all still-unclaimed logical rows for cancellation finalization."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM episodes AS episode WHERE run_id = ? AND status = 'planned'
                   AND NOT EXISTS (
                     SELECT 1 FROM baseten_outbox AS outbox
                     WHERE outbox.run_id = episode.run_id AND outbox.episode_id = episode.episode_id
                   ) ORDER BY policy, task, start_id, episode_id""",
                (run_id,),
            ).fetchall()
            return [self._episode_from_row(row) for row in rows]

    def running_episodes(self, run_id: str) -> List[Dict[str, Any]]:
        """Return in-flight records for recovery reconciliation only."""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM episodes AS episode WHERE run_id = ? AND status = 'running'
                   AND NOT EXISTS (
                     SELECT 1 FROM baseten_outbox AS outbox
                     WHERE outbox.run_id = episode.run_id AND outbox.episode_id = episode.episode_id
                   ) ORDER BY policy, task, start_id, episode_id""",
                (run_id,),
            ).fetchall()
            return [self._episode_from_row(row) for row in rows]

    def reclaim_orphaned_attempts(self, run_id: str, owner_token: str, reason: str) -> int:
        """Return unreconciled running rows to planned without erasing attempt lineage.

        The caller must already have acquired the current run owner lease. A
        later completion from the old worker cannot pass the owner+attempt CAS.
        """

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute("SELECT owner_token FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise KeyError("run not found: %s" % run_id)
            if run["owner_token"] != owner_token:
                connection.commit()
                return 0
            rows = connection.execute(
                """SELECT episode_id, attempt_count, attempt_token FROM episodes AS episode
                   WHERE run_id = ? AND status = 'running'
                     AND NOT EXISTS (
                       SELECT 1 FROM baseten_outbox AS outbox
                       WHERE outbox.run_id = episode.run_id AND outbox.episode_id = episode.episode_id
                     )""",
                (run_id,),
            ).fetchall()
            now = utc_now()
            for row in rows:
                connection.execute(
                    """UPDATE attempts SET status = 'orphaned', completed_at = ?, error_json = ?
                       WHERE run_id = ? AND episode_id = ? AND attempt_no = ?
                         AND (attempt_token IS ? OR attempt_token = ?) AND status = 'running'""",
                    (
                        now,
                        canonical_json({"reason": reason}),
                        run_id,
                        row["episode_id"],
                        row["attempt_count"],
                        row["attempt_token"],
                        row["attempt_token"],
                    ),
                )
                connection.execute(
                    """UPDATE episodes SET status = 'planned', attempt_token = NULL, lease_expires_at = NULL,
                       updated_at = ? WHERE episode_id = ? AND status = 'running'
                         AND (attempt_token IS ? OR attempt_token = ?)""",
                    (now, row["episode_id"], row["attempt_token"], row["attempt_token"]),
                )
                self._append_event(
                    connection,
                    run_id,
                    "episode_reclaimed",
                    {"episode_id": row["episode_id"], "prior_attempt": row["attempt_count"], "reason": reason},
                )
            connection.commit()
            return len(rows)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _finalise_run_state(self, connection: sqlite3.Connection, run_id: str) -> None:
        run = connection.execute("SELECT cancellation_requested FROM runs WHERE id = ?", (run_id,)).fetchone()
        stats = self._stats(connection, run_id)
        if stats["terminal"] != stats["total"]:
            return
        status = "cancelled" if run and run["cancellation_requested"] else "completed"
        existing = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if existing and existing["status"] != status:
            now = utc_now()
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, completed_at = ? WHERE id = ?",
                (status, now, now, run_id),
            )
            self._append_event(connection, run_id, "run_terminal", {"status": status, **stats})

    def finalise_run(self, run_id: str, owner_token: Optional[str] = None) -> Dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if owner_token is not None:
                run = connection.execute("SELECT owner_token FROM runs WHERE id = ?", (run_id,)).fetchone()
                if run is None:
                    raise KeyError("run not found: %s" % run_id)
                if run["owner_token"] != owner_token:
                    connection.commit()
                    return self.get_run(run_id)
            self._finalise_run_state(connection, run_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)
