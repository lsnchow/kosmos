"""SQLite persistence for PLUMB logical runs, episodes, attempts, and events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .records import ConfigurationError, canonical_json, lease_deadline, utc_now


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
)


class LeaseActiveError(RuntimeError):
    """A safe recovery was requested while another owner still has a lease."""


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

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(%s)" % table).fetchall()}
        if column not in existing:
            connection.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, definition))

    @staticmethod
    def _row_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

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
            }
        )
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
                       WHERE run_id = ? AND status = 'running'""",
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
                """SELECT * FROM episodes WHERE run_id = ? AND status = 'planned'
                   ORDER BY policy, task, start_id, episode_id LIMIT 1""",
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
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._current_owned_attempt(connection, run_id, episode_id, owner_token, attempt_token)
            if row is None:
                connection.commit()
                return False
            now = utc_now()
            connection.execute(
                """UPDATE episodes SET status = 'failed', validity = 'unknown', binary_success = NULL,
                   progress_score = NULL, missing_reason = 'service_failure', artifact_refs_json = ?,
                   error_json = ?, completed_at = ?, lease_expires_at = NULL, updated_at = ?
                   WHERE episode_id = ? AND status = 'running' AND attempt_token = ?""",
                (canonical_json(dict(artifact_refs)), canonical_json(dict(error)), now, now, episode_id, attempt_token),
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
            "SELECT episode_id FROM episodes WHERE run_id = ? AND status = 'planned'", (run_id,)
        ).fetchall()
        if not rows:
            return 0
        now = utc_now()
        connection.execute(
            """UPDATE episodes SET status = 'cancelled', validity = 'unknown', binary_success = NULL,
               progress_score = NULL, missing_reason = ?, completed_at = ?, updated_at = ?
               WHERE run_id = ? AND status = 'planned'""",
            (reason, now, now, run_id),
        )
        for row in rows:
            self._append_event(connection, run_id, "episode_cancelled", {"episode_id": row["episode_id"], "reason": reason})
        return len(rows)

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
                self._append_event(connection, run_id, "run_cancellation_requested", {"cancelled_planned": cancelled})
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
                self._append_event(connection, run_id, "run_cancellation_requested", {"pending_artifacts": True})
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
                "SELECT * FROM episodes WHERE run_id = ? AND status = 'planned' ORDER BY policy, task, start_id, episode_id",
                (run_id,),
            ).fetchall()
            return [self._episode_from_row(row) for row in rows]

    def running_episodes(self, run_id: str) -> List[Dict[str, Any]]:
        """Return in-flight records for recovery reconciliation only."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND status = 'running' ORDER BY policy, task, start_id, episode_id",
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
                "SELECT episode_id, attempt_count, attempt_token FROM episodes WHERE run_id = ? AND status = 'running'",
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
