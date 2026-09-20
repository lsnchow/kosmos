"""Engine backend that executes logical episodes on a deployed Baseten Chain.

``plumb.platform.BasetenChainClient`` already implements the whole HTTP surface
PLUMB uses -- ``/async_run_remote`` submission, queue status behind a verified
route, Chain deployment replicas, instance prices behind a verified basis, and
HMAC callback authentication.  Nothing called it.  This module is the call site.

Three pieces of durability the platform client deliberately leaves to the
application, because spec section 7 assigns them here:

**An outbox.**  A logical row exists before submission, keyed by
``(run_id, policy_variant, task, start_id, world_seed, protocol_hash)``, and the
platform request ID is persisted against it.  Transport retries create attempts,
never new statistical episodes.

**An explicitly verified result store, where available.** A callback is a
notification, not an output store. The rehearsal supplies a durable result
store so dropped callbacks can be recovered. Production does not infer a
result URL from the async submission URL: an account-tested result-store adapter
must be injected before it is used.

**Ambiguous submissions are never re-POSTed.**  ``SubmissionRetryPolicy``
refuses a nonzero retry count until a durable server-side entrypoint
compare-and-set is deployed, and ``AmbiguousSubmissionError`` routes the row to
reconciliation instead.  This backend honours that: an ambiguous POST becomes an
``awaiting_reconciliation`` row, not a second episode.

The backend never fabricates a frame, a score, a GPU-second or a dollar. A
completed Chain result can be legitimately unevaluable (for example, a clip
whose judge did not reach quorum). A terminal Chain failure or block is instead
an execution failure: it did not complete the requested control horizon and
must remain visible in the intent-to-evaluate denominator.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ..platform import (
    AmbiguousSubmissionError,
    AsyncChainRequestOptions,
    AuthenticatedCallback,
    BasetenChainClient,
    BasetenPlatformConfig,
    CallbackAssociation,
    PlatformError,
    VerifiedChainQueueRoute,
    VerifiedPriceBasis,
)
from ..freeplay import SourceBranchAdapter
from ..records import canonical_json, file_digest, json_digest, utc_now
from ..result_store import ResultStoreBinding, S3ResultStore, request_payload_digest
from ..starts import StartResolutionError, StartResolver


OUTBOX_SCHEMA_VERSION = 1

#: Stage names the Chain controller reports, in execution order.
CHAIN_STAGES: Tuple[str, ...] = ("policy", "world", "validity", "judge")

#: Terminal outbox states.  ``awaiting_reconciliation`` is deliberately not
#: terminal: an ambiguous submission must be resolved by the reconciler, never
#: by guessing.
OUTBOX_TERMINAL = ("completed", "failed", "cancelled")


class BackendNotConfigured(RuntimeError):
    """The backend cannot run because its external configuration is absent."""


class RemoteChainExecutionError(RuntimeError):
    """A terminal Chain result that cannot be recorded as a completed episode.

    ``artifact_refs`` points at the immutable received-result record.  The
    engine deliberately carries those references into its failed terminal
    record, so the raw failure evidence is not lost when it catches this typed
    exception and records the logical failure.
    """

    def __init__(
        self,
        error: Mapping[str, Any],
        artifact_refs: Mapping[str, Any],
        ledger_projection: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.error = dict(error)
        self.artifact_refs = dict(artifact_refs)
        self.ledger_projection = dict(ledger_projection or {})
        super().__init__(
            canonical_json(
                {
                    "chain_error": self.error,
                    "artifact_refs": self.artifact_refs,
                    "ledger_projection": self.ledger_projection,
                }
            )
        )


class ChainResultStore(Protocol):
    """A verified result lookup, deliberately separate from queue status."""

    def get_result(self, binding: ResultStoreBinding) -> Optional[Mapping[str, Any]]:
        """Return one completed request result, or ``None`` while unavailable."""


@dataclass(frozen=True)
class BasetenBackendSettings:
    """Everything the backend needs that is not a credential.

    ``operating_point`` records the confirmed cost setting so an episode can
    never be silently scored at a different resolution/step count than the one
    the protocol froze.
    """

    webhook_endpoint: str
    operating_point_id: str
    resolution: int = 256
    denoise_steps: int = 30
    action_chunk_size: int = 16
    batch_size: int = 1
    priority: Optional[int] = None
    max_time_in_queue_seconds: Optional[int] = None
    request_deadline_seconds: float = 1800.0
    poll_interval_seconds: float = 0.5
    feedback_mode: str = "unqualified"
    parity_status: str = "unqualified"

    def __post_init__(self) -> None:
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.request_deadline_seconds <= 0:
            raise ValueError("request_deadline_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "operating_point_id": self.operating_point_id,
            "resolution": self.resolution,
            "denoise_steps": self.denoise_steps,
            "action_chunk_size": self.action_chunk_size,
            "batch_size": self.batch_size,
            "feedback_mode": self.feedback_mode,
            "parity_status": self.parity_status,
            "request_deadline_seconds": self.request_deadline_seconds,
        }


class SubmissionOutbox:
    """Durable submission state and callback receipt store for the adapter."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                  logical_key TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  episode_id TEXT NOT NULL,
                  policy_variant TEXT NOT NULL,
                  task TEXT NOT NULL,
                  start_id TEXT NOT NULL,
                  world_seed INTEGER NOT NULL,
                  protocol_hash TEXT,
                  state TEXT NOT NULL,
                  request_id TEXT,
                  result_key TEXT,
                  attempt_count INTEGER NOT NULL DEFAULT 0,
                  request_payload_sha256 TEXT NOT NULL,
                  submitted_at TEXT,
                  settled_at TEXT,
                  result_json TEXT,
                  error_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_request
                  ON outbox(request_id) WHERE request_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_outbox_state ON outbox(state, run_id);

                CREATE TABLE IF NOT EXISTS callbacks (
                  request_id TEXT PRIMARY KEY,
                  -- Nullable: a webhook can legitimately arrive before the local
                  -- outbox has committed its request ID. Dropping such a callback
                  -- would lose a real result, so it is parked unassociated and
                  -- bound once the submission lands.
                  run_id TEXT,
                  episode_id TEXT,
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  received_at TEXT NOT NULL,
                  associated INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            # An outbox created by an earlier revision is upgraded in place.
            # SQLite's ALTER TABLE is limited, so keep the additions explicit and
            # backwards-compatible, exactly as plumb/ledger.py does.
            for table, column, definition in (
                ("callbacks", "associated", "INTEGER NOT NULL DEFAULT 0"),
                ("outbox", "result_key", "TEXT"),
            ):
                self._ensure_column(connection, table, column, definition)
            # Legacy outbox rows were written before result-store bindings
            # existed. Give each a secret-free opaque key during migration so a
            # recovery can bind it without changing its logical identity.
            rows = connection.execute("SELECT logical_key FROM outbox WHERE result_key IS NULL OR result_key = ''").fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE outbox SET result_key = ? WHERE logical_key = ?",
                    (uuid.uuid4().hex, row["logical_key"]),
                )
            connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_result_key ON outbox(result_key)")

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(%s)" % table).fetchall()}
        if column not in existing:
            connection.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, definition))

    @staticmethod
    def logical_key(episode: Mapping[str, Any], protocol_hash: Optional[str], variant: str) -> str:
        """Spec section 7's logical identity tuple, rendered as one key."""

        return "\x1f".join(
            [
                str(episode.get("run_id")),
                variant,
                str(episode.get("task")),
                str(episode.get("start_id")),
                str(episode.get("world_seed")),
                str(protocol_hash or ""),
            ]
        )

    def reserve(
        self,
        key: str,
        episode: Mapping[str, Any],
        variant: str,
        protocol_hash: Optional[str],
        payload_sha256: Optional[str],
    ) -> Dict[str, Any]:
        """Create or fetch the logical row.  Never creates a second row for one cell."""

        now = utc_now()
        requested_digest = payload_sha256 or "pending_result_store_binding"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM outbox WHERE logical_key = ?", (key,)).fetchone()
            if existing is not None:
                if payload_sha256 is not None and existing["request_payload_sha256"] not in (
                    payload_sha256,
                    "pending_result_store_binding",
                ):
                    connection.rollback()
                    raise ValueError(
                        "logical episode %s already exists with a different request payload; "
                        "a changed payload is a new protocol, not a retry" % key
                    )
                connection.commit()
                return dict(existing)
            connection.execute(
                """INSERT INTO outbox(
                     logical_key, run_id, episode_id, policy_variant, task, start_id, world_seed,
                     protocol_hash, state, result_key, request_payload_sha256, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)""",
                (
                    key,
                    str(episode.get("run_id")),
                    str(episode.get("episode_id")),
                    variant,
                    str(episode.get("task")),
                    str(episode.get("start_id")),
                    int(episode.get("world_seed") or 0),
                    protocol_hash,
                    uuid.uuid4().hex,
                    requested_digest,
                    now,
                    now,
                ),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM outbox WHERE logical_key = ?", (key,)).fetchone()
            return dict(row)

    def bind_request_payload(self, key: str, payload_sha256: str) -> Dict[str, Any]:
        """Bind the key to the final request digest before the POST boundary."""

        if not isinstance(payload_sha256, str) or not payload_sha256.startswith("sha256:"):
            raise ValueError("request payload digest must be a SHA-256 digest")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM outbox WHERE logical_key = ?", (key,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError("outbox record not found: %s" % key)
            existing = str(row["request_payload_sha256"])
            if existing not in ("pending_result_store_binding", payload_sha256):
                connection.rollback()
                raise ValueError("logical episode already exists with a different request payload")
            if row["state"] != "planned":
                if existing == "pending_result_store_binding":
                    connection.commit()
                    return dict(row)
                if existing != payload_sha256:
                    connection.rollback()
                    raise ValueError("result-store request binding must precede the POST boundary")
                connection.commit()
                return dict(row)
            connection.execute(
                "UPDATE outbox SET request_payload_sha256 = ?, updated_at = ? WHERE logical_key = ?",
                (payload_sha256, utc_now(), key),
            )
            bound = connection.execute("SELECT * FROM outbox WHERE logical_key = ?", (key,)).fetchone()
            connection.commit()
            return dict(bound)

    def result_binding(self, key: str) -> ResultStoreBinding:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM outbox WHERE logical_key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError("outbox record not found: %s" % key)
        return ResultStoreBinding(
            result_key=str(row["result_key"]),
            run_id=str(row["run_id"]),
            episode_id=str(row["episode_id"]),
            protocol_hash=str(row["protocol_hash"]),
            request_payload_sha256=str(row["request_payload_sha256"]),
        )

    def begin_submission(self, key: str) -> bool:
        """Commit the irreversible-POST boundary before bytes leave the process.

        A restart that finds ``dispatching`` cannot know whether the process
        died just before or just after the HTTP send, so it must turn the row
        ambiguous rather than retry it.
        """

        with self._lock, self._connect() as connection:
            changed = connection.execute(
                """UPDATE outbox SET state = 'dispatching', attempt_count = attempt_count + 1,
                   updated_at = ? WHERE logical_key = ? AND state = 'planned'""",
                (utc_now(), key),
            )
            return changed.rowcount == 1

    def mark_submitted(self, key: str, request_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE outbox SET state = 'submitted', request_id = ?, submitted_at = ?,
                   attempt_count = attempt_count + CASE WHEN state = 'planned' THEN 1 ELSE 0 END,
                   updated_at = ? WHERE logical_key = ? AND state IN ('planned', 'dispatching')""",
                (request_id, utc_now(), utc_now(), key),
            )

    def mark_ambiguous(self, key: str, error: Mapping[str, Any]) -> None:
        """An unresolved POST outcome.  Never re-POSTed; the reconciler owns it."""

        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE outbox SET state = 'awaiting_reconciliation', error_json = ?,
                   attempt_count = attempt_count + CASE WHEN state = 'planned' THEN 1 ELSE 0 END,
                   updated_at = ? WHERE logical_key = ? AND state IN ('planned', 'dispatching')""",
                (canonical_json(dict(error)), utc_now(), key),
            )

    def settle(self, key: str, state: str, result: Optional[Mapping[str, Any]], error: Optional[Mapping[str, Any]]) -> None:
        if state not in OUTBOX_TERMINAL:
            raise ValueError("settle() requires a terminal state, got %r" % state)
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE outbox SET state = ?, result_json = ?, error_json = ?, settled_at = ?,
                   updated_at = ? WHERE logical_key = ?""",
                (
                    state,
                    canonical_json(dict(result)) if result is not None else None,
                    canonical_json(dict(error)) if error is not None else None,
                    utc_now(),
                    utc_now(),
                    key,
                ),
            )

    def association(self, request_id: str) -> Optional[CallbackAssociation]:
        """Resolve a callback's request ID against the durable ledger."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, episode_id FROM outbox WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        return CallbackAssociation(request_id=request_id, run_id=row["run_id"], episode_id=row["episode_id"])

    def record_callback(self, callback: AuthenticatedCallback) -> bool:
        """Store an authenticated callback once.

        A duplicate delivery is ignored rather than double-counted.  A callback
        whose request ID is not yet in the outbox is *parked*, not rejected: the
        platform can deliver before the local submission commits, and discarding
        it would lose a real result.  ``bind_pending_callbacks`` attaches it once
        the submission lands, and the reconciler settles it either way.

        The HMAC has already been verified by the caller, so parking an
        unassociated callback does not trust an unauthenticated one.
        """

        association = callback.association
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_id, associated FROM callbacks WHERE request_id = ?", (callback.request_id,)
            ).fetchone()
            if existing is not None:
                # A later delivery may carry the association a parked one lacked.
                if association is not None and not int(existing["associated"] or 0):
                    connection.execute(
                        "UPDATE callbacks SET run_id = ?, episode_id = ?, associated = 1 WHERE request_id = ?",
                        (association.run_id, association.episode_id, callback.request_id),
                    )
                connection.commit()
                return False
            connection.execute(
                """INSERT INTO callbacks(
                     request_id, run_id, episode_id, event_type, payload_json, received_at, associated
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    callback.request_id,
                    None if association is None else association.run_id,
                    None if association is None else association.episode_id,
                    callback.event_type,
                    canonical_json(dict(callback.raw)),
                    utc_now(),
                    0 if association is None else 1,
                ),
            )
            connection.commit()
            return True

    def bind_pending_callbacks(self) -> int:
        """Attach parked callbacks to submissions that have since committed."""

        bound = 0
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT request_id FROM callbacks WHERE associated = 0"
            ).fetchall()
            for row in rows:
                request_id = str(row["request_id"])
                match = connection.execute(
                    "SELECT run_id, episode_id FROM outbox WHERE request_id = ?", (request_id,)
                ).fetchone()
                if match is None:
                    continue
                connection.execute(
                    "UPDATE callbacks SET run_id = ?, episode_id = ?, associated = 1 WHERE request_id = ?",
                    (match["run_id"], match["episode_id"], request_id),
                )
                bound += 1
            connection.commit()
        return bound

    def callback_for(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM callbacks WHERE request_id = ?", (request_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def unsettled(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM outbox WHERE state NOT IN ('completed','failed','cancelled')"
        params: Tuple[Any, ...] = ()
        if run_id is not None:
            query += " AND run_id = ?"
            params = (run_id,)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params).fetchall()]

    def counts(self, run_id: Optional[str] = None) -> Dict[str, int]:
        query = "SELECT state, COUNT(*) AS total FROM outbox"
        params: Tuple[Any, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        query += " GROUP BY state"
        with self._connect() as connection:
            return {str(row["state"]): int(row["total"]) for row in connection.execute(query, params).fetchall()}


@dataclass
class AllocationLedger:
    """Non-overlapping resource allocations, for the two spec section 7 cost views.

    Each interval is counted exactly once.  ``estimated_usd`` is ``None`` unless
    a ``VerifiedPriceBasis`` supplies currency and unit -- spec section 7 is
    explicit that an API field named ``price`` does not imply hourly USD.
    """

    intervals: List[Dict[str, Any]] = field(default_factory=list)
    price_basis: Optional[VerifiedPriceBasis] = None
    rates_per_hour: Dict[str, float] = field(default_factory=dict)

    def allocate(
        self,
        resource: str,
        started_at: float,
        ended_at: float,
        *,
        count: int = 1,
        view: str = "marginal_execution",
        note: Optional[str] = None,
    ) -> None:
        if ended_at < started_at:
            raise ValueError("an allocation cannot end before it starts")
        if view not in ("marginal_execution", "prewarm", "cooldown", "idle_capacity"):
            raise ValueError("unknown allocation view %r" % view)
        self.intervals.append(
            {
                "resource": resource,
                "count": int(count),
                "seconds": float(ended_at - started_at),
                "started_at": float(started_at),
                "ended_at": float(ended_at),
                "view": view,
                "note": note,
            }
        )

    def _sum_hours(self, views: Sequence[str]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for interval in self.intervals:
            if interval["view"] not in views:
                continue
            hours = interval["seconds"] * interval["count"] / 3600.0
            totals[interval["resource"]] = totals.get(interval["resource"], 0.0) + hours
        return totals

    def _usd(self, hours: Mapping[str, float]) -> Optional[float]:
        if self.price_basis is None or not self.rates_per_hour:
            return None
        total = 0.0
        for resource, resource_hours in hours.items():
            rate = self.rates_per_hour.get(resource)
            if rate is None:
                # An unpriced resource makes the whole view unknown rather than
                # silently cheaper.
                return None
            total += rate * resource_hours
        return total

    def report(self) -> Dict[str, Any]:
        marginal_hours = self._sum_hours(("marginal_execution",))
        total_hours = self._sum_hours(("marginal_execution", "prewarm", "cooldown", "idle_capacity"))
        return {
            "price_basis": None if self.price_basis is None else {
                "currency": self.price_basis.currency,
                "unit": getattr(self.price_basis.unit, "value", str(self.price_basis.unit)),
                "evidence_uri": self.price_basis.evidence_uri,
                "retrieved_at": self.price_basis.retrieved_at,
            },
            "allocation_count": len(self.intervals),
            "marginal_execution": {
                "resource_hours": marginal_hours,
                "estimated_usd": self._usd(marginal_hours),
                "definition": "attributed stage work, retries and transfers for this run only",
            },
            "total_demonstration_run": {
                "resource_hours": total_hours,
                "estimated_usd": self._usd(total_hours),
                "definition": "every allocation from prewarm start through cooldown, counted once",
            },
            "reconciliation_status": "estimated_pending_billing_reconciliation",
            "intervals": list(self.intervals),
        }


class BasetenChainBackend:
    """Engine backend: one logical episode, one Chain entrypoint invocation."""

    label = "real-model execution on a deployed Baseten Chain"
    name = "baseten"

    def __init__(
        self,
        settings: BasetenBackendSettings,
        data_dir: Path,
        client: Optional[BasetenChainClient] = None,
        config: Optional[BasetenPlatformConfig] = None,
        queue_route: Optional[VerifiedChainQueueRoute] = None,
        price_basis: Optional[VerifiedPriceBasis] = None,
        clock: Callable[[], float] = time.monotonic,
        transport: Optional[Any] = None,
        transport_kind: str = "network",
        start_resolver: Optional[StartResolver] = None,
        result_store: Optional[ChainResultStore] = None,
        source_branch_adapter: Optional[SourceBranchAdapter] = None,
    ) -> None:
        if transport_kind not in ("network", "simulated"):
            raise ValueError("transport_kind must be 'network' or 'simulated'")
        self.transport_kind = transport_kind
        # Without a resolver no submission can be built, because the Chain
        # requires the start's real frame and state. That refusal is deliberate.
        self.start_resolver = start_resolver
        self.settings = settings
        self.data_dir = Path(data_dir)
        self.outbox = SubmissionOutbox(self.data_dir / "baseten-outbox.sqlite3")
        # Never manufacture a Chain result URL. The rehearsal transport exposes
        # this narrow interface directly; a production deployment must inject a
        # separately verified adapter.
        inferred_store = transport if callable(getattr(transport, "get_result", None)) else None
        self.result_store: Optional[ChainResultStore] = result_store or inferred_store
        # Generic free-play resolves a task start for each request.  It must
        # never be used as an implicit recording continuation, so source-bound
        # control requires this separately configured adapter and an explicit
        # checkpoint manifest.
        self.source_branch_adapter = source_branch_adapter
        self.allocations = AllocationLedger(price_basis=price_basis)
        self._clock = clock
        self._queue_route = queue_route
        if client is not None:
            self.client: Optional[BasetenChainClient] = client
            self._config = config
        else:
            self._config = config
            self.client = None
            if config is not None:
                self.client = BasetenChainClient(
                    config, transport=transport, queue_route=queue_route
                )

    # -- configuration --------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        settings: BasetenBackendSettings,
        data_dir: Path,
        environ: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> "BasetenChainBackend":
        """Build from ``BASETEN_*`` environment values.

        Raises ``BackendNotConfigured`` with the missing variable names rather
        than half-constructing a client that would fail at submission time.
        """

        try:
            config = BasetenPlatformConfig.from_env(environ)
        except PlatformError as exc:
            raise BackendNotConfigured(str(exc)) from exc
        return cls(settings=settings, data_dir=data_dir, config=config, **kwargs)

    def configuration_status(self) -> Dict[str, Any]:
        """What is configured and what is still missing, for ``/api/health``."""

        missing: List[str] = []
        if self.client is None:
            missing.append("BASETEN_API_KEY and BASETEN_CHAIN_ASYNC_URL")
        if self.transport_kind == "network" and self.result_store is None:
            missing.append(
                "a configured immutable S3-compatible PLUMB result store "
                "(bucket, region, HTTPS endpoint, prefix, evidence, and deployment secrets)"
            )
        if self.transport_kind == "network" and isinstance(self.result_store, S3ResultStore) and not self.result_store.sdk_available():
            missing.append("boto3 result-store optional dependency is not installed")
        if self._config is not None and not self._config.webhook_secret:
            missing.append("BASETEN_WEBHOOK_SECRET (callback authentication unavailable)")
        if self._queue_route is None:
            missing.append("a tested VerifiedChainQueueRoute (platform queue depth unavailable)")
        if self.allocations.price_basis is None:
            missing.append("a VerifiedPriceBasis (USD unavailable)")
        return {
            "backend": self.name,
            "configured": self.client is not None and (
                self.transport_kind == "simulated" or self.result_store is not None
            ),
            "transport_kind": self.transport_kind,
            "simulated": self.transport_kind == "simulated",
            "chain_async_url": getattr(self.client, "chain_async_url", None) if self.client is not None else None,
            "operating_point": self.settings.to_mapping(),
            "missing": missing,
            "rehearsal_warning": (
                "This backend talks to a SIMULATED Chain. It exercises the integration path and "
                "measures nothing about a model, a latency or a cost."
                if self.transport_kind == "simulated"
                else None
            ),
        }

    # -- execution ------------------------------------------------------------

    def execute(self, episode: Mapping[str, Any], config: Mapping[str, Any], artifact_dir: Path) -> Dict[str, Any]:
        """Submit one logical episode and return a score-shaped result.

        Synchronous by contract because ``plumb.engine`` runs episodes on a
        worker pool.  The waiting is bounded by ``request_deadline_seconds``; a
        a deadline is a failed lifecycle, never a guessed score or a retry.
        """

        if self.client is None or (self.transport_kind == "network" and self.result_store is None):
            raise BackendNotConfigured(
                "Baseten backend is not configured: %s" % ", ".join(self.configuration_status()["missing"])
            )

        variant = str(episode.get("policy_variant") or episode.get("policy"))
        protocol_hash = episode.get("protocol_hash") or config.get("protocol_hash")
        try:
            unsigned_input = self._build_entrypoint_input(episode, config, variant, protocol_hash)
        except StartResolutionError as exc:
            # An unresolvable start is a known, named gap (Gate C), not a service
            # failure and not a scored outcome.
            return self._unevaluable(episode, "start_unresolved", str(exc))
        key = self.outbox.logical_key(episode, protocol_hash, variant)
        # Persist an opaque object key before it becomes part of the request.
        # The digest binds the entire unsigned request; the binding itself is
        # excluded from that digest to avoid a circular hash definition.
        row = self.outbox.reserve(key, episode, variant, protocol_hash, None)
        binding = ResultStoreBinding(
            result_key=str(row["result_key"]),
            run_id=str(episode["run_id"]),
            episode_id=str(episode["episode_id"]),
            protocol_hash=str(protocol_hash or ""),
            request_payload_sha256=request_payload_digest(unsigned_input),
        )
        entrypoint_input = self._build_entrypoint_input(
            episode, config, variant, protocol_hash, result_store=binding.as_mapping()
        )
        row = self.outbox.bind_request_payload(key, json_digest(entrypoint_input))

        if row["state"] == "completed" and row["result_json"]:
            # Idempotent replay of an already-settled logical episode.
            saved_result = json.loads(row["result_json"])
            failure = self._terminal_execution_failure(saved_result, episode)
            if failure is not None:
                refs = self._persist_terminal_failure(episode, artifact_dir, failure, saved_result, None)
                raise RemoteChainExecutionError(failure, refs)
            return self._result_from_chain(saved_result, episode, artifact_dir, replayed=True)
        if row["state"] == "failed":
            # A terminal remote failure is evidence, not permission to submit
            # the same logical episode again.  The engine will already have
            # terminalized its row, but this also protects a direct retry of the
            # adapter from becoming a second request.
            saved_error = json.loads(row["error_json"]) if row.get("error_json") else {
                "reason": "previous_terminal_chain_failure"
            }
            saved_result = json.loads(row["result_json"]) if row.get("result_json") else None
            refs = self._persist_terminal_failure(episode, artifact_dir, saved_error, saved_result, None)
            saved_request_id = row.get("request_id")
            projection = self._failure_ledger_projection(
                [str(saved_request_id)] if saved_request_id else (), saved_error, saved_result
            )
            raise RemoteChainExecutionError(saved_error, refs, projection)
        if row["state"] == "awaiting_reconciliation":
            return self._unevaluable(
                episode,
                "awaiting_reconciliation",
                "a prior submission outcome was ambiguous; the reconciler owns this row",
            )
        if row["state"] == "dispatching":
            # The old process may have sent bytes after committing this state.
            # There is no safe retry without a server-side idempotency CAS.
            self.outbox.mark_ambiguous(
                key,
                {"reason": "crash_after_pre_post_boundary", "detail": "request ID was not durably recorded"},
            )
            return self._unevaluable(
                episode,
                "awaiting_reconciliation",
                "a prior process crossed the pre-POST boundary; this episode is never re-sent",
            )

        request_id = row.get("request_id")
        if row["state"] == "submitted" and isinstance(request_id, str) and request_id:
            # The POST was already durably acknowledged. A new process must
            # observe that request's callback/result-store entry, never issue a
            # second POST for the same logical cell.
            outcome = self._await_result(key, request_id)
        else:
            started = self._clock()
            if not self.outbox.begin_submission(key):
                return self._unevaluable(
                    episode,
                    "awaiting_reconciliation",
                    "the durable submission state changed before this process could send",
                )
            options = AsyncChainRequestOptions(
                webhook_endpoint=self._webhook_for_run(str(episode.get("run_id"))),
                priority=self.settings.priority,
                max_time_in_queue_seconds=self.settings.max_time_in_queue_seconds,
            )
            try:
                receipt = _run_async(self.client.submit_async(entrypoint_input, options))
            except AmbiguousSubmissionError as exc:
                self.outbox.mark_ambiguous(
                    key,
                    {
                        "reason": "ambiguous_submission",
                        "detail": str(exc),
                        "attempts": len(getattr(exc, "attempts", ()) or ()),
                    },
                )
                self.allocations.allocate("controller_cpu", started, self._clock(), note="ambiguous submission")
                return self._unevaluable(
                    episode,
                    "ambiguous_submission",
                    "the POST outcome was unresolved; it is never re-sent as a second episode",
                )
            except PlatformError as exc:
                self.outbox.settle(key, "failed", None, {"reason": "submission_failed", "detail": str(exc)})
                self.allocations.allocate("controller_cpu", started, self._clock(), note="failed submission")
                raise

            request_id = receipt.request_id
            self.outbox.mark_submitted(key, request_id)
            outcome = self._await_result(key, request_id)
            self.allocations.allocate("controller_cpu", started, self._clock(), note="episode controller")

        if outcome is None:
            error = {
                "reason": "deadline_exceeded",
                "seconds": self.settings.request_deadline_seconds,
                "request_id": request_id,
            }
            refs = self._persist_terminal_failure(
                episode, artifact_dir, error, None, [str(request_id)]
            )
            error["failure_artifact"] = refs["chain_terminal_failure"]
            self.outbox.settle(key, "failed", None, error)
            raise RemoteChainExecutionError(
                error,
                refs,
                self._failure_ledger_projection([str(request_id)], error),
            )

        failure = self._terminal_execution_failure(outcome, episode)
        if failure is not None:
            failure["request_id"] = request_id
            refs = self._persist_terminal_failure(
                episode, artifact_dir, failure, outcome, [str(request_id)]
            )
            failure["failure_artifact"] = refs["chain_terminal_failure"]
            self.outbox.settle(key, "failed", None, failure)
            raise RemoteChainExecutionError(
                failure,
                refs,
                self._failure_ledger_projection([str(request_id)], failure, outcome),
            )

        self.outbox.settle(key, "completed", outcome, None)
        return self._result_from_chain(
            outcome, episode, artifact_dir, request_ids=[str(request_id)]
        )

    def _webhook_for_run(self, run_id: str) -> str:
        """Compose the per-run callback URL.

        The receiver is ``POST /api/callbacks/{run_id}`` so the run identity can
        be checked against the callback's ledger association.  A configured base
        that already names this run is left alone, so an operator can point at an
        exact URL if they need to.
        """

        base = self.settings.webhook_endpoint.rstrip("/")
        if base.endswith("/" + run_id):
            return base
        return "%s/%s" % (base, run_id)

    def _build_entrypoint_input(
        self,
        episode: Mapping[str, Any],
        config: Mapping[str, Any],
        variant: str,
        protocol_hash: Optional[str],
        result_store: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build the ``RolloutRequest`` the Chain entrypoint actually accepts.

        The controller validates ``request.policy.payload`` as
        ``EpisodeControlPayload``, which requires the task prompt, the
        conditioning frame's pixels and the 8-D Bridge state -- an episode cannot
        begin without them.  Those come from the ScenarioManifest via a
        ``StartResolver``; with no resolver, or an unresolvable start, this raises
        instead of submitting a request the Chain would reject.

        Every model is ``extra="forbid"``, so a drift between these field names
        and ``deploy/baseten/chain.py`` fails loudly at validation rather than
        silently nulling a measurement.
        """

        if self.start_resolver is None:
            raise StartResolutionError(
                "no start resolver is configured: a Chain submission requires the start's real "
                "conditioning frame and 8-D Bridge state (Gate C / scenarios.jsonl)"
            )
        episode_id = str(episode.get("episode_id"))
        task = str(episode.get("task"))
        start = self.start_resolver.resolve(
            task, str(episode.get("start_id")), str(episode.get("start_lineage_id") or "")
        )
        seeds = dict(episode.get("seeds") or {})
        world_seed = int(seeds.get("world") or episode.get("world_seed") or 0)
        stage_ref = {"episode_id": episode_id, "protocol_hash": str(protocol_hash or "")}

        episode_payload: Dict[str, Any] = {
            "policy": variant,
            "task_id": task,
            "prompt": start.prompt,
            "horizon_actions": int(episode.get("horizon_actions") or start.horizon_actions),
            "initial_frame": start.frame_payload(),
            "initial_state": [float(value) for value in start.state],
            "policy_seed": int(seeds.get("policy") or 0) or None,
            "bridge_control_profile_id": str(
                config.get("bridge_control_profile_id") or "bridge-default"
            ),
            "feedback_mode": self.settings.feedback_mode,
            "return_frames": True,
        }
        goal = start.goal_payload()
        if goal is not None:
            episode_payload["goal_image"] = goal

        # The controller builds each segment's WorldStagePayload itself; from the
        # entrypoint it reads only these four keys (chain.py ``_world_setup``).
        # Sending a conditioning image or an empty action list here would be
        # misleading, because neither is read.
        world_payload: Dict[str, Any] = {
            "compatibility_profile_id": self.settings.operating_point_id,
            "domain": str(config.get("world_domain") or "bridge_orig_lerobot"),
            "seed": world_seed,
            "source_state_lineage_id": start.start_lineage_id,
        }

        scoring_payload = {
            "task_id": task,
            "start_provenance": start.provenance,
            "operating_point": self.settings.to_mapping(),
            "judge_seeds": [int(value) for value in (seeds.get("judge_samples") or [])],
        }

        request = {
            "run_id": str(episode.get("run_id")),
            "episode_id": episode_id,
            "protocol_hash": str(protocol_hash or ""),
            "policy": {**stage_ref, "payload": episode_payload},
            "world": {**stage_ref, "payload": world_payload},
            "validity": {**stage_ref, "payload": scoring_payload},
            "judge": {**stage_ref, "payload": scoring_payload},
        }
        if result_store is not None:
            request["result_store"] = dict(result_store)
        return request

    def _await_result(self, key: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Wait for the Chain's own persisted result, notified by callback.

        The callback is a notification only.  If it never arrives the reconciler
        is expected to settle the row; this loop simply times out rather than
        inventing an outcome.
        """

        deadline = self._clock() + self.settings.request_deadline_seconds
        while self._clock() < deadline:
            # A webhook can land before the submission committed its request ID.
            self.outbox.bind_pending_callbacks()
            stored = self.outbox.callback_for(request_id)
            if stored is not None:
                payload = json.loads(stored["payload_json"])
                data = payload.get("data")
                if isinstance(data, Mapping):
                    return dict(data)
                return {"status": "failed", "missing_reason": "callback_carried_no_result"}
            if self.result_store is not None:
                # The store is injected only after its route/evidence was
                # verified. It is especially important for rehearsal: a
                # deliberately dropped webhook must not turn completed remote
                # work into a made-up timeout failure.
                result = self.result_store.get_result(self.outbox.result_binding(key))
                if isinstance(result, Mapping):
                    return dict(result)
            time.sleep(self.settings.poll_interval_seconds)
        return None

    # -- result translation ---------------------------------------------------

    def _result_from_chain(
        self,
        chain_result: Mapping[str, Any],
        episode: Mapping[str, Any],
        artifact_dir: Path,
        request_ids: Optional[Sequence[str]] = None,
        replayed: bool = False,
    ) -> Dict[str, Any]:
        """Translate a ``RolloutResult`` into the engine's result shape.

        ``execute`` guards this method with :meth:`_terminal_execution_failure`.
        Thus it translates only completed Chain results; the completed result
        may still be unevaluable when a real clip was generated but the validity
        or judge stages could not support a score.
        """

        status = str(chain_result.get("status") or "unknown")
        stages = chain_result.get("stages") or {}
        if not isinstance(stages, Mapping):
            stages = {}

        if status != "completed":
            raise ValueError("_result_from_chain only accepts completed Chain results")

        validity = str(chain_result.get("validity") or "unknown")
        if validity not in ("valid", "invalid", "unknown"):
            validity = "unknown"
        progress = chain_result.get("progress_score")
        success = chain_result.get("binary_success")
        if validity != "valid":
            progress = None
            success = None
        elif success is None:
            progress = None

        refs = self._persist_provenance(episode, artifact_dir, chain_result, request_ids)
        timing = _chain_timing(chain_result)
        timing.setdefault("source", "baseten_chain")
        if replayed:
            timing["replayed_from_outbox"] = True

        result: Dict[str, Any] = {
            "validity": validity,
            "binary_success": success,
            "progress_score": progress,
            "missing_reason": chain_result.get("missing_reason"),
            "horizon_actions": int(chain_result.get("horizon_actions") or episode["horizon_actions"]),
            "artifact_refs": refs,
            "timing": timing,
            "backend_metadata": {
                "label": self.label,
                "transport": self.transport_kind,
                "simulated_transport": self.transport_kind == "simulated",
                "operating_point": self.settings.to_mapping(),
                "stages": {stage: dict(stages.get(stage) or {}) for stage in CHAIN_STAGES if stage in stages},
                "replayed": replayed,
            },
            # Spec section 8 measurement/cost fields.  Absent values stay None.
            # The Chain reports these under nested names (``segments``,
            # ``gpu_seconds.*``); a flat alias is also accepted so the two sides
            # cannot silently disagree and null out every measurement.
            "executed_actions": chain_result.get("executed_actions"),
            "n_segments": _chain_segment_count(chain_result),
            "compute_gpu_seconds": _chain_gpu_seconds(chain_result, "total_instrumented_gpu_seconds"),
            "allocated_gpu_seconds": _chain_gpu_seconds(chain_result, "allocated_gpu_seconds"),
            "estimated_usd": chain_result.get("estimated_usd"),
            "cost_basis_ref": chain_result.get("cost_basis_ref"),
            "raw_judge_samples_ref": chain_result.get("raw_judge_samples_ref"),
            "segments_manifest_ref": chain_result.get("segments_manifest_ref"),
            "video_ref": chain_result.get("video_ref"),
            "exclusion_reason": chain_result.get("exclusion_reason"),
            "feedback_mode": (
                "unqualified"
                if self.transport_kind == "simulated"
                else str(chain_result.get("feedback_mode") or self.settings.feedback_mode)
            ),
            "parity_status": (
                "unqualified"
                if self.transport_kind == "simulated"
                else str(chain_result.get("parity_status") or self.settings.parity_status)
            ),
            "platform_request_ids": list(request_ids or []),
            # Per-segment frames drive the console's rollout wall. They are
            # carried verbatim from the Chain so the tile shows exactly what was
            # persisted, with the certified frame count the Chain reported.
            "segments": _chain_segments(chain_result),
            "world_identity": {
                **dict(chain_result.get("world_identity") or {}),
                "transport": self.transport_kind,
                "simulated_transport": self.transport_kind == "simulated",
                "operating_point_id": self.settings.operating_point_id,
            },
        }
        return result

    def _failure_ledger_projection(
        self,
        request_ids: Sequence[str],
        error: Mapping[str, Any],
        chain_result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return failure facts that belong on the logical ledger row.

        The artifact retains the complete remote payload, but operators and
        study checks need enough durable row-level provenance to distinguish a
        simulated transport failure from an unlabelled local exception without
        opening every artifact.
        """

        remote = dict(chain_result or {})
        world_identity = dict(remote.get("world_identity") or {})
        world_identity.update(
            {
                "transport": self.transport_kind,
                "simulated_transport": self.transport_kind == "simulated",
                "operating_point_id": self.settings.operating_point_id,
            }
        )
        remote_missing = remote.get("missing_reason", error.get("missing_reason", error.get("reason")))
        return {
            "world_identity": world_identity,
            "platform_request_ids": [str(request_id) for request_id in request_ids if request_id],
            "feedback_mode": (
                "unqualified"
                if self.transport_kind == "simulated"
                else str(remote.get("feedback_mode") or self.settings.feedback_mode)
            ),
            "parity_status": (
                "unqualified"
                if self.transport_kind == "simulated"
                else str(remote.get("parity_status") or self.settings.parity_status)
            ),
            # ``missing_reason`` remains the stable lifecycle value
            # ``service_failure``. The remote reason is an exclusion fact, so
            # it remains queryable without relabelling a failed execution as a
            # completed-but-unevaluable episode.
            "exclusion_reason": None if remote_missing is None else str(remote_missing),
        }

    @staticmethod
    def _terminal_execution_failure(
        chain_result: Mapping[str, Any], episode: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a classified error for a Chain result that did not execute.

        A completed result with ``validity`` unknown is still a completed
        episode: the clip exists and its missing score belongs in analysis
        missingness. By contrast, a non-completed lifecycle or a result that
        does not declare exactly the planned horizon means no complete control
        trajectory was generated and is a service failure.
        """

        status = str(chain_result.get("status") or "unknown")
        stages = chain_result.get("stages")
        if status != "completed":
            blocked_stage = None
            if isinstance(stages, Mapping):
                for stage in CHAIN_STAGES:
                    entry = stages.get(stage)
                    if isinstance(entry, Mapping) and entry.get("status") in ("blocked", "failed"):
                        blocked_stage = stage
                        break
            return {
                "reason": "remote_chain_terminal_%s" % status,
                "chain_status": status,
                "missing_reason": chain_result.get("missing_reason"),
                "detail": chain_result.get("reason"),
                "failed_stage": blocked_stage,
            }

        planned = episode.get("horizon_actions")
        reported = chain_result.get("horizon_actions")
        executed = chain_result.get("executed_actions")
        if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
            return {"reason": "invalid_planned_horizon", "planned_horizon_actions": planned}
        if isinstance(reported, bool) or not isinstance(reported, int) or reported != planned:
            return {
                "reason": "remote_chain_horizon_mismatch",
                "planned_horizon_actions": planned,
                "reported_horizon_actions": reported,
            }
        if isinstance(executed, bool) or not isinstance(executed, int) or executed != planned:
            return {
                "reason": "remote_chain_incomplete_horizon",
                "planned_horizon_actions": planned,
                "executed_actions": executed,
            }
        return None

    def _unevaluable(
        self,
        episode: Mapping[str, Any],
        reason: str,
        detail: Any = None,
        request_ids: Optional[Sequence[str]] = None,
        chain_result: Optional[Mapping[str, Any]] = None,
        artifact_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """An explicit unevaluable outcome.  Never a zero, never a failure score."""

        refs: Dict[str, Any] = {}
        if artifact_dir is not None and chain_result is not None:
            refs = self._persist_provenance(episode, artifact_dir, chain_result, request_ids)
        return {
            "validity": "unknown",
            "binary_success": None,
            "progress_score": None,
            "missing_reason": reason,
            "horizon_actions": int(episode["horizon_actions"]),
            "artifact_refs": refs,
            "timing": {"source": "baseten_chain"},
            "backend_metadata": {
                "label": self.label,
                "transport": self.transport_kind,
                "simulated_transport": self.transport_kind == "simulated",
                "operating_point": self.settings.to_mapping(),
                "detail": None if detail is None else str(detail),
            },
            "feedback_mode": self.settings.feedback_mode,
            "parity_status": self.settings.parity_status,
            "platform_request_ids": list(request_ids or []),
            "world_identity": {
                "transport": self.transport_kind,
                "simulated_transport": self.transport_kind == "simulated",
                "operating_point_id": self.settings.operating_point_id,
            },
        }

    def _persist_provenance(
        self,
        episode: Mapping[str, Any],
        artifact_dir: Path,
        chain_result: Mapping[str, Any],
        request_ids: Optional[Sequence[str]],
    ) -> Dict[str, Any]:
        """Write the attempt's own provenance manifest and reference it.

        ``plumb.records`` requires a manifest whose ``real_robot``,
        ``physical_control`` and ``qualified_measurement`` claims are all false.
        ``world_model`` is legitimately true here, because one did run.
        """

        artifact_dir.mkdir(parents=True, exist_ok=True)
        label = (
            "rehearsal transport; simulated Chain; not a learned-model result; unqualified"
            if self.transport_kind == "simulated"
            else "real-model Baseten Chain attempt; unqualified until gate review"
        )
        manifest = {
            "schema_version": 1,
            "kind": "plumb_real_model_attempt",
            "label": label,
            "created_at": utc_now(),
            "run_id": episode.get("run_id"),
            "episode_id": episode.get("episode_id"),
            "policy": episode.get("policy"),
            "policy_variant": episode.get("policy_variant"),
            "task": episode.get("task"),
            "start_id": episode.get("start_id"),
            "world_seed": episode.get("world_seed"),
            "protocol_hash": episode.get("protocol_hash"),
            "operating_point": self.settings.to_mapping(),
            "platform_request_ids": list(request_ids or []),
            # The transport is recorded on the artifact itself, so a rehearsal
            # result can never be mistaken for a real one after the fact.
            "transport": self.transport_kind,
            "simulated_transport": self.transport_kind == "simulated",
            "chain_status": chain_result.get("status"),
            "chain_result_sha256": json_digest(dict(chain_result)),
            "claims": {
                "real_robot": False,
                "physical_control": False,
                # A generative world model did run; saying otherwise would be
                # false, and saying it is physics would also be false.
                # A simulated transport did not run a world model; saying it did
                # would be the exact false claim this project refuses to make.
                "world_model": self.transport_kind != "simulated",
                "qualified_measurement": False,
            },
        }
        path = artifact_dir / "baseten_attempt.json"
        _atomic_json(path, manifest)
        data_root = artifact_dir.resolve().parents[3]
        relative = path.resolve().relative_to(data_root).as_posix()
        return {
            "manifest": {
                "uri": "artifact://%s" % relative,
                "relative_path": relative,
                "artifact_path": relative,
                "url": "/api/artifacts/%s" % relative,
                "sha256": file_digest(path),
                "media_type": "application/json",
                "label": manifest["label"],
            }
        }

    def _persist_terminal_failure(
        self,
        episode: Mapping[str, Any],
        artifact_dir: Path,
        error: Mapping[str, Any],
        chain_result: Optional[Mapping[str, Any]],
        request_ids: Optional[Sequence[str]],
    ) -> Dict[str, Dict[str, str]]:
        """Persist an immutable, typed record before raising a Chain failure.

        This keeps the exact received terminal result and its classification
        beside the engine's later failed terminal record. The latter references
        this artifact through :class:`RemoteChainExecutionError`, so an error
        lifecycle cannot erase the only evidence of what the remote service
        returned.
        """

        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "plumb_baseten_terminal_failure",
            "label": (
                "rehearsal transport; simulated Chain; not a learned-model result; unqualified"
                if self.transport_kind == "simulated"
                else "real-model Baseten Chain terminal execution failure; unqualified"
            ),
            "created_at": utc_now(),
            "run_id": episode.get("run_id"),
            "episode_id": episode.get("episode_id"),
            "policy": episode.get("policy"),
            "policy_variant": episode.get("policy_variant"),
            "task": episode.get("task"),
            "start_id": episode.get("start_id"),
            "world_seed": episode.get("world_seed"),
            "protocol_hash": episode.get("protocol_hash"),
            "transport": self.transport_kind,
            "simulated_transport": self.transport_kind == "simulated",
            "platform_request_ids": list(request_ids or []),
            "failure": dict(error),
            # Keep the received terminal object verbatim in the durable
            # artifact. It is evidence of a failed execution, not an inferred
            # score or a replacement result.
            "chain_result": None if chain_result is None else dict(chain_result),
        }
        path = artifact_dir / "baseten_terminal_failure.json"
        _atomic_json(path, payload)
        data_root = artifact_dir.resolve().parents[3]
        relative = path.resolve().relative_to(data_root).as_posix()
        return {
            "chain_terminal_failure": {
                "uri": "artifact://%s" % relative,
                "relative_path": relative,
                "artifact_path": relative,
                "url": "/api/artifacts/%s" % relative,
                "sha256": file_digest(path),
                "media_type": "application/json",
                "label": payload["label"],
            }
        }

    # -- free play ------------------------------------------------------------

    def source_branch_status(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        """Ask an injected exact-source adapter; never fall back to task start."""

        if self.source_branch_adapter is None:
            return {
                "available": False,
                "exact_branch_supported": False,
                "reason": (
                    "The configured Baseten free-play path can start an unscored task, but no "
                    "source-branch adapter is configured to restore this recording's checkpoint."
                ),
                "missing": ["a source-branch adapter bound to this checkpoint format"],
            }
        return self.source_branch_adapter.source_branch_status(source)

    def source_branch_step(
        self,
        *,
        session_id: str,
        source: Mapping[str, Any],
        actions: Sequence[Sequence[float]],
        protocol_hash: str,
        seed: int,
        resolution: int,
    ) -> Mapping[str, Any]:
        """Run only an explicit exact-source adapter, never generic free-play."""

        if self.source_branch_adapter is None:
            raise BackendNotConfigured("no source-branch adapter is configured")
        return self.source_branch_adapter.source_branch_step(
            session_id=session_id,
            source=source,
            actions=actions,
            protocol_hash=protocol_hash,
            seed=seed,
            resolution=resolution,
        )

    def _persist_freeplay_frames(self, session_id: str, outcome: Mapping[str, Any]) -> List[str]:
        """Write the Chain's returned free-play frames and return their URLs.

        The scored path never does this: its frames are persisted by the control
        plane from segment events. Free-play has no ledger row, so its pixels
        arrive on the result or not at all.

        A declared ``png_sha256`` is verified before the bytes are written. A
        frame that fails its own hash is dropped rather than shown, because the
        whole point of this beat is that the pixels are what the world model
        actually produced.
        """

        payloads = outcome.get("freeplay_frames") or []
        if not isinstance(payloads, (list, tuple)):
            return []
        directory = self.data_dir / "freeplay" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        # A session is a live viewport, not an archive: clear the previous chunk
        # so a slow frame from an earlier keypress cannot appear in this one.
        for stale in directory.glob("*.png"):
            stale.unlink()
        urls: List[str] = []
        self._last_freeplay_size: Optional[Tuple[int, int]] = None
        for index, item in enumerate(payloads):
            if not isinstance(item, Mapping):
                continue
            encoded = item.get("data")
            if not isinstance(encoded, str) or not encoded:
                continue
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                continue
            declared = item.get("png_sha256")
            if isinstance(declared, str) and declared:
                actual = hashlib.sha256(raw).hexdigest()
                if actual != declared.removeprefix("sha256:"):
                    continue
            path = directory / ("%03d.png" % index)
            path.write_bytes(raw)
            height, width = item.get("height"), item.get("width")
            if isinstance(height, int) and isinstance(width, int):
                self._last_freeplay_size = (height, width)
            urls.append("/api/artifacts/freeplay/%s/%s" % (session_id, path.name))
        return urls


    def freeplay_step(
        self,
        session_id: str,
        task: str,
        actions: Sequence[Sequence[float]],
        protocol_hash: str,
        seed: int,
        resolution: int = 480,
    ) -> Dict[str, Any]:
        """Send one constant action chunk straight to the world model.

        No policy, no language model, no validity gate and no judge are in this
        loop, which is what makes it the most robust thing in the demo: the
        keypress becomes an action vector directly.  The result is unscored and
        never reaches the ledger.

        Raises rather than returning a placeholder when the Chain is unavailable.
        """

        if self.client is None:
            raise BackendNotConfigured("free-play needs a configured Baseten Chain")
        if not actions:
            raise ValueError("free-play needs at least one action row")
        if any(len(row) != 7 for row in actions):
            raise ValueError("free-play actions must be 7-D Bridge rows")
        # The Chain's RolloutRequest requires a non-empty protocol_hash on the
        # envelope and on every stage. Free-play is unscored, but it is still
        # parameterised by the protocol -- denoise_steps below comes from these
        # settings -- so the hash records which world configuration produced the
        # frames. Sending "" made the Chain reject the request with a bare HTTP
        # 400 that named nothing; refuse here instead, where the cause is known.
        if not protocol_hash:
            raise ValueError(
                "free-play needs the active protocol hash: the Chain rejects an envelope without one, "
                "and an unscored frame still has to record which world configuration produced it"
            )
        if self.start_resolver is None:
            raise StartResolutionError(
                "free-play needs a start resolver: the world model is conditioned on a real start frame, "
                "and a fabricated one would defeat the point of showing it live"
            )
        start = self.start_resolver.resolve(task, "freeplay", "")
        episode_id = "freeplay-%s" % session_id
        stage_ref = {"episode_id": episode_id, "protocol_hash": protocol_hash}
        # Free-play skips the policy, the validity gate and the judge, so those
        # three payloads are empty by design: the Chain's freeplay branch reads
        # none of them. The stage refs stay present because the controller still
        # checks episode identity across every stage on this path.
        entrypoint_input = {
            "run_id": "freeplay",
            "episode_id": episode_id,
            "protocol_hash": protocol_hash,
            "policy": {**stage_ref, "payload": {}},
            "world": {**stage_ref, "payload": {}},
            "validity": {**stage_ref, "payload": {}},
            "judge": {**stage_ref, "payload": {}},
            "freeplay": {
                "task_id": task,
                "prompt": start.prompt,
                "compatibility_profile_id": self.settings.operating_point_id,
                "domain": "bridge_orig_lerobot",
                "seed": seed,
                "conditioning_image": start.frame_payload(),
                "actions": [list(row) for row in actions],
                "source_state_lineage_id": start.start_lineage_id or None,
            },
        }
        options = AsyncChainRequestOptions(
            webhook_endpoint=self._webhook_for_run("freeplay"),
            priority=0,
            max_time_in_queue_seconds=30,
        )
        started = self._clock()
        receipt = _run_async(self.client.submit_async(entrypoint_input, options))
        outcome = self._await_result("freeplay:%s" % session_id, receipt.request_id)
        self.allocations.allocate("controller_cpu", started, self._clock(), note="freeplay")
        if outcome is None:
            raise TimeoutError("the world model did not return frames before the free-play deadline")
        frames = self._persist_freeplay_frames(session_id, outcome)
        return {
            "frame_urls": list(frames),
            "frame_count": len(frames),
            "commanded_rows": len(actions),
            "requested_resolution": int(resolution),
            # What the frames measurably are, which is not always what was asked
            # for. None when the Chain did not report dimensions -- "480p" is a
            # claim made on stage, so it is never assumed from the request.
            "frame_height": (self._last_freeplay_size or (None, None))[0],
            "frame_width": (self._last_freeplay_size or (None, None))[1],
            "request_id": receipt.request_id,
            "scored": False,
            "qualified": False,
        }

    # -- callbacks and reconciliation -----------------------------------------

    def authenticate_and_store_callback(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> Tuple[AuthenticatedCallback, bool]:
        """Authenticate a webhook and store it once.

        ``BasetenChainClient.authenticate_callback`` takes an already-resolved
        association, so the request ID is read from the body first in order to
        look it up.  That read is not trusted: ``authenticate_callback`` verifies
        the HMAC over the raw body *and* re-checks that the association's request
        ID matches the payload's, so a forged body cannot borrow a real
        association.

        A lookup miss is not an error.  The platform can deliver before the local
        submission has committed its request ID, and rejecting that callback would
        lose a real result the platform will not send again.  It is parked and
        bound later by ``bind_pending_callbacks``.
        """

        if self.client is None:
            raise BackendNotConfigured("cannot authenticate a callback without platform configuration")
        association = self.outbox.association(_peek_request_id(raw_body))
        callback = self.client.authenticate_callback(raw_body, headers, association=association)
        stored = self.outbox.record_callback(callback)
        return callback, stored

    def reconcile(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Detect rows whose callback never arrived or whose POST was ambiguous.

        This reports; it does not guess.  Spec section 7 requires a reconciler
        precisely because queue status is not an output store and a lost webhook
        can lose an output.
        """

        bound = self.outbox.bind_pending_callbacks()
        rows = self.outbox.unsettled(run_id)
        resolved: List[str] = []
        still_unknown: List[Dict[str, Any]] = []
        for row in rows:
            request_id = row.get("request_id")
            if request_id:
                stored = self.outbox.callback_for(str(request_id))
                if stored is not None:
                    payload = json.loads(stored["payload_json"])
                    data = payload.get("data")
                    self.outbox.settle(
                        row["logical_key"],
                        "completed" if isinstance(data, Mapping) else "failed",
                        data if isinstance(data, Mapping) else None,
                        None if isinstance(data, Mapping) else {"reason": "callback_carried_no_result"},
                    )
                    resolved.append(row["logical_key"])
                    continue
            # The opaque key is committed before the POST, so it can recover a
            # result even when the process never durably learned Baseten's
            # request ID. This is the only permitted ambiguous-POST recovery;
            # it never submits a second request.
            if self.result_store is not None:
                result = self.result_store.get_result(self.outbox.result_binding(str(row["logical_key"])))
                if isinstance(result, Mapping):
                    terminal_state = "completed" if result.get("status") == "completed" else "failed"
                    self.outbox.settle(
                        row["logical_key"],
                        terminal_state,
                        result,
                        None
                        if terminal_state == "completed"
                        else {
                            "reason": "remote_chain_terminal_%s" % str(result.get("status") or "unknown"),
                            "request_id": request_id,
                        },
                    )
                    resolved.append(row["logical_key"])
                    continue
            still_unknown.append(
                {
                    "logical_key": row["logical_key"],
                    "episode_id": row["episode_id"],
                    "state": row["state"],
                    "request_id": request_id,
                    "attempt_count": row["attempt_count"],
                }
            )
        return {
            "run_id": run_id,
            "reconciled_at": utc_now(),
            "late_callbacks_bound": bound,
            "resolved": resolved,
            "unresolved": still_unknown,
            "counts": self.outbox.counts(run_id),
            "note": "unresolved rows stay unevaluable; a lost outcome is never replaced by a guess",
        }

    # -- telemetry ------------------------------------------------------------

    def telemetry(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Platform queue, replicas and cost, each with an explicit availability.

        Queue depth and USD stay ``unavailable`` without a verified route and a
        verified price basis respectively; the application's own ledger counts
        are always available and are never relabelled as platform queue depth.
        """

        payload: Dict[str, Any] = {
            "source": "baseten_chain" if self.transport_kind == "network" else "simulated_chain",
            "transport_kind": self.transport_kind,
            "simulated": self.transport_kind == "simulated",
            "fetched_at": utc_now(),
            "outbox_counts": self.outbox.counts(run_id),
            "platform_queue": {"status": "unavailable", "reason": "no verified Chain queue route configured"},
            "replicas": {"status": "unavailable", "reason": "chain deployment not queried"},
            "economics": self.allocations.report(),
        }
        if self.client is None:
            payload["platform_queue"]["reason"] = "backend not configured"
            return payload
        try:
            queue = _run_async(self.client.get_queue_status())
            payload["platform_queue"] = {
                "status": queue.availability,
                "queued": queue.queued_requests,
                "in_progress": queue.in_progress_requests,
                "reason": queue.reason,
                "source_url": queue.source_url,
            }
        except PlatformError as exc:
            payload["platform_queue"] = {"status": "error", "reason": str(exc)}
        return payload


def _chain_timing(chain_result: Mapping[str, Any]) -> Dict[str, Any]:
    """Read per-stage timings from either the flat or the nested Chain shape.

    ``deploy/baseten/chain.py`` emits ``timings`` as a list of
    ``{stage, seconds}`` entries; a flat ``timing`` mapping is also accepted.
    Reading only one of them silently produced an empty timing record.
    """

    flat = chain_result.get("timing")
    if isinstance(flat, Mapping):
        return dict(flat)
    entries = chain_result.get("timings")
    timing: Dict[str, Any] = {}
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            stage = entry.get("stage")
            seconds = entry.get("seconds", entry.get("wall_seconds"))
            if stage is None or seconds is None:
                continue
            key = "%s_seconds" % stage
            try:
                value = float(seconds)
            except (TypeError, ValueError):
                continue
            timing[key] = timing.get(key, 0.0) + value
    return timing


def _chain_segment_count(chain_result: Mapping[str, Any]) -> Optional[int]:
    """Segment count from ``n_segments`` or the length of ``segments``."""

    flat = chain_result.get("n_segments")
    if isinstance(flat, int) and not isinstance(flat, bool):
        return flat
    segments = chain_result.get("segments")
    if isinstance(segments, (list, tuple)):
        return len(segments)
    return None


def _chain_segments(chain_result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalise the Chain's per-segment records for the console's wall."""

    raw = chain_result.get("segments")
    if not isinstance(raw, (list, tuple)):
        return []
    segments: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            continue
        urls = entry.get("frame_urls")
        segments.append(
            {
                "segment_index": int(entry.get("index", index)),
                "frame_urls": [str(item) for item in urls] if isinstance(urls, (list, tuple)) else [],
                # None, not a guess: a segment that did not report a certified
                # count must be shown as unreported rather than assumed to be 16.
                "certified_frame_count": entry.get("certified_frame_count"),
                "provenance": entry.get("provenance"),
                "status": entry.get("status"),
            }
        )
    return segments


def _chain_gpu_seconds(chain_result: Mapping[str, Any], nested_key: str) -> Optional[float]:
    """GPU seconds from a flat alias or the nested ``gpu_seconds`` accounting.

    Returns ``None`` rather than 0.0 when neither is present: an unmeasured
    GPU-second is unknown, not free.
    """

    flat = chain_result.get(nested_key)
    if flat is None:
        flat = chain_result.get("compute_gpu_seconds" if "total" in nested_key else "allocated_gpu_seconds")
    if isinstance(flat, (int, float)) and not isinstance(flat, bool):
        return float(flat)
    accounting = chain_result.get("gpu_seconds")
    if isinstance(accounting, Mapping):
        value = accounting.get(nested_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _peek_request_id(raw_body: bytes) -> str:
    """Read the request ID from an unverified body, for association lookup only.

    The value is used solely to find a candidate ledger row.  Authentication
    happens afterwards and re-checks the ID against the verified payload, so a
    forged body cannot use this to claim someone else's association.
    """

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get("request_id")
    return str(value) if isinstance(value, str) else ""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_async(awaitable: Any) -> Any:
    """Run one coroutine from the engine's synchronous worker thread.

    ``asyncio.run`` is safe here because engine workers are plain threads with no
    running loop.  If a loop is already running the caller is misusing the
    backend, and an explicit error beats a deadlock.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("BasetenChainBackend.execute must not be called from a running event loop")


__all__ = [
    "AllocationLedger",
    "BackendNotConfigured",
    "BasetenBackendSettings",
    "BasetenChainBackend",
    "ChainResultStore",
    "CHAIN_STAGES",
    "OUTBOX_SCHEMA_VERSION",
    "RemoteChainExecutionError",
    "SubmissionOutbox",
]
