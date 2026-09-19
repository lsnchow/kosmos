"""Local, durable execution control plane for explicitly synthetic PLUMB runs."""

from __future__ import annotations

import inspect
import json
import os
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .ledger import LeaseActiveError, Ledger
from .records import (
    ConfigurationError,
    decode_backend_output,
    deterministic_seed,
    file_digest,
    horizon_for,
    normalise_backend_result,
    normalise_config,
    start_id_for,
    start_lineage_for,
    utc_now,
)
from .synthetic import SyntheticBackend


BackendCallable = Callable[..., Any]


class RunService:
    """Owns local run creation, bounded execution, cancellation, and read models.

    The default backend is deliberately synthetic and records an unqualified
    label in every run, event, episode, and artifact.  ``backend`` is an
    adapter hook for future offline/process workers; accepting it does not make
    a run qualified or enable any external service.
    """

    def __init__(self, data_dir: Path, backend: Optional[Any] = None, lease_seconds: float = 30.0) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.data_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.data_dir / "plumb.sqlite3")
        self.backend = backend
        self.lease_seconds = float(lease_seconds)

    def _make_episodes(self, run_id: str, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        for policy in config["policies"]:
            for task in config["tasks"]:
                for index in range(int(config["starts_per_task"])):
                    start_id = start_id_for(index)
                    logical_key = "%s\x1f%s\x1f%s" % (policy, task, start_id)
                    episodes.append(
                        {
                            "episode_id": "episode-" + uuid.uuid4().hex,
                            "logical_key": logical_key,
                            "policy": policy,
                            "task": task,
                            "start_id": start_id,
                            "start_lineage_id": start_lineage_for(task, index),
                            "world_seed": deterministic_seed(int(config["seed"]), policy, task, start_id),
                            "mode": "synthetic/unqualified",
                            "horizon_actions": horizon_for(task),
                        }
                    )
        return episodes

    def create_run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        normalised = normalise_config(config)
        run_id = "run-" + uuid.uuid4().hex
        return self.ledger.create_run(run_id, normalised, self._make_episodes(run_id, normalised))

    def list_runs(self) -> List[Dict[str, Any]]:
        return self.ledger.list_runs()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self.ledger.get_run(run_id)

    def list_episodes(self, run_id: str, limit: int = 1500) -> List[Dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1 or limit > 10_000:
            raise ValueError("limit must be an integer from 1 through 10000")
        # Check the run even if zero episodes are returned in a future API.
        self.ledger.get_run(run_id)
        return self.ledger.list_episodes(run_id, limit)

    def list_events(self, run_id: str, after: int = 0) -> List[Dict[str, Any]]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be a non-negative integer")
        self.ledger.get_run(run_id)
        return self.ledger.list_events(run_id, after)

    def _episode_dir(self, episode: Mapping[str, Any]) -> Path:
        return self.artifacts_dir / str(episode["run_id"]) / str(episode["episode_id"])

    def _attempt_dir(self, episode: Mapping[str, Any]) -> Path:
        """Each execution attempt gets a distinct, never-reused artifact path."""

        return self._episode_dir(episode) / ("attempt-%04d" % int(episode["attempt_count"]))

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".%s.tmp-%d-%s" % (path.name, os.getpid(), uuid.uuid4().hex))
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))

    def _immutable_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Atomically publish a final record once; late workers cannot replace it."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".%s.tmp-%d-%s" % (path.name, os.getpid(), uuid.uuid4().hex))
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # link(2) fails if the destination exists, unlike replace().
                os.link(str(temporary), str(path))
            except FileExistsError:
                pass
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _artifact_ref(self, path: Path, media_type: str) -> Dict[str, str]:
        relative = path.relative_to(self.data_dir).as_posix()
        return {
            "uri": "artifact://%s" % relative,
            "relative_path": relative,
            "artifact_path": relative,
            "url": "/api/artifacts/%s" % relative,
            "sha256": file_digest(path),
            "media_type": media_type,
            "label": "synthetic engineering test artifact; unqualified; not real robot physics",
        }

    def _write_terminal_record(
        self,
        episode: Mapping[str, Any],
        state: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, str]:
        """Persist a per-attempt immutable record before terminal ledger state."""

        path = self._attempt_dir(episode) / "terminal_record.json"
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "kind": "plumb_synthetic_terminal_record",
            "label": "synthetic engineering test artifact; unqualified; not real robot physics",
            "terminal_state": state,
            "written_at": utc_now(),
            "episode": {
                key: episode[key]
                for key in (
                    "run_id",
                    "episode_id",
                    "policy",
                    "task",
                    "start_id",
                    "start_lineage_id",
                    "world_seed",
                    "mode",
                )
            },
        }
        if result is not None:
            payload["result"] = dict(result)
        if error is not None:
            payload["error"] = dict(error)
        self._immutable_json(path, payload)
        return self._artifact_ref(path, "application/json")

    def _invoke_backend(self, episode: Mapping[str, Any], config: Mapping[str, Any], artifact_dir: Path) -> Dict[str, Any]:
        backend = self.backend or SyntheticBackend()
        if hasattr(backend, "execute"):
            output = backend.execute(episode, config, artifact_dir)
        elif callable(backend):
            # Adapter hooks normally receive these three positional values.  A
            # zero-argument worker factory is also useful in integration tests.
            try:
                output = backend(episode, config, artifact_dir)
            except TypeError as exc:
                try:
                    signature = inspect.signature(backend)
                except (TypeError, ValueError):
                    raise exc
                if len(signature.parameters) != 0:
                    raise exc
                output = backend()
        else:
            raise TypeError("backend must be callable or expose execute()")
        return normalise_backend_result(decode_backend_output(output), episode, artifact_dir)

    def _run_one(self, episode: Mapping[str, Any], config: Mapping[str, Any], owner_token: str) -> None:
        episode_id = str(episode["episode_id"])
        attempt_token = str(episode["attempt_token"])
        if self.ledger.cancellation_requested(str(episode["run_id"])):
            terminal = self._write_terminal_record(episode, "cancelled")
            self.ledger.cancel_episode(
                str(episode["run_id"]),
                episode_id,
                "cancelled_before_execution",
                {"terminal_record": terminal},
                owner_token,
                attempt_token,
            )
            return
        try:
            result = self._invoke_backend(episode, config, self._attempt_dir(episode))
            terminal = self._write_terminal_record(episode, "completed", result=result)
            artifact_refs = dict(result["artifact_refs"])
            artifact_refs["terminal_record"] = terminal
            result = dict(result)
            result["artifact_refs"] = artifact_refs
            if self.ledger.cancellation_requested(str(episode["run_id"])):
                self.ledger.cancel_episode(
                    str(episode["run_id"]),
                    episode_id,
                    "cancelled_during_execution",
                    artifact_refs,
                    owner_token,
                    attempt_token,
                )
            else:
                self.ledger.complete_episode(
                    str(episode["run_id"]), episode_id, result, owner_token, attempt_token
                )
        except Exception as exc:  # Record all worker errors as service failures; never drop a planned cell.
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            terminal = self._write_terminal_record(episode, "failed", error=error)
            if self.ledger.cancellation_requested(str(episode["run_id"])):
                self.ledger.cancel_episode(
                    str(episode["run_id"]),
                    episode_id,
                    "cancelled_during_execution",
                    {"terminal_record": terminal},
                    owner_token,
                    attempt_token,
                )
            else:
                self.ledger.fail_episode(
                    str(episode["run_id"]),
                    episode_id,
                    error,
                    {"terminal_record": terminal},
                    owner_token,
                    attempt_token,
                )

    def _reconcile_terminal_artifact(self, episode: Mapping[str, Any], owner_token: str) -> bool:
        """Commit a complete durable attempt record left behind before a crash.

        Invalid/partial JSON is deliberately not repaired: the attempt remains
        an orphan and is retried as a new attempt with the same deterministic
        inputs. This avoids fabricating a final score from an incomplete write.
        """

        path = self._attempt_dir(episode) / "terminal_record.json"
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if not isinstance(record, dict) or record.get("terminal_state") not in ("completed", "failed"):
                return False
            source = record.get("episode")
            if not isinstance(source, dict) or source.get("run_id") != episode["run_id"] or source.get("episode_id") != episode["episode_id"]:
                return False
            terminal_ref = self._artifact_ref(path, "application/json")
            if record["terminal_state"] == "completed":
                result = normalise_backend_result(decode_backend_output(record.get("result")), episode)
                artifact_refs = dict(result["artifact_refs"])
                artifact_refs["terminal_record"] = terminal_ref
                result = dict(result)
                result["artifact_refs"] = artifact_refs
                return self.ledger.complete_episode(
                    str(episode["run_id"]),
                    str(episode["episode_id"]),
                    result,
                    owner_token,
                    str(episode["attempt_token"]),
                )
            error = record.get("error")
            if not isinstance(error, Mapping):
                return False
            return self.ledger.fail_episode(
                str(episode["run_id"]),
                str(episode["episode_id"]),
                dict(error),
                {"terminal_record": terminal_ref},
                owner_token,
                str(episode["attempt_token"]),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _execute_owned_run(self, run_id: str, owner_token: str) -> None:
        """Execute under an already-acquired owner lease, renewing heartbeats."""

        run = self.ledger.get_run(run_id)
        if not self.ledger.begin_run(run_id, owner_token):
            return
        config = run["config"]
        worker_count = min(int(config["max_workers"]), max(1, int(run["total"])))
        futures: Dict[Future[Any], Mapping[str, Any]] = {}
        owner_lost = False
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="plumb-fixture") as executor:
            while True:
                if not self.ledger.heartbeat_run_lease(run_id, owner_token, self.lease_seconds):
                    owner_lost = True
                while not owner_lost and len(futures) < worker_count and not self.ledger.cancellation_requested(run_id):
                    episode = self.ledger.claim_next_episode(
                        run_id, owner_token, uuid.uuid4().hex, self.lease_seconds
                    )
                    if episode is None:
                        break
                    futures[executor.submit(self._run_one, episode, config, owner_token)] = episode

                if not futures:
                    # Either every planned row was claimed/terminal, or the
                    # cancellation request atomically terminalized remaining rows.
                    break
                # Do not block longer than a fraction of the lease while a
                # backend call is slow; otherwise a live controller could look
                # orphaned and be safely but unnecessarily reclaimed.
                done, _ = wait(
                    tuple(futures),
                    timeout=max(0.05, min(1.0, self.lease_seconds / 3.0)),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    futures.pop(future, None)
                    # _run_one accounts for its own exceptions. This preserves
                    # executor robustness if an unexpected future error escapes.
                    future.result()

        if not owner_lost:
            self.ledger.finalise_run(run_id, owner_token)

    def execute_run(self, run_id: str) -> None:
        """Blocking, bounded execution. Web/API callers may background this call."""

        owner_token = uuid.uuid4().hex
        try:
            acquired = self.ledger.acquire_run_lease(run_id, owner_token, self.lease_seconds)
        except LeaseActiveError:
            # An idempotent API request can race a currently-running worker;
            # leave that owner untouched rather than treating it as a failure.
            return
        if not acquired:
            return
        try:
            # A restarted controller may own a run whose prior process died
            # after claiming rows. Reconcile a durable terminal record first;
            # otherwise preserve that attempt as orphaned and re-run only its
            # deterministic logical cell under a new attempt token.
            for episode in self.ledger.running_episodes(run_id):
                self._reconcile_terminal_artifact(episode, owner_token)
            self.ledger.reclaim_orphaned_attempts(run_id, owner_token, "expired_owner_lease")
            self._execute_owned_run(run_id, owner_token)
        finally:
            self.ledger.release_run_lease(run_id, owner_token)

    def recover_run(self, run_id: str, force: bool = False) -> Dict[str, Any]:
        """Safely resume a crashed fixture run, preserving attempt lineage.

        Normal recovery refuses a live owner lease. ``force=True`` is an
        explicit operator override; token CAS prevents the displaced worker
        from overwriting recovered results if it later returns.
        """

        owner_token = uuid.uuid4().hex
        acquired = self.ledger.acquire_run_lease(
            run_id, owner_token, self.lease_seconds, force=force, recovery=True
        )
        if not acquired:
            return self.ledger.get_run(run_id)
        try:
            for episode in self.ledger.running_episodes(run_id):
                self._reconcile_terminal_artifact(episode, owner_token)
            self.ledger.reclaim_orphaned_attempts(
                run_id, owner_token, "forced_recovery" if force else "expired_owner_lease"
            )
            self._execute_owned_run(run_id, owner_token)
        finally:
            self.ledger.release_run_lease(run_id, owner_token)
        return self.ledger.get_run(run_id)

    def cancel_run(self, run_id: str) -> Dict[str, Any]:
        # Stop new claims first. Then every never-started planned cell gets its
        # own persisted cancellation record before it becomes terminal.
        self.ledger.start_cancellation(run_id)
        for episode in self.ledger.planned_episodes(run_id):
            terminal = self._write_terminal_record(episode, "cancelled")
            self.ledger.cancel_episode(
                run_id,
                str(episode["episode_id"]),
                "cancelled_before_execution",
                {"terminal_record": terminal},
            )
        return self.ledger.finalise_run(run_id)


__all__ = ["RunService", "ConfigurationError"]
