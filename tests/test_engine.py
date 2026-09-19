from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plumb.engine import RunService
from plumb.backends.baseten import RemoteChainExecutionError
from plumb.ledger import LeaseActiveError
from plumb.records import ConfigurationError, file_digest, normalise_backend_result


def small_config(**changes):
    config = {
        "mode": "synthetic",
        "backend": "synthetic",
        "policies": ["Policy-A"],
        "tasks": ["open_drawer"],
        "starts_per_task": 2,
        "seed": 42,
        "max_workers": 2,
    }
    config.update(changes)
    return config


class SlowFixtureBackend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, episode, config, artifact_dir):
        self.started.set()
        self.release.wait(timeout=3)
        return {
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "artifact_refs": {},
            "timing": {"source": "slow-test"},
        }


class LateWorkerBackend:
    """First attempt blocks; recovered second attempt must win permanently."""

    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def execute(self, episode, config, artifact_dir):
        with self.lock:
            self.calls += 1
            call = self.calls
        if call == 1:
            self.first_started.set()
            self.release_first.wait(timeout=3)
            return {
                "validity": "valid",
                "binary_success": True,
                "progress_score": 5,
                "artifact_refs": {},
                "timing": {"source": "late-worker"},
            }
        return {
            "validity": "valid",
            "binary_success": False,
            "progress_score": 0,
            "artifact_refs": {},
            "timing": {"source": "recovered-worker"},
        }


class ExternalUrlBackend:
    def execute(self, episode, config, artifact_dir):
        return {
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "artifact_refs": {"frame_url": "/api/artifacts/cluster-evidence/real-looking.mp4"},
            "timing": {},
        }


class ArtifactBearingRemoteFailureBackend:
    """Models the typed failure emitted after a Chain's terminal result."""

    def execute(self, episode, config, artifact_dir):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "baseten_terminal_failure.json"
        path.write_text(json.dumps({"received_status": "failed", "run_id": episode["run_id"]}), encoding="utf-8")
        relative = path.relative_to(artifact_dir.parents[3]).as_posix()
        ref = {
            "uri": "artifact://%s" % relative,
            "relative_path": relative,
            "artifact_path": relative,
            "url": "/api/artifacts/%s" % relative,
            "sha256": file_digest(path),
            "media_type": "application/json",
        }
        raise RemoteChainExecutionError(
            {"reason": "remote_chain_terminal_failed"}, {"chain_terminal_failure": ref}
        )


class RunServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_idempotency_reuses_the_original_logical_run(self):
        service = RunService(self.data_dir)
        first = service.create_run(small_config(idempotency_key="browser-request-17"))
        second = service.create_run(small_config(idempotency_key="browser-request-17"))
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(1, len(service.list_runs()))
        self.assertEqual(2, first["total"])
        self.assertEqual(2, len(service.list_episodes(first["id"])))
        with self.assertRaises(ConfigurationError):
            service.create_run(small_config(idempotency_key="browser-request-17", seed=999))

    def test_full_matrix_counts_are_logical_episodes_not_threads_or_attempts(self):
        service = RunService(self.data_dir)
        run = service.create_run(
            small_config(policies=["Policy-A", "Policy-B"], tasks=["open_drawer", "close_drawer"], starts_per_task=2)
        )
        service.execute_run(run["id"])
        final = service.get_run(run["id"])
        self.assertEqual("completed", final["status"])
        self.assertEqual(8, final["total"])
        self.assertEqual(8, final["completed"])
        self.assertEqual(0, final["failed"])
        self.assertEqual(8, final["evaluable"])
        events = service.list_events(run["id"])
        self.assertEqual(list(range(1, len(events) + 1)), [event["sequence"] for event in events])
        self.assertTrue(all(episode["mode"] == "synthetic/unqualified" for episode in service.list_episodes(run["id"])))

    def test_failures_remain_in_the_denominator_and_are_not_scored(self):
        service = RunService(self.data_dir)
        run = service.create_run(small_config(synthetic_failure_start_ids=["start-000"]))
        service.execute_run(run["id"])
        final = service.get_run(run["id"])
        self.assertEqual(2, final["total"])
        self.assertEqual(1, final["completed"])
        self.assertEqual(1, final["failed"])
        self.assertEqual(1, final["evaluable"])
        failed = [item for item in service.list_episodes(run["id"]) if item["status"] == "failed"]
        self.assertEqual("service_failure", failed[0]["missing_reason"])
        self.assertIsNone(failed[0]["binary_success"])

    def test_cancellation_leaves_terminal_cancelled_records(self):
        backend = SlowFixtureBackend()
        service = RunService(self.data_dir, backend=backend)
        run = service.create_run(small_config(starts_per_task=3, max_workers=1))
        execution = threading.Thread(target=service.execute_run, args=(run["id"],))
        execution.start()
        self.assertTrue(backend.started.wait(timeout=2))
        cancelled = service.cancel_run(run["id"])
        self.assertGreaterEqual(cancelled["cancelled"], 2)
        backend.release.set()
        execution.join(timeout=3)
        self.assertFalse(execution.is_alive())
        final = service.get_run(run["id"])
        self.assertEqual("cancelled", final["status"])
        self.assertEqual(3, final["cancelled"])
        self.assertEqual(3, final["total"])

    def test_missing_fixture_labels_stay_nullable_and_unqualified(self):
        service = RunService(self.data_dir)
        run = service.create_run(small_config(synthetic_missing_start_ids=["start-000"]))
        service.execute_run(run["id"])
        final = service.get_run(run["id"])
        self.assertEqual(1, final["evaluable"])
        missing = [item for item in service.list_episodes(run["id"]) if item["start_id"] == "start-000"][0]
        self.assertEqual("unknown", missing["validity"])
        self.assertIsNone(missing["binary_success"])
        self.assertEqual("synthetic_fixture_missing_label", missing["missing_reason"])
        self.assertIn("unqualified", missing["artifact_refs"]["fixture_frame"]["label"])
        self.assertTrue(missing["frame_url"].startswith("/api/artifacts/artifacts/"))

    def test_qualified_mode_is_rejected_while_gates_are_unpassed(self):
        service = RunService(self.data_dir)
        with self.assertRaises(ConfigurationError):
            service.create_run(small_config(mode="qualified"))

    def test_local_matrix_cap_prevents_unreadable_runs(self):
        service = RunService(self.data_dir)
        with self.assertRaises(ConfigurationError):
            service.create_run(small_config(policies=["p-%d" % index for index in range(100)], starts_per_task=101))

    def test_recovery_reclaims_an_orphan_without_creating_another_logical_episode(self):
        service = RunService(self.data_dir)
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        old_owner = "simulated-crashed-owner"
        self.assertTrue(service.ledger.acquire_run_lease(run["id"], old_owner, 30))
        self.assertTrue(service.ledger.begin_run(run["id"], old_owner))
        claimed = service.ledger.claim_next_episode(run["id"], old_owner, "old-attempt-token", 30)
        self.assertIsNotNone(claimed)
        # Releasing the owner represents a process that died after durable
        # claim/attempt creation but before it could write a terminal result.
        service.ledger.release_run_lease(run["id"], old_owner)
        recovered = service.recover_run(run["id"])
        episode = service.list_episodes(run["id"])[0]
        self.assertEqual("completed", recovered["status"])
        self.assertEqual(1, recovered["total"])
        self.assertEqual(2, episode["attempt_count"])
        self.assertTrue(any(event["type"] == "episode_reclaimed" for event in service.list_events(run["id"])))

    def test_recovery_reconciles_a_durable_terminal_record_before_retrying(self):
        service = RunService(self.data_dir)
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        old_owner = "crashed-after-artifact"
        self.assertTrue(service.ledger.acquire_run_lease(run["id"], old_owner, 30))
        self.assertTrue(service.ledger.begin_run(run["id"], old_owner))
        claimed = service.ledger.claim_next_episode(run["id"], old_owner, "artifact-attempt-token", 30)
        result = {
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "artifact_refs": {},
            "timing": {"source": "durable-crash-fixture"},
        }
        service._write_terminal_record(claimed, "completed", result=result)
        service.ledger.release_run_lease(run["id"], old_owner)
        recovered = service.recover_run(run["id"])
        episode = service.list_episodes(run["id"])[0]
        self.assertEqual("completed", recovered["status"])
        self.assertEqual(1, recovered["successes"])
        self.assertEqual(1, episode["attempt_count"])
        self.assertIn("terminal_record", episode["artifact_refs"])

    def test_live_owner_requires_force_and_late_worker_cannot_overwrite_recovery(self):
        backend = LateWorkerBackend()
        service = RunService(self.data_dir, backend=backend, lease_seconds=5)
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        original = threading.Thread(target=service.execute_run, args=(run["id"],))
        original.start()
        self.assertTrue(backend.first_started.wait(timeout=2))
        with self.assertRaises(LeaseActiveError):
            service.recover_run(run["id"])
        recovered = service.recover_run(run["id"], force=True)
        self.assertEqual("completed", recovered["status"])
        self.assertEqual(1, recovered["completed"])
        backend.release_first.set()
        original.join(timeout=3)
        self.assertFalse(original.is_alive())
        episode = service.list_episodes(run["id"])[0]
        self.assertFalse(episode["binary_success"])
        self.assertEqual(2, episode["attempt_count"])
        old_terminal = self.data_dir / "artifacts" / run["id"] / episode["episode_id"] / "attempt-0001" / "terminal_record.json"
        new_terminal = self.data_dir / "artifacts" / run["id"] / episode["episode_id"] / "attempt-0002" / "terminal_record.json"
        self.assertTrue(old_terminal.is_file())
        self.assertTrue(new_terminal.is_file())

    def test_execute_run_recovers_released_crash_claims_before_scheduling(self):
        service = RunService(self.data_dir)
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        old_owner = "released-crashed-owner"
        self.assertTrue(service.ledger.acquire_run_lease(run["id"], old_owner, 30))
        self.assertTrue(service.ledger.begin_run(run["id"], old_owner))
        self.assertIsNotNone(service.ledger.claim_next_episode(run["id"], old_owner, "abandoned-token", 30))
        service.ledger.release_run_lease(run["id"], old_owner)
        service.execute_run(run["id"])
        final = service.get_run(run["id"])
        episode = service.list_episodes(run["id"])[0]
        self.assertEqual("completed", final["status"])
        self.assertEqual((1, 1), (final["total"], final["completed"]))
        self.assertEqual(2, episode["attempt_count"])

    def test_malformed_progress_and_external_artifact_urls_are_rejected(self):
        episode = {"horizon_actions": 70, "run_id": "r", "episode_id": "e"}
        for result in (
            {"validity": "valid", "binary_success": True, "progress_score": 4},
            {"validity": "valid", "binary_success": False, "progress_score": 5},
            {"validity": "valid", "binary_success": None, "progress_score": 1},
            {"validity": "unknown", "binary_success": False, "progress_score": 0},
        ):
            with self.assertRaises(ValueError):
                normalise_backend_result({**result, "artifact_refs": {}, "timing": {}}, episode)
        service = RunService(self.data_dir, backend=ExternalUrlBackend())
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        service.execute_run(run["id"])
        failed = service.list_episodes(run["id"])[0]
        self.assertEqual("failed", failed["status"])
        self.assertEqual("service_failure", failed["missing_reason"])

    def test_typed_remote_failure_keeps_its_raw_artifact_through_engine_terminalization(self):
        service = RunService(self.data_dir, backend=ArtifactBearingRemoteFailureBackend())
        run = service.create_run(small_config(starts_per_task=1, max_workers=1))
        service.execute_run(run["id"])

        episode = service.list_episodes(run["id"])[0]
        self.assertEqual((episode["status"], episode["missing_reason"]), ("failed", "service_failure"))
        self.assertIn("chain_terminal_failure", episode["artifact_refs"])
        terminal = self.data_dir / episode["artifact_refs"]["terminal_record"]["relative_path"]
        terminal_record = json.loads(terminal.read_text())
        self.assertIn("chain_terminal_failure", terminal_record["error"]["backend_failure_artifacts"])


if __name__ == "__main__":
    unittest.main()
