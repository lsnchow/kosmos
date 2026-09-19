"""Crash and ordering contracts for the durable Baseten application outbox."""

from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Union

from plumb.ledger import Ledger
from plumb.records import ConfigurationError
from plumb.engine import RunService
from plumb.outbox import BasetenOutbox, RemoteRequestController
from plumb.platform import (
    AsyncChainRequestOptions,
    BasetenChainClient,
    BasetenPlatformConfig,
    ChainAsyncEndpoint,
    HttpRequest,
    HttpResponse,
    PlatformTransportError,
)


class RecordedTransport:
    def __init__(self, responses: List[Union[HttpResponse, Exception]]) -> None:
        self.responses = list(responses)
        self.requests: List[HttpRequest] = []

    async def request(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(status: int, payload: object) -> HttpResponse:
    return HttpResponse(status_code=status, headers={}, body=json.dumps(payload).encode("utf-8"))


def config() -> BasetenPlatformConfig:
    return BasetenPlatformConfig(
        api_key="test-api-key",
        webhook_secret="webhook-test-secret",
        chain_endpoint=ChainAsyncEndpoint(
            "https://chain-chain_test.api.baseten.co/deployment/deploy_test/async_run_remote"
        ),
    )


def callback_body(request_id: str, output: object = None) -> bytes:
    return json.dumps(
        {
            "request_id": request_id,
            "type": "async_request_completed",
            "deployment_id": "deploy_test",
            "data": {"output": output},
            "errors": [],
        },
        separators=(",", ":"),
    ).encode("utf-8")


def callback_headers(body: bytes, request_id: str) -> dict[str, str]:
    signature = hmac.new(b"webhook-test-secret", body, hashlib.sha256).hexdigest()
    return {
        "X-BASETEN-SIGNATURE": "v1=" + signature,
        "X-BASETEN-REQUEST-ID": request_id,
    }


class FakeController(RemoteRequestController):
    def __init__(self, statuses: List[str], cancellation: str = "CANCELED") -> None:
        self.statuses = list(statuses)
        self.cancellation = cancellation
        self.status_requests: List[str] = []
        self.cancel_requests: List[str] = []

    async def get_status(self, request_id: str) -> str:
        self.status_requests.append(request_id)
        return self.statuses.pop(0)

    async def cancel_queued(self, request_id: str) -> str:
        self.cancel_requests.append(request_id)
        return self.cancellation


class OutboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Ledger(Path(self.temp.name) / "plumb.sqlite3")
        self.run_id = "run-test"
        self.episode_id = "episode-test"
        self.ledger.create_run(
            self.run_id,
            {"mode": "synthetic", "protocol_hash": "sha256:frozen-protocol"},
            [
                {
                    "episode_id": self.episode_id,
                    "logical_key": "logical-test",
                    "policy": "OpenVLA",
                    "task": "open_drawer",
                    "start_id": "start-001",
                    "start_lineage_id": "lineage-001",
                    "world_seed": 1,
                    "mode": "synthetic",
                    "horizon_actions": 70,
                }
            ],
        )
        self.options = AsyncChainRequestOptions("https://control.example.test/baseten/callback")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_outbox(self, transport: RecordedTransport) -> BasetenOutbox:
        return BasetenOutbox(self.ledger, BasetenChainClient(config(), transport=transport), dispatch_lease_seconds=0.01)

    def enqueue(self, outbox: BasetenOutbox, outbox_id: str = "outbox-test") -> None:
        outbox.enqueue(
            outbox_id,
            self.run_id,
            self.episode_id,
            {
                "request": {
                    "run_id": self.run_id,
                    "episode_id": self.episode_id,
                    "protocol_hash": "sha256:frozen-protocol",
                    "policy": {"episode_id": self.episode_id, "protocol_hash": "sha256:frozen-protocol"},
                    "world": {"episode_id": self.episode_id, "protocol_hash": "sha256:frozen-protocol"},
                    "validity": {"episode_id": self.episode_id, "protocol_hash": "sha256:frozen-protocol"},
                    "judge": {"episode_id": self.episode_id, "protocol_hash": "sha256:frozen-protocol"},
                }
            },
            self.options,
            "2030-01-01T00:00:00Z",
        )

    @staticmethod
    def completed_result() -> dict[str, object]:
        return {
            "run_id": "run-test", "episode_id": "episode-test", "protocol_hash": "sha256:frozen-protocol",
            "validity": "valid",
            "binary_success": True,
            "progress_score": 5,
            "missing_reason": None,
            "horizon_actions": 70,
            "executed_actions": 70,
            "timing": {"total_seconds": 1.0},
        }

    @staticmethod
    def artifact(uri: str) -> dict:
        return {"final_artifact": {"uri": uri, "sha256": "sha256:" + "a" * 64, "media_type": "application/json"}}

    async def test_enqueue_rejects_identity_mismatch_non_utc_deadline_and_nonfinite_lease(self) -> None:
        outbox = self.make_outbox(RecordedTransport([]))
        invalid_request = {
            "request": {
                "run_id": self.run_id,
                "episode_id": "wrong-episode",
                "protocol_hash": "sha256:frozen-protocol",
                "policy": {"episode_id": "wrong-episode", "protocol_hash": "sha256:frozen-protocol"},
                "world": {"episode_id": "wrong-episode", "protocol_hash": "sha256:frozen-protocol"},
                "validity": {"episode_id": "wrong-episode", "protocol_hash": "sha256:frozen-protocol"},
                "judge": {"episode_id": "wrong-episode", "protocol_hash": "sha256:frozen-protocol"},
            }
        }
        with self.assertRaises(ConfigurationError):
            outbox.enqueue("outbox-test", self.run_id, self.episode_id, invalid_request, self.options, "2030-01-01T00:00:00Z")
        with self.assertRaises(ValueError):
            outbox.enqueue("outbox-test", self.run_id, self.episode_id, invalid_request, self.options, "2030-01-01T00:00:00+01:00")
        with self.assertRaises(ValueError):
            BasetenOutbox(self.ledger, BasetenChainClient(config(), transport=RecordedTransport([])), dispatch_lease_seconds=float("nan"))

    async def test_post_ack_callback_finalization_is_idempotent_and_scores_once(self) -> None:
        transport = RecordedTransport([response(202, {"request_id": "request-accepted"})])
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)

        dispatched = await outbox.dispatch("outbox-test", "dispatch-one")
        self.assertEqual((dispatched.disposition, dispatched.request_id), ("accepted", "request-accepted"))
        self.assertEqual(1, len(transport.requests))
        body = callback_body("request-accepted", {"final": "value"})
        received = outbox.ingest_callback(body, callback_headers(body, "request-accepted"))
        duplicate = outbox.ingest_callback(body, callback_headers(body, "request-accepted"))
        self.assertTrue(received.associated)
        self.assertFalse(received.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(self.run_id, received.callback.association.run_id)
        self.assertEqual(body, self.ledger.get_baseten_callback("request-accepted")["raw_body"])
        restarted = Ledger(Path(self.temp.name) / "plumb.sqlite3")
        self.assertEqual(body, restarted.get_baseten_callback("request-accepted")["raw_body"])

        claimed = outbox.claim_callback_finalization("outbox-test", "artifact-writer")
        self.assertEqual("processing", claimed["state"])
        self.assertTrue(
            outbox.finalise_callback(
                "outbox-test",
                "artifact-writer",
                "completed",
                self.artifact("object://final/request-accepted.json"),
                result=self.completed_result(),
            )
        )
        self.assertFalse(
            outbox.finalise_callback(
                "outbox-test",
                "artifact-writer",
                "completed",
                self.artifact("object://final/request-accepted.json"),
                result=self.completed_result(),
            )
        )
        episode = self.ledger.list_episodes(self.run_id)[0]
        self.assertEqual((episode["status"], episode["attempt_count"], episode["binary_success"]), ("completed", 1, True))
        self.assertEqual("finalized", self.ledger.get_baseten_submission("outbox-test")["state"])

    async def test_finalization_rejects_label_coercion_changed_horizon_and_unbound_artifacts(self) -> None:
        outbox = self.make_outbox(RecordedTransport([response(202, {"request_id": "strict-result"})]))
        self.enqueue(outbox)
        await outbox.dispatch("outbox-test")
        body = callback_body("strict-result", {"run_id": self.run_id, "episode_id": self.episode_id})
        outbox.ingest_callback(body, callback_headers(body, "strict-result"))
        outbox.claim_callback_finalization("outbox-test", "strict-writer")
        for replacement in ({"binary_success": "false"}, {"binary_success": 1},
                            {"horizon_actions": 16}, {"executed_actions": 16}, {"run_id": "wrong-run"},
                            {"missing_reason": "service_failure"},
                            {"validity": "unknown"}, {"progress_score": 4},
                            {"timing": {"total_seconds": float("nan")}}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                outbox.finalise_callback("outbox-test", "strict-writer", "completed",
                    self.artifact("object://final/strict.json"), result={**self.completed_result(), **replacement})
            self.assertEqual(0, self.ledger.get_run(self.run_id)["completed"])
        with self.assertRaises(ValueError):
            outbox.finalise_callback("outbox-test", "strict-writer", "completed",
                {"final_artifact": "object://no-digest.json"}, result=self.completed_result())
        self.assertTrue(outbox.finalise_callback("outbox-test", "strict-writer", "completed",
            self.artifact("object://final/strict.json"), result=self.completed_result()))

    async def test_callback_declared_result_identity_must_match_request(self) -> None:
        outbox = self.make_outbox(RecordedTransport([response(202, {"request_id": "wrong-identity"})]))
        self.enqueue(outbox)
        await outbox.dispatch("outbox-test")
        body = callback_body("wrong-identity", {"episode_id": "another-episode"})
        outbox.ingest_callback(body, callback_headers(body, "wrong-identity"))
        outbox.claim_callback_finalization("outbox-test", "strict-writer")
        with self.assertRaises(ValueError):
            outbox.finalise_callback("outbox-test", "strict-writer", "completed",
                self.artifact("object://final/wrong.json"), result=self.completed_result())
        self.assertEqual(0, self.ledger.get_run(self.run_id)["completed"])

    async def test_callback_before_ack_is_held_then_atomically_associated(self) -> None:
        class CallbackBeforeAckTransport(RecordedTransport):
            def __init__(self) -> None:
                super().__init__([response(202, {"request_id": "request-early"})])
                self.deliver = None

            async def request(self, request: HttpRequest) -> HttpResponse:
                self.requests.append(request)
                assert self.deliver is not None
                body = callback_body("request-early", "arrived-before-ack")
                early = self.deliver(body, callback_headers(body, "request-early"))
                assert not early.associated
                return self.responses.pop(0)  # type: ignore[return-value]

        transport = CallbackBeforeAckTransport()
        outbox = self.make_outbox(transport)
        transport.deliver = outbox.ingest_callback
        self.enqueue(outbox)

        result = await outbox.dispatch("outbox-test", "dispatch-early")
        self.assertEqual("accepted", result.disposition)
        submission = self.ledger.get_baseten_submission("outbox-test")
        callback = self.ledger.get_baseten_callback("request-early")
        self.assertEqual((submission["state"], callback["outbox_id"], callback["state"]), (
            "callback_received", "outbox-test", "ready"
        ))

    async def test_cancellation_during_post_keeps_dispatch_state_until_request_id_is_durable(self) -> None:
        class CancelDuringPostTransport(RecordedTransport):
            def __init__(self) -> None:
                super().__init__([response(202, {"request_id": "request-cancel-during-post"})])
                self.cancel = None

            async def request(self, request: HttpRequest) -> HttpResponse:
                self.requests.append(request)
                assert self.cancel is not None
                self.cancel()
                return self.responses.pop(0)  # type: ignore[return-value]

        transport = CancelDuringPostTransport()
        outbox = self.make_outbox(transport)
        transport.cancel = lambda: outbox.request_cancellation("outbox-test")
        self.enqueue(outbox)

        result = await outbox.dispatch("outbox-test", "dispatch-cancel-race")
        saved = self.ledger.get_baseten_submission("outbox-test")
        self.assertEqual("accepted", result.disposition)
        self.assertEqual(
            ("cancellation_requested", "request-cancel-during-post", True),
            (saved["state"], saved["platform_request_id"], saved["cancellation_requested"]),
        )

    async def test_ambiguous_post_is_never_reposted_after_restart(self) -> None:
        transport = RecordedTransport([PlatformTransportError("network dropped")])
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)

        first = await outbox.dispatch("outbox-test", "dispatch-ambiguous")
        self.assertEqual("ambiguous", first.disposition)
        self.assertEqual(1, len(transport.requests))
        restarted = BasetenOutbox(self.ledger, BasetenChainClient(config(), transport=transport))
        second = await restarted.dispatch("outbox-test", "after-restart")
        self.assertEqual("not_dispatchable", second.disposition)
        self.assertEqual(1, len(transport.requests))
        self.assertEqual("ambiguous", self.ledger.get_baseten_submission("outbox-test")["state"])

    async def test_crashed_pre_ack_dispatch_recovers_to_ambiguous_without_post(self) -> None:
        outbox = self.make_outbox(RecordedTransport([]))
        self.enqueue(outbox)
        self.assertIsNotNone(self.ledger.claim_baseten_dispatch("outbox-test", "crashed", 0.01))
        self.assertEqual(1, self.ledger.recover_expired_baseten_dispatches("9999-01-01T00:00:00Z"))
        self.assertEqual("ambiguous", self.ledger.get_baseten_submission("outbox-test")["state"])
        self.assertEqual(
            "not_dispatchable", (await outbox.dispatch("outbox-test", "must-not-post")).disposition
        )

    async def test_known_rejection_can_only_be_manually_requeued_on_same_logical_episode(self) -> None:
        transport = RecordedTransport(
            [response(429, {"error": "rate_limited"}), response(202, {"request_id": "request-retry"})]
        )
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)
        rejected = await outbox.dispatch("outbox-test", "first-post")
        self.assertEqual("rejected", rejected.disposition)
        self.assertEqual(1, len(transport.requests))
        self.assertTrue(self.ledger.requeue_baseten_known_rejection("outbox-test", "rate limit passed"))
        accepted = await outbox.dispatch("outbox-test", "operator-retry")
        self.assertEqual("accepted", accepted.disposition)
        self.assertEqual(2, len(transport.requests))
        episode = self.ledger.list_episodes(self.run_id)[0]
        self.assertEqual((episode["attempt_count"], episode["status"]), (2, "running"))

    async def test_cancellation_after_known_rejection_never_posts_again(self) -> None:
        transport = RecordedTransport([response(422, {"error": "invalid"})])
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)
        self.assertEqual("rejected", (await outbox.dispatch("outbox-test", "rejected-then-cancel")).disposition)
        requested = outbox.request_cancellation("outbox-test")
        self.assertEqual("cancelled_pre_dispatch", requested["state"])
        self.assertEqual("planned", self.ledger.list_episodes(self.run_id)[0]["status"])
        self.assertTrue(
            outbox.finalise_pre_dispatch_cancellation(
                "outbox-test", self.artifact("object://final/rejected-cancelled.json")
            )
        )
        self.assertEqual(1, len(transport.requests))
        self.assertEqual("cancelled", self.ledger.list_episodes(self.run_id)[0]["status"])

    async def test_cancellation_preserves_late_callback_and_uses_verified_controller_only(self) -> None:
        transport = RecordedTransport([response(202, {"request_id": "request-cancel"})])
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)
        self.assertEqual("accepted", (await outbox.dispatch("outbox-test", "dispatch-cancel")).disposition)
        outbox.request_cancellation("outbox-test")

        without_controller = await outbox.reconcile()
        self.assertEqual(("outbox-test",), without_controller.unavailable)
        controller = FakeController(["QUEUED"])
        reconciled = await outbox.reconcile(controller)
        self.assertEqual(("outbox-test",), reconciled.cancellation_attempted)
        self.assertEqual(["request-cancel"], controller.cancel_requests)
        self.assertEqual(
            "remote_terminal_without_callback", self.ledger.get_baseten_submission("outbox-test")["state"]
        )

        body = callback_body("request-cancel", "late-output")
        late = outbox.ingest_callback(body, callback_headers(body, "request-cancel"))
        self.assertTrue(late.associated)
        self.assertTrue(self.ledger.get_baseten_submission("outbox-test")["cancellation_requested"])
        self.assertIsNotNone(outbox.claim_callback_finalization("outbox-test", "cancel-writer"))
        self.assertTrue(
            outbox.finalise_callback(
                "outbox-test",
                "cancel-writer",
                "cancelled",
                self.artifact("object://final/late-output.json"),
                cancellation_reason="cancelled_after_remote_work",
            )
        )
        self.assertEqual("cancelled", self.ledger.list_episodes(self.run_id)[0]["status"])

    async def test_callbackless_remote_cancellation_requires_explicit_terminal_artifact(self) -> None:
        transport = RecordedTransport([response(202, {"request_id": "request-no-callback"})])
        outbox = self.make_outbox(transport)
        self.enqueue(outbox)
        self.assertEqual("accepted", (await outbox.dispatch("outbox-test", "dispatch-no-callback")).disposition)
        outbox.request_cancellation("outbox-test")
        controller = FakeController(["QUEUED"])
        await outbox.reconcile(controller)
        self.assertTrue(
            outbox.finalise_remote_terminal(
                "outbox-test",
                "cancelled",
                self.artifact("object://final/cancelled-request.json"),
                cancellation_reason="remote_queue_cancelled",
            )
        )
        self.assertEqual("cancelled", self.ledger.list_episodes(self.run_id)[0]["status"])
        self.assertEqual("finalized", self.ledger.get_baseten_submission("outbox-test")["state"])

    async def test_generic_engine_execution_recovery_and_cancel_preserve_outbox_ownership(self) -> None:
        service = RunService(Path(self.temp.name) / "mixed-run")
        run = service.create_run(
            {
                "mode": "synthetic",
                "backend": "synthetic",
                "starts_per_task": 1,
                "max_workers": 1,
                "policies": ["OpenVLA"],
                "tasks": ["open_drawer"],
                "protocol_hash": "sha256:mixed-protocol",
            }
        )
        run_id = str(run["id"])
        episode = service.ledger.list_episodes(run_id)[0]
        transport = RecordedTransport([PlatformTransportError("lost after POST")])
        outbox = BasetenOutbox(service.ledger, BasetenChainClient(config(), transport=transport))
        request = {
            "request": {
                "run_id": run_id,
                "episode_id": episode["episode_id"],
                "protocol_hash": "sha256:mixed-protocol",
                "policy": {"episode_id": episode["episode_id"], "protocol_hash": "sha256:mixed-protocol"},
                "world": {"episode_id": episode["episode_id"], "protocol_hash": "sha256:mixed-protocol"},
                "validity": {"episode_id": episode["episode_id"], "protocol_hash": "sha256:mixed-protocol"},
                "judge": {"episode_id": episode["episode_id"], "protocol_hash": "sha256:mixed-protocol"},
            }
        }
        outbox.enqueue("mixed-outbox", run_id, episode["episode_id"], request, self.options, "2030-01-01T00:00:00Z")
        service.execute_run(run_id)
        before_dispatch = service.ledger.list_episodes(run_id)[0]
        self.assertEqual(("planned", 0), (before_dispatch["status"], before_dispatch["attempt_count"]))
        self.assertEqual([], transport.requests)
        self.assertEqual("ambiguous", (await outbox.dispatch("mixed-outbox", "remote-owner")).disposition)
        self.assertEqual(1, len(transport.requests))

        # A generic fixture recovery sees the same run but cannot reclaim,
        # execute, cancel-terminalize, or re-POST its cloud-owned episode.
        service.recover_run(run_id)
        after_recovery = service.ledger.list_episodes(run_id)[0]
        self.assertEqual(("running", 1), (after_recovery["status"], after_recovery["attempt_count"]))
        self.assertEqual("ambiguous", service.ledger.get_baseten_submission("mixed-outbox")["state"])
        self.assertEqual(1, len(transport.requests))

        service.cancel_run(run_id)
        after_cancel = service.ledger.get_baseten_submission("mixed-outbox")
        self.assertTrue(after_cancel["cancellation_requested"])
        self.assertEqual("cancellation_requested", after_cancel["state"])
        self.assertEqual("running", service.ledger.list_episodes(run_id)[0]["status"])


if __name__ == "__main__":
    unittest.main()
