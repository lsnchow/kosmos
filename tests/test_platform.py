"""Offline contract tests for PLUMB's deliberately small Baseten client."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import unittest
from decimal import Decimal
from typing import List, Union

from plumb.platform import (
    AmbiguousSubmissionError,
    AsyncChainRequestOptions,
    BasetenChainClient,
    BasetenPlatformConfig,
    CallbackAssociation,
    CallbackVerificationError,
    ChainAsyncEndpoint,
    HttpRequest,
    HttpResponse,
    PlatformConfigurationError,
    PlatformHTTPError,
    PlatformSchemaError,
    PriceUnit,
    SubmissionRetryPolicy,
    VerifiedChainQueueRoute,
    VerifiedPriceBasis,
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


def _response(status_code: int, payload: object) -> HttpResponse:
    return HttpResponse(status_code=status_code, headers={}, body=json.dumps(payload).encode("utf-8"))


def _config(webhook_secret: str = "webhook-test-secret") -> BasetenPlatformConfig:
    return BasetenPlatformConfig(
        api_key="api-test-secret",
        chain_endpoint=ChainAsyncEndpoint(
            "https://chain-chain_123.api.baseten.co/deployment/deploy_456/async_run_remote"
        ),
        webhook_secret=webhook_secret,
    )


class ChainEndpointTests(unittest.TestCase):
    def test_accepts_only_documented_chain_async_routes(self) -> None:
        endpoint = ChainAsyncEndpoint.for_target("chain_123", "environments/staging")
        self.assertEqual(
            endpoint.url,
            "https://chain-chain_123.api.baseten.co/environments/staging/async_run_remote",
        )
        with self.assertRaises(PlatformConfigurationError):
            ChainAsyncEndpoint("https://model-model_123.api.baseten.co/production/async_predict")
        with self.assertRaises(PlatformConfigurationError):
            ChainAsyncEndpoint("https://chain-chain_123.api.baseten.co/production/async_predict")

    def test_env_configuration_requires_explicit_chain_url_and_hides_key(self) -> None:
        config = BasetenPlatformConfig.from_env(
            {
                "BASETEN_API_KEY": "do-not-print",
                "BASETEN_CHAIN_ASYNC_URL": "https://chain-a.api.baseten.co/production/async_run_remote",
            }
        )
        self.assertNotIn("do-not-print", repr(config))
        with self.assertRaises(PlatformConfigurationError):
            BasetenPlatformConfig.from_env({"BASETEN_API_KEY": "x"})


class ChainClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_uses_chain_async_run_remote_and_async_envelope(self) -> None:
        transport = RecordedTransport([_response(202, {"request_id": "request-1"})])
        client = BasetenChainClient(_config(), transport=transport)

        receipt = await client.submit_async(
            {"request": {"run_id": "run-1", "episode_id": "episode-1"}},
            AsyncChainRequestOptions(webhook_endpoint="https://control.example.test/baseten/callback"),
        )

        self.assertEqual(receipt.request_id, "request-1")
        self.assertEqual(len(receipt.attempts), 1)
        self.assertTrue(transport.requests[0].url.endswith("/async_run_remote"))
        self.assertNotIn("async_predict", transport.requests[0].url)
        self.assertEqual(transport.requests[0].headers["Authorization"], "Bearer api-test-secret")
        self.assertEqual(
            json.loads(transport.requests[0].body.decode("utf-8")),
            {
                "model_input": {"request": {"run_id": "run-1", "episode_id": "episode-1"}},
                "webhook_endpoint": "https://control.example.test/baseten/callback",
            },
        )

    async def test_503_is_ambiguous_and_never_causes_a_second_post(self) -> None:
        transport = RecordedTransport(
            [_response(503, {"error": "unavailable"}), _response(200, {"request_id": "request-2"})]
        )
        observed = []

        client = BasetenChainClient(_config(), transport=transport)
        with self.assertRaises(AmbiguousSubmissionError) as raised:
            await client.submit_async(
                {"request": {"run_id": "run-2"}},
                AsyncChainRequestOptions(webhook_endpoint="https://control.example.test/baseten/callback"),
                attempt_observer=observed.append,
            )

        self.assertEqual([attempt.state for attempt in raised.exception.attempts], ["http_error"])
        self.assertEqual([attempt.state for attempt in observed], ["started", "http_error"])
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(1, len(transport.responses))

    async def test_no_optional_flag_can_enable_client_side_submission_retries(self) -> None:
        with self.assertRaises(PlatformConfigurationError):
            SubmissionRetryPolicy(max_transport_retries=1)

    async def test_429_is_a_single_known_rejection_not_an_automatic_retry(self) -> None:
        transport = RecordedTransport(
            [_response(429, {"error": "rate_limited"}), _response(200, {"request_id": "must-not-send"})]
        )
        client = BasetenChainClient(_config(), transport=transport)
        with self.assertRaises(PlatformHTTPError) as raised:
            await client.submit_async(
                {"request": {"run_id": "run-rate-limit"}},
                AsyncChainRequestOptions(webhook_endpoint="https://control.example.test/baseten/callback"),
            )
        self.assertEqual(429, raised.exception.status_code)
        self.assertEqual(1, len(transport.requests))

    async def test_submission_response_must_return_request_id(self) -> None:
        transport = RecordedTransport([_response(200, {"accepted": True})])
        client = BasetenChainClient(_config(), transport=transport)
        with self.assertRaises(AmbiguousSubmissionError):
            await client.submit_async(
                {"request": {"run_id": "run-3"}},
                AsyncChainRequestOptions(webhook_endpoint="https://control.example.test/baseten/callback"),
            )

    async def test_queue_is_unavailable_without_an_explicit_verified_route(self) -> None:
        transport = RecordedTransport([])
        client = BasetenChainClient(_config(), transport=transport)
        status = await client.get_queue_status()
        self.assertEqual(status.availability, "unavailable")
        self.assertEqual(transport.requests, [])

    async def test_verified_queue_route_is_parsed_but_not_inferred(self) -> None:
        route = VerifiedChainQueueRoute(
            url="https://chain-chain_123.api.baseten.co/deployment/deploy_456/async_queue_status",
            evidence_uri="https://artifacts.example.test/queue-route-fixture.json",
            verified_at="2026-09-19T00:00:00Z",
        )
        transport = RecordedTransport(
            [_response(200, {"num_queued_requests": 7, "num_in_progress_requests": 2})]
        )
        client = BasetenChainClient(_config(), transport=transport, queue_route=route)
        status = await client.get_queue_status()
        self.assertEqual((status.queued_requests, status.in_progress_requests), (7, 2))
        self.assertEqual(transport.requests[0].url, route.url)

    async def test_management_deployment_exposes_active_not_configured_max_as_active(self) -> None:
        transport = RecordedTransport(
            [
                _response(
                    200,
                    {
                        "id": "deploy_456",
                        "chain_id": "chain_123",
                        "status": "ACTIVE",
                        "chainlets": [
                            {
                                "name": "WorldWorker",
                                "active_replica_count": 3,
                                "instance_type_name": "H100:1x80",
                                "autoscaling_settings": {"min_replica": 1, "max_replica": 100},
                            }
                        ],
                    },
                )
            ]
        )
        deployment = await BasetenChainClient(_config(), transport=transport).get_chain_deployment(
            "chain_123", "deploy_456"
        )
        worker = deployment.chainlets[0]
        self.assertEqual(worker.active_replica_count, 3)
        self.assertEqual(worker.configured_max_replica, 100)
        self.assertIsNone(worker.desired_replica_count)
        self.assertIsNone(worker.starting_replica_count)

    async def test_prices_have_no_currency_or_hourly_rate_until_verified(self) -> None:
        payload = {
            "instance_types": [
                {"instance_type": {"id": "h100", "name": "H100:1x80"}, "price": 2.5}
            ]
        }
        unverified = await BasetenChainClient(
            _config(), transport=RecordedTransport([_response(200, payload)])
        ).get_instance_type_prices()
        self.assertEqual(unverified.prices[0].raw_price, Decimal("2.5"))
        self.assertIsNone(unverified.prices[0].currency)
        self.assertIsNone(unverified.prices[0].rate_per_hour)

        verified = await BasetenChainClient(
            _config(), transport=RecordedTransport([_response(200, payload)])
        ).get_instance_type_prices(
            VerifiedPriceBasis(
                currency="CAD",
                unit=PriceUnit.PER_MINUTE,
                evidence_uri="https://artifacts.example.test/contract-snapshot",
                retrieved_at="2026-09-19T00:00:00Z",
            )
        )
        self.assertEqual(verified.prices[0].currency, "CAD")
        self.assertEqual(verified.prices[0].rate_per_hour, Decimal("150.0"))

    async def test_pricing_rejects_unknown_price_unit_and_nonfinite_value(self) -> None:
        with self.assertRaises(PlatformConfigurationError):
            VerifiedPriceBasis(
                currency="usd",
                unit=PriceUnit.PER_HOUR,
                evidence_uri="https://artifacts.example.test/contract-snapshot",
                retrieved_at="not-a-timestamp",
            )
        client = BasetenChainClient(
            _config(),
            transport=RecordedTransport(
                [
                    _response(
                        200,
                        {
                            "instance_types": [
                                {"instance_type": {"id": "h100", "name": "H100:1x80"}, "price": "NaN"}
                            ]
                        },
                    )
                ]
            ),
        )
        with self.assertRaises(PlatformSchemaError):
            await client.get_instance_type_prices()


class CallbackTests(unittest.TestCase):
    def test_callback_uses_raw_body_hmac_and_request_association(self) -> None:
        body = json.dumps(
            {
                "request_id": "request-3",
                "type": "async_request_completed",
                "data": {"status": "completed"},
                "errors": [],
                "deployment_id": "deploy_456",
            },
            separators=(",", ":"),
        ).encode("utf-8")
        signature = hmac.new(b"webhook-test-secret", body, hashlib.sha256).hexdigest()
        callback = BasetenChainClient(_config()).authenticate_callback(
            body,
            {
                "X-BASETEN-SIGNATURE": "v1={0}".format(signature),
                "X-BASETEN-REQUEST-ID": "request-3",
            },
            association=CallbackAssociation("request-3", "run-3", "episode-3"),
        )
        self.assertEqual(callback.association.run_id, "run-3")
        self.assertEqual(callback.data, {"status": "completed"})

    def test_callback_rejects_signature_or_request_id_mismatch(self) -> None:
        body = b'{"request_id":"request-4","type":"async_request_completed","errors":[]}'
        with self.assertRaises(CallbackVerificationError):
            BasetenChainClient(_config()).authenticate_callback(
                body,
                {
                    "X-BASETEN-SIGNATURE": "v1=" + ("0" * 64),
                    "X-BASETEN-REQUEST-ID": "request-4",
                },
            )

        signature = hmac.new(b"webhook-test-secret", body, hashlib.sha256).hexdigest()
        with self.assertRaises(CallbackVerificationError):
            BasetenChainClient(_config()).authenticate_callback(
                body,
                {
                    "X-BASETEN-SIGNATURE": "v1={0}".format(signature),
                    "X-BASETEN-REQUEST-ID": "another-request",
                },
            )


class DeploymentTemplateTests(unittest.TestCase):
    def test_template_imports_without_remote_chains_sdk(self) -> None:
        template = importlib.import_module("deploy.baseten.chain")
        request = template.RolloutRequest(
            run_id="run-4",
            episode_id="episode-4",
            protocol_hash="sha256:protocol",
            policy={"episode_id": "episode-4", "protocol_hash": "sha256:protocol"},
            world={"episode_id": "episode-4", "protocol_hash": "sha256:protocol"},
            validity={"episode_id": "episode-4", "protocol_hash": "sha256:protocol"},
            judge={"episode_id": "episode-4", "protocol_hash": "sha256:protocol"},
        )
        self.assertTrue(request.matching_episode_requests())
        if not template.CHAINS_RUNTIME_AVAILABLE:
            with self.assertRaises(template.ChainRuntimeUnavailable):
                template.require_chains_runtime()


if __name__ == "__main__":
    unittest.main()
