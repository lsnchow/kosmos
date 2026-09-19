"""Narrow, fail-closed Baseten integration primitives for PLUMB.

This module deliberately owns transport and platform observations, not the run
ledger.  The application must persist each logical episode and each submission
attempt before it calls these methods.  In particular, a retry is never a new
statistical episode.

The asynchronous Chain contract is intentionally kept separate from the
model-only ``/async_predict`` API.  The official Chain endpoint is
``.../async_run_remote`` and accepts the entrypoint arguments under the async
inference ``model_input`` envelope.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


JsonValue = Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]]


class PlatformError(RuntimeError):
    """Base class for platform integration failures safe to surface to callers."""


class PlatformConfigurationError(PlatformError):
    """The local configuration cannot express a supported Baseten request."""


class PlatformTransportError(PlatformError):
    """No usable HTTP response was received."""


class PlatformHTTPError(PlatformError):
    """Baseten returned a non-success HTTP status."""

    def __init__(self, status_code: int, message: Optional[str] = None) -> None:
        self.status_code = status_code
        super().__init__(message or "Baseten request failed with HTTP {0}".format(status_code))


class PlatformSchemaError(PlatformError):
    """A response or webhook did not meet the documented minimum schema."""


class CallbackVerificationError(PlatformError):
    """A callback was not authenticated or was not associated with its request."""


class AmbiguousSubmissionError(PlatformError):
    """A POST may have been accepted although no durable request ID was recorded.

    The control plane must reconcile its durable outbox/ledger before any later
    submission.  Retrying this HTTP POST in a client can create duplicate world
    work and cost, so this exception intentionally contains the one attempted
    submission rather than causing another POST.
    """

    def __init__(self, attempts: Sequence["SubmissionAttempt"], cause: PlatformError) -> None:
        self.attempts = tuple(attempts)
        self.cause = cause
        super().__init__("Async Chain submission outcome is ambiguous; reconcile before resubmitting")


# Kept as a source-compatible name for callers which imported the old error.
# Automatic submission retries have been removed; all ambiguous failures use
# AmbiguousSubmissionError instead.
SubmissionRetryExhausted = AmbiguousSubmissionError


@dataclass(frozen=True)
class HttpRequest:
    """A small transport-neutral HTTP request used by the async client."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: Optional[bytes] = None


@dataclass(frozen=True)
class HttpResponse:
    """A small transport-neutral HTTP response used by the async client."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


class AsyncHttpTransport(Protocol):
    """Injectable async HTTP transport; tests use this instead of the network."""

    async def request(self, request: HttpRequest) -> HttpResponse:
        """Send *request* and return an HTTP response, or raise PlatformTransportError."""


class UrllibAsyncTransport:
    """A dependency-free async wrapper around :mod:`urllib.request`."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise PlatformConfigurationError("timeout_seconds must be a finite positive value")
        self._timeout_seconds = timeout_seconds

    async def request(self, request: HttpRequest) -> HttpResponse:
        return await asyncio.to_thread(self._request_blocking, request)

    def _request_blocking(self, request: HttpRequest) -> HttpResponse:
        urllib_request = Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urlopen(urllib_request, timeout=self._timeout_seconds) as response:  # nosec B310: caller validates HTTPS URLs
                return HttpResponse(
                    status_code=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=response.read(),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=int(error.code),
                headers={key.lower(): value for key, value in error.headers.items()} if error.headers else {},
                body=error.read(),
            )
        except URLError as error:
            raise PlatformTransportError("Unable to reach Baseten") from error


@dataclass(frozen=True)
class ChainAsyncEndpoint:
    """Validated deployed-Chain URL, restricted to the documented async route."""

    url: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise PlatformConfigurationError("A Chain async endpoint must be an absolute HTTPS URL")
        hostname = parsed.hostname.lower()
        if not hostname.startswith("chain-") or not hostname.endswith(".api.baseten.co"):
            raise PlatformConfigurationError("A Chain async endpoint must use a chain-*.api.baseten.co host")
        if parsed.port not in (None, 443):
            raise PlatformConfigurationError("A Chain async endpoint must not use a nonstandard port")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise PlatformConfigurationError("A Chain async endpoint may not contain credentials, query, or fragment")
        parts = [part for part in parsed.path.split("/") if part]
        valid_route = (
            len(parts) == 2 and parts[0] in {"production", "development"} and parts[1] == "async_run_remote"
        ) or (
            len(parts) == 3
            and parts[0] in {"deployment", "environments"}
            and _is_path_identifier(parts[1])
            and parts[2] == "async_run_remote"
        )
        if not valid_route:
            raise PlatformConfigurationError(
                "Chain async endpoint must end in a documented /async_run_remote route"
            )

    @classmethod
    def for_target(cls, chain_id: str, target: str) -> "ChainAsyncEndpoint":
        """Build a documented Chain endpoint from an explicit deployment target.

        ``target`` is deliberately explicit: ``production``, ``development``,
        ``deployment/<id>``, or ``environments/<name>``.  It is never inferred
        from a model deployment identifier.
        """

        _validate_path_identifier(chain_id, "chain_id")
        target_parts = [part for part in target.split("/") if part]
        valid_target = (
            len(target_parts) == 1 and target_parts[0] in {"production", "development"}
        ) or (
            len(target_parts) == 2
            and target_parts[0] in {"deployment", "environments"}
            and _is_path_identifier(target_parts[1])
        )
        if not valid_target:
            raise PlatformConfigurationError(
                "target must be production, development, deployment/<id>, or environments/<name>"
            )
        encoded_target = "/".join(quote(part, safe="") for part in target_parts)
        return cls("https://chain-{0}.api.baseten.co/{1}/async_run_remote".format(chain_id, encoded_target))


@dataclass(frozen=True)
class BasetenPlatformConfig:
    """Configuration loaded from explicit environment values without logging secrets."""

    api_key: str = field(repr=False)
    chain_endpoint: ChainAsyncEndpoint
    webhook_secret: Optional[str] = field(default=None, repr=False)
    management_base_url: str = "https://api.baseten.co/v1"

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise PlatformConfigurationError("BASETEN_API_KEY must be non-empty")
        if self.webhook_secret is not None and not self.webhook_secret:
            raise PlatformConfigurationError("BASETEN_WEBHOOK_SECRET must be non-empty when set")
        parsed = urlsplit(self.management_base_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.baseten.co"
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
            or parsed.path.rstrip("/") != "/v1"
        ):
            raise PlatformConfigurationError("management_base_url must be the HTTPS Baseten management API")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "BasetenPlatformConfig":
        values = os.environ if environ is None else environ
        api_key = values.get("BASETEN_API_KEY", "")
        endpoint = values.get("BASETEN_CHAIN_ASYNC_URL", "")
        missing = [
            name
            for name, value in (("BASETEN_API_KEY", api_key), ("BASETEN_CHAIN_ASYNC_URL", endpoint))
            if not value
        ]
        if missing:
            raise PlatformConfigurationError("Missing required environment variable(s): {0}".format(", ".join(missing)))
        return cls(
            api_key=api_key,
            chain_endpoint=ChainAsyncEndpoint(endpoint),
            webhook_secret=values.get("BASETEN_WEBHOOK_SECRET"),
            management_base_url=values.get("BASETEN_MANAGEMENT_BASE_URL", "https://api.baseten.co/v1"),
        )


@dataclass(frozen=True)
class AsyncChainRequestOptions:
    """Supported optional fields from Baseten's async-inference envelope."""

    webhook_endpoint: str
    priority: Optional[int] = None
    max_time_in_queue_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_webhook_endpoint(self.webhook_endpoint)
        if self.priority is not None and self.priority not in {0, 1, 2}:
            raise PlatformConfigurationError("priority must be 0, 1, or 2")
        if self.max_time_in_queue_seconds is not None and self.max_time_in_queue_seconds <= 0:
            raise PlatformConfigurationError("max_time_in_queue_seconds must be positive")

    def as_payload_fields(self) -> Dict[str, JsonValue]:
        fields: Dict[str, JsonValue] = {"webhook_endpoint": self.webhook_endpoint}
        if self.priority is not None:
            fields["priority"] = self.priority
        if self.max_time_in_queue_seconds is not None:
            fields["max_time_in_queue_seconds"] = self.max_time_in_queue_seconds
        return fields


@dataclass(frozen=True)
class SubmissionRetryPolicy:
    """Compatibility guard which explicitly disables client-side POST retries.

    A future durable server entrypoint compare-and-set may make a narrow retry
    policy safe. Until then no optional flag may turn one logical submission
    into blind repeated HTTP POSTs.
    """

    max_transport_retries: int = 0
    initial_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_transport_retries != 0:
            raise PlatformConfigurationError(
                "automatic async submission retries are disabled until a durable entrypoint CAS is deployed"
            )
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise PlatformConfigurationError("retry delays cannot be negative")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise PlatformConfigurationError("max_delay_seconds must be at least initial_delay_seconds")

    def delay_for_retry(self, retry_number: int) -> float:
        del retry_number
        return 0.0


@dataclass(frozen=True)
class SubmissionAttempt:
    ordinal: int
    state: str
    http_status: Optional[int] = None
    error_kind: Optional[str] = None


SubmissionAttemptObserver = Callable[[SubmissionAttempt], Union[None, Awaitable[None]]]


@dataclass(frozen=True)
class AsyncSubmissionReceipt:
    request_id: str
    response: Mapping[str, JsonValue]
    attempts: Tuple[SubmissionAttempt, ...]


@dataclass(frozen=True)
class VerifiedChainQueueRoute:
    """An account-tested Chain queue route, never inferred from model-only docs."""

    url: str
    evidence_uri: str
    verified_at: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.lower().startswith("chain-")
            or not parsed.hostname.lower().endswith(".api.baseten.co")
            or parsed.port not in (None, 443)
            or parsed.query
            or parsed.fragment
        ):
            raise PlatformConfigurationError("A verified queue route must be an absolute HTTPS URL without query")
        if not self.evidence_uri.strip() or not self.verified_at.strip():
            raise PlatformConfigurationError("A verified queue route requires evidence_uri and verified_at")
        _validate_timestamp(self.verified_at, "verified_at")


@dataclass(frozen=True)
class QueueStatus:
    availability: str
    reason: Optional[str]
    queued_requests: Optional[int]
    in_progress_requests: Optional[int]
    source_url: Optional[str]

    @classmethod
    def unavailable(cls, reason: str) -> "QueueStatus":
        return cls(
            availability="unavailable",
            reason=reason,
            queued_requests=None,
            in_progress_requests=None,
            source_url=None,
        )


@dataclass(frozen=True)
class ChainletReplicaStatus:
    name: str
    active_replica_count: Optional[int]
    instance_type_name: Optional[str]
    configured_min_replica: Optional[int]
    configured_max_replica: Optional[int]
    desired_replica_count: Optional[int] = None
    starting_replica_count: Optional[int] = None


@dataclass(frozen=True)
class ChainDeploymentStatus:
    chain_id: str
    deployment_id: str
    status: Optional[str]
    chainlets: Tuple[ChainletReplicaStatus, ...]
    raw: Mapping[str, JsonValue]


class PriceUnit(str, Enum):
    PER_HOUR = "per_hour"
    PER_MINUTE = "per_minute"
    PER_SECOND = "per_second"


@dataclass(frozen=True)
class VerifiedPriceBasis:
    """Price semantics supplied from the account contract/evidence, not guessed."""

    currency: str
    unit: PriceUnit
    evidence_uri: str
    retrieved_at: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise PlatformConfigurationError("currency must be an explicit three-letter uppercase code")
        if not self.evidence_uri.strip() or not self.retrieved_at.strip():
            raise PlatformConfigurationError("A price basis requires evidence_uri and retrieved_at")
        _validate_timestamp(self.retrieved_at, "retrieved_at")


@dataclass(frozen=True)
class InstanceTypePrice:
    instance_type_id: str
    instance_type_name: str
    raw_price: Decimal
    rate_per_hour: Optional[Decimal]
    currency: Optional[str]


@dataclass(frozen=True)
class InstanceTypePriceSnapshot:
    basis: Optional[VerifiedPriceBasis]
    prices: Tuple[InstanceTypePrice, ...]
    raw: Mapping[str, JsonValue]


@dataclass(frozen=True)
class CallbackAssociation:
    """The request association resolved from the durable application ledger."""

    request_id: str
    run_id: str
    episode_id: str

    def __post_init__(self) -> None:
        if not self.request_id or not self.run_id or not self.episode_id:
            raise PlatformConfigurationError("CallbackAssociation fields must be non-empty")


@dataclass(frozen=True)
class AuthenticatedCallback:
    request_id: str
    event_type: str
    data: JsonValue
    errors: Tuple[Mapping[str, JsonValue], ...]
    deployment_id: Optional[str]
    model_id: Optional[str]
    event_time: Optional[str]
    association: Optional[CallbackAssociation]
    raw: Mapping[str, JsonValue]


class BasetenChainClient:
    """Typed async client for the small, documented Baseten surface PLUMB uses."""

    def __init__(
        self,
        config: BasetenPlatformConfig,
        transport: Optional[AsyncHttpTransport] = None,
        queue_route: Optional[VerifiedChainQueueRoute] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._transport = transport if transport is not None else UrllibAsyncTransport()
        self._queue_route = queue_route
        self._sleep = sleep

    @property
    def chain_async_url(self) -> str:
        return self._config.chain_endpoint.url

    async def submit_async(
        self,
        entrypoint_input: Mapping[str, JsonValue],
        options: AsyncChainRequestOptions,
        retry_policy: Optional[SubmissionRetryPolicy] = None,
        attempt_observer: Optional[SubmissionAttemptObserver] = None,
    ) -> AsyncSubmissionReceipt:
        """Queue one Chain entrypoint invocation through ``/async_run_remote``.

        ``entrypoint_input`` is the typed argument object accepted by the
        controller's ``run_remote`` method.  It is deliberately placed in
        ``model_input`` because the async Chain API uses the async-inference
        envelope.  No internal Chainlet call reaches this external queue.
        """

        _ensure_json_compatible(entrypoint_input, "entrypoint_input")
        # Constructing a nonzero policy raises immediately. Keep the argument
        # for source compatibility, but never loop/re-POST an ambiguous job.
        if retry_policy is not None:
            _ = retry_policy
        payload: Dict[str, JsonValue] = {"model_input": dict(entrypoint_input)}
        payload.update(options.as_payload_fields())
        encoded = _encode_json(payload)
        attempts: List[SubmissionAttempt] = []
        started = SubmissionAttempt(ordinal=1, state="started")
        await _notify_attempt(attempt_observer, started)
        try:
            response = await self._transport.request(
                HttpRequest(
                    method="POST",
                    url=self._config.chain_endpoint.url,
                    headers=self._inference_headers(),
                    body=encoded,
                )
            )
        except PlatformTransportError as error:
            completed = SubmissionAttempt(ordinal=1, state="transport_error", error_kind=type(error).__name__)
            attempts.append(completed)
            await _notify_attempt(attempt_observer, completed)
            raise AmbiguousSubmissionError(attempts, error) from error

        if not 200 <= response.status_code < 300:
            error = PlatformHTTPError(response.status_code)
            completed = SubmissionAttempt(
                ordinal=1,
                state="http_error",
                http_status=response.status_code,
                error_kind=type(error).__name__,
            )
            attempts.append(completed)
            await _notify_attempt(attempt_observer, completed)
            # A 429 is a known rejection, but still receives no automatic
            # retry. A transport timeout, 408, or 5xx may have run the Chain
            # after accepting the body, so force ledger/outbox reconciliation.
            if _ambiguous_submission_status(response.status_code):
                raise AmbiguousSubmissionError(attempts, error) from error
            raise error

        try:
            response_object = _decode_json_object(response.body, "async Chain submission response")
            request_id = _required_string(response_object, "request_id", "async Chain submission response")
        except PlatformSchemaError as error:
            completed = SubmissionAttempt(
                ordinal=1,
                state="accepted_response_unreadable",
                http_status=response.status_code,
                error_kind=type(error).__name__,
            )
            attempts.append(completed)
            await _notify_attempt(attempt_observer, completed)
            raise AmbiguousSubmissionError(attempts, error) from error
        completed = SubmissionAttempt(ordinal=1, state="accepted", http_status=response.status_code)
        attempts.append(completed)
        await _notify_attempt(attempt_observer, completed)
        return AsyncSubmissionReceipt(
            request_id=request_id,
            response=response_object,
            attempts=tuple(attempts),
        )

    async def get_queue_status(self) -> QueueStatus:
        """Return queue depth only when an account-tested Chain route was supplied."""

        if self._queue_route is None:
            return QueueStatus.unavailable(
                "No verified Chain async_queue_status route is configured; model-only routes are not assumed"
            )
        response = await self._transport.request(
            HttpRequest(
                method="GET",
                url=self._queue_route.url,
                headers=self._inference_headers(),
            )
        )
        _require_success(response)
        payload = _decode_json_object(response.body, "Chain queue-status response")
        return QueueStatus(
            availability="available",
            reason=None,
            queued_requests=_required_nonnegative_int(payload, "num_queued_requests", "Chain queue-status response"),
            in_progress_requests=_required_nonnegative_int(
                payload, "num_in_progress_requests", "Chain queue-status response"
            ),
            source_url=self._queue_route.url,
        )

    async def get_chain_deployment(
        self, chain_id: str, deployment_id: str
    ) -> ChainDeploymentStatus:
        """Fetch documented active replica counts from Chain deployment management."""

        _validate_path_identifier(chain_id, "chain_id")
        _validate_path_identifier(deployment_id, "deployment_id")
        url = "{0}/chains/{1}/deployments/{2}".format(
            self._config.management_base_url.rstrip("/"),
            quote(chain_id, safe=""),
            quote(deployment_id, safe=""),
        )
        response = await self._transport.request(
            HttpRequest(method="GET", url=url, headers=self._management_headers())
        )
        _require_success(response)
        payload = _decode_json_object(response.body, "Chain deployment response")
        raw_chainlets = payload.get("chainlets")
        if not isinstance(raw_chainlets, list):
            raise PlatformSchemaError("Chain deployment response.chainlets must be a list")
        chainlets: List[ChainletReplicaStatus] = []
        for raw_chainlet in raw_chainlets:
            if not isinstance(raw_chainlet, dict):
                raise PlatformSchemaError("Chain deployment response.chainlets entries must be objects")
            autoscaling = raw_chainlet.get("autoscaling_settings")
            if autoscaling is not None and not isinstance(autoscaling, dict):
                raise PlatformSchemaError("chainlet autoscaling_settings must be an object or null")
            autoscaling_object = autoscaling if isinstance(autoscaling, dict) else {}
            chainlets.append(
                ChainletReplicaStatus(
                    name=_required_string(raw_chainlet, "name", "chainlet"),
                    active_replica_count=_optional_nonnegative_int(raw_chainlet, "active_replica_count", "chainlet"),
                    instance_type_name=_optional_string(raw_chainlet, "instance_type_name", "chainlet"),
                    configured_min_replica=_optional_nonnegative_int(
                        autoscaling_object, "min_replica", "chainlet.autoscaling_settings"
                    ),
                    configured_max_replica=_optional_nonnegative_int(
                        autoscaling_object, "max_replica", "chainlet.autoscaling_settings"
                    ),
                )
            )
        return ChainDeploymentStatus(
            chain_id=_required_string(payload, "chain_id", "Chain deployment response"),
            deployment_id=_required_string(payload, "id", "Chain deployment response"),
            status=_optional_string(payload, "status", "Chain deployment response"),
            chainlets=tuple(chainlets),
            raw=payload,
        )

    async def get_instance_type_prices(
        self, basis: Optional[VerifiedPriceBasis] = None
    ) -> InstanceTypePriceSnapshot:
        """Fetch prices without inventing their currency or billing unit.

        Baseten's response currently exposes an unqualified numeric ``price``.
        Passing a verified account contract/evidence basis is required before
        ``rate_per_hour`` and ``currency`` are populated.  In particular, this
        function never assumes USD.
        """

        url = "{0}/instance_type_prices".format(self._config.management_base_url.rstrip("/"))
        response = await self._transport.request(
            HttpRequest(method="GET", url=url, headers=self._management_headers())
        )
        _require_success(response)
        payload = _decode_json_object(response.body, "instance type prices response")
        raw_prices = payload.get("instance_types")
        if not isinstance(raw_prices, list):
            raise PlatformSchemaError("instance type prices response.instance_types must be a list")
        prices: List[InstanceTypePrice] = []
        for entry in raw_prices:
            if not isinstance(entry, dict) or not isinstance(entry.get("instance_type"), dict):
                raise PlatformSchemaError("instance type price entry must contain instance_type object")
            instance_type = entry["instance_type"]
            raw_price = _nonnegative_decimal(entry.get("price"), "instance type price")
            if basis is None:
                rate_per_hour = None
                currency = None
            else:
                rate_per_hour = _to_hourly_rate(raw_price, basis.unit)
                currency = basis.currency
            prices.append(
                InstanceTypePrice(
                    instance_type_id=_required_string(instance_type, "id", "instance type"),
                    instance_type_name=_required_string(instance_type, "name", "instance type"),
                    raw_price=raw_price,
                    rate_per_hour=rate_per_hour,
                    currency=currency,
                )
            )
        return InstanceTypePriceSnapshot(basis=basis, prices=tuple(prices), raw=payload)

    def authenticate_callback(
        self,
        body: bytes,
        headers: Mapping[str, str],
        association: Optional[CallbackAssociation] = None,
    ) -> AuthenticatedCallback:
        """Verify a raw Baseten webhook body and bind it to the persisted request.

        Baseten documents ``X-BASETEN-SIGNATURE: v1=<hmac>`` over the *raw*
        request body and ``X-BASETEN-REQUEST-ID``.  A valid signature alone is
        insufficient: the header and JSON request IDs must agree, and callers
        should pass the association loaded from their durable ledger.
        """

        if self._config.webhook_secret is None:
            raise CallbackVerificationError("Webhook verification is unavailable: no webhook secret is configured")
        signature = _header_value(headers, "x-baseten-signature")
        if signature is None or not verify_baseten_webhook_signature(body, signature, self._config.webhook_secret):
            raise CallbackVerificationError("Invalid or missing Baseten webhook signature")
        payload = _decode_json_object(body, "Baseten webhook payload")
        request_id = _required_string(payload, "request_id", "Baseten webhook payload")
        header_request_id = _header_value(headers, "x-baseten-request-id")
        if not header_request_id or not hmac.compare_digest(header_request_id, request_id):
            raise CallbackVerificationError("Baseten webhook request ID header does not match payload")
        if association is not None and not hmac.compare_digest(association.request_id, request_id):
            raise CallbackVerificationError("Baseten webhook request ID does not match the ledger association")
        errors = _parse_errors(payload)
        return AuthenticatedCallback(
            request_id=request_id,
            event_type=_required_string(payload, "type", "Baseten webhook payload"),
            data=payload.get("data"),
            errors=errors,
            deployment_id=_optional_string(payload, "deployment_id", "Baseten webhook payload"),
            model_id=_optional_string(payload, "model_id", "Baseten webhook payload"),
            event_time=_optional_string(payload, "time", "Baseten webhook payload"),
            association=association,
            raw=payload,
        )

    def _inference_headers(self) -> Dict[str, str]:
        return {
            "Authorization": "Bearer {0}".format(self._config.api_key),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _management_headers(self) -> Dict[str, str]:
        return {"Authorization": "Bearer {0}".format(self._config.api_key), "Accept": "application/json"}


def verify_baseten_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    """Return whether ``signature`` has a valid documented ``v1=`` HMAC value."""

    if not isinstance(body, bytes) or not isinstance(signature, str) or not secret:
        return False
    actual: Optional[str] = None
    for part in signature.split(","):
        candidate = part.strip()
        if candidate.startswith("v1="):
            actual = candidate[3:]
            break
    if actual is None or not re.fullmatch(r"[0-9a-fA-F]{64}", actual):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, actual.lower())


def _validate_webhook_endpoint(value: str) -> None:
    parsed = urlsplit(value)
    is_localhost = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise PlatformConfigurationError("webhook_endpoint must be an absolute URL without credentials, query, or fragment")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_localhost):
        raise PlatformConfigurationError("webhook_endpoint must use HTTPS (except localhost during development)")


def _validate_path_identifier(value: str, field_name: str) -> None:
    if not _is_path_identifier(value):
        raise PlatformConfigurationError("{0} must be a non-empty path-safe identifier".format(field_name))


def _is_path_identifier(value: str) -> bool:
    return value not in {".", ".."} and bool(re.fullmatch(r"[A-Za-z0-9._-]+", value))


def _validate_timestamp(value: str, field_name: str) -> None:
    """Require an offset-bearing ISO-8601 timestamp for evidence freshness."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PlatformConfigurationError("{0} must be an ISO-8601 timestamp".format(field_name)) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PlatformConfigurationError("{0} must include a UTC offset".format(field_name))


def _ensure_json_compatible(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise PlatformConfigurationError("{0} must be JSON-compatible with finite numbers".format(field_name)) from error


def _encode_json(value: Mapping[str, JsonValue]) -> bytes:
    _ensure_json_compatible(value, "request payload")
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _decode_json_object(body: bytes, context: str) -> Dict[str, JsonValue]:
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlatformSchemaError("{0} is not a UTF-8 JSON object".format(context)) from error
    if not isinstance(decoded, dict):
        raise PlatformSchemaError("{0} must be a JSON object".format(context))
    return decoded


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PlatformSchemaError("{0}.{1} must be a non-empty string".format(context, key))
    return value


def _optional_string(payload: Mapping[str, Any], key: str, context: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PlatformSchemaError("{0}.{1} must be a non-empty string or null".format(context, key))
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlatformSchemaError("{0}.{1} must be a non-negative integer".format(context, key))
    return value


def _optional_nonnegative_int(payload: Mapping[str, Any], key: str, context: str) -> Optional[int]:
    if key not in payload or payload[key] is None:
        return None
    return _required_nonnegative_int(payload, key, context)


def _nonnegative_decimal(value: Any, context: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise PlatformSchemaError("{0} must be a finite non-negative number".format(context))
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise PlatformSchemaError("{0} must be a finite non-negative number".format(context)) from error
    if not decimal_value.is_finite() or decimal_value < 0:
        raise PlatformSchemaError("{0} must be a finite non-negative number".format(context))
    return decimal_value


def _to_hourly_rate(raw_price: Decimal, unit: PriceUnit) -> Decimal:
    if unit == PriceUnit.PER_HOUR:
        return raw_price
    if unit == PriceUnit.PER_MINUTE:
        return raw_price * Decimal(60)
    if unit == PriceUnit.PER_SECOND:
        return raw_price * Decimal(3600)
    raise PlatformConfigurationError("Unsupported verified price unit")


def _parse_errors(payload: Mapping[str, JsonValue]) -> Tuple[Mapping[str, JsonValue], ...]:
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list) or not all(isinstance(error, dict) for error in raw_errors):
        raise PlatformSchemaError("Baseten webhook payload.errors must be a list of objects")
    return tuple(raw_errors)


def _header_value(headers: Mapping[str, str], expected_name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == expected_name:
            return value
    return None


def _ambiguous_submission_status(status_code: int) -> bool:
    return status_code == 408 or 500 <= status_code <= 599


async def _notify_attempt(observer: Optional[SubmissionAttemptObserver], attempt: SubmissionAttempt) -> None:
    if observer is None:
        return
    outcome = observer(attempt)
    if hasattr(outcome, "__await__"):
        await outcome  # type: ignore[misc]


def _require_success(response: HttpResponse) -> None:
    if not 200 <= response.status_code < 300:
        raise PlatformHTTPError(response.status_code)
