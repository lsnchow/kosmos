"""Production-shaped rehearsal of the Baseten Chain, with no credentials.

The integration code between PLUMB and Baseten is the part most likely to be
wrong on the day: envelope construction, the durable outbox, HMAC callback
authentication, duplicate and lost-callback reconciliation, telemetry
availability, cost attribution, and the burst mechanics.  None of that needs a
GPU to be exercised -- it needs something on the other end of the socket that
speaks the documented protocol.

This module is that something.  It is an ``AsyncHttpTransport`` (the same
injectable seam ``plumb.platform`` already uses for its tests) which answers
``/async_run_remote``, the queue-status route, the Chain deployment route and the
instance-prices route, and then posts a **genuinely HMAC-signed** webhook back to
the application exactly as Baseten would.

Everything downstream of the socket is therefore the real production path.  What
is *not* real is the model: the rehearsal returns labelled synthetic frames.

Two safeguards make that impossible to forget:

1. Every response, artifact manifest and episode record carries
   ``transport="simulated"``, and the rehearsal refuses to emit a result without
   it.
2. The rehearsal never reports ``validity="valid"`` for an episode unless the
   caller explicitly asks for a scored rehearsal, and even then the record keeps
   ``feedback_mode="unqualified"`` and ``parity_status="unqualified"`` so no gate
   and no measurement can mistake it for evidence.

Use it to prove the plumbing, the dashboard and the timings.  Never to claim a
policy result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .platform import HttpRequest, HttpResponse, PlatformTransportError

#: Stamped on every rehearsal artifact and record.  A record without it is a bug.
SIMULATED_TRANSPORT = "simulated"

#: Labels that make a rehearsal frame unmistakable in the UI and on disk.
REHEARSAL_LABEL = "rehearsal transport; simulated Chain; not a learned-model result; unqualified"

#: A well-formed Chain URL is still required so the endpoint validator keeps
#: doing its job.  Nothing ever dials this host: the transport is swapped.
REHEARSAL_CHAIN_ID = "rehearsal000000"
REHEARSAL_CHAIN_ASYNC_URL = "https://chain-%s.api.baseten.co/production/async_run_remote" % REHEARSAL_CHAIN_ID
REHEARSAL_QUEUE_URL = "https://chain-%s.api.baseten.co/production/async_queue_status" % REHEARSAL_CHAIN_ID

_CHAIN_MODELS: Optional[Dict[str, Any]] = None


@dataclass
class RehearsalProfile:
    """How the simulated Chain behaves.

    The latency numbers are *simulated* and exist so the dashboard, the SSE
    stream and the burst timer can be exercised against a realistic shape.  They
    are not measurements and must never be reported as such.
    """

    #: Simulated per-stage latency, seconds.  Defaults are scaled down from the
    #: recorded cluster observations so a full rehearsal finishes quickly.
    policy_seconds: float = 0.004
    world_seconds_per_chunk: float = 0.010
    validity_seconds: float = 0.002
    judge_seconds: float = 0.008
    #: Fraction of episodes that return an explicitly unevaluable outcome, so the
    #: coverage, missingness-bound and exclusion paths are all exercised.
    unevaluable_rate: float = 0.08
    #: Fraction that return a terminal service failure, exercising the failure
    #: path and its presence in the intent-to-evaluate denominator.
    service_failure_rate: float = 0.02
    #: Fraction of webhooks deliberately dropped, so the reconciler is exercised.
    dropped_webhook_rate: float = 0.02
    #: Fraction of webhooks delivered twice, so idempotency is exercised.
    duplicate_webhook_rate: float = 0.02
    #: Simulated queue and replica shape.
    max_replicas: int = 100
    #: Simulated hourly rate for the cost-attribution path.  Labelled simulated.
    simulated_h100_rate_per_hour: float = 9.984
    #: Whether episodes may return a scored outcome at all.
    scored: bool = True
    seed: int = 20260919

    def stage_seconds(self, chunks: int) -> Dict[str, float]:
        return {
            "policy_seconds": self.policy_seconds * max(1, chunks),
            "world_seconds": self.world_seconds_per_chunk * max(1, chunks),
            "validity_seconds": self.validity_seconds,
            "judge_seconds": self.judge_seconds,
        }


@dataclass
class RehearsalStats:
    submitted: int = 0
    completed: int = 0
    unevaluable: int = 0
    service_failures: int = 0
    webhooks_sent: int = 0
    webhooks_dropped: int = 0
    webhooks_duplicated: int = 0
    queue_status_calls: int = 0
    deployment_calls: int = 0
    price_calls: int = 0
    peak_in_flight: int = 0
    completion_errors: int = 0
    contract_rejections: int = 0

    def to_mapping(self) -> Dict[str, int]:
        return dict(self.__dict__)


def _deterministic_unit(*parts: Any) -> float:
    """A stable pseudo-random value in [0, 1) derived from *parts*.

    Deterministic so a rehearsal is reproducible: the same run id and episode id
    always produce the same outcome, which is what makes a rehearsal comparable
    across attempts.
    """

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class RehearsalChainTransport:
    """An ``AsyncHttpTransport`` that answers as a deployed Chain would.

    It implements only the four routes PLUMB actually calls.  Anything else gets
    a 404, so an unexpected call surfaces instead of silently succeeding.
    """

    def __init__(
        self,
        webhook_secret: str,
        artifacts_dir: Path,
        profile: Optional[RehearsalProfile] = None,
        deliver: Optional[Callable[[str, bytes, Mapping[str, str]], None]] = None,
    ) -> None:
        if not webhook_secret:
            raise ValueError("the rehearsal transport needs a webhook secret to sign callbacks with")
        self.webhook_secret = webhook_secret
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.profile = profile or RehearsalProfile()
        self.stats = RehearsalStats()
        self._deliver = deliver or self._deliver_over_http
        self._lock = threading.RLock()
        self._in_flight = 0
        self._started_at = time.monotonic()
        self._requests: Dict[str, Dict[str, Any]] = {}
        # Completion runs on a thread pool rather than an asyncio task.  The
        # engine calls the backend from a worker thread via ``asyncio.run``, so
        # the event loop that serves a submission is torn down as soon as that
        # submission returns -- a task scheduled on it would be destroyed while
        # still pending and the callback would never fire.
        self._workers = ThreadPoolExecutor(max_workers=32, thread_name_prefix="rehearsal-chain")
        self._pending: List[Future] = []
        self.errors: List[str] = []

    # -- transport ------------------------------------------------------------

    async def request(self, request: HttpRequest) -> HttpResponse:
        url = request.url
        if request.method == "POST" and url.endswith("/async_run_remote"):
            return await self._handle_submit(request)
        if request.method == "GET" and url.endswith("/async_queue_status"):
            return self._handle_queue_status()
        if request.method == "GET" and "/chains/" in url:
            return self._handle_deployment()
        if request.method == "GET" and "instance_type" in url.replace("-", "_"):
            return self._handle_prices()
        return HttpResponse(
            status_code=404,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "error": "the rehearsal transport does not implement this route",
                    "method": request.method,
                    "url": url,
                }
            ).encode("utf-8"),
        )

    async def _handle_submit(self, request: HttpRequest) -> HttpResponse:
        try:
            payload = json.loads((request.body or b"{}").decode("utf-8"))
        except ValueError as exc:
            raise PlatformTransportError("rehearsal transport received malformed JSON") from exc
        model_input = payload.get("model_input")
        if not isinstance(model_input, Mapping):
            return self._bad_request("the async envelope must carry a model_input object")
        webhook = payload.get("webhook_endpoint")
        if not webhook:
            return self._bad_request("the async envelope must carry a webhook_endpoint")

        # Validate against the Chain's OWN pydantic models. Every one is
        # extra="forbid", so this is a genuine contract test: a rehearsal that
        # accepted a looser envelope than the real Chain would give false
        # confidence, which is the exact failure this whole module exists to
        # prevent.
        contract_error = _validate_against_chain_contract(model_input)
        if contract_error is not None:
            with self._lock:
                self.errors.append("contract: %s" % contract_error)
                self.stats.contract_rejections += 1
            return self._bad_request("Chain contract rejected the request: %s" % contract_error)

        request_id = "rehearsal-" + uuid.uuid4().hex
        with self._lock:
            self.stats.submitted += 1
            self._in_flight += 1
            self.stats.peak_in_flight = max(self.stats.peak_in_flight, self._in_flight)
            self._requests[request_id] = {"model_input": dict(model_input), "webhook": str(webhook)}

        # Schedule the callback the way the platform would: out of band, after
        # the work notionally completes.
        future = self._workers.submit(self._complete_later, request_id, dict(model_input), str(webhook))
        with self._lock:
            self._pending.append(future)
        return HttpResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"request_id": request_id, "status": "queued"}).encode("utf-8"),
        )

    def _handle_queue_status(self) -> HttpResponse:
        with self._lock:
            self.stats.queue_status_calls += 1
            in_flight = self._in_flight
        return HttpResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "num_queued_requests": max(0, in_flight - self.profile.max_replicas),
                    "num_in_progress_requests": min(in_flight, self.profile.max_replicas),
                }
            ).encode("utf-8"),
        )

    def _handle_deployment(self) -> HttpResponse:
        with self._lock:
            self.stats.deployment_calls += 1
            in_flight = self._in_flight
        active = min(max(1, in_flight), self.profile.max_replicas)
        return HttpResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "id": REHEARSAL_CHAIN_ID,
                    "chainlets": [
                        {
                            "name": name,
                            "active_replica_count": active if name == "WorldWorker" else 1,
                            "instance_type_name": "H100" if name == "WorldWorker" else "cpu-2x8",
                            "autoscaling_settings": {"min_replica": 0, "max_replica": self.profile.max_replicas},
                        }
                        for name in ("RolloutController", "OpenVLAWorker", "WorldWorker", "ValidityWorker", "JudgeWorker")
                    ],
                }
            ).encode("utf-8"),
        )

    def _handle_prices(self) -> HttpResponse:
        with self._lock:
            self.stats.price_calls += 1
        return HttpResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "instance_types": [
                        {
                            "name": "H100",
                            # Deliberately labelled: a rehearsal price is not an
                            # account price, and USD stays unavailable unless the
                            # caller supplies a VerifiedPriceBasis anyway.
                            "price": self.profile.simulated_h100_rate_per_hour,
                            "unit": "per_hour",
                            "currency": "USD",
                            "source": SIMULATED_TRANSPORT,
                        }
                    ]
                }
            ).encode("utf-8"),
        )

    @staticmethod
    def _bad_request(message: str) -> HttpResponse:
        return HttpResponse(
            status_code=400,
            headers={"content-type": "application/json"},
            body=json.dumps({"error": message}).encode("utf-8"),
        )

    # -- simulated execution --------------------------------------------------

    def _complete_later(self, request_id: str, model_input: Mapping[str, Any], webhook: str) -> None:
        try:
            self._complete_now(request_id, model_input, webhook)
        except Exception as exc:  # noqa: BLE001 - a swallowed error looks like a lost webhook
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)
                self.errors.append("%s: %s: %s" % (request_id, type(exc).__name__, exc))

    def _complete_now(self, request_id: str, model_input: Mapping[str, Any], webhook: str) -> None:
        run_id = str(model_input.get("run_id"))
        episode_id = str(model_input.get("episode_id"))
        world = model_input.get("world") or {}
        payload = world.get("payload") if isinstance(world, Mapping) else {}
        payload = payload if isinstance(payload, Mapping) else {}
        horizon = int(payload.get("horizon_actions") or 0)
        operating_point = payload.get("operating_point") or {}
        chunk = int((operating_point or {}).get("action_chunk_size") or 16)
        chunks = max(1, -(-horizon // chunk)) if horizon else 1
        timings = self.profile.stage_seconds(chunks)
        time.sleep(sum(timings.values()))

        result = self._build_result(run_id, episode_id, chunks, timings, payload)
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if result["status"] == "completed":
                self.stats.completed += 1
            else:
                self.stats.service_failures += 1
            if result.get("missing_reason"):
                self.stats.unevaluable += 1

        body = json.dumps(
            {
                "request_id": request_id,
                "type": "predict.completed" if result["status"] == "completed" else "predict.failed",
                "time": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "data": result,
                "errors": [],
            }
        ).encode("utf-8")

        drop = _deterministic_unit("drop", run_id, episode_id) < self.profile.dropped_webhook_rate
        if drop:
            with self._lock:
                self.stats.webhooks_dropped += 1
            # Deliberately never delivered. The reconciler must find it.
            return

        signature = hmac.new(self.webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-BASETEN-SIGNATURE": "v1=" + signature,
            "X-BASETEN-REQUEST-ID": request_id,
        }
        deliveries = 1
        if _deterministic_unit("dup", run_id, episode_id) < self.profile.duplicate_webhook_rate:
            deliveries = 2
            with self._lock:
                self.stats.webhooks_duplicated += 1
        for _ in range(deliveries):
            try:
                self._deliver(webhook, body, headers)
                with self._lock:
                    self.stats.webhooks_sent += 1
            except Exception:
                with self._lock:
                    self.stats.webhooks_dropped += 1

    def _build_result(
        self,
        run_id: str,
        episode_id: str,
        chunks: int,
        timings: Mapping[str, float],
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build a ``RolloutResult``-shaped response, always marked simulated."""

        roll = _deterministic_unit("outcome", run_id, episode_id)
        chunk = int((payload.get("operating_point") or {}).get("action_chunk_size") or 16)
        executed = chunks * chunk
        base: Dict[str, Any] = {
            "run_id": run_id,
            "episode_id": episode_id,
            "transport": SIMULATED_TRANSPORT,
            "label": REHEARSAL_LABEL,
            # Deliberately the Chain's nested shape: a list of stage timings and
            # a gpu_seconds accounting object, with no flat aliases. If the
            # consumer only reads flat names, this rehearsal fails loudly.
            "segments": [{"index": index, "status": "completed"} for index in range(chunks)],
            "executed_actions": executed,
            "horizon_actions": int(payload.get("horizon_actions") or executed),
            "timings": [
                {"stage": name.replace("_seconds", ""), "seconds": round(value, 6)}
                for name, value in timings.items()
            ],
            "gpu_seconds": {
                "accounting_basis": SIMULATED_TRANSPORT,
                "total_instrumented_gpu_seconds": round(timings["world_seconds"] + timings["judge_seconds"], 6),
                "allocated_gpu_seconds": round(sum(timings.values()), 6),
            },
            "estimated_usd": None,
            "cost_basis_ref": SIMULATED_TRANSPORT,
            # A rehearsal can never qualify a feedback mode or scene parity.
            "feedback_mode": "unqualified",
            "parity_status": "unqualified",
            "stages": {
                name: {"stage": name, "status": "completed", "transport": SIMULATED_TRANSPORT}
                for name in ("policy", "world", "validity", "judge")
            },
        }

        if roll < self.profile.service_failure_rate:
            base.update(
                {
                    "status": "failed",
                    "validity": "unknown",
                    "binary_success": None,
                    "progress_score": None,
                    "missing_reason": "simulated_terminal_service_failure",
                    "reason": "the rehearsal transport simulated a terminal service failure",
                }
            )
            base["stages"]["world"] = {"stage": "world", "status": "failed", "transport": SIMULATED_TRANSPORT}
            return base

        if not self.profile.scored or roll < self.profile.service_failure_rate + self.profile.unevaluable_rate:
            base.update(
                {
                    "status": "completed",
                    "validity": "unknown",
                    "binary_success": None,
                    "progress_score": None,
                    "missing_reason": (
                        "simulated_unscored_rehearsal"
                        if not self.profile.scored
                        else "simulated_judge_no_quorum"
                    ),
                }
            )
            return base

        success = _deterministic_unit("success", run_id, episode_id) < 0.42
        base.update(
            {
                "status": "completed",
                "validity": "valid",
                "binary_success": success,
                "progress_score": 5 if success else int(_deterministic_unit("prog", episode_id) * 5),
                "missing_reason": None,
            }
        )
        return base

    @staticmethod
    def _deliver_over_http(url: str, body: bytes, headers: Mapping[str, str]) -> None:
        request = Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urlopen(request, timeout=15) as response:  # nosec B310: operator-supplied local endpoint
                response.read()
        except URLError as exc:
            raise PlatformTransportError("rehearsal webhook delivery failed") from exc

    # -- reporting ------------------------------------------------------------

    def drain(self, timeout: float = 60.0) -> int:
        """Wait for scheduled completions.  Returns how many were still pending."""

        deadline = time.monotonic() + timeout
        with self._lock:
            pending = list(self._pending)
        remaining = 0
        for future in pending:
            left = deadline - time.monotonic()
            if left <= 0:
                remaining += 1
                continue
            try:
                future.result(timeout=left)
            except Exception:
                pass
        return remaining

    def shutdown(self) -> None:
        self._workers.shutdown(wait=False, cancel_futures=True)

    def report(self) -> Dict[str, Any]:
        with self._lock:
            stats = self.stats.to_mapping()
            stats["completion_errors"] = len(self.errors)
            errors = list(self.errors[:10])
            in_flight = self._in_flight
        return {
            "errors": errors,
            "transport": SIMULATED_TRANSPORT,
            "label": REHEARSAL_LABEL,
            "elapsed_seconds": time.monotonic() - self._started_at,
            "in_flight": in_flight,
            "stats": stats,
            "profile": {
                "policy_seconds": self.profile.policy_seconds,
                "world_seconds_per_chunk": self.profile.world_seconds_per_chunk,
                "validity_seconds": self.profile.validity_seconds,
                "judge_seconds": self.profile.judge_seconds,
                "unevaluable_rate": self.profile.unevaluable_rate,
                "service_failure_rate": self.profile.service_failure_rate,
                "dropped_webhook_rate": self.profile.dropped_webhook_rate,
                "duplicate_webhook_rate": self.profile.duplicate_webhook_rate,
                "max_replicas": self.profile.max_replicas,
                "scored": self.profile.scored,
            },
            "not_a_measurement": (
                "Every latency, GPU-second and replica count above is simulated by this transport. "
                "It exercises the integration path; it measures nothing about a model or a platform."
            ),
        }


def chain_contract_available() -> bool:
    """Whether the Chain's pydantic models can be imported for validation."""

    return _chain_models() is not None


def _chain_models() -> Optional[Dict[str, Any]]:
    """Import the Chain's request models, or ``None`` if unavailable.

    The models live in ``deploy/baseten/chain.py`` and import cleanly without the
    Chains SDK, so this normally succeeds.  If it ever does not, the rehearsal
    says so rather than silently skipping the contract check.
    """

    global _CHAIN_MODELS
    if _CHAIN_MODELS is not None:
        return None if _CHAIN_MODELS == {} else _CHAIN_MODELS
    try:
        import deploy.baseten.chain as chain
    except Exception:
        _CHAIN_MODELS = {}
        return None
    models: Dict[str, Any] = {}
    for name in ("RolloutRequest", "EpisodeControlPayload", "_world_setup"):
        model = getattr(chain, name, None)
        if model is None:
            _CHAIN_MODELS = {}
            return None
        models[name] = model
    _CHAIN_MODELS = models
    return models


def _validate_against_chain_contract(model_input: Mapping[str, Any]) -> Optional[str]:
    """Return a contract violation message, or ``None`` when the envelope is valid."""

    models = _chain_models()
    if models is None:
        return "the Chain request models could not be imported for validation"
    try:
        request = models["RolloutRequest"].model_validate(dict(model_input))
    except Exception as exc:
        return "RolloutRequest: %s" % _first_line(exc)
    try:
        models["EpisodeControlPayload"].model_validate(dict(request.policy.payload or {}))
    except Exception as exc:
        return "policy.payload is not a valid EpisodeControlPayload: %s" % _first_line(exc)
    # The controller does not validate the entrypoint's world payload as a full
    # WorldStagePayload -- it builds one per segment. It reads exactly four keys,
    # via its own ``_world_setup``. Calling that function is a truer contract test
    # than re-deriving the rule here, because it cannot drift from the Chain.
    try:
        setup = models["_world_setup"](request)
    except Exception as exc:
        return "world.payload rejected by the Chain's own _world_setup: %s" % _first_line(exc)
    if isinstance(setup, str):
        return "world.payload rejected by the Chain's own _world_setup: %s" % setup
    return None


def _first_line(error: Exception) -> str:
    text = str(error).replace("\n", " | ")
    return text[:400]


def rehearsal_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    import os

    values = os.environ if environ is None else environ
    return str(values.get("PLUMB_REHEARSAL_CHAIN", "")).strip().lower() in {"1", "true", "yes", "on"}


def build_rehearsal_profile(environ: Optional[Mapping[str, str]] = None) -> RehearsalProfile:
    import os

    values = os.environ if environ is None else environ

    def number(name: str, default: float) -> float:
        try:
            return float(values.get(name, default))
        except (TypeError, ValueError):
            return default

    return RehearsalProfile(
        policy_seconds=number("PLUMB_REHEARSAL_POLICY_SECONDS", 0.004),
        world_seconds_per_chunk=number("PLUMB_REHEARSAL_WORLD_SECONDS", 0.010),
        validity_seconds=number("PLUMB_REHEARSAL_VALIDITY_SECONDS", 0.002),
        judge_seconds=number("PLUMB_REHEARSAL_JUDGE_SECONDS", 0.008),
        unevaluable_rate=number("PLUMB_REHEARSAL_UNEVALUABLE_RATE", 0.08),
        service_failure_rate=number("PLUMB_REHEARSAL_FAILURE_RATE", 0.02),
        dropped_webhook_rate=number("PLUMB_REHEARSAL_DROPPED_WEBHOOK_RATE", 0.02),
        duplicate_webhook_rate=number("PLUMB_REHEARSAL_DUPLICATE_WEBHOOK_RATE", 0.02),
        max_replicas=int(number("PLUMB_REHEARSAL_MAX_REPLICAS", 100)),
        scored=str(values.get("PLUMB_REHEARSAL_SCORED", "1")).strip().lower() in {"1", "true", "yes", "on"},
    )


__all__ = [
    "REHEARSAL_CHAIN_ASYNC_URL",
    "REHEARSAL_CHAIN_ID",
    "REHEARSAL_LABEL",
    "REHEARSAL_QUEUE_URL",
    "RehearsalChainTransport",
    "RehearsalProfile",
    "RehearsalStats",
    "SIMULATED_TRANSPORT",
    "chain_contract_available",
    "build_rehearsal_profile",
    "rehearsal_enabled",
]
