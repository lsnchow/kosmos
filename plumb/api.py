"""PLUMB control plane HTTP surface.

Two backends are reachable: an explicitly labelled synthetic engineering fixture
and real-model execution on a deployed Baseten Chain.  The real backend is
registered only when its environment is configured *and* its declared gates
carry passing evidence, so the fail-closed behaviour the project depends on is
preserved rather than removed.

What this layer will not do:

* fabricate telemetry.  Platform queue depth stays ``unavailable`` without a
  tested Chain queue route, USD stays ``unavailable`` without a verified price
  basis, and the application's own ledger counts are never relabelled as
  platform queue depth.
* assert qualification.  ``/api/gates`` reports evidence;
  ``plumb.gates.QualificationValidator`` is the only thing that decides.
* invent a frame.  Free-play returns ``503`` with a named reason when no
  certified world backend is configured, rather than a placeholder image.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import math
import os
import threading
import time
import zlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from plumb.artifacts import ArtifactStore
from plumb.engine import RunService
from plumb.evidence import experiments_payload
from plumb.freeplay import recording_branch_status, source_branch_step, source_identity, task_start_status
from plumb.gates import GATE_DESCRIPTIONS, GateLedger, GateStatus
from plumb.measurement import analyze
from plumb.protocol import (
    ProtocolDocument,
    ProtocolError,
    analysis_options_for_run,
    protocol_payload,
)
from plumb.records import BACKENDS, COHORTS, RUN_MODES, ConfigurationError
from plumb.reference import reference_payload
from plumb.ledger import CallbackConflictError
from plumb.live_demo import LiveDemoService, register_live_demo_routes
from plumb.comparisons import ComparisonService, register_comparison_routes
from plumb.demo_judge import DemoJudgeService, register_demo_judge_routes
from plumb.outbox import BasetenOutbox
from plumb.platform import BasetenChainClient, BasetenPlatformConfig, CallbackVerificationError, PlatformSchemaError

LOGGER = logging.getLogger("plumb.api")

POLICIES = ["OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL"]
TASKS = ["open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth"]

#: Spec section 11: twelve tiles is a viewport choice, not a policy count.
TILE_SLOTS = 12

#: Spec section 7 asks for one-second telemetry where it is supported.
TELEMETRY_INTERVAL_SECONDS = 1.0

#: How many recent events to replay to a newly connected stream, so a reload
#: rebuilds the wall rather than starting blank.
SEGMENT_REPLAY_EVENTS = 400

#: Snapshots to keep sending after a run reaches a terminal status.  The old
#: implementation broke immediately, so the telemetry strip froze the instant a
#: burst finished -- exactly when a stage audience is looking at it.  A bounded
#: tail keeps the final numbers arriving and still lets the stream end, so a
#: client that reads to completion is not left hanging.
POST_TERMINAL_SNAPSHOTS = 3

#: The single cell the pitch names out loud (spec section 4).
CALLED_SHOT_CELL = ("OpenVLA", "close_drawer")

_ARTIFACT_MEDIA = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "synthetic"
    backend: str = "synthetic"
    cohort: str = "primary"
    policies: List[str] = Field(default_factory=lambda: list(POLICIES))
    tasks: List[str] = Field(default_factory=lambda: list(TASKS))
    starts_per_task: int = Field(default=50, ge=1, le=500)
    seed: int = Field(default=20260919, ge=0, le=2**31 - 1)
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    operating_point_id: Optional[str] = Field(default=None, max_length=200)
    #: Free-text task strings. Every policy here is language-conditioned, so a
    #: task string is already the system's primary input -- this exposes it
    #: rather than adding a capability. A run carrying these is forced into the
    #: `exploration` cohort, which the lineage validator keeps disjoint from
    #: every scored cohort, so a typed prompt can never land in a primary cell.
    prompts: List[str] = Field(default_factory=list, max_length=8)


#: Actions per chunk, and chunks per free-prompt rollout. Five chunks of sixteen
#: is eighty actions: long enough to show an intent developing, short enough that
#: the tile finishes while the presenter is still talking over the wall.
PROMPT_CHUNK_ACTIONS = 16
PROMPT_CHUNKS = 5


def custom_task_id(prompt: str) -> str:
    """A stable task id for a free-text prompt.

    Derived from the prompt so the same words resolve to the same task, which is
    what lets the ledger's idempotency and the wall's slot keying behave exactly
    as they do for a benchmark task. The `custom:` prefix is load-bearing: every
    consumer that needs to know whether a cell has human ground truth tests it.
    """

    digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:10]
    return "custom:%s" % digest


class FreeplayInput(BaseModel):
    """Free-play takes a *direction*, not a raw action vector.

    Spec section 9 step 5a: one keypress maps to a constant action chunk which is
    sent straight to the world model.  Expanding the chunk on the server keeps
    the client from being able to submit an arbitrary trajectory.
    """

    model_config = ConfigDict(extra="forbid")
    session_id: Optional[str] = None
    direction: str = Field(pattern="^(up|down|left|right|forward|back|stop)$")
    task: str = Field(default="close_drawer")
    # A source-bound request names a catalog identity and repeats its content
    # hash.  The server re-resolves both; clients never supply a path, frame, or
    # timestamp from which a branch could be guessed.
    video_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    source_sha256: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def source_binding_is_complete(self) -> "FreeplayInput":
        if (self.video_id is None) != (self.source_sha256 is None):
            raise ValueError("video_id and source_sha256 must be supplied together for source-bound free-play")
        return self


class AnnotationInput(BaseModel):
    """One blinded annotation.  No policy/backend/command field exists by design."""

    model_config = ConfigDict(extra="forbid")
    clip_id: str = Field(min_length=1, max_length=200)
    annotator_id: str = Field(min_length=1, max_length=200)
    annotator_type: str = Field(pattern="^(human|model|external_label)$")
    integrity: str = Field(pattern="^(intact|artifact|uncertain)$")
    collision: str = Field(pattern="^(none_visible|visible|uncertain)$")
    progress: Optional[int] = Field(default=None, ge=0, le=5)
    completion_evidence: str = Field(pattern="^(met|not_met|uncertain)$")
    evidence_frame_indices: List[int] = Field(default_factory=list, max_length=16)
    observable_reasons: str = Field(default="", max_length=2000)
    annotator_model: Optional[str] = Field(default=None, max_length=200)
    annotator_model_revision: Optional[str] = Field(default=None, max_length=200)


async def _callback_body(request: Request) -> bytes:
    """Bound both callback routes without changing the signed byte stream."""
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 1024 * 1024:
            raise HTTPException(413, "Callback exceeds the 1 MiB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _load_protocol(root: Path, repo_root: Path) -> Tuple[Optional[ProtocolDocument], Optional[str]]:
    """Load ``protocol.json`` if one has been frozen, else report why not."""

    for candidate in (root / "protocol.json", repo_root / "protocol.json"):
        if candidate.is_file():
            try:
                return ProtocolDocument.load(candidate), None
            except ProtocolError as exc:
                return None, str(exc)
    return None, "no protocol.json has been frozen; run 'plumb freeze-protocol'"


def _backend_settings(webhook: str) -> Any:
    from plumb.backends.baseten import BasetenBackendSettings

    return BasetenBackendSettings(
        webhook_endpoint=webhook,
        operating_point_id=os.environ.get("PLUMB_OPERATING_POINT", "unconfirmed"),
        resolution=int(os.environ.get("PLUMB_RESOLUTION", "256")),
        denoise_steps=int(os.environ.get("PLUMB_DENOISE_STEPS", "30")),
        action_chunk_size=int(os.environ.get("PLUMB_ACTION_CHUNK", "16")),
        batch_size=int(os.environ.get("PLUMB_BATCH_SIZE", "1")),
        feedback_mode=os.environ.get("PLUMB_FEEDBACK_MODE", "unqualified"),
        parity_status=os.environ.get("PLUMB_PARITY_STATUS", "unqualified"),
    )


def _register_rehearsal_backend(root: Path) -> Tuple[Optional[Any], List[str]]:
    """Build a backend that talks to a SIMULATED Chain.

    This exists so the whole production path -- envelope, outbox, HMAC callback,
    reconciler, telemetry, cost attribution, ledger, measurement, artifacts and
    the dashboard -- can be exercised end to end without credentials or a GPU.
    It measures nothing about a model.  Every record it produces carries
    ``transport="simulated"`` and keeps ``feedback_mode``/``parity_status``
    unqualified, and the gate ledger still decides qualification.
    """

    from plumb.backends.baseten import BasetenBackendSettings, BasetenChainBackend
    from plumb.platform import (
        BasetenPlatformConfig,
        ChainAsyncEndpoint,
        VerifiedChainQueueRoute,
    )
    from plumb.rehearsal import (
        REHEARSAL_CHAIN_ASYNC_URL,
        REHEARSAL_QUEUE_URL,
        RehearsalChainTransport,
        build_rehearsal_profile,
    )

    webhook = os.environ.get("PLUMB_WEBHOOK_ENDPOINT") or "http://127.0.0.1:%s/api/callbacks" % os.environ.get(
        "PLUMB_PORT", "8787"
    )
    secret = os.environ.get("BASETEN_WEBHOOK_SECRET") or "rehearsal-secret-do-not-use-in-production"
    try:
        settings: BasetenBackendSettings = _backend_settings(webhook)
    except ValueError as exc:
        return None, [str(exc)]

    transport = RehearsalChainTransport(
        webhook_secret=secret,
        artifacts_dir=root / "rehearsal",
        profile=build_rehearsal_profile(),
    )
    # A rehearsal that stalls should say so in seconds, not sit on the
    # production 30-minute per-episode deadline.
    settings = dataclasses.replace(
        settings,
        request_deadline_seconds=float(os.environ.get("PLUMB_REHEARSAL_DEADLINE_SECONDS", "45")),
        poll_interval_seconds=float(os.environ.get("PLUMB_REHEARSAL_POLL_SECONDS", "0.05")),
    )
    config = BasetenPlatformConfig(
        api_key="rehearsal-key",
        chain_endpoint=ChainAsyncEndpoint(REHEARSAL_CHAIN_ASYNC_URL),
        webhook_secret=secret,
    )
    # The queue route is "verified" against the rehearsal transport only, and the
    # evidence URI says exactly that so it cannot be mistaken for account evidence.
    queue_route = VerifiedChainQueueRoute(
        url=REHEARSAL_QUEUE_URL,
        evidence_uri="plumb://rehearsal-transport/simulated-queue-route",
        verified_at="2026-09-19T00:00:00+00:00",
    )
    from plumb.starts import RehearsalStartResolver, ScenarioStartResolver, StartResolutionError

    # Prefer a real ScenarioManifest even in rehearsal, so a rehearsal against
    # real Gate C panels exercises the real provenance path.
    resolver: Any
    try:
        resolver = ScenarioStartResolver(root / "scenarios.jsonl", artifact_root=root)
    except StartResolutionError:
        resolver = RehearsalStartResolver(resolution=settings.resolution)
    backend = BasetenChainBackend(
        settings=settings,
        data_dir=root,
        config=config,
        queue_route=queue_route,
        transport=transport,
        transport_kind="simulated",
        start_resolver=resolver,
    )
    backend.rehearsal_transport = transport
    return backend, ["REHEARSAL MODE: the Chain is simulated; no model, latency or cost is measured"]


def _register_baseten_backend(root: Path) -> Tuple[Optional[Any], List[str]]:
    """Build the Baseten backend when its environment is present.

    Absence is normal and is reported, not raised: the console must be able to
    run and show *why* the real backend is unavailable.
    """

    from plumb.backends.baseten import BackendNotConfigured, BasetenChainBackend
    from plumb.rehearsal import rehearsal_enabled
    from plumb.result_store import ResultStoreConfigurationError, S3ResultStore

    if rehearsal_enabled():
        return _register_rehearsal_backend(root)

    webhook = os.environ.get("PLUMB_WEBHOOK_ENDPOINT")
    if not webhook:
        return None, ["PLUMB_WEBHOOK_ENDPOINT is not set"]
    from plumb.starts import ScenarioStartResolver, StartResolutionError

    result_store = None
    result_store_reason: Optional[str] = None
    try:
        # Construction only validates explicit configuration. It does not make
        # an S3 call; preflight is an operator-invoked action below.
        result_store = S3ResultStore.from_env()
    except ResultStoreConfigurationError as exc:
        result_store_reason = str(exc)
    try:
        settings = _backend_settings(webhook)
        backend = BasetenChainBackend.from_env(settings, root, result_store=result_store)
    except (BackendNotConfigured, ValueError) as exc:
        return None, [str(exc)]
    missing = list(backend.configuration_status()["missing"])
    if result_store_reason is not None:
        missing.append(result_store_reason)
    # A real submission needs real starts. There is no synthetic fallback on this
    # path: a rehearsal start must never reach a production run.
    try:
        backend.start_resolver = ScenarioStartResolver(root / "scenarios.jsonl", artifact_root=root)
    except StartResolutionError as exc:
        missing.append("scenarios.jsonl (Gate C): %s" % exc)
    return backend, missing


def _cloud_diagnostic_service(root: Path) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Load only a fixed hash-bound fixture; never inspect CLI credentials."""

    from plumb.cloud_diagnostic import CloudDiagnosticConfig, CloudDiagnosticService, CloudDiagnosticValidationError

    required = {
        "model_id": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_MODEL_ID"),
        "deployment_id": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_DEPLOYMENT_ID"),
        "cli_profile": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_PROFILE"),
        "model_revision": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_MODEL_REVISION"),
        "checkpoint_sha256": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_CHECKPOINT_SHA256"),
        "soar_revision": os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_SOAR_REVISION"),
    }
    if any(not isinstance(value, str) or not value.strip() for value in required.values()):
        return None, {"configured": False, "available": False, "reason": "cloud diagnostic model/deployment/profile and pinned model bindings are not configured", "policy": "SuSIE_LL", "qualified": False, "fixture": None}
    fixture_dir = root / "cloud-diagnostic-fixtures" / "susie-v1"
    manifest_path = fixture_dir / "manifest.json"
    try:
        timeout_seconds = float(os.environ.get("PLUMB_CLOUD_DIAGNOSTIC_TIMEOUT_SECONDS", "300"))
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
            raise ValueError("cloud diagnostic timeout must be from 1 through 600 seconds")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = manifest.get("frames") if isinstance(manifest, Mapping) else None
        if not isinstance(frames, list) or len(frames) != 2:
            raise ValueError("fixture must bind current and goal frames")
        encoded = []
        urls = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError("fixture frame is invalid")
            name, digest = frame.get("output_file"), frame.get("output_file_sha256")
            path = (fixture_dir / str(name)).resolve()
            if fixture_dir.resolve() not in path.parents or not path.is_file() or path.is_symlink():
                raise ValueError("fixture image is unavailable")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("fixture image digest mismatch")
            encoded.append(base64.b64encode(raw).decode("ascii"))
            urls.append("/api/artifacts/" + path.relative_to(root).as_posix())
        config = CloudDiagnosticConfig(
            **{key: str(value) for key, value in required.items()},
            current_png_base64=encoded[0],
            goal_png_base64=encoded[1],
            current_png_sha256="sha256:" + str(frames[0]["output_file_sha256"]),
            goal_png_sha256="sha256:" + str(frames[1]["output_file_sha256"]),
            prompt="Fixed static current/goal diagnostic; goal conditioning only, not language or task success.",
            timeout_seconds=timeout_seconds,
        )
    except (OSError, ValueError, KeyError, CloudDiagnosticValidationError) as error:
        return None, {"configured": False, "available": False, "reason": "cloud diagnostic fixture is unavailable: %s" % error, "policy": "SuSIE_LL", "qualified": False, "fixture": None}
    return CloudDiagnosticService(root, config), {
        "configured": True,
        "available": True,
        "reason": "configured; CLI authentication is exercised only by an explicit diagnostic request",
        "model_id": config.model_id,
        "deployment_id": config.deployment_id,
        "policy": "SuSIE_LL",
        "qualified": False,
        "fixture": {"current_url": urls[0], "goal_url": urls[1], "description": manifest.get("preprocessing", {}).get("limitation")},
    }



class _SinglePageApp(StaticFiles):
    """Static files that fall back to ``index.html`` for client-side routes.

    The console moved from one scrolling page to six routed pages, so a reload
    on ``/results`` asks this mount for a file that does not exist. Plain
    ``StaticFiles`` answers 404, which would turn a refresh mid-demo into a
    blank page. API routes are unaffected: they are declared on the app and
    match before this mount is consulted.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code != 404:
                raise
            # A missing asset is a real 404 -- only extensionless paths, which is
            # what a route looks like, are handed to the client router.
            if "." in path.rsplit("/", 1)[-1]:
                raise
            return await super().get_response("index.html", scope)


def create_app(
    data_dir: Optional[Path] = None,
    *,
    baseten_client: Optional[BasetenChainClient] = None,
    development_review_root: Optional[Path] = None,
    development_review_private_root: Optional[Path] = None,
    live_demo_service: Optional[LiveDemoService] = None,
    comparison_service: Optional[ComparisonService] = None,
    demo_judge_service: Optional[DemoJudgeService] = None,
) -> FastAPI:
    root = (data_dir or Path(os.environ.get("PLUMB_DATA_DIR", "data"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    results_root = Path(os.environ.get("PLUMB_RESULTS_DIR", str(repo_root)))
    store = ArtifactStore(results_root)

    document, protocol_reason = _load_protocol(root, repo_root)
    baseten_backend, baseten_missing = _register_baseten_backend(root)
    registered: Dict[str, Any] = {}
    if baseten_backend is not None:
        registered["baseten"] = baseten_backend

    service = RunService(root, backends=registered, gates_path=store.gates_path)
    if baseten_client is None and os.environ.get("PLUMB_ENABLE_BASETEN_CALLBACKS") == "1":
        config = BasetenPlatformConfig.from_env()
        if not config.webhook_secret:
            raise ValueError("PLUMB_ENABLE_BASETEN_CALLBACKS requires BASETEN_WEBHOOK_SECRET")
        baseten_client = BasetenChainClient(config)
    outbox = BasetenOutbox(service.ledger, baseten_client) if baseten_client is not None else None
    pool = ThreadPoolExecutor(
        max_workers=int(os.environ.get("PLUMB_MAX_WORKERS", "8")), thread_name_prefix="plumb-run"
    )
    cloud_diagnostic, cloud_diagnostic_status = _cloud_diagnostic_service(root)
    cloud_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plumb-cloud-diagnostic")
    live_demo = live_demo_service or LiveDemoService(root)
    comparisons = comparison_service or ComparisonService(root)
    demo_judge = demo_judge_service or DemoJudgeService(root)
    submitted: set = set()
    lock = threading.RLock()
    freeplay_sessions: Dict[str, Dict[str, Any]] = {}
    annotations: List[Dict[str, Any]] = []

    @asynccontextmanager
    async def lifespan(application):
        yield
        pool.shutdown(wait=True)
        cloud_pool.shutdown(wait=True)
        live_demo.shutdown()
        comparisons.shutdown()
        demo_judge.shutdown()

    app = FastAPI(title="Kosmos", version="0.2.0", lifespan=lifespan)
    app.state.service = service
    app.state.data_dir = root
    app.state.store = store
    app.state.protocol = document
    app.state.baseten = baseten_backend
    app.state.baseten_outbox = outbox
    app.state.cloud_diagnostic = cloud_diagnostic
    app.state.cloud_diagnostic_status = cloud_diagnostic_status
    app.state.live_demo = live_demo
    app.state.comparisons = comparisons
    app.state.demo_judge = demo_judge
    register_live_demo_routes(app, live_demo)
    register_comparison_routes(app, comparisons)
    register_demo_judge_routes(app, demo_judge)
    # Development review is intentionally isolated from the annotation/Gate D
    # routes. Its SQLite database is private data, never an artifact endpoint.
    from plumb.development_review import DevelopmentReviewStore, register_development_review_routes

    review_root = development_review_root or (root / "review")
    private_root = development_review_private_root or root.parent / (root.name + "-development-review-private")
    try:
        review_session_ttl = int(os.environ.get("PLUMB_DEVELOPMENT_REVIEW_SESSION_TTL_SECONDS", str(8 * 60 * 60)))
    except ValueError as error:
        raise ValueError("PLUMB_DEVELOPMENT_REVIEW_SESSION_TTL_SECONDS must be an integer") from error
    development_review = DevelopmentReviewStore(
        repo_root=repo_root,
        review_root=review_root,
        served_root=root,
        private_root=private_root,
        session_ttl_seconds=review_session_ttl,
    )
    app.state.development_review = development_review
    register_development_review_routes(app, development_review)

    def cloud_public(record: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = str(record["request_id"])
        result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
        status = str(record.get("status"))
        artifact_name = "result.json" if status == "completed" else "failure.json"
        def artifact_url(name: str) -> Optional[str]:
            path = root / "cloud-diagnostics" / request_id / name
            return "/api/artifacts/cloud-diagnostics/%s/%s" % (request_id, name) if path.is_file() else None
        return {
            "request_id": request_id,
            "status": status,
            "qualified": False,
            "diagnostic_only": True,
            "physical_action": result.get("physical_action"),
            "modelrevision": result.get("modelrevision"),
            "timing": result.get("timing"),
            "error": record.get("error"),
            "report_url": artifact_url(artifact_name),
            "raw_response_url": artifact_url("raw-response.json"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }

    @app.get("/api/cloud-diagnostics/status")
    def cloud_diagnostic_readiness() -> dict:
        return dict(cloud_diagnostic_status)

    @app.get("/api/cloud-diagnostics")
    def cloud_diagnostic_list() -> dict:
        if cloud_diagnostic is None:
            return {"requests": [], **dict(cloud_diagnostic_status)}
        return {"requests": [cloud_public(record) for record in cloud_diagnostic.list_requests()], **dict(cloud_diagnostic_status)}

    @app.post("/api/cloud-diagnostics", status_code=202)
    async def cloud_diagnostic_submit(request: Request) -> dict:
        from plumb.development_review import _safe_origin

        _safe_origin(request)
        if cloud_diagnostic is None:
            raise HTTPException(409, cloud_diagnostic_status)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(422, "cloud diagnostic body must be valid JSON")
        if not isinstance(body, Mapping) or set(body) - {"request_id", "prompt"}:
            raise HTTPException(422, "cloud diagnostic body accepts only request_id and fixed prompt")
        if "prompt" in body and body["prompt"] != cloud_diagnostic.config.prompt:
            raise HTTPException(422, "cloud diagnostic prompt is fixed by the pinned goal fixture")
        try:
            record = cloud_diagnostic.submit(body.get("request_id"))
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error))
        if record["status"] == "pending":
            cloud_pool.submit(cloud_diagnostic.execute, record["request_id"])
        return cloud_public(record)

    @app.get("/api/cloud-diagnostics/{request_id}")
    def cloud_diagnostic_get(request_id: str) -> dict:
        if cloud_diagnostic is None:
            raise HTTPException(409, cloud_diagnostic_status)
        try:
            return cloud_public(cloud_diagnostic.get(request_id))
        except KeyError:
            raise HTTPException(404, "cloud diagnostic request not found")

    # ------------------------------------------------------------------ helpers

    def require_run(run_id: str) -> dict:
        try:
            result = service.get_run(run_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "Run not found")
        if result is None:
            raise HTTPException(404, "Run not found")
        return result

    def gate_ledger() -> GateLedger:
        return GateLedger.load(str(store.gates_path))

    def gates() -> dict:
        """Report gate evidence.  ``qualified`` is always a separate decision."""

        ledger = gate_ledger()
        values = []
        for gate_id, description in GATE_DESCRIPTIONS.items():
            record = ledger.records.get(gate_id)
            mapping = record.to_mapping() if record is not None else {"status": GateStatus.NOT_RUN.value}
            errors = list(record.pass_evidence_errors()) if record is not None else []
            reasons = list(mapping.get("reasons") or [])
            values.append(
                {
                    "id": gate_id,
                    "description": description,
                    "status": mapping.get("status", GateStatus.NOT_RUN.value),
                    "summary": (reasons[-1] if reasons else description),
                    "evidence_errors": errors,
                    "evidence": mapping,
                }
            )
        blocking = sorted({name for value in values for name in value["evidence_errors"]})
        return {
            "gates": values,
            "protocol_hash": ledger.protocol_hash,
            "qualified": False,
            "reason": (
                "Full qualification requires reviewed, protocol-bound evidence decided by "
                "QualificationValidator; an evidence report never qualifies a run by itself."
            ),
            "blocking_evidence_errors": blocking,
            "source": store.gates_path.as_posix(),
        }

    def available_backends() -> List[Dict[str, Any]]:
        result = []
        for name, spec in BACKENDS.items():
            reasons = service.backend_blocking_reasons(name)
            entry = {
                "name": name,
                "description": spec["description"],
                "requires_gates": spec["requires_gates"],
                "available": not reasons,
                "blocking_reasons": reasons,
            }
            if name == "baseten":
                entry["configuration_missing"] = baseten_missing
            result.append(entry)
        return result

    def sweep_points() -> dict:
        """Read real sweep artifacts.  Never synthesises a cost curve."""

        path = store.results / "sweeps.json"
        if not path.is_file():
            return {
                "points": [],
                "status": "unavailable",
                "reason": "No independent real-model cost-fidelity sweep has completed.",
                "selected_operating_point": None,
            }
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return {"points": [], "status": "error", "reason": "sweeps.json is unreadable: %s" % exc}
        points = payload.get("points")
        if not isinstance(points, list):
            return {"points": [], "status": "error", "reason": "sweeps.json has no points array"}
        return {
            "points": points,
            "status": "available",
            "reason": None,
            "selected_operating_point": payload.get("selected_operating_point"),
            "confirmed_on_disjoint_panel": payload.get("confirmed_on_disjoint_panel", False),
            "source": path.as_posix(),
        }

    def telemetry_snapshot(run: Mapping[str, Any]) -> Dict[str, Any]:
        """Telemetry with explicit availability per field.

        Ledger counts are always present.  Platform queue depth, replicas and USD
        appear only when the platform actually supplies them; an absent value is
        ``None`` with a status string, never a zero that would read as a
        measurement of an empty queue.
        """

        now = time.time()
        payload: Dict[str, Any] = {
            "source": "application_ledger",
            "timestamp": now,
            "fresh_at": now,
            "stale": False,
            "mode": run.get("mode", "synthetic"),
            "episodes_planned": int(run.get("total") or 0),
            "episodes_completed": int(run.get("completed") or 0),
            "episodes_failed": int(run.get("failed") or 0),
            "episodes_cancelled": int(run.get("cancelled") or 0),
            "platform_queue": None,
            "platform_queue_status": "unavailable",
            "platform_in_progress": None,
            "active_replicas": None,
            "desired_replicas": None,
            "replicas_status": "unavailable",
            "gpu_seconds": None,
            "marginal_estimated_usd": None,
            "total_estimated_usd": None,
            "cost_status": "unavailable",
            "reconciliation_status": "not_applicable",
        }
        backend_name = str((run.get("config") or {}).get("backend") or "")
        if backend_name != "baseten" or baseten_backend is None:
            payload["platform_queue_status"] = "not_applicable_synthetic_backend"
            payload["replicas_status"] = "not_applicable_synthetic_backend"
            payload["cost_status"] = "not_applicable_synthetic_backend"
            return payload
        try:
            live = baseten_backend.telemetry(run.get("id") or run.get("run_id"))
        except Exception as exc:  # pragma: no cover - platform failure path
            payload["platform_queue_status"] = "error"
            payload["stale"] = True
            payload["reconciliation_status"] = "telemetry_error: %s" % exc
            return payload
        queue = live.get("platform_queue") or {}
        payload["platform_queue_status"] = queue.get("status", "unavailable")
        payload["platform_queue"] = queue.get("queued")
        payload["platform_in_progress"] = queue.get("in_progress")
        replicas = live.get("replicas") or {}
        payload["replicas_status"] = replicas.get("status", "unavailable")
        payload["active_replicas"] = replicas.get("active")
        payload["desired_replicas"] = replicas.get("desired")
        economics = live.get("economics") or {}
        payload["marginal_estimated_usd"] = (economics.get("marginal_execution") or {}).get("estimated_usd")
        payload["total_estimated_usd"] = (economics.get("total_demonstration_run") or {}).get("estimated_usd")
        payload["cost_status"] = "available" if payload["marginal_estimated_usd"] is not None else "unavailable"
        payload["reconciliation_status"] = economics.get("reconciliation_status", "unknown")
        payload["outbox_counts"] = live.get("outbox_counts")
        payload["source"] = "application_ledger+baseten_chain"
        return payload

    def tile_slots(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Twelve stable slots keyed by ``(policy, task)``.

        The previous implementation sent the last twelve non-planned episodes, so
        a 1,500-row burst churned tiles instead of letting them grow.  A stable
        key means a tile belongs to one cell for the whole run and can accumulate
        frames.
        """

        chosen: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for episode in episodes:
            key = (str(episode.get("policy")), str(episode.get("task")))
            existing = chosen.get(key)
            if existing is None or _tile_rank(episode) > _tile_rank(existing):
                chosen[key] = episode
        ordered = sorted(chosen.items(), key=lambda item: (item[0][1], item[0][0]))
        slots: List[Dict[str, Any]] = []
        for index, ((policy, task), episode) in enumerate(ordered[:TILE_SLOTS]):
            slots.append(
                {
                    "slot": index,
                    "policy": policy,
                    "task": task,
                    # Presentation timing only (spec section 11), 40-60 ms.
                    "ignition_delay_ms": 40 + (index * 2),
                    "provenance": _tile_provenance(episode),
                    "certified_frame_count": episode.get("n_segments"),
                    "episode": episode,
                }
            )
        return slots

    # ------------------------------------------------------------------- routes

    @app.get("/api/health")
    def health() -> dict:
        backends = available_backends()
        return {
            "status": "ok",
            "version": "0.2.0",
            "qualified": False,
            "available_backends": [entry["name"] for entry in backends if entry["available"]],
            "registered_backends": backends,
            "modes": sorted(RUN_MODES),
            "cohorts": list(COHORTS),
            "data_dir": str(root),
            "protocol_frozen": bool(document and document.frozen),
            "protocol_reason": protocol_reason,
            "transport": (
                baseten_backend.transport_kind if baseten_backend is not None else None
            ),
            "simulated_transport": bool(
                baseten_backend is not None and baseten_backend.transport_kind == "simulated"
            ),
        }

    @app.get("/api/rehearsal")
    def rehearsal_report() -> dict:
        """Report the simulated Chain's own counters, or say there is none."""

        transport = getattr(baseten_backend, "rehearsal_transport", None)
        if transport is None:
            return {
                "status": "not_active",
                "reason": "the Chain transport is not simulated; set PLUMB_REHEARSAL_CHAIN=1 to rehearse",
            }
        return {"status": "active", **transport.report()}

    @app.get("/api/protocol")
    def protocol() -> dict:
        ledger = gate_ledger()
        base: Dict[str, Any] = {
            "policies": POLICIES,
            "tasks": TASKS,
            "starts_per_task": 50,
            "total": 1500,
            "reference": reference_payload(),
            "target": {"episodes": 1500, "seconds": 60, "usd": 11.25, "status": "unmeasured"},
            "mode": "synthetic",
            "qualified": False,
            # The console renders these three; they were previously absent, so
            # the provenance row always read "not recorded".
            "backend_revision": _backend_revision(ledger),
            "judge_revision": _judge_revision(ledger),
            "parity": _parity_status(ledger),
            "protocol_hash": ledger.protocol_hash,
            "protocol_frozen": False,
            "preregistration": {
                "status": "unregistered",
                "blocking_reasons": [protocol_reason or "no protocol.json"],
            },
        }
        if document is None:
            return base
        base.update(protocol_payload(document, repo_root=repo_root))
        base["protocol_frozen"] = document.frozen
        base["backend_revision"] = _backend_revision(ledger)
        base["judge_revision"] = _judge_revision(ledger)
        base["parity"] = _parity_status(ledger)
        base["mode"] = "qualification" if document.frozen else "synthetic"
        return base

    @app.get("/api/gates")
    def gate_report() -> dict:
        return gates()

    @app.get("/api/sweeps")
    def sweeps() -> dict:
        return sweep_points()

    @app.get("/api/experiments")
    def experiments() -> dict:
        return experiments_payload(root)

    @app.get("/api/world-videos")
    def world_videos() -> dict:
        """Existing, hash-bound model outputs; never enqueue inference jobs."""
        from plumb.world_videos import world_videos_payload
        return world_videos_payload(root)

    def freeplay_source(video_id: str) -> Optional[Dict[str, Any]]:
        """Look up one server-derived source identity from the current catalog."""

        from plumb.world_videos import world_videos_payload

        matches = [
            video for video in world_videos_payload(root).get("videos", [])
            if isinstance(video, Mapping) and video.get("id") == video_id
        ]
        # A source-bound control cannot choose arbitrarily if an old catalog
        # happened to create duplicate IDs. The catalog now incorporates the
        # evidence root and digest, but preserve the refusal for old data.
        if len(matches) != 1:
            return None
        return dict(matches[0])

    @app.get("/api/freeplay/status")
    def freeplay_status(video_id: Optional[str] = None) -> dict:
        """Read-only capability report; it never submits a generation request."""

        protocol_hash = document.sha256 if (document is not None and document.frozen) else None
        if video_id is None:
            return task_start_status(baseten_backend, baseten_missing, protocol_hash, protocol_reason)
        if not video_id or len(video_id) > 200:
            raise HTTPException(422, "Invalid video ID")
        source = freeplay_source(video_id)
        if source is None:
            return {
                "available": False,
                "mode": "recording_branch",
                "reason": "No current saved recording matches this video_id, so no source-bound branch can be restored.",
                "source": None,
                "binding": {
                    "requested": True,
                    "exact_branch_supported": False,
                    "checkpoint": None,
                    "adapter": None,
                },
                "missing": ["a current hash-bound catalog recording matching video_id"],
                "scored": False,
                "qualified": False,
            }
        return recording_branch_status(baseten_backend, source)

    @app.get("/api/artifact-inventory")
    def artifact_inventory() -> dict:
        """Which spec section 8 artifacts exist.  Used by the e2e smoke test."""

        return store.inventory()

    @app.get("/api/calledshot")
    def called_shot() -> dict:
        """The one cell the pitch names: 92% human vs 4% simulator.

        ``plumb_estimate`` is ``None`` until a frozen real-model run has measured
        it, and the response says so rather than showing a placeholder number.
        """

        policy, task = CALLED_SHOT_CELL
        payload = reference_payload()
        human = _reference_cell(payload, "human", policy, task)
        simpler = _reference_cell(payload, "simpler", policy, task)
        estimate, estimate_status, run_id = _called_shot_estimate(service, policy, task)
        prereg = {"status": "unregistered", "uri": None, "sha256": None}
        if document is not None:
            record = document.preregistration_status(repo_root=repo_root)
            prereg = {"status": record.status, "uri": record.uri, "sha256": record.protocol_sha256}
        return {
            "cell": "%s/%s" % (policy, task),
            "policy": policy,
            "task": task,
            "human_rate": human.get("rate"),
            "human_successes": human.get("successes"),
            "human_n": 50,
            "simpler_rate": simpler.get("rate"),
            "simpler_successes": simpler.get("successes"),
            "simpler_n": 50,
            "published_gap_points": 88.0,
            "plumb_estimate": estimate,
            "plumb_status": estimate_status,
            "plumb_run_id": run_id,
            "preregistration": prereg,
            "note": (
                "The reference value is already published, so this is a preregistered comparison, "
                "not a blind prediction."
            ),
        }

    @app.get("/api/clips/sixclip")
    def six_clip() -> dict:
        """Six clips with provenance sealed until the reveal is requested."""

        return _six_clip_payload(store, revealed=False)

    @app.post("/api/clips/sixclip/reveal")
    def six_clip_reveal() -> dict:
        return _six_clip_payload(store, revealed=True)

    @app.get("/api/baseten/outbox")
    def platform_submissions(run_id: Optional[str] = None) -> dict:
        if run_id is not None:
            require_run(run_id)
        # Never return payloads, callback bodies, credentials, or ownership tokens.
        public_fields = ("outbox_id", "run_id", "episode_id", "state", "platform_request_id",
                         "post_attempt_count", "remote_status", "remote_status_observed_at",
                         "cancellation_requested", "deadline_at", "created_at", "updated_at")
        rows = service.ledger.list_baseten_submissions(run_id=run_id)
        return {"submissions": [{key: row.get(key) for key in public_fields} for row in rows],
                "callback_ingress_enabled": outbox is not None,
                "automatic_submission_enabled": False, "qualified": False}

    @app.post("/api/baseten/callback", status_code=202)
    async def outbox_callback(request: Request) -> dict:
        if outbox is None:
            raise HTTPException(503, "Baseten callback ingress is not configured")
        raw = await _callback_body(request)
        try:
            receipt = outbox.ingest_callback(raw, dict(request.headers))
        except CallbackVerificationError:
            raise HTTPException(401, "Invalid callback authentication")
        except CallbackConflictError:
            raise HTTPException(409, "Conflicting callback for a persisted request")
        except (PlatformSchemaError, ValueError):
            raise HTTPException(422, "Invalid callback payload")
        return {"request_id": receipt.callback.request_id, "persisted": True,
                "duplicate": receipt.duplicate, "associated": receipt.associated,
                "logical_result_finalized": False}

    @app.post("/api/baseten/reconcile")
    async def reconcile_platform() -> dict:
        if outbox is None:
            raise HTTPException(503, "Baseten delivery is not configured")
        # Only local recovery here. No guessed Chain lifecycle URL or network
        # retry is reachable from this endpoint.
        result = await outbox.reconcile()
        return {**asdict(result), "scope": "local_recovery_only", "submission_posts": 0}

    @app.get("/api/runs")
    def list_runs() -> dict:
        return {"runs": service.list_runs()}

    @app.post("/api/runs", status_code=201)
    def create_run(config: RunConfig) -> dict:
        if config.backend not in BACKENDS:
            raise HTTPException(422, "Unknown backend")
        blocking = service.backend_blocking_reasons(config.backend)
        if blocking:
            raise HTTPException(
                409,
                {
                    "reason": "backend '%s' cannot execute yet" % config.backend,
                    "blocking_reasons": blocking,
                    "gates": gates(),
                },
            )
        # A free-text prompt becomes a task whose id encodes the prompt, so the
        # rest of the pipeline treats it exactly like a benchmark task. There is
        # no second code path: same planner, same backend, same ledger.
        prompt_text = {custom_task_id(text): text.strip() for text in config.prompts if text.strip()}
        tasks = list(config.tasks) + [task for task in prompt_text if task not in config.tasks]

        if not config.policies or not tasks:
            raise HTTPException(422, "At least one policy and task is required")
        if len(set(config.policies)) != len(config.policies) or len(set(tasks)) != len(tasks):
            raise HTTPException(422, "Duplicate policies/tasks are not allowed")
        if set(config.policies) - set(POLICIES) or set(tasks) - set(TASKS) - set(prompt_text):
            raise HTTPException(422, "Unknown policy or task")

        payload = config.model_dump(exclude_none=True)
        payload.pop("prompts", None)
        if prompt_text:
            payload["tasks"] = tasks
            payload["task_prompts"] = prompt_text
            # Forced, not defaulted. The caller does not get to put a typed
            # prompt in a scored cohort, and the ledger records which cohort it
            # actually ran in rather than which one was asked for.
            payload["cohort"] = "exploration"
            payload["horizons"] = {task: PROMPT_CHUNK_ACTIONS * PROMPT_CHUNKS for task in prompt_text}
        if document is not None and document.frozen:
            payload.setdefault("protocol_hash", document.sha256)
            if document.scenario_manifest_hash:
                payload.setdefault("scenario_manifest_hash", document.scenario_manifest_hash)
        try:
            result = service.create_run(payload)
        except ConfigurationError as exc:
            raise HTTPException(422, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        run_id = result.get("id", result.get("run_id"))
        if not run_id:
            raise HTTPException(500, "Engine returned no run identity")
        with lock:
            if run_id not in submitted and result.get("status") not in {"completed", "failed", "cancelled"}:
                submitted.add(run_id)
                future = pool.submit(service.execute_run, run_id)

                def finished(task):
                    error = task.exception()
                    if error is not None:
                        LOGGER.error(
                            "Run execution failed for %s",
                            run_id,
                            exc_info=(type(error), error, error.__traceback__),
                        )
                    with lock:
                        submitted.discard(run_id)

                future.add_done_callback(finished)
        return result

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return require_run(run_id)

    @app.get("/api/runs/{run_id}/episodes")
    def episodes(run_id: str) -> dict:
        require_run(run_id)
        return {"episodes": service.list_episodes(run_id, limit=10000)}

    def _analysis_for(
        run: Mapping[str, Any], rows: List[Dict[str, Any]], full: bool = False
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Analyse a run.

        ``full`` decides whether the headline statistics run.  They are genuinely
        expensive -- 10,000 bootstrap replicates, 1,000 split-half repetitions and
        10,000 MDD simulations per gap -- and ``plumb.measurement`` says so
        explicitly: "MDD power simulation is opt-in and intentionally not run on
        every analysis poll."  A dashboard polling this endpoint must not pay that
        cost, so the default is the fast per-cell view and ``publish`` (or
        ``?full=1``) computes the rest.
        """

        options: Any = run.get("config")
        note: Optional[str] = None
        if document is not None and document.frozen:
            try:
                options = analysis_options_for_run(
                    document, rows, cohort=str((run.get("config") or {}).get("cohort") or "primary")
                )
                if not full:
                    for heavy in ("reliability", "mdd", "advanced_inference", "drift", "cost_fidelity"):
                        options.pop(heavy, None)
                    note = (
                        "Fast view: split-half, MDD and the family-adjusted inference are computed by "
                        "POST /api/runs/{run_id}/publish or GET ...?full=1, not on a dashboard poll."
                    )
            except ProtocolError as exc:
                note = "protocol manifest could not be built: %s" % exc
        else:
            note = protocol_reason or "no frozen protocol; headline statistics stay not_computed"
        return analyze(rows, options), note

    @app.get("/api/runs/{run_id}/analysis")
    def analysis(run_id: str, full: bool = False) -> dict:
        run = require_run(run_id)
        rows = service.list_episodes(run_id, limit=10000)
        result, note = _analysis_for(run, rows, full=full)
        result.update({"mode": run.get("mode", "synthetic"), "qualified": False})
        if note:
            result["protocol_note"] = note
        return result

    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id: str):
        require_run(run_id)
        rows = service.list_episodes(run_id, limit=10000)
        return StreamingResponse(
            (json.dumps(row, allow_nan=False) + "\n" for row in rows),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="episodes.jsonl"'},
        )

    @app.post("/api/runs/{run_id}/publish")
    def publish_run(run_id: str) -> dict:
        """Write the spec section 8 artifacts for this run."""

        run = require_run(run_id)
        rows = service.list_episodes(run_id, limit=10000)
        # Publishing is the explicit, operator-initiated step, so it pays for the
        # full statistics.
        result, _ = _analysis_for(run, rows, full=True)
        proto = protocol()
        receipts = {
            "per_episode": store.write_per_episode(run_id, rows, run).to_mapping(),
            "run_ledger": store.write_run_ledger(service.list_runs()).to_mapping(),
            "leaderboard": store.write_leaderboard(result, run, proto).to_mapping(),
            "reliability": store.write_reliability(result, sweep_points()).to_mapping(),
            "exclusions": store.write_exclusions(result).to_mapping(),
            "gates": store.write_gates(gate_ledger()).to_mapping(),
        }
        return {"run_id": run_id, "receipts": receipts, "inventory": store.inventory()}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict:
        require_run(run_id)
        return service.cancel_run(run_id)

    @app.get("/api/runs/{run_id}/telemetry")
    def run_telemetry(run_id: str) -> dict:
        return telemetry_snapshot(require_run(run_id))

    @app.post("/api/runs/{run_id}/reconcile")
    def reconcile(run_id: str) -> dict:
        """Sweep for lost callbacks and ambiguous submissions."""

        require_run(run_id)
        if baseten_backend is None:
            raise HTTPException(409, {"reason": "no Baseten backend is configured", "missing": baseten_missing})
        return baseten_backend.reconcile(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request) -> StreamingResponse:
        require_run(run_id)

        async def snapshots():
            # Replay recent segment events so a client that connects mid-run (or
            # reloads) rebuilds its wall instead of starting empty.
            last_sequence = max(0, _latest_sequence(service, run_id) - SEGMENT_REPLAY_EVENTS)
            terminal_snapshots = 0
            while not await request.is_disconnected():
                run = require_run(run_id)
                rows = service.list_episodes(run_id, limit=10000)
                slots = tile_slots(rows)
                segment_events = []
                for event in service.list_events(run_id, after=last_sequence):
                    last_sequence = max(last_sequence, int(event.get("sequence") or 0))
                    # The ledger projects the event kind as "type", not
                    # "event_type". Reading the wrong key matched nothing and the
                    # wall silently stayed empty while 1,022 segments existed.
                    kind = event.get("type")
                    if kind not in ("segment_completed", "episode_completed"):
                        continue
                    payload = event.get("payload")
                    # Flatten the payload onto the event: the console's wall
                    # reducer reads episode_id/segment_index/frame_urls directly
                    # off the event, not from a nested object.
                    flattened = {
                        "type": kind,
                        "sequence": event.get("sequence"),
                        "timestamp": event.get("timestamp"),
                    }
                    if isinstance(payload, Mapping):
                        flattened.update(payload)
                    segment_events.append(flattened)
                data = {
                    "run": run,
                    "tiles": slots,
                    # Retained for clients that read a flat episode list.
                    "episodes": [slot["episode"] for slot in slots],
                    "segment_events": segment_events,
                    "telemetry": telemetry_snapshot(run),
                }
                yield "event: snapshot\ndata: " + json.dumps(data, allow_nan=False) + "\n\n"
                if run.get("status") in {"completed", "failed", "cancelled"}:
                    terminal_snapshots += 1
                    if terminal_snapshots >= POST_TERMINAL_SNAPSHOTS:
                        yield "event: complete\ndata: " + json.dumps(
                            {"run_id": run_id, "status": run.get("status")}, allow_nan=False
                        ) + "\n\n"
                        return
                    await asyncio.sleep(_terminal_tail_interval())
                    continue
                await asyncio.sleep(TELEMETRY_INTERVAL_SECONDS)

        return StreamingResponse(
            snapshots(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str):
        def private_namespace(parts):
            return any(part.casefold() == "private" or part.casefold().endswith("-development-review-private") for part in parts)

        if private_namespace(Path(artifact_path).parts):
            raise HTTPException(404, "Artifact not found")
        path = (root / artifact_path).resolve()
        suffix = path.suffix.lower()
        if root not in path.parents or not path.is_file() or suffix not in _ARTIFACT_MEDIA:
            raise HTTPException(404, "Artifact not found")
        if private_namespace(path.relative_to(root).parts):
            raise HTTPException(404, "Artifact not found")
        return FileResponse(
            path,
            media_type=_ARTIFACT_MEDIA[suffix],
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.post("/api/callbacks/{run_id}")
    async def platform_callback(run_id: str, request: Request) -> Response:
        """Authenticated Baseten webhook receiver.

        The signature is verified over the *raw* body and the request ID is
        resolved against the durable outbox, so an unknown or unsigned callback is
        rejected rather than trusted.  Delivery is only a notification: the Chain
        persists its own result, and a duplicate delivery is ignored.
        """

        if baseten_backend is None:
            raise HTTPException(409, {"reason": "no Baseten backend is configured", "missing": baseten_missing})
        raw = await _callback_body(request)
        try:
            callback, stored = baseten_backend.authenticate_and_store_callback(raw, dict(request.headers))
        except Exception as exc:
            LOGGER.warning("Rejected platform callback for %s: %s", run_id, exc)
            raise HTTPException(401, "Callback authentication failed")
        association = callback.association
        if association is not None and association.run_id != run_id:
            raise HTTPException(404, "Callback does not belong to this run")
        # A ``None`` association means the submission has not committed its
        # request ID locally yet.  The HMAC already proved authenticity, so the
        # callback is accepted and parked rather than discarded -- dropping it
        # would lose a real result that the platform will not send again.
        return Response(
            content=json.dumps(
                {
                    "accepted": True,
                    "duplicate": not stored,
                    "request_id": callback.request_id,
                    "association": "bound" if association is not None else "parked_pending_submission_commit",
                }
            ),
            media_type="application/json",
        )

    @app.post("/api/freeplay/step")
    def freeplay_step(body: FreeplayInput) -> dict:
        """Expand one direction into a constant action chunk and generate frames.

        No policy and no language model are in this loop: the keypress becomes an
        action vector directly.  When no certified world backend is configured
        this returns ``503`` with a named reason -- it never returns a placeholder
        frame, because the entire point of this beat is that the video is really
        being generated in response to the keypress.  A ``video_id`` path is
        stricter: it can use only a verified exact-source branch adapter and
        never falls back to this task-start path.
        """

        source: Optional[Dict[str, Any]] = None
        source_status: Optional[Dict[str, Any]] = None
        if body.video_id is not None:
            source = freeplay_source(body.video_id)
            if source is None:
                raise HTTPException(
                    409,
                    {
                        "available": False,
                        "mode": "recording_branch",
                        "reason": "No current saved recording matches this video_id, so no source-bound branch can be restored.",
                        "source": None,
                        "binding": {
                            "requested": True,
                            "exact_branch_supported": False,
                            "checkpoint": None,
                            "adapter": None,
                        },
                        "missing": ["a current hash-bound catalog recording matching video_id"],
                        "scored": False,
                        "qualified": False,
                    },
                )
            if body.source_sha256 != source.get("sha256"):
                raise HTTPException(
                    409,
                    {
                        "available": False,
                        "mode": "recording_branch",
                        "reason": "The selected recording's hash no longer matches source_sha256; reload its capability status before commanding it.",
                        "source": source_identity(source),
                        "binding": {
                            "requested": True,
                            "exact_branch_supported": False,
                            "checkpoint": None,
                            "adapter": None,
                        },
                        "missing": ["the current source_sha256 returned by /api/freeplay/status"],
                        "scored": False,
                        "qualified": False,
                    },
                )
            # This is a side-effect-free adapter capability check.  It occurs
            # before a session is created and, crucially, before any world-model
            # generation call. Unsupported MP4-only recordings never fall
            # through to generic task-start free-play.
            source_status = recording_branch_status(baseten_backend, source)
            if not source_status["available"]:
                raise HTTPException(409, source_status)

        session_id = body.session_id or uuid.uuid4().hex
        if not session_id.isalnum() or len(session_id) > 64:
            raise HTTPException(422, "Invalid session ID")
        requested_mode = "recording_branch" if source is not None else "task_start"
        source_binding = None
        if source is not None and source_status is not None:
            checkpoint = source_status["binding"].get("checkpoint") or {}
            source_binding = {
                "video_id": source["id"],
                "source_sha256": source["sha256"],
                "checkpoint_sha256": checkpoint.get("sha256"),
                "adapter": source_status["binding"].get("adapter"),
            }
        with lock:
            if len(freeplay_sessions) >= 256 and session_id not in freeplay_sessions:
                raise HTTPException(429, "Free-play session limit reached")
            session = freeplay_sessions.get(session_id)
            if session is None:
                session = {
                    "created_at": time.time(),
                    "steps": 0,
                    "frames": [],
                    "mode": requested_mode,
                    "source_binding": source_binding,
                    "busy": False,
                }
                freeplay_sessions[session_id] = session
            else:
                # Pre-source-binding sessions are legacy generic sessions. They
                # can continue their old task-start semantics, but cannot be
                # silently upgraded into a recording continuation.
                existing_mode = session.setdefault("mode", "task_start")
                existing_binding = session.setdefault("source_binding", None)
                session.setdefault("busy", False)
                if existing_mode != requested_mode or existing_binding != source_binding:
                    raise HTTPException(
                        409,
                        {
                            "reason": "A free-play session is permanently bound to its original task-start mode or exact recording checkpoint.",
                            "session_id": session_id,
                            "existing_mode": existing_mode,
                            "requested_mode": requested_mode,
                        },
                    )
            if session.get("source_branch_failure"):
                raise HTTPException(
                    409,
                    {
                        "reason": session["source_branch_failure"],
                        "session_id": session_id,
                    },
                )
            if session["busy"]:
                raise HTTPException(409, {"reason": "Free-play session already has a generation in flight.", "session_id": session_id})
        if body.direction == "stop":
            return {
                "session_id": session_id,
                "direction": "stop",
                "mode": "recording_branch" if source is not None else "task_start",
                "source": source_identity(source) if source is not None else None,
                "binding": source_status.get("binding") if source_status is not None else {
                    "requested": False,
                    "exact_branch_supported": False,
                    "checkpoint": None,
                    "adapter": None,
                },
                "generating": False,
                "frame_urls": list(session["frames"]),
                "frame_count": len(session["frames"]),
                "commanded_rows": 0,
                "qualified": False,
                "scored": False,
                "reason": "Release-to-stop: no action was commanded.",
            }
        chunk = _freeplay_chunk(body.direction, baseten_backend)
        if source is None and (baseten_backend is None or not hasattr(baseten_backend, "freeplay_step")):
            raise HTTPException(
                503,
                {
                    "reason": (
                        "Free-play needs a certified world-model backend. It is deliberately not "
                        "simulated locally: a placeholder frame would defeat the purpose of the beat."
                    ),
                    "missing": baseten_missing or ["a world-model free-play path on the configured backend"],
                    "commanded_chunk": {"rows": len(chunk), "direction": body.direction},
                },
            )
        # Free-play is unscored, but the Chain still requires a protocol hash on
        # the envelope, and an unscored frame should record which world
        # configuration produced it. Refuse with a named blocker rather than
        # letting the Chain answer with a bare 400.
        protocol_hash = document.sha256 if (document is not None and document.frozen) else None
        if not protocol_hash:
            raise HTTPException(
                503,
                {
                    "reason": (
                        "Free-play needs a frozen protocol. The world model is configured by it, so an "
                        "unscored frame still has to record which configuration produced it."
                    ),
                    "missing": [protocol_reason or "no frozen protocol.json is loaded"],
                    "commanded_chunk": {"rows": len(chunk), "direction": body.direction},
                },
            )
        with lock:
            # The first lock covers source/mode binding. This second, short
            # lock marks the session busy only after all non-generation
            # validation has passed; a second keypress can never overlap a
            # source-branch step or observe it as a completed frame set.
            if session["busy"]:
                raise HTTPException(409, {"reason": "Free-play session already has a generation in flight.", "session_id": session_id})
            session["busy"] = True
        started = time.perf_counter()
        try:
            seed = zlib.crc32(("%s:%d" % (session_id, session["steps"])).encode("utf-8"))
            if source is not None:
                outcome = source_branch_step(
                    baseten_backend,
                    session_id=session_id,
                    source=source,
                    actions=chunk,
                    protocol_hash=protocol_hash,
                    seed=seed,
                    resolution=480,
                )
            else:
                outcome = baseten_backend.freeplay_step(
                    session_id=session_id,
                    task=body.task,
                    actions=chunk,
                    protocol_hash=protocol_hash,
                    # Derived from the session and step so a free-play frame is
                    # reproducible, and so holding a key does not redraw the same
                    # noise every chunk. Not a protocol seed: nothing here is scored.
                    seed=seed,
                    resolution=480,
                )
            if not isinstance(outcome, Mapping):
                raise ValueError("world-model adapter returned a non-object result")
            frame_urls = outcome.get("frame_urls") or []
            if not isinstance(frame_urls, (list, tuple)) or not all(isinstance(url, str) and url for url in frame_urls):
                raise ValueError("world-model adapter returned invalid frame URLs")
            frame_urls = list(frame_urls)
            frame_count = int(outcome.get("frame_count") or 0)
            requested_resolution = outcome.get("requested_resolution")
            frame_height = outcome.get("frame_height")
            frame_width = outcome.get("frame_width")
        except Exception as exc:
            with lock:
                session["busy"] = False
                if source is not None:
                    # Once a source-bound request left this process, a transport
                    # failure cannot prove whether the remote world advanced.
                    # Preserve the exact-source contract by refusing blind
                    # retries under the same session/seed.
                    session["source_branch_failure"] = (
                        "The source-bound generation failed after dispatch; its remote state may be ambiguous, "
                        "so this exact-source session cannot be retried automatically."
                    )
            raise HTTPException(502, {"reason": "world-model generation failed", "detail": str(exc)})
        with lock:
            session["steps"] += 1
            session["frames"] = frame_urls
            session["busy"] = False
        return {
            "session_id": session_id,
            "direction": body.direction,
            "mode": "recording_branch" if source is not None else "task_start",
            "source": source_identity(source) if source is not None else None,
            "binding": source_status.get("binding") if source_status is not None else {
                "requested": False,
                "exact_branch_supported": False,
                "checkpoint": None,
                "adapter": None,
            },
            "generating": False,
            "frame_urls": frame_urls,
            "frame_count": frame_count,
            "commanded_rows": len(chunk),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            # What was asked for, and separately what the frames measurably are.
            # Reporting the request as though it were the result is how a 64px
            # rehearsal frame would end up captioned "480p" on stage.
            "requested_resolution": requested_resolution,
            "frame_height": frame_height,
            "frame_width": frame_width,
            "backend": "baseten",
            "protocol_hash": protocol_hash,
            "qualified": False,
            "scored": False,
            "reason": (
                "Unscored exact source-bound free-play. No policy, no language model, no validity gate, no judge."
                if source is not None
                else "Unscored task-start free-play. No policy, no language model, no validity gate, no judge."
            ),
        }

    @app.get("/api/annotation/packets")
    def annotation_packets() -> dict:
        """Blinded annotation packets, or an explicit reason none exist."""

        path = store.results / "calibration_panel.json"
        if not path.is_file():
            return {
                "packets": [],
                "status": "unavailable",
                "reason": "No frozen 150-clip calibration panel exists yet.",
            }
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return {"packets": [], "status": "error", "reason": str(exc)}
        return {
            "packets": payload.get("packets", []),
            "status": "available",
            "split_counts": payload.get("split_counts"),
            "blinded": True,
            "reason": None,
        }

    @app.post("/api/annotation/labels", status_code=201)
    def annotation_label(body: AnnotationInput) -> dict:
        """Record one blinded annotation.

        ``annotator_type`` is a closed enum recorded verbatim, and a model
        annotator must additionally supply its model id and revision -- so a
        model can never be stored as a human annotator.
        """

        if body.annotator_type == "model" and not (body.annotator_model and body.annotator_model_revision):
            raise HTTPException(422, "a model annotator must record its model id and revision")
        if body.completion_evidence == "met" and body.progress is not None and body.progress != 5:
            raise HTTPException(422, "completion_evidence='met' requires progress 5")
        if body.completion_evidence == "not_met" and body.progress == 5:
            raise HTTPException(422, "completion_evidence='not_met' cannot have progress 5")
        record = {**body.model_dump(), "recorded_at": time.time()}
        with lock:
            annotations.append(record)
        return {"recorded": True, "count": len(annotations), "annotator_type": body.annotator_type}

    @app.get("/api/annotation/labels")
    def annotation_labels() -> dict:
        with lock:
            rows = list(annotations)
        by_type: Dict[str, int] = {}
        for row in rows:
            by_type[row["annotator_type"]] = by_type.get(row["annotator_type"], 0) + 1
        return {
            "labels": rows,
            "count": len(rows),
            "by_annotator_type": by_type,
            "human_annotation_present": by_type.get("human", 0) > 0,
        }

    frontend = repo_root / "web" / "dist"
    if frontend.is_dir():
        @app.get("/review", include_in_schema=False)
        def development_review_page():
            index = frontend / "index.html"
            if not index.is_file():
                raise HTTPException(404, "Development review UI is not built")
            return FileResponse(index, media_type="text/html", headers={"X-Content-Type-Options": "nosniff"})

        # _SinglePageApp is what makes the other six routes survive a reload;
        # /review keeps its own handler above because that one also sets
        # nosniff, which the static mount does not.
        app.mount("/", _SinglePageApp(directory=str(frontend), html=True), name="dashboard")
    else:

        @app.get("/")
        def root_status() -> dict:
            return {
                "service": "Nightshift",
                "dashboard": "Run npm --prefix web install && npm --prefix web run build",
                "docs": "/docs",
            }

    return app


# --------------------------------------------------------------------- helpers


def _latest_sequence(service: RunService, run_id: str) -> int:
    try:
        events = service.list_events(run_id, after=0)
    except Exception:  # pragma: no cover - a missing run is handled by the caller
        return 0
    return max((int(event.get("sequence") or 0) for event in events), default=0)


def _terminal_tail_interval() -> float:
    """Short tail interval so a finished run closes its stream promptly."""

    return min(0.05, TELEMETRY_INTERVAL_SECONDS)


def _tile_rank(episode: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Prefer rows with media, then more segments, then terminal status."""

    has_media = 1 if (episode.get("video_url") or episode.get("frame_url")) else 0
    segments = int(episode.get("n_segments") or 0)
    terminal = 1 if episode.get("status") in ("completed", "failed", "cancelled") else 0
    return (has_media, segments, terminal)


def _tile_provenance(episode: Mapping[str, Any]) -> str:
    """Label what a tile is actually showing."""

    mode = str(episode.get("mode") or "")
    if "synthetic" in mode:
        return "qualitative"
    timing = episode.get("timing") or {}
    if isinstance(timing, Mapping) and timing.get("replayed_from_outbox"):
        return "replayed"
    if episode.get("status") in ("completed", "failed", "cancelled"):
        return "cached"
    return "live"


def _judge_revision(ledger: GateLedger) -> Optional[str]:
    record = ledger.records.get("D")
    if record is None:
        return None
    revisions = dict(record.judge_revisions or {})
    return revisions.get("model_revision") or revisions.get("revision") or None


def _backend_revision(ledger: GateLedger) -> Optional[str]:
    record = ledger.records.get("A")
    if record is None:
        return None
    revisions = dict(record.backend_revisions or {})
    return revisions.get("backend_profile_hash") or revisions.get("revision") or None


def _parity_status(ledger: GateLedger) -> str:
    record = ledger.records.get("C")
    if record is None or record.status is not GateStatus.PASS:
        return "unqualified"
    comparability = dict(record.measurements.get("comparability") or {})
    if not comparability:
        return "unqualified"
    if all(str(value) == "matched_provenance" for value in comparability.values()):
        return "matched_provenance"
    return "matched_distribution_with_limitations"


def _reference_cell(payload: Mapping[str, Any], source: str, policy: str, task: str) -> Dict[str, Any]:
    from plumb.reference import HUMAN, SIMPLER

    tables = payload.get("tables") if isinstance(payload.get("tables"), Mapping) else payload
    table = tables.get(source) if isinstance(tables, Mapping) else None
    if isinstance(table, Mapping):
        row = table.get(policy)
        if isinstance(row, Mapping) and isinstance(row.get(task), (int, float)):
            successes = int(row[task])
            return {"successes": successes, "rate": successes / 50.0}
    source_table = HUMAN if source == "human" else SIMPLER
    row = source_table.get(policy) or {}
    if task in row:
        return {"successes": int(row[task]), "rate": float(row[task]) / 50.0}
    return {"successes": None, "rate": None}


def _called_shot_estimate(
    service: RunService, policy: str, task: str
) -> Tuple[Optional[float], str, Optional[str]]:
    """Find a frozen real-model run's estimate for the called-shot cell."""

    for run in service.list_runs():
        config = dict(run.get("config") or {})
        if config.get("backend") != "baseten" or config.get("cohort") != "primary":
            continue
        if run.get("status") != "completed":
            continue
        run_id = run.get("id") or run.get("run_id")
        rows = [
            row
            for row in service.list_episodes(str(run_id), limit=10000)
            if str(row.get("policy")) == policy and str(row.get("task")) == task
        ]
        if not rows:
            continue
        positives = sum(
            1
            for row in rows
            if row.get("status") == "completed"
            and row.get("validity") == "valid"
            and row.get("binary_success") is True
        )
        return positives / float(len(rows)), "measured_observed_positive_rate", str(run_id)
    return None, "pending_no_frozen_real_model_run", None


def _six_clip_payload(store: ArtifactStore, revealed: bool) -> Dict[str, Any]:
    """Six clips for the opening beat, from a declared manifest only.

    The manifest must name which clip is real.  Without it this reports
    unavailable rather than picking six arbitrary files and guessing provenance.
    """

    path = store.results / "sixclip.json"
    if not path.is_file():
        return {
            "clips": [],
            "revealed": revealed,
            "status": "unavailable",
            "reason": (
                "No six-clip manifest exists. It must declare one real robot clip and five "
                "generated clips with their sources; provenance is never inferred from a filename."
            ),
            "audience_accuracy": None,
        }
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"clips": [], "revealed": revealed, "status": "error", "reason": str(exc)}
    clips = []
    for entry in payload.get("clips", []):
        item: Dict[str, Any] = {"id": entry.get("id"), "url": entry.get("url")}
        if revealed:
            item["provenance"] = entry.get("provenance")
            item["source"] = entry.get("source")
        else:
            item["provenance_sealed"] = True
        clips.append(item)
    return {
        "clips": clips,
        "revealed": revealed,
        "status": "available",
        "order_seed": payload.get("order_seed"),
        # Only report an accuracy figure if answers were actually collected.
        "audience_accuracy": payload.get("audience_accuracy"),
        "reason": None,
    }


def _freeplay_chunk(direction: str, backend: Any) -> List[List[float]]:
    """Expand a direction into a constant, clamped action chunk.

    One keypress commands the whole chunk, which is what makes the wait
    meaningful: the world model is inventing that much video from a single
    command.  Magnitudes stay inside the observed Bridge action ranges (xyz
    around +/-0.015), so free-play cannot command an implausible jump.
    """

    step = 0.015
    axes = {
        "right": (0, step),
        "left": (0, -step),
        "forward": (1, step),
        "back": (1, -step),
        "up": (2, step),
        "down": (2, -step),
    }
    rows = int(getattr(getattr(backend, "settings", None), "action_chunk_size", 16) or 16)
    index, magnitude = axes.get(direction, (0, 0.0))
    chunk: List[List[float]] = []
    for _ in range(rows):
        row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        row[index] = magnitude
        chunk.append(row)
    return chunk
