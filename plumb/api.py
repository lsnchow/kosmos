"""Local control plane. Real-model qualification is never inferred from fixtures."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from plumb.engine import RunService
from plumb.evidence import experiments_payload
from plumb.measurement import analyze
from plumb.reference import reference_payload
from plumb.ledger import CallbackConflictError
from plumb.outbox import BasetenOutbox
from plumb.platform import BasetenChainClient, BasetenPlatformConfig, CallbackVerificationError, PlatformSchemaError

POLICIES = ["OpenVLA", "OpenPiZero", "Octo", "MiniVLA", "SuSIE", "SuSIE_LL"]
TASKS = ["open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth"]
GATE_DESCRIPTIONS = {
    "A": "Pinned model/backend conformance and measured GPU inference",
    "B": "Action grounding, native feedback, and state conversion",
    "C": "Matched real starting states for all five tasks",
    "D": "Blinded human judge calibration",
    "E": "Primary measurement and independent cost confirmation",
    "F": "Three fresh full-scale latency/cost rehearsals",
}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "synthetic"
    backend: str = "synthetic"
    policies: List[str] = Field(default_factory=lambda: list(POLICIES))
    tasks: List[str] = Field(default_factory=lambda: list(TASKS))
    starts_per_task: int = Field(default=50, ge=1, le=500)
    seed: int = Field(default=20260919, ge=0, le=2**31 - 1)
    idempotency_key: Optional[str] = Field(default=None, max_length=200)


class FreeplayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: Optional[str] = None
    action: List[float] = Field(min_length=7, max_length=7)


def create_app(data_dir: Optional[Path] = None, *, baseten_client: Optional[BasetenChainClient] = None) -> FastAPI:
    root = (data_dir or Path(os.environ.get("PLUMB_DATA_DIR", "data"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    service = RunService(root)
    if baseten_client is None and os.environ.get("PLUMB_ENABLE_BASETEN_CALLBACKS") == "1":
        config = BasetenPlatformConfig.from_env()
        if not config.webhook_secret:
            raise ValueError("PLUMB_ENABLE_BASETEN_CALLBACKS requires BASETEN_WEBHOOK_SECRET")
        baseten_client = BasetenChainClient(config)
    outbox = BasetenOutbox(service.ledger, baseten_client) if baseten_client is not None else None
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="plumb-run")
    submitted: set = set()
    lock = threading.RLock()
    freeplay: Dict[str, Dict[str, float]] = {}
    @asynccontextmanager
    async def lifespan(application):
        yield
        pool.shutdown(wait=True)

    app = FastAPI(title="PLUMB", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.data_dir = root
    app.state.baseten_outbox = outbox

    def require_run(run_id: str) -> dict:
        try:
            result = service.get_run(run_id)
        except (KeyError, ValueError):
            raise HTTPException(404, "Run not found")
        if result is None:
            raise HTTPException(404, "Run not found")
        return result

    def gates() -> dict:
        # Evidence-only gate results may be supplied by the cluster smoke tooling.
        path = root / "gates.json"
        saved: Any = {}
        if path.is_file():
            try:
                saved = json.loads(path.read_text())
            except (ValueError, OSError):
                saved = {}
        if isinstance(saved, dict):
            saved = saved.get("gates", saved)
        if isinstance(saved, list):
            saved = {str(g.get("id")): g for g in saved if isinstance(g, dict)}
        if not isinstance(saved, dict):
            saved = {}
        values = []
        for key, description in GATE_DESCRIPTIONS.items():
            evidence = saved.get(key, {})
            if not isinstance(evidence, dict):
                evidence = {}
            values.append({"id": key, "description": description,
                           "status": evidence.get("status", "not_run"),
                           "summary": evidence.get("reason", description),
                           "evidence": evidence})
        return {"gates": values, "qualified": False,
                "reason": "Full qualification requires reviewed, protocol-bound evidence; fixture runs cannot qualify."}

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": "0.1.0", "qualified": False,
                "available_backends": ["synthetic"], "data_dir": str(root)}

    @app.get("/api/protocol")
    def protocol() -> dict:
        return {"policies": POLICIES, "tasks": TASKS, "starts_per_task": 50,
                "total": 1500, "reference": reference_payload(),
                "target": {"episodes": 1500, "seconds": 60, "usd": 11.25,
                           "status": "unmeasured"},
                "mode": "synthetic", "qualified": False}

    @app.get("/api/gates")
    def gate_report() -> dict:
        return gates()

    @app.get("/api/sweeps")
    def sweeps() -> dict:
        return {"points": [], "status": "unavailable",
                "reason": "No independent real-model cost-fidelity sweep has completed."}

    @app.get("/api/experiments")
    def experiments() -> dict:
        return experiments_payload(root)

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
    async def platform_callback(request: Request) -> dict:
        if outbox is None:
            raise HTTPException(503, "Baseten callback ingress is not configured")
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1024 * 1024:
                raise HTTPException(413, "Callback exceeds the 1 MiB limit")
            chunks.append(chunk)
        try:
            receipt = outbox.ingest_callback(b"".join(chunks), dict(request.headers))
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
        if config.mode != "synthetic" or config.backend != "synthetic":
            raise HTTPException(409, {"reason": "Real-policy runs require qualified adapters and scenario manifests.", "gates": gates()})
        if not config.policies or not config.tasks:
            raise HTTPException(422, "At least one policy and task is required")
        if len(set(config.policies)) != len(config.policies) or len(set(config.tasks)) != len(config.tasks):
            raise HTTPException(422, "Duplicate policies/tasks are not allowed")
        if set(config.policies) - set(POLICIES) or set(config.tasks) - set(TASKS):
            raise HTTPException(422, "Unknown policy or task")
        try:
            result = service.create_run(config.model_dump(exclude_none=True))
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
                        logging.getLogger("plumb").error(
                            "Run execution failed for %s", run_id,
                            exc_info=(type(error), error, error.__traceback__))
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

    @app.get("/api/runs/{run_id}/analysis")
    def analysis(run_id: str) -> dict:
        run = require_run(run_id)
        result = analyze(service.list_episodes(run_id, limit=10000), run.get("config"))
        result.update({"mode": run.get("mode", "synthetic"), "qualified": False})
        return result

    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id: str):
        require_run(run_id)
        rows = service.list_episodes(run_id, limit=10000)
        return StreamingResponse((json.dumps(row, allow_nan=False) + "\n" for row in rows),
                                 media_type="application/x-ndjson",
                                 headers={"Content-Disposition": 'attachment; filename="episodes.jsonl"'})

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict:
        require_run(run_id)
        return service.cancel_run(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request) -> StreamingResponse:
        require_run(run_id)

        async def snapshots():
            while not await request.is_disconnected():
                run = require_run(run_id)
                rows = service.list_episodes(run_id, limit=10000)
                visible = [row for row in rows if row.get("status") not in {"planned", "pending"}][-12:]
                data = {"run": run, "episodes": visible,
                        "telemetry": {"source": "application_ledger", "platform_queue": None,
                                      "active_replicas": None, "estimated_usd": None,
                                      "mode": run.get("mode", "synthetic"), "timestamp": time.time()}}
                yield "event: snapshot\ndata: " + json.dumps(data, allow_nan=False) + "\n\n"
                if run.get("status") in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(snapshots(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/artifacts/{artifact_path:path}")
    def artifact(artifact_path: str):
        path = (root / artifact_path).resolve()
        if root not in path.parents or not path.is_file() or path.suffix.lower() not in {".svg", ".png", ".jpg", ".jpeg", ".mp4", ".json", ".jsonl"}:
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})

    @app.post("/api/freeplay/step")
    def freeplay_step(body: FreeplayInput) -> dict:
        import math
        started = time.perf_counter()
        if not all(math.isfinite(x) for x in body.action):
            raise HTTPException(422, "Actions must be finite")
        session_id = body.session_id or uuid.uuid4().hex
        if not session_id.isalnum() or len(session_id) > 64:
            raise HTTPException(422, "Invalid session ID")
        with lock:
            if len(freeplay) >= 1000 and session_id not in freeplay:
                raise HTTPException(429, "Free-play session limit reached")
            state = freeplay.setdefault(session_id, {"x": 0.5, "y": 0.5, "z": 0.5, "gripper": 0.5})
            for index, key in enumerate(["x", "y", "z"]):
                state[key] = max(0.0, min(1.0, state[key] + max(-0.03, min(0.03, body.action[index]))))
            state["gripper"] = max(0.0, min(1.0, body.action[6]))
            output = dict(state)
        return {"session_id": session_id, "mode": "synthetic", "state": output,
                "qualified": False, "latency_ms": (time.perf_counter() - started) * 1000,
                "reason": "Bounded UI fixture only; no learned world dynamics or robot hardware."}

    frontend = Path(__file__).resolve().parents[1] / "web" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="dashboard")
    else:
        @app.get("/")
        def root_status() -> dict:
            return {"service": "PLUMB", "dashboard": "Run npm --prefix web install && npm --prefix web run build", "docs": "/docs"}
    return app
