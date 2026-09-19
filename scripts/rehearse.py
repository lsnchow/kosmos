#!/usr/bin/env python3
"""Drive a full production-path rehearsal against a simulated Chain.

This exists to answer one question without a GPU: **does the entire production
path work end to end?**  It seeds the gate ledger with rehearsal evidence, starts
the server in rehearsal mode, dispatches a real run through the real Baseten code
path, waits for the real HMAC-signed callbacks, runs the reconciler over the
deliberately-dropped ones, publishes the run-derived section 8 artifacts, and reports
what happened.

Nothing it produces is a scientific result.  Every record carries
``transport="simulated"``, the gate evidence is marked ``rehearsal``, and Gate F
cannot pass on it.  What it does prove is that the plumbing, the dashboard feed,
the reconciler, the cost ledger and the artifact writers work for the exercised
cases. Real deployment, resource limits and model behavior remain unverified.

    python scripts/rehearse.py --starts 2
    python scripts/rehearse.py --starts 50 --keep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REHEARSAL_SECRET = "rehearsal-secret-do-not-use-in-production"


def _request(url: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310: localhost
            return json.loads(response.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError("%s %s -> %d %s" % (method, url, error.code, detail[:400])) from error


def seed_rehearsal_gates(results_dir: Path, protocol_hash: str) -> Path:
    """Write unrun gate records explicitly marked as rehearsal, not qualification.

    A configured simulated transport can execute independently of model gates.
    Its metadata must never become passed evidence after switching to a real
    transport in the same results directory.
    """

    from plumb.artifacts import ArtifactStore
    from plumb.gates import GateLedger, GateRecord, GateStatus

    store = ArtifactStore(results_dir)
    ledger = GateLedger.new(protocol_hash)
    evidence_uri = "plumb://rehearsal-transport/gate-evidence"

    ledger.record(
        GateRecord(
            gate_id="A",
            status=GateStatus.NOT_RUN,
            protocol_hash=protocol_hash,
            fixture_ids=("rehearsal-vendor-fixture",),
            evidence_uris=(evidence_uri,),
            thresholds={"frame_count_exact": True, "rehearsal": True},
            evidence_kind="rehearsal_transport",
            backend_revisions={"backend_profile_hash": "rehearsal", "revision": "rehearsal"},
            measurements={
                "backend_calls": 1,
                "wall_seconds": 0.01,
                "gpu_peak_memory_bytes": 1,
                "rehearsal": True,
                "certified_action_lengths": [16],
            },
            reasons=("rehearsal evidence: exercises the integration path, certifies no hardware",),
        )
    )
    ledger.record(
        GateRecord(
            gate_id="B",
            status=GateStatus.NOT_RUN,
            protocol_hash=protocol_hash,
            fixture_ids=("rehearsal-bridge-fixture",),
            evidence_uris=(evidence_uri,),
            thresholds={"max_suffix_invariance_mae": 0.25, "rehearsal": True},
            evidence_kind="rehearsal_transport",
            measurements={
                "interventions": {
                    "original": 1,
                    "zero": 1,
                    "temporally_permuted": 1,
                    "sign_reversed": 1,
                    "cross_episode": 1,
                },
                "suffix_causality": {"invariant": None, "rehearsal": True},
                "feedback_mode": "unqualified",
                "rehearsal": True,
            },
            reasons=("rehearsal evidence: no action fidelity is established",),
        )
    )
    ledger.record(
        GateRecord(
            gate_id="C",
            status=GateStatus.NOT_RUN,
            protocol_hash=protocol_hash,
            start_ids=tuple("rehearsal-start-%03d" % index for index in range(50)),
            evidence_uris=(evidence_uri,),
            thresholds={"min_starts_per_task": 50, "rehearsal": True},
            evidence_kind="rehearsal_transport",
            measurements={
                "starts_per_task": {
                    "open_drawer": 50,
                    "close_drawer": 50,
                    "to_basket": 50,
                    "to_sink": 50,
                    "fold_cloth": 50,
                },
                "comparability": {task: "rehearsal_placeholder" for task in
                                  ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")},
                "rehearsal": True,
            },
            reasons=("rehearsal evidence: no scenario parity is established",),
        )
    )
    store.write_gates(ledger)
    return store.gates_path


def wait_for(url: str, attempts: int = 60, delay: float = 0.25) -> None:
    for _ in range(attempts):
        try:
            _request(url, timeout=2.0)
            return
        except Exception:
            time.sleep(delay)
    raise RuntimeError("server did not become ready at %s" % url)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--starts", type=int, default=2, help="Starts per task. 50 is the full matrix.")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--policies", default=None, help="Comma-separated subset; default is all six.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep", action="store_true", help="Keep the rehearsal data directory.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from plumb.protocol import default_protocol, freeze

    # Each invocation owns a new directory; --keep must not cause a later run
    # to overwrite its protocol, logs, receipts, or artifacts.
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="rehearsal-run-", dir=str(REPO_ROOT / "data")))
    data_dir = workspace / "data"
    results_dir = workspace / "results-root"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    document = freeze(default_protocol(starts_per_task=args.starts, protocol_id="protocol-rehearsal"))
    protocol_path = data_dir / "protocol.json"
    document.save(protocol_path)
    gates_path = seed_rehearsal_gates(results_dir, document.sha256)

    base = "http://127.0.0.1:%d" % args.port
    environment = dict(os.environ)
    environment.update(
        {
            "PLUMB_DATA_DIR": str(data_dir),
            "PLUMB_RESULTS_DIR": str(results_dir),
            "PLUMB_REHEARSAL_CHAIN": "1",
            "BASETEN_WEBHOOK_SECRET": REHEARSAL_SECRET,
            "PLUMB_WEBHOOK_ENDPOINT": "%s/api/callbacks" % base,
            "PLUMB_PORT": str(args.port),
            "PLUMB_MAX_WORKERS": "16",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )

    # Log to a file, never a pipe. An undrained pipe fills at 64 KB and the
    # server then blocks on write, which looks exactly like a crash except that
    # the process is still alive and no longer accepting connections.
    log_path = workspace / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", "plumb.api:create_app", "--host", "127.0.0.1",
         "--port", str(args.port), "--log-level", "warning"],
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    findings: Dict[str, Any] = {"protocol_hash": document.sha256, "gates_path": str(gates_path), "workspace": str(workspace)}

    def server_output() -> str:
        try:
            return log_path.read_text(encoding="utf-8", errors="ignore") or "(no output)"
        except OSError:
            return "(unreadable)"

    try:
        wait_for("%s/api/health" % base)
        health = _request("%s/api/health" % base)
        findings["health"] = {
            "available_backends": health["available_backends"],
            "transport": health.get("transport"),
            "simulated_transport": health.get("simulated_transport"),
            "protocol_frozen": health.get("protocol_frozen"),
        }
        if "baseten" not in health["available_backends"]:
            registered = {item["name"]: item for item in health["registered_backends"]}
            raise RuntimeError(
                "the baseten backend is still blocked: %s" % registered["baseten"]["blocking_reasons"]
            )
        if not health.get("simulated_transport"):
            raise RuntimeError("rehearsal mode did not engage; refusing to run against a real Chain")

        policies = args.policies.split(",") if args.policies else None
        payload: Dict[str, Any] = {
            "mode": "qualification",
            "backend": "baseten",
            "cohort": "primary",
            "starts_per_task": args.starts,
            "seed": 20260919,
            "idempotency_key": "rehearsal-%d-%s" % (args.starts, document.sha256[-8:]),
        }
        if policies:
            payload["policies"] = policies

        started = time.monotonic()
        run = _request("%s/api/runs" % base, method="POST", payload=payload)
        run_id = run["id"]
        findings["run_id"] = run_id
        findings["planned"] = run.get("total")

        # Idempotency: the same key must return the same run, not a second one.
        again = _request("%s/api/runs" % base, method="POST", payload=payload)
        findings["idempotent_resubmission"] = again["id"] == run_id

        deadline = time.monotonic() + args.timeout
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("the server exited mid-run:\n%s" % server_output()[-4000:])
            try:
                last = _request("%s/api/runs/%s" % (base, run_id))
            except Exception as exc:
                raise RuntimeError(
                    "lost contact with the server (%s). Server log tail:\n%s" % (exc, server_output()[-4000:])
                ) from exc
            if last.get("status") in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.5)
        findings["elapsed_seconds"] = round(time.monotonic() - started, 3)
        findings["run_status"] = last.get("status")
        findings["counts"] = {
            key: last.get(key) for key in ("total", "completed", "failed", "cancelled") if key in last
        }

        telemetry = _request("%s/api/runs/%s/telemetry" % (base, run_id), timeout=60.0)
        findings["telemetry"] = {
            key: telemetry.get(key)
            for key in (
                "platform_queue_status",
                "platform_queue",
                "replicas_status",
                "cost_status",
                "reconciliation_status",
                "outbox_counts",
                "source",
            )
        }

        rehearsal = _request("%s/api/rehearsal" % base)
        findings["simulated_chain"] = rehearsal.get("stats")
        findings["dropped_webhooks"] = (rehearsal.get("stats") or {}).get("webhooks_dropped")
        findings["duplicate_webhooks"] = (rehearsal.get("stats") or {}).get("webhooks_duplicated")

        reconcile = _request("%s/api/runs/%s/reconcile" % (base, run_id), method="POST", timeout=300.0)
        findings["reconciler"] = {
            "resolved": len(reconcile.get("resolved") or []),
            "unresolved": len(reconcile.get("unresolved") or []),
            "counts": reconcile.get("counts"),
        }

        fast_started = time.monotonic()
        analysis = _request("%s/api/runs/%s/analysis" % (base, run_id), timeout=120.0)
        findings["fast_analysis_seconds"] = round(time.monotonic() - fast_started, 3)
        findings["analysis"] = {
            "identity": analysis["analysis_identity"]["status"],
            "reliability": analysis["reliability"].get("status"),
            "mmrv": analysis["mmrv"].get("status"),
            "advanced_inference": analysis["advanced_inference"].get("status"),
            "mdd": analysis["mdd"].get("status"),
            "cells": len(analysis["cells"]),
        }

        publish_started = time.monotonic()
        published = _request(
            "%s/api/runs/%s/publish" % (base, run_id), method="POST", timeout=1800.0
        )
        findings["publish_seconds"] = round(time.monotonic() - publish_started, 3)
        findings["artifacts_written"] = sorted(published["receipts"])
        findings["artifact_inventory_missing"] = published["inventory"]["missing"]

        episodes = _request("%s/api/runs/%s/episodes" % (base, run_id), timeout=120.0)["episodes"]
        # Transport-level failures never produced video, so they legitimately
        # carry no measurements. The response-shape checks below apply only to
        # episodes the Chain actually generated and returned.
        transport_failures = ("service_failure", "deadline_exceeded", "ambiguous_submission",
                              "awaiting_reconciliation", "callback_carried_no_result")
        generated = [
            row
            for row in episodes
            if not any(marker in str(row.get("missing_reason") or "") for marker in transport_failures)
        ]
        findings["episodes"] = {
            "total": len(episodes),
            "marked_simulated_transport": sum(
                1 for row in episodes if (row.get("world_identity") or {}).get("simulated_transport")
            ),
            "terminal": sum(1 for row in episodes if row.get("status") in ("completed", "failed", "cancelled")),
            "evaluable": sum(1 for row in episodes if row.get("binary_success") is not None),
            "unevaluable": sum(1 for row in episodes if row.get("binary_success") is None),
            "unqualified_feedback_mode": all(row.get("feedback_mode") == "unqualified" for row in episodes),
            "carry_protocol_hash": all(row.get("protocol_hash") == document.sha256 for row in episodes),
            "platform_request_ids_present": sum(1 for row in episodes if row.get("platform_request_ids")),
            # A service failure legitimately has no measurements, so the
            # response-shape checks below are scoped to episodes the Chain
            # actually generated.
            "generated": len(generated),
            "segments_recorded": sum(1 for row in generated if (row.get("n_segments") or 0) > 0),
            "gpu_seconds_recorded": sum(1 for row in generated if row.get("compute_gpu_seconds") is not None),
            "timing_recorded": sum(
                1 for row in generated if (row.get("timing") or {}).get("world_seconds") is not None
            ),
        }
        findings["gates_after"] = {
            item["id"]: item["status"] for item in _request("%s/api/gates" % base)["gates"]
        }
        findings["still_unqualified"] = _request("%s/api/gates" % base)["qualified"] is False
    except Exception as exc:
        findings["error"] = str(exc)
        findings["server_log_tail"] = server_output()[-3000:]
        with (workspace / "rehearsal-report.json").open("x", encoding="utf-8") as report:
            json.dump(findings, report, indent=2)
        print(json.dumps(findings, indent=2))
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log_handle.close()

    problems: List[str] = []
    if findings.get("run_status") != "completed":
        problems.append("run did not complete: %s" % findings.get("run_status"))
    if not findings.get("idempotent_resubmission"):
        problems.append("a repeated submission created a second run")
    counts = findings.get("counts") or {}
    if counts.get("total") and counts.get("total") != (counts.get("completed", 0) + counts.get("failed", 0)):
        problems.append("not every planned episode reached a terminal state")
    simulated = findings.get("simulated_chain") or {}
    if counts.get("failed") != simulated.get("service_failures"):
        problems.append("logical failures do not match injected terminal service failures")
    if counts.get("completed") != simulated.get("completed"):
        problems.append("logical completions do not match successful simulated executions")
    if simulated.get("submitted") != counts.get("total"):
        problems.append("submission count differs from planned episodes (missing or repeated POST)")
    if simulated.get("completion_errors"):
        problems.append("simulated executions raised unexpected errors")
    if (findings.get("reconciler") or {}).get("unresolved"):
        problems.append("dropped callbacks left unresolved submissions")
    outbox = (findings.get("telemetry") or {}).get("outbox_counts") or {}
    if outbox.get("failed", 0) != simulated.get("service_failures", 0):
        problems.append("outbox failures differ from injected failures; callback recovery incomplete")
    episodes = findings.get("episodes") or {}
    if episodes.get("total") != episodes.get("terminal"):
        problems.append("terminal episode count does not equal the planned count")
    if not episodes.get("unqualified_feedback_mode"):
        problems.append("a rehearsal episode claimed a qualified feedback mode")
    if episodes.get("marked_simulated_transport") != episodes.get("total"):
        problems.append(
            "only %s of %s episodes carry the simulated-transport marker; a rehearsal row must be "
            "impossible to mistake for a real one"
            % (episodes.get("marked_simulated_transport"), episodes.get("total"))
        )
    if episodes.get("platform_request_ids_present") != episodes.get("total"):
        problems.append("not every episode persisted its platform request ID")
    # These three catch a field-name mismatch between the Chain's response shape
    # and the backend's reader, which would otherwise null every measurement.
    generated_count = episodes.get("generated")
    if generated_count:
        if episodes.get("segments_recorded") != generated_count:
            problems.append("segment counts did not survive the Chain response shape")
        if episodes.get("gpu_seconds_recorded") != generated_count:
            problems.append("GPU-seconds did not survive the Chain response shape")
        if episodes.get("timing_recorded") != generated_count:
            problems.append("per-stage timings did not survive the Chain response shape")
    if not episodes.get("carry_protocol_hash"):
        problems.append("an episode lost its protocol hash")
    if findings.get("analysis", {}).get("identity") != "accepted":
        problems.append("the analysis manifest was not accepted")
    if not findings.get("still_unqualified"):
        problems.append("a rehearsal made the system report itself qualified")
    if any(value != "not_run" for value in findings.get("gates_after", {}).values()):
        problems.append("a rehearsal changed a model qualification gate")
    findings["problems"] = problems
    findings["verdict"] = "production path verified (simulated Chain)" if not problems else "problems found"
    with (workspace / "rehearsal-report.json").open("x", encoding="utf-8") as report:
        json.dump(findings, report, indent=2)

    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        print("\nPLUMB production-path rehearsal")
        print("=" * 78)
        print(json.dumps(findings, indent=2))
        print("=" * 78)
        print(findings["verdict"])
        print(
            "\nThis exercised the real submission envelope, durable outbox, HMAC callback\n"
            "authentication, duplicate/lost-callback reconciliation, telemetry, cost ledger,\n"
            "measurement and all artifact writers. The Chain was SIMULATED: no model ran, and\n"
            "no latency or cost figure here is a measurement."
        )
    if not args.keep:
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
