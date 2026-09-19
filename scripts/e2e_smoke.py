#!/usr/bin/env python3
"""End-to-end readiness check for PLUMB.

The point of this script is to answer one question honestly: **is anything
broken, and what is merely waiting on hardware?**  Those are different failures
and conflating them is how a team walks on stage believing it is ready.

Every check reports one of three outcomes:

``pass``
    Verified now.
``pending``
    Correct but not yet satisfiable, because it needs a GPU, Baseten
    credentials, human annotation, or a gate that has not been run.  A pending
    check names its blocker.
``fail``
    Something is actually wrong in the code or the artifacts.

Exit code is non-zero only when a check ``fail``s.  Pass ``--require-qualified``
for the final pre-demo run, which additionally turns every ``pending`` into a
failure -- that is the mode to run once the GPU work is done.

    python scripts/e2e_smoke.py
    python scripts/e2e_smoke.py --require-qualified
    python scripts/e2e_smoke.py --json > readiness.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Importable whether invoked as `python scripts/e2e_smoke.py` or from elsewhere.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PASS = "pass"
PENDING = "pending"
FAIL = "fail"

#: Phrases BUILD-SPEC.md forbids.  Each is matched as a whole word/phrase so that
#: legitimate uses ("we could not find a published version") do not trip it.
NEVER_SAY = (
    r"\bSOTA\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bsolves\b",
    r"\bmatches human performance\b",
    r"\bfirst ever\b",
    r"\bworld[- ]first\b",
    r"\btime to first token\b",
    r"\btokens? per second\b",
    r"\btokens/sec\b",
    r"\bTTFT\b",
)

#: Files that legitimately quote the forbidden phrases in order to forbid them:
#: the specs that define the list, this checker, and any test asserting absence.
NEVER_SAY_EXEMPT = {
    "BUILD-SPEC.md",
    "AGENT-BUILD-SPEC.md",
    "SCRIPT.md",
    "PASS-OFF.md",
    "HANDOFF.md",
    "scripts/e2e_smoke.py",
}

#: Filename markers for test sources.  A test that checks a phrase never reaches
#: the UI has to name the phrase, so tests are scanned for nothing.
TEST_FILE_MARKERS = (".test.", ".spec.", "/tests/", "test_")


@dataclass
class Check:
    name: str
    status: str
    detail: str
    blocker: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "blocker": self.blocker,
            "data": self.data,
        }


class Report:
    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(
        self,
        name: str,
        status: str,
        detail: str,
        blocker: Optional[str] = None,
        **data: Any,
    ) -> Check:
        check = Check(name=name, status=status, detail=detail, blocker=blocker, data=dict(data))
        self.checks.append(check)
        return check

    def run(self, name: str, function: Callable[[], Tuple[str, str, Optional[str], Dict[str, Any]]]) -> Check:
        """Run a check, turning an unexpected exception into a ``fail``.

        A check that crashes is a real problem, so it is never reported as
        pending.
        """

        try:
            status, detail, blocker, data = function()
        except Exception as exc:  # noqa: BLE001 - a crashing check is a failure
            return self.add(name, FAIL, "check raised %s: %s" % (type(exc).__name__, exc))
        return self.add(name, status, detail, blocker, **data)

    def counts(self) -> Dict[str, int]:
        result = {PASS: 0, PENDING: 0, FAIL: 0}
        for check in self.checks:
            result[check.status] = result.get(check.status, 0) + 1
        return result


# --------------------------------------------------------------------- checks


def check_imports() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Every production module must import on a CPU-only machine."""

    modules = [
        "plumb.api",
        "plumb.artifacts",
        "plumb.backends.baseten",
        "plumb.calibration",
        "plumb.engine",
        "plumb.evidence",
        "plumb.gates",
        "plumb.ledger",
        "plumb.measurement",
        "plumb.platform",
        "plumb.protocol",
        "plumb.records",
        "plumb.reference",
        "plumb.rollout",
        "deploy.baseten.chain",
    ]
    optional = ["plumb.sweeps", "plumb.scenarios", "plumb.validity", "plumb.annotation"]
    failures: List[str] = []
    missing: List[str] = []
    import importlib

    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append("%s: %s" % (name, exc))
    for name in optional:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            missing.append(name)
        except Exception as exc:  # noqa: BLE001
            failures.append("%s: %s" % (name, exc))
    if failures:
        return FAIL, "modules failed to import", None, {"failures": failures}
    if missing:
        return (
            PENDING,
            "core modules import; %d optional module(s) absent" % len(missing),
            "not yet implemented: " + ", ".join(missing),
            {"missing": missing},
        )
    return PASS, "all %d production modules import on CPU" % len(modules + optional), None, {}


def check_reference_tables() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """The published tables must match spec section 4 exactly."""

    from plumb.reference import HUMAN, SIMPLER

    expected_human = {
        "OpenVLA": {"open_drawer": 40, "close_drawer": 46, "to_basket": 1, "to_sink": 0, "fold_cloth": 12},
        "OpenPiZero": {"open_drawer": 24, "close_drawer": 45, "to_basket": 7, "to_sink": 47, "fold_cloth": 3},
        "Octo": {"open_drawer": 0, "close_drawer": 0, "to_basket": 0, "to_sink": 0, "fold_cloth": 2},
        "MiniVLA": {"open_drawer": 32, "close_drawer": 49, "to_basket": 38, "to_sink": 0, "fold_cloth": 8},
        "SuSIE": {"open_drawer": 2, "close_drawer": 13, "to_basket": 0, "to_sink": 0, "fold_cloth": 10},
        "SuSIE_LL": {"open_drawer": 0, "close_drawer": 0, "to_basket": 0, "to_sink": 0, "fold_cloth": 0},
    }
    expected_simpler = {
        "OpenVLA": {"open_drawer": 32, "close_drawer": 2, "to_basket": 1, "to_sink": 0},
        "OpenPiZero": {"open_drawer": 34, "close_drawer": 24, "to_basket": 45, "to_sink": 6},
        "Octo": {"open_drawer": 3, "close_drawer": 0, "to_basket": 6, "to_sink": 3},
        "MiniVLA": {"open_drawer": 30, "close_drawer": 23, "to_basket": 10, "to_sink": 2},
        "SuSIE": {"open_drawer": 0, "close_drawer": 41, "to_basket": 7, "to_sink": 0},
        "SuSIE_LL": {"open_drawer": 1, "close_drawer": 0, "to_basket": 0, "to_sink": 0},
    }
    problems: List[str] = []
    if dict(HUMAN) != expected_human:
        problems.append("HUMAN table does not match spec section 4")
    if dict(SIMPLER) != expected_simpler:
        problems.append("SIMPLER table does not match spec section 4")
    if any("fold_cloth" in row for row in SIMPLER.values()):
        problems.append("SIMPLER must not have a fold_cloth column")
    totals = {policy: sum(row.values()) for policy, row in HUMAN.items()}
    for policy, expected in (("Octo", 2), ("MiniVLA", 127), ("OpenPiZero", 126)):
        if totals[policy] != expected:
            problems.append("%s total is %d, expected %d" % (policy, totals[policy], expected))
    # Four of five leave-one-task-out comparisons must reverse MiniVLA/OpenPiZero.
    reversals = [
        task
        for task in ("open_drawer", "close_drawer", "to_basket", "to_sink", "fold_cloth")
        if (totals["MiniVLA"] - HUMAN["MiniVLA"][task]) < (totals["OpenPiZero"] - HUMAN["OpenPiZero"][task])
    ]
    if len(reversals) != 4:
        problems.append("expected 4 leave-one-task-out reversals, found %d" % len(reversals))
    if problems:
        return FAIL, "; ".join(problems), None, {"problems": problems}
    return (
        PASS,
        "both tables exact; Octo 2/250, MiniVLA 127/250, OpenPiZero 126/250, 4-of-5 reversal",
        None,
        {"reversal_tasks": reversals},
    )


def check_task_prompts() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Task prompts are verbatim, including the lowercase 'fold'."""

    from plumb.gates import EXACT_TASK_PROMPTS

    expected = {
        "close_drawer": "Close the drawer",
        "open_drawer": "Open the drawer",
        "to_basket": "Put the eggplant in the yellow basket",
        "to_sink": "Put the eggplant in the blue sink",
        "fold_cloth": "fold the cloth from top right to bottom left",
    }
    if dict(EXACT_TASK_PROMPTS) != expected:
        return FAIL, "task prompts do not match spec section 3 verbatim", None, {"actual": dict(EXACT_TASK_PROMPTS)}
    if not EXACT_TASK_PROMPTS["fold_cloth"].startswith("fold"):
        return FAIL, "task 5 must keep its lowercase 'fold'", None, {}
    return PASS, "all five prompts verbatim, including lowercase 'fold'", None, {}


def check_horizons() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Horizons are 70/70/100/100/80, and 70 is not reachable by multiples of 4."""

    from plumb.protocol import TASK_HORIZONS

    expected = {"open_drawer": 70, "close_drawer": 70, "to_basket": 100, "to_sink": 100, "fold_cloth": 80}
    if dict(TASK_HORIZONS) != expected:
        return FAIL, "task horizons do not match spec section 3", None, {"actual": dict(TASK_HORIZONS)}
    note = (
        "70 mod 4 == 2, so the drawer tasks need a certified non-multiple-of-4 action length "
        "or a terminal-padding certificate"
    )
    return PASS, "horizons exact; " + note, None, {}


def check_protocol_and_preregistration(require: bool) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    from plumb.protocol import ProtocolDocument, default_protocol

    path = REPO_ROOT / "protocol.json"
    if not path.is_file():
        return (
            PENDING,
            "no protocol.json has been frozen",
            "run: plumb freeze-protocol",
            {"expected_path": str(path)},
        )
    document = ProtocolDocument.load(path)
    if not document.frozen:
        return PENDING, "protocol.json exists but is not frozen", "run: plumb freeze-protocol", {}
    if document.total_planned_episodes != 1500:
        return (
            FAIL,
            "frozen protocol plans %d episodes, not 1500" % document.total_planned_episodes,
            None,
            {},
        )
    record = document.preregistration_status(repo_root=REPO_ROOT)
    data = {"protocol_hash": document.sha256, "preregistration": record.to_mapping()}
    if record.status == "unregistered" or record.blocking_reasons():
        return (
            PENDING,
            "protocol frozen but preregistration is not verified: %s" % "; ".join(record.blocking_reasons()),
            "git tag -s %s and push it; a local hash is not preregistration"
            % document.preregistration_tag,
            data,
        )
    # A default protocol must not be what got registered by accident.
    if document.sha256 == default_protocol(protocol_id=document.protocol_id).sha256:
        return FAIL, "the registered hash equals an unfrozen default protocol", None, data
    return PASS, "protocol frozen and preregistered (%s)" % record.status, None, data


def check_asset_lock() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    from plumb.protocol import AssetLock, build_asset_lock

    path = REPO_ROOT / "assets.lock.json"
    if not path.is_file():
        return PENDING, "no assets.lock.json", "run: plumb assets-lock", {}
    lock = AssetLock.load(path)
    plan_ids = set(build_asset_lock().assets)
    missing_rows = sorted(plan_ids - set(lock.assets))
    if missing_rows:
        return FAIL, "asset lock is missing planned rows", None, {"missing_rows": missing_rows}
    unresolved = list(lock.unresolved_asset_ids)
    if unresolved:
        return (
            PENDING,
            "%d of %d assets unresolved" % (len(unresolved), len(lock.assets)),
            "download and hash the assets; unresolved rows block Gate A",
            {"unresolved": unresolved},
        )
    return PASS, "all %d assets verified with hashes and licenses" % len(lock.assets), None, {}


def check_gates() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    from plumb.artifacts import ArtifactStore
    from plumb.gates import GATE_IDS, GateLedger, GateStatus

    store = ArtifactStore(REPO_ROOT)
    if not store.gates_path.is_file():
        return PENDING, "results/gates.json does not exist", "no gate has been run", {}
    ledger = GateLedger.load(str(store.gates_path))
    statuses = {gate_id: ledger.records[gate_id].status.value for gate_id in GATE_IDS}
    evidence_errors: Dict[str, List[str]] = {}
    for gate_id in GATE_IDS:
        errors = list(ledger.records[gate_id].pass_evidence_errors())
        if errors:
            evidence_errors[gate_id] = errors
    if evidence_errors:
        return (
            FAIL,
            "a gate claims a pass without the evidence its own contract requires",
            None,
            {"statuses": statuses, "evidence_errors": evidence_errors},
        )
    not_run = [gate_id for gate_id in GATE_IDS if ledger.records[gate_id].status is GateStatus.NOT_RUN]
    failed = [gate_id for gate_id in GATE_IDS if ledger.records[gate_id].status is GateStatus.FAIL]
    if failed:
        return (
            PENDING,
            "gate(s) %s failed" % ", ".join(failed),
            "a failed gate keeps its feature in scope and reports its blocker",
            {"statuses": statuses},
        )
    if not_run:
        return (
            PENDING,
            "gate(s) %s not run" % ", ".join(not_run),
            "requires GPU execution on the deployed Chain",
            {"statuses": statuses},
        )
    return PASS, "all six gates carry passing, self-consistent evidence", None, {"statuses": statuses}


def check_artifacts(require: bool) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    from plumb.artifacts import ArtifactStore

    store = ArtifactStore(REPO_ROOT)
    inventory = store.inventory()
    missing = list(inventory["missing"])
    if not missing:
        return PASS, "all spec section 8 artifacts present", None, {"count": len(inventory["artifacts"])}
    return (
        PENDING,
        "%d of %d section 8 artifacts missing" % (len(missing), len(inventory["artifacts"])),
        "produced by a real run: plumb publish <run_id>",
        {"missing": missing},
    )


def check_synthetic_run_and_measurement() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Exercise the whole local pipeline and assert the invariants that matter."""

    from plumb.artifacts import ArtifactStore
    from plumb.engine import RunService
    from plumb.measurement import analyze, validate_source_lineage_leakage
    from plumb.protocol import analysis_options_for_run, default_protocol, freeze

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        document = freeze(default_protocol(starts_per_task=4))
        service = RunService(root / "data")
        run = service.create_run(
            {
                "mode": "synthetic",
                "backend": "synthetic",
                "starts_per_task": 4,
                "seed": 4242,
                "cohort": "primary",
                "protocol_hash": document.sha256,
            }
        )
        run_id = run["id"]
        service.execute_run(run_id)
        rows = service.list_episodes(run_id, limit=10000)
        run = service.get_run(run_id)

        problems: List[str] = []
        expected = 6 * 5 * 4
        if len(rows) != expected:
            problems.append("expected %d planned episodes, got %d" % (expected, len(rows)))
        if any(row.get("cohort") != "primary" for row in rows):
            problems.append("an episode lost its cohort identity")
        if any(row.get("protocol_hash") != document.sha256 for row in rows):
            problems.append("an episode lost its protocol hash")

        options = analysis_options_for_run(document, rows)
        options["mdd"]["gap_grid"] = [0.1, 0.3]
        analysis = analyze(rows, options)

        if analysis["analysis_identity"]["status"] != "accepted":
            problems.append(
                "a frozen protocol must yield an accepted primary analysis, got %s"
                % analysis["analysis_identity"]["status"]
            )
        for key in ("reliability", "mmrv", "advanced_inference", "mdd"):
            if analysis[key].get("status") not in ("computed", "available"):
                problems.append("%s did not compute: %s" % (key, analysis[key].get("status")))

        # Horowitz-Manski width must equal the missing-outcome rate exactly.
        for cell in analysis["cells"]:
            width = cell.get("missing_bounds_width")
            missingness = cell.get("missingness_rate")
            if width is not None and missingness is not None and abs(width - missingness) > 1e-9:
                problems.append("bound width != missingness for %s/%s" % (cell["policy"], cell["task"]))
                break

        leakage = validate_source_lineage_leakage(rows)
        if leakage.get("status") not in ("clean", "ok", "pass", "no_conflicts"):
            problems.append("lineage leakage check reported %s" % leakage.get("status"))

        # No episode may appear in both halves of a reliability split.
        reliability = analysis["reliability"]
        splits = reliability.get("splits") or reliability.get("split_membership") or []
        for split in splits if isinstance(splits, list) else []:
            if not isinstance(split, Mapping):
                continue
            first = set(split.get("first_half") or [])
            second = set(split.get("second_half") or [])
            if first & second:
                problems.append("a reliability split reused IDs across halves")
                break

        store = ArtifactStore(root)
        store.write_per_episode(run_id, rows, run)
        store.write_leaderboard(analysis, run, {})
        store.write_reliability(analysis)
        store.write_exclusions(analysis)
        text = store.leaderboard_path.read_text()
        if "| OpenVLA | 40 | 46 | 1 | 0 | 12 | 99 |" not in text:
            problems.append("leaderboard did not reproduce the published human row exactly")

        written = [json.loads(line) for line in store.per_episode_path(run_id).read_text().splitlines()]
        if len(written) != expected:
            problems.append("per-episode export dropped rows")
        if any(row.get("estimated_usd") == 0 for row in written):
            problems.append("an unmeasured cost was written as zero instead of null")

    if problems:
        return FAIL, "; ".join(problems), None, {"problems": problems}
    return (
        PASS,
        "synthetic 120-episode run: identities preserved, all four headline statistics computed, "
        "bounds identity holds, artifacts written",
        None,
        {"episodes": expected},
    )


def check_determinism() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """The same seed must reproduce the same outcome, byte for byte."""

    from plumb.engine import RunService
    from plumb.records import deterministic_seed

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        outcomes = []
        for attempt in range(2):
            service = RunService(root / ("data-%d" % attempt))
            run = service.create_run(
                {"mode": "synthetic", "backend": "synthetic", "starts_per_task": 2, "seed": 99}
            )
            service.execute_run(run["id"])
            rows = sorted(
                service.list_episodes(run["id"], limit=10000),
                key=lambda row: (row["policy"], row["task"], row["start_id"]),
            )
            outcomes.append(
                [(row["policy"], row["task"], row["start_id"], row["world_seed"], row["binary_success"]) for row in rows]
            )
        if outcomes[0] != outcomes[1]:
            return FAIL, "the same seed produced different outcomes", None, {}
        first = deterministic_seed(99, "OpenVLA", "close_drawer", "start-000")
        second = deterministic_seed(99, "OpenVLA", "close_drawer", "start-000")
        if first != second:
            return FAIL, "deterministic_seed is not deterministic", None, {}
    return PASS, "identical seeds reproduce identical outcomes and world seeds", None, {}


def check_backend_fail_closed() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """A gated backend must refuse to run without passing gates."""

    from fastapi.testclient import TestClient

    from plumb.api import create_app

    with tempfile.TemporaryDirectory() as temporary:
        with TestClient(create_app(Path(temporary) / "data")) as client:
            health = client.get("/api/health").json()
            registered = {entry["name"]: entry for entry in health["registered_backends"]}
            if "baseten" not in registered:
                return FAIL, "the baseten backend is not registered at all", None, {}
            response = client.post(
                "/api/runs", json={"mode": "qualification", "backend": "baseten", "starts_per_task": 1}
            )
            if response.status_code == 201:
                return (
                    PASS,
                    "the baseten backend is configured and its gates pass",
                    None,
                    {"available": True},
                )
            if response.status_code != 409:
                return FAIL, "expected 409 or 201 from a gated backend, got %d" % response.status_code, None, {}
            detail = response.json()["detail"]
            if not detail.get("blocking_reasons"):
                return FAIL, "a refusal must name its blockers", None, {}
            if detail["gates"]["qualified"] is not False:
                return FAIL, "a blocked backend must not report qualified", None, {}
            # A qualified claim must be impossible to request directly.
            if client.post("/api/runs", json={"mode": "qualified"}).status_code != 422:
                return FAIL, "'qualified' must not be a settable mode", None, {}
            return (
                PENDING,
                "baseten backend correctly refuses with named blockers",
                "; ".join(detail["blocking_reasons"][:4]),
                {"blocking_reasons": detail["blocking_reasons"]},
            )


def check_freeplay_refuses_placeholders() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Free-play must never return a fake frame."""

    from fastapi.testclient import TestClient

    from plumb.api import create_app

    with tempfile.TemporaryDirectory() as temporary:
        with TestClient(create_app(Path(temporary) / "data")) as client:
            response = client.post("/api/freeplay/step", json={"direction": "right"})
            if response.status_code == 200:
                body = response.json()
                if not body.get("frame_urls"):
                    return FAIL, "free-play returned 200 with no frames", None, {}
                return PASS, "free-play generated %d frames" % body["frame_count"], None, {}
            if response.status_code != 503:
                return FAIL, "expected 503 or 200 from free-play, got %d" % response.status_code, None, {}
            detail = response.json()["detail"]
            if detail.get("commanded_chunk", {}).get("rows", 0) < 4:
                return FAIL, "one keypress must expand to a whole action chunk", None, {}
            return (
                PENDING,
                "free-play refuses to fabricate a frame and names its blocker",
                "; ".join(detail.get("missing") or []),
                {"commanded_rows": detail["commanded_chunk"]["rows"]},
            )


def check_telemetry_never_fabricated() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Unavailable telemetry must be null with a status, never zero."""

    from fastapi.testclient import TestClient

    from plumb.api import create_app

    with tempfile.TemporaryDirectory() as temporary:
        with TestClient(create_app(Path(temporary) / "data")) as client:
            run = client.post(
                "/api/runs", json={"mode": "synthetic", "backend": "synthetic", "starts_per_task": 1}
            ).json()
            telemetry = client.get("/api/runs/%s/telemetry" % run["id"]).json()
            problems = []
            for field_name in ("platform_queue", "active_replicas", "desired_replicas", "marginal_estimated_usd"):
                if telemetry.get(field_name) == 0:
                    problems.append("%s was fabricated as zero" % field_name)
            for field_name in ("platform_queue_status", "replicas_status", "cost_status"):
                if not telemetry.get(field_name):
                    problems.append("%s has no availability status" % field_name)
            sweeps = client.get("/api/sweeps").json()
            if sweeps["points"] and sweeps["status"] != "available":
                problems.append("sweeps reported points without being available")
            if not sweeps["points"] and not sweeps.get("reason"):
                problems.append("an empty sweep must say why it is empty")
            if problems:
                return FAIL, "; ".join(problems), None, {"problems": problems}
            return (
                PASS,
                "absent telemetry is null with an explicit availability status, never zero",
                None,
                {"cost_status": telemetry["cost_status"]},
            )


def check_never_say() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """The forbidden phrases must not reach any user-visible surface."""

    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in NEVER_SAY]
    roots = [REPO_ROOT / "plumb", REPO_ROOT / "web" / "src", REPO_ROOT / "deploy", REPO_ROOT / "README.md"]
    dist = REPO_ROOT / "web" / "dist"
    if dist.is_dir():
        roots.append(dist)
    hits: List[Dict[str, Any]] = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix.lower() not in (".py", ".ts", ".tsx", ".js", ".css", ".html", ".md", ".json"):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative in NEVER_SAY_EXEMPT:
                continue
            if any(marker in relative for marker in TEST_FILE_MARKERS):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in patterns:
                for match in pattern.finditer(text):
                    line = text[: match.start()].count("\n") + 1
                    hits.append({"file": relative, "line": line, "phrase": match.group(0)})
    if hits:
        return FAIL, "%d forbidden phrase(s) found" % len(hits), None, {"hits": hits[:20]}
    return PASS, "no forbidden phrase in any shipped surface", None, {}


def check_python_tests() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    tail = (completed.stdout or completed.stderr).strip().splitlines()[-1:] or [""]
    if completed.returncode != 0:
        return FAIL, "pytest failed: %s" % tail[0], None, {"tail": tail}
    return PASS, "pytest: %s" % tail[0], None, {}


def check_static_names() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    import importlib.util

    if importlib.util.find_spec("pyflakes") is None:
        return PENDING, "pyflakes is not installed", "pip install -e '.[test]'", {}
    targets = ["plumb", "deploy", "cluster", "reverse_validation.py", "scripts"]
    completed = subprocess.run(
        [sys.executable, "-m", "pyflakes", *targets],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    undefined = [
        line
        for line in (completed.stdout or "").splitlines()
        if "undefined name" in line or "syntax" in line.lower()
    ]
    if undefined:
        return FAIL, "%d undefined-name/syntax problem(s)" % len(undefined), None, {"lines": undefined[:20]}
    return PASS, "no undefined names or syntax errors in production sources", None, {}


def check_frontend_build() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    web = REPO_ROOT / "web"
    if not (web / "node_modules").is_dir():
        return PENDING, "web/node_modules is absent", "run: npm --prefix web install", {}
    completed = subprocess.run(
        ["npm", "--prefix", str(web), "run", "build"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-6:]
        return FAIL, "frontend build failed", None, {"tail": tail}
    if not (web / "dist" / "index.html").is_file():
        return FAIL, "build succeeded but web/dist/index.html is absent", None, {}
    return PASS, "frontend builds and web/dist is servable", None, {}


def check_frontend_tests() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    web = REPO_ROOT / "web"
    if not (web / "node_modules").is_dir():
        return PENDING, "web/node_modules is absent", "run: npm --prefix web install", {}
    completed = subprocess.run(
        ["npm", "--prefix", str(web), "run", "test"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-8:]
        return FAIL, "frontend tests failed", None, {"tail": tail}
    return PASS, "frontend tests pass", None, {}


def check_chain_deployable() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """The Chain template must import and declare real per-stage compute."""

    import deploy.baseten.chain as chain

    contracts = REPO_ROOT / "deploy" / "baseten" / "model-contracts.json"
    data: Dict[str, Any] = {"chains_runtime_available": chain.CHAINS_RUNTIME_AVAILABLE}
    if not contracts.is_file():
        return FAIL, "deploy/baseten/model-contracts.json is missing", None, data
    payload = json.loads(contracts.read_text())
    nulls = _count_nulls(payload)
    data["unresolved_contract_fields"] = nulls
    if not chain.CHAINS_RUNTIME_AVAILABLE:
        # truss pins pydantic/fastapi and must not destabilise the app venv, so
        # it lives in a separate deploy venv (see docs/GO_LIVE.md step 2). Check
        # there before reporting it absent.
        deploy_python = REPO_ROOT / ".venv-deploy" / "bin" / "python"
        if deploy_python.is_file():
            probe = subprocess.run(
                [
                    str(deploy_python),
                    "-c",
                    "import truss; import deploy.baseten.chain; from truss_chains import framework; "
                    "framework.raise_validation_errors(); print(truss.__version__)",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if probe.returncode == 0:
                version = (probe.stdout or "").strip().splitlines()[-1:] or ["unknown"]
                data["truss_version"] = version[0]
                data["truss_venv"] = str(deploy_python.parent.parent)
                return (
                    PASS,
                    "chain definition passed the truss %s SDK validator; deployment/runtime remain unverified"
                    % version[0],
                    None,
                    data,
                )
            return FAIL, "Chain SDK validation failed", None, {
                **data, "validator_output": (probe.stdout + probe.stderr)[-4000:],
            }
        return (
            PENDING,
            "chain template imports; truss_chains is not installed",
            "python -m venv .venv-deploy && .venv-deploy/bin/pip install truss==0.18.30",
            data,
        )
    try:
        from truss_chains import framework
        framework.raise_validation_errors()
    except Exception as error:
        return FAIL, "Chain SDK validation failed", None, {**data, "error": str(error)}
    return PASS, "chain definition passed SDK validation; deployment/runtime remain unverified", None, data


def check_chain_contract() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """The submission payload must satisfy the Chain's own pydantic contract.

    This is the check that catches a field-name drift between
    ``plumb/backends/baseten.py`` and ``deploy/baseten/chain.py`` -- a class of bug
    that silently nulls every measurement rather than failing.
    """

    import tempfile as _tempfile

    from plumb.backends.baseten import BasetenBackendSettings, BasetenChainBackend
    from plumb.rehearsal import _validate_against_chain_contract, chain_contract_available
    from plumb.starts import RehearsalStartResolver

    if not chain_contract_available():
        return FAIL, "the Chain request models could not be imported", None, {}
    with _tempfile.TemporaryDirectory() as temporary:
        backend = BasetenChainBackend(
            settings=BasetenBackendSettings(
                webhook_endpoint="https://plumb.example.com/api/callbacks",
                operating_point_id="op-256-30",
            ),
            data_dir=Path(temporary),
            start_resolver=RehearsalStartResolver(256),
        )
        episode = {
            "run_id": "run-1",
            "episode_id": "ep-1",
            "policy": "OpenVLA",
            "policy_variant": "OpenVLA",
            "task": "close_drawer",
            "start_id": "start-000",
            "start_lineage_id": "lineage-000",
            "world_seed": 4242,
            "horizon_actions": 70,
            "protocol_hash": "sha256:" + "a" * 64,
            "seeds": {"world": 4242, "policy": 7, "judge_samples": [1, 2, 3, 4, 5]},
        }
        payload = backend._build_entrypoint_input(episode, {}, "OpenVLA", episode["protocol_hash"])
        violation = _validate_against_chain_contract(payload)
    if violation is not None:
        return FAIL, "the submission payload violates the Chain contract", None, {"violation": violation}
    return PASS, "the submission payload validates against the Chain's own models", None, {}


def check_production_rehearsal() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Run the whole production path against a simulated Chain."""

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "rehearse.py"), "--starts", "1",
         "--policies", "OpenVLA", "--port", "8791", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    text = completed.stdout or completed.stderr
    try:
        start = text.index("{")
        findings = json.loads(text[start:])
    except (ValueError, json.JSONDecodeError):
        return FAIL, "the rehearsal produced no parseable report", None, {"tail": text[-800:]}
    problems = findings.get("problems") or []
    if completed.returncode != 0 or problems:
        return FAIL, "the production-path rehearsal found problems", None, {"problems": problems}
    return (
        PASS,
        "production path verified against a simulated Chain: %s episodes, %s webhooks, reconciler clean"
        % (findings.get("planned"), (findings.get("simulated_chain") or {}).get("webhooks_sent")),
        None,
        {
            "elapsed_seconds": findings.get("elapsed_seconds"),
            "fast_analysis_seconds": findings.get("fast_analysis_seconds"),
            "publish_seconds": findings.get("publish_seconds"),
        },
    )


def check_reverse_validation() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    from reverse_validation import build_demonstration, run_sweep

    checkpoints, provenance = build_demonstration(checkpoints=6, items=80, seed=5)
    report = run_sweep(checkpoints, repetitions=15, seed=5)
    conclusion = report["conclusion"]
    if not conclusion["mean_aggregate_loss_correlation_invariant"]:
        return FAIL, "the loss correlation moved under scrambling; the mechanism is wrong", None, {}
    if provenance["kind"] != "analytic_demonstration":
        return FAIL, "the demonstration must be labelled analytic, not empirical", None, {}
    return (
        PASS,
        "loss correlation exactly invariant (shift %s) while item agreement fell %.3f -> %.3f"
        % (
            conclusion["mean_aggregate_correlation_shift_at_full_scramble"],
            conclusion["item_agreement_unscrambled"],
            conclusion["item_agreement_at_full_scramble"],
        ),
        None,
        {"baseline": report["baseline_loss_correlation"]["mean"]},
    )


def check_human_annotation() -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """Gate D's human requirement is tracked, never quietly satisfied."""

    from plumb.artifacts import ArtifactStore

    store = ArtifactStore(REPO_ROOT)
    path = store.judge_calibration_path
    if not path.is_file():
        return (
            PENDING,
            "no judge calibration report",
            "Gate D needs a frozen 150-clip panel and blinded annotation",
            {},
        )
    payload = json.loads(path.read_text())
    calibration_class = str(payload.get("calibration_class") or "unknown")
    if "human" not in calibration_class:
        return (
            PENDING,
            "calibration exists but its class is %r" % calibration_class,
            "two blinded human annotators are still an open dependency",
            {"calibration_class": calibration_class},
        )
    return PASS, "human calibration recorded (%s)" % calibration_class, None, {}


def _count_nulls(payload: Any, prefix: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = "%s.%s" % (prefix, key) if prefix else str(key)
            if value is None:
                found.append(path)
            else:
                found.extend(_count_nulls(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found.extend(_count_nulls(value, "%s[%d]" % (prefix, index)))
    return found


# ----------------------------------------------------------------------- main


def build_report(require_qualified: bool, skip_slow: bool) -> Report:
    report = Report()
    report.run("imports", check_imports)
    report.run("static_names", check_static_names)
    report.run("reference_tables", check_reference_tables)
    report.run("task_prompts", check_task_prompts)
    report.run("task_horizons", check_horizons)
    report.run("protocol_and_preregistration", lambda: check_protocol_and_preregistration(require_qualified))
    report.run("asset_lock", check_asset_lock)
    report.run("gates", check_gates)
    report.run("section_8_artifacts", lambda: check_artifacts(require_qualified))
    report.run("synthetic_run_and_measurement", check_synthetic_run_and_measurement)
    report.run("determinism", check_determinism)
    report.run("backend_fail_closed", check_backend_fail_closed)
    report.run("freeplay_refuses_placeholders", check_freeplay_refuses_placeholders)
    report.run("telemetry_never_fabricated", check_telemetry_never_fabricated)
    report.run("chain_deployable", check_chain_deployable)
    report.run("chain_contract", check_chain_contract)
    report.run("reverse_validation", check_reverse_validation)
    report.run("human_annotation", check_human_annotation)
    report.run("never_say_list", check_never_say)
    if not skip_slow:
        report.run("production_rehearsal", check_production_rehearsal)
        report.run("python_tests", check_python_tests)
        report.run("frontend_build", check_frontend_build)
        report.run("frontend_tests", check_frontend_tests)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--require-qualified",
        action="store_true",
        help="Treat every pending check as a failure. Use for the final pre-demo run.",
    )
    parser.add_argument("--skip-slow", action="store_true", help="Skip pytest and the frontend build/tests.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    report = build_report(args.require_qualified, args.skip_slow)
    counts = report.counts()
    failed = counts[FAIL]
    pending = counts[PENDING]

    payload = {
        "schema_version": 1,
        "require_qualified": args.require_qualified,
        "counts": counts,
        "checks": [check.to_mapping() for check in report.checks],
        "verdict": (
            "broken"
            if failed
            else ("ready" if not pending else "ready_pending_external_dependencies")
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        symbols = {PASS: "  ok  ", PENDING: "pending", FAIL: " FAIL "}
        print("PLUMB readiness\n" + "=" * 78)
        for check in report.checks:
            print("[%s] %-32s %s" % (symbols[check.status], check.name, check.detail))
            if check.blocker:
                print("%s        blocker: %s" % (" " * 9, check.blocker))
        print("=" * 78)
        print("pass %d   pending %d   fail %d" % (counts[PASS], pending, failed))
        print("verdict: %s" % payload["verdict"])
        if failed:
            print("\nSomething is broken. Fix the FAIL lines before anything else.")
        elif pending:
            print(
                "\nNothing is broken. %d check(s) wait on external dependencies (GPU, Baseten\n"
                "credentials, human annotation). Re-run with --require-qualified once the GPU\n"
                "work is done to confirm demo readiness." % pending
            )
        else:
            print("\nFully qualified.")

    if failed:
        return 1
    if args.require_qualified and pending:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
