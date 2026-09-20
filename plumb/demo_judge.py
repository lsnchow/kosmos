"""Durable saved-clip assessments using the experimental semantic Qwen adapter.

This service is deliberately small and separate from the rollout ledger.  It
only accepts one server-resolved saved clip, materialises its exact 16 frames,
and invokes one explicitly configured private Baseten model through the local
CLI profile.  It cannot fall back to a base model, another adapter, a browser
URL, or an automatic retry.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from .policies.judge import parse_rubric_json
from .policies.provenance import image_pixel_hash
from .policies.tasks import BENCHMARK_TASK_REGISTRY
from .policies.demo_tasks import POT_TASK_ID, POT_INSTRUCTION, POT_RUBRIC_HASH, CUSTOM_TASK_ID, custom_task_rubric
from .policies.provenance import canonical_json_sha256
from .records import canonical_json, utc_now
from .world_videos import world_videos_payload


DEMO_PROFILE_ID = "semantic_pilot_epoch_02"
DEMO_TASK_ID = "close_drawer"
DEMO_TASK_LABEL = "Close the drawer"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SEMANTIC_ADAPTER_TREE_SHA256 = "sha256:7a0b7796a0ac8b78fbb8ba8f4c4db01fe8d35c2edc3cb7a0fbeec9f4fb20523e"
SEMANTIC_ADAPTER_PATH = "/scratch/lchow432/plumb/experiments/judge-lora-pilot-v1-trained/adapters/epoch-02"
FRAME_COUNT = 16
SAMPLE_COUNT = 5
QUORUM = 3
MAX_VIDEO_FRAMES = 600
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_CLI_RESPONSE_BYTES = 4 * 1024 * 1024
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class DemoJudgeError(RuntimeError):
    """A public, non-secret error for the experimental judge path."""


class DemoJudgeUnavailable(DemoJudgeError):
    """The selected trained-adapter endpoint is not ready for a submission."""


class DemoJudgeInputError(DemoJudgeError):
    """A saved clip does not meet the frozen input contract."""


class DemoJudgmentInput(BaseModel):
    """The browser may name a catalog clip, never a filesystem path or URL."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1, max_length=200)
    profile: str = Field(default=DEMO_PROFILE_ID, pattern="^semantic_pilot_epoch_02$")
    idempotency_key: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True)
class DemoJudgeConfig:
    model_id: str
    deployment_id: str
    cli_profile: str
    timeout_seconds: float = 600.0
    budget_cap_usd: float = 100.0
    worker_url: Optional[str] = None
    worker_token_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["DemoJudgeConfig"]:
        values = {
            "model_id": os.environ.get("PLUMB_DEMO_JUDGE_MODEL_ID"),
            "deployment_id": os.environ.get("PLUMB_DEMO_JUDGE_DEPLOYMENT_ID"),
            "cli_profile": os.environ.get("PLUMB_DEMO_JUDGE_PROFILE"),
        }
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            return None
        try:
            timeout = float(os.environ.get("PLUMB_DEMO_JUDGE_TIMEOUT_SECONDS", "600"))
            budget = float(os.environ.get("PLUMB_DEMO_JUDGE_BUDGET_CAP_USD", "100"))
        except ValueError as error:
            raise DemoJudgeError("judge timeout and budget cap must be numeric") from error
        if not math.isfinite(timeout) or not 1 <= timeout <= 600:
            raise DemoJudgeError("judge timeout must be from 1 through 600 seconds")
        if not math.isfinite(budget) or budget <= 0 or budget > 100:
            raise DemoJudgeError("judge budget cap must be greater than zero and at most $100")
        return cls(
            model_id=str(values["model_id"]).strip(),
            deployment_id=str(values["deployment_id"]).strip(),
            cli_profile=str(values["cli_profile"]).strip(),
            timeout_seconds=timeout,
            budget_cap_usd=budget,
            worker_url=os.environ.get("PLUMB_DEMO_JUDGE_WORKER_URL"),
            worker_token_file=os.environ.get("PLUMB_DEMO_JUDGE_WORKER_TOKEN_FILE"),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=str))


def _normalise_sha(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.lower().removeprefix("sha256:")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return "sha256:" + text


def _tree_sha256(directory: Path) -> str:
    """The immutable pilot's recorded directory-hash algorithm."""

    records: List[Dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DemoJudgeError("semantic adapter contains a symlink")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not records:
        raise DemoJudgeError("semantic adapter tree has no regular files")
    return "sha256:" + hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _under(root: Path, value: str, *, file: bool = True) -> Path:
    path = (root / value).resolve()
    if root not in path.parents or (not path.is_file() if file else not path.is_dir()) or path.is_symlink():
        raise DemoJudgeInputError("saved clip artifact is unavailable")
    return path


def _frame_indices(frame_count: int) -> Tuple[int, ...]:
    if frame_count < FRAME_COUNT:
        raise DemoJudgeInputError(
            "This saved clip has fewer than 16 distinct decoded frames. Choose a longer saved clip."
        )
    if frame_count > MAX_VIDEO_FRAMES:
        raise DemoJudgeInputError("This saved clip exceeds the assessment frame safety limit.")
    indices = tuple((position * (frame_count - 1)) // (FRAME_COUNT - 1) for position in range(FRAME_COUNT))
    if len(set(indices)) != FRAME_COUNT or tuple(sorted(indices)) != indices:
        raise DemoJudgeInputError("Could not derive 16 ordered distinct frame indexes from this saved clip.")
    return indices


def _probe_video(path: Path) -> Tuple[int, float]:
    """Read real decoded-frame count and FPS; container metadata alone is not enough."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, timeout=30)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams") if isinstance(payload, Mapping) else None
        stream = streams[0] if isinstance(streams, list) and streams and isinstance(streams[0], Mapping) else {}
        frame_count = int(stream.get("nb_read_frames"))
        numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise DemoJudgeInputError("Could not verify the complete saved video frame sequence.") from error
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        raise DemoJudgeInputError("Saved clip frame metadata is invalid.")
    return frame_count, fps


def _extract_frames(video: Path, indices: Sequence[int], destination: Path) -> List[Path]:
    # `n` is the decoded frame ordinal. The expression selects only the frozen
    # manifest indexes, in-order, rather than seeking and accidentally repeating
    # a nearby keyframe.
    expression = "+".join("eq(n\\,%d)" % index for index in indices)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        "select=" + expression,
        "-fps_mode",
        "passthrough",
        "-start_number",
        "0",
        str(destination / "frame-%02d.png"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=90)
    except (OSError, subprocess.SubprocessError) as error:
        raise DemoJudgeInputError("Could not decode the selected saved-video frames.") from error
    frames = [destination / ("frame-%02d.png" % index) for index in range(FRAME_COUNT)]
    if any(not frame.is_file() or frame.is_symlink() or frame.stat().st_size > MAX_FRAME_BYTES for frame in frames):
        raise DemoJudgeInputError("Saved-video sampling did not produce exactly 16 usable frames.")
    return frames


def _derive_seeds(material: str) -> Tuple[int, ...]:
    values: List[int] = []
    counter = 0
    while len(values) < SAMPLE_COUNT:
        digest = hashlib.sha256(("plumb-demo-judge-v1:" + material + ":" + str(counter)).encode("utf-8")).digest()
        # Baseten CLI JSON output passes numbers through floating-point JSON.
        # Keep seeds exactly representable across Python, Go and browser JSON.
        value = int.from_bytes(digest[:8], "big") & ((1 << 31) - 1)
        if value not in values:
            values.append(value)
        counter += 1
    return tuple(values)


def _latest_parsed(sample: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    attempts = sample.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise DemoJudgeError("judge response has an invalid bounded-attempt record")
    final = attempts[-1]
    if not isinstance(final, Mapping):
        raise DemoJudgeError("judge response has an invalid attempt")
    raw = final.get("raw_output")
    parsed = final.get("parsed")
    if parsed is None:
        return None
    if not isinstance(raw, str) or not isinstance(parsed, Mapping):
        raise DemoJudgeError("judge response has inconsistent raw and parsed sample data")
    # Parsing the retained raw text locally prevents a remote summary from
    # turning malformed model text into a displayable assessment.
    reparsed = parse_rubric_json(raw).as_dict()
    if dict(parsed) != dict(reparsed):
        raise DemoJudgeError("judge response parsed fields do not match its raw output")
    return reparsed


def _assessment_from_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    sampling = report.get("sampling")
    expected_sampling = {
        "sample_count": SAMPLE_COUNT,
        "quorum": QUORUM,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_new_tokens": 512,
        "retries_per_sample": 1,
    }
    if not isinstance(sampling, Mapping) or any(sampling.get(key) != value for key, value in expected_sampling.items()):
        raise DemoJudgeError("judge response sampling configuration differs from the frozen protocol")
    raw_samples = report.get("raw_judge_samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != SAMPLE_COUNT:
        raise DemoJudgeError("judge response does not retain five raw samples")
    parsed_samples: List[Mapping[str, Any]] = []
    for expected_index, sample in enumerate(raw_samples):
        if not isinstance(sample, Mapping) or sample.get("sample_index") != expected_index:
            raise DemoJudgeError("judge response sample ordering is invalid")
        parsed = _latest_parsed(sample)
        if parsed is not None:
            parsed_samples.append(parsed)
    decisive: Dict[bool, List[Mapping[str, Any]]] = {True: [], False: []}
    for sample in parsed_samples:
        integrity = sample.get("integrity")
        progress = sample.get("progress")
        completion = sample.get("completion_evidence")
        if integrity == "intact" and isinstance(progress, int) and not isinstance(progress, bool) and 0 <= progress <= 5 and completion in {"met", "not_met"}:
            decisive[completion == "met"].append(sample)
    winner: Optional[bool] = None
    votes: Sequence[Mapping[str, Any]] = ()
    for result in (True, False):
        if len(decisive[result]) >= QUORUM:
            winner, votes = result, decisive[result]
            break
    integrity_values = [str(value.get("integrity")) for value in parsed_samples]
    collision_values = [str(value.get("collision")) for value in parsed_samples]

    def majority(values: Sequence[str]) -> str:
        if not values:
            return "Unable to assess"
        counts = {value: values.count(value) for value in set(values)}
        value, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        return value if count >= QUORUM else "Unable to assess"

    if winner is None:
        return {
            "status": "unable_to_assess",
            "label": "Unable to assess",
            "progress": None,
            "visual_integrity": majority(integrity_values),
            "collision": majority(collision_values),
            "completion": "Unable to assess",
            "explanation": None,
            "evidence_frame_indices": [],
            "quorum": max(len(decisive[True]), len(decisive[False])),
            "missing_reason": report.get("missing_reason") or "judge_insufficient_quorum",
        }
    sorted_progress = sorted(int(sample["progress"]) for sample in votes)
    chosen = votes[0]
    return {
        "status": "evaluable",
        "label": "Assessment complete",
        "progress": sorted_progress[(len(sorted_progress) - 1) // 2],
        "visual_integrity": majority(integrity_values),
        "collision": majority(collision_values),
        # This application label is deliberately derived here from the retained
        # quorum, never requested as a separate VLM response field.
        "completion": "Completed" if winner else "Not completed",
        "explanation": chosen.get("observable_reasons"),
        "evidence_frame_indices": list(chosen.get("evidence_frame_indices") or []),
        "quorum": len(votes),
    }


class DemoJudgeService:
    """SQLite journal and one-item remote queue for saved-clip judging."""

    def __init__(self, root: Path, config: Optional[DemoJudgeConfig] = None):
        self.root = Path(root).resolve()
        self.config = config if config is not None else DemoJudgeConfig.from_env()
        self.judgments_root = self.root / "demo-judgments"
        self.db_path = self.root / "demo-judgments.sqlite3"
        self._lock = threading.RLock()
        self._deployment_observation: Dict[str, Any] = {}
        self._deployment_observed_at = 0.0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plumb-demo-judge")
        self._init_schema()
        self._recover()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _init_schema(self) -> None:
        self.judgments_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_judgments (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_sha256 TEXT NOT NULL,
                    clip_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT
                );
                CREATE TABLE IF NOT EXISTS demo_judgment_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    judgment_id TEXT NOT NULL REFERENCES demo_judgments(id),
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS demo_judgments_clip_created ON demo_judgments(clip_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS demo_judgment_events_lookup ON demo_judgment_events(judgment_id, sequence);
                """
            )
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    def _recover(self) -> None:
        now = utc_now()
        failure = {
            "kind": "InterruptedRemoteJudgment",
            "message": "Local process ended while an assessment was outstanding; it was not retried automatically.",
            "automatic_retry_allowed": False,
        }
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM demo_judgments WHERE status IN ('queued', 'warming', 'assessing')"
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE demo_judgments SET status = 'interrupted', error_json = ?, finished_at = ?, updated_at = ? WHERE id = ?",
                    (canonical_json(failure), now, now, row["id"]),
                )
                self._event(connection, row["id"], "interrupted", "interrupted", failure)
            connection.commit()

    @staticmethod
    def _event(connection: sqlite3.Connection, judgment_id: str, stage: str, status: str, detail: Mapping[str, Any]) -> None:
        connection.execute(
            "INSERT INTO demo_judgment_events(judgment_id, stage, status, detail_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (judgment_id, stage, status, canonical_json(_safe_json(detail)), utc_now()),
        )

    def _artifact_dir(self, judgment_id: str) -> Path:
        path = (self.judgments_root / judgment_id).resolve()
        if self.judgments_root.resolve() not in path.parents:
            raise DemoJudgeError("judgment artifact path escapes its root")
        return path

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(_safe_json(value)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _supported_clip(self, clip_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if clip_id and clip_id.startswith("live-demo:"):
            session_id = clip_id.removeprefix("live-demo:")
            if not re.fullmatch(r"[a-f0-9]{32}", session_id):
                return None
            db_path = self.root / "live-demo/live-demo.sqlite3"
            if not db_path.is_file():
                return None
            with sqlite3.connect(str(db_path)) as db:
                db.row_factory = sqlite3.Row
                row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not row or row["state"] != "completed":
                return None
            pot = json.loads(row["source_json"]).get("kind") == "demo_pot_fixture"
            task_id, instruction = (POT_TASK_ID, POT_INSTRUCTION) if pot else (DEMO_TASK_ID, DEMO_TASK_LABEL)
            if json.loads(row["source_json"]).get("kind") not in {"demo_drawer_fixture", "demo_pot_fixture"}:
                return None
            if row["prompt"] != instruction:
                task_id, instruction = CUSTOM_TASK_ID, row["prompt"]
                custom_task_rubric(instruction)
            video = _under(self.root, "live-demo/" + str(row["latest_video_file"]))
            receipt = json.loads(video.with_suffix(".manifest.json").read_text())
            if receipt["session_id"] != session_id or receipt["sha256"] != _sha256_file(video):
                return None
            scene_name = "pot-scene.png" if pot else "scene-reference.png"
            scene = _under(self.root, "demo-judge-fixtures/" + scene_name)
            if _sha256_file(scene) != ("sha256:65db577ed68303af3931781428da7d017337b3b7be8c2eaac4fa192b2cb9dd69" if pot else "sha256:42ad1fe7593823c0aa33888ffd10927c2dde7cce8b65544e48dd9a5754e7f24e"):
                return None
            return {"id": clip_id, "title": row["title"], "video_path": video, "scene_path": scene,
                    "video_url": "/api/artifacts/live-demo/" + video.name,
                    "scene_url": "/api/artifacts/demo-judge-fixtures/" + scene_name,
                    "video_sha256": receipt["sha256"], "task_id": task_id, "task_label": instruction,
                    "action_source": "Supplied manual right actions" if json.loads(row["source_json"]).get("world_model") == "cosmos" else "OpenVLA actions generated in this run", "controller_identity": "Cosmos" if json.loads(row["source_json"]).get("world_model") == "cosmos" else "OpenVLA",
                    "scene_reference_role": "scene_reference_not_goal"}
        catalog = world_videos_payload(self.root)
        candidates: List[Dict[str, Any]] = []
        for video in catalog.get("videos", []):
            if not isinstance(video, Mapping) or video.get("kind") != "kosmos_baseten_world_probe":
                continue
            if clip_id is not None and video.get("id") != clip_id:
                continue
            report_url = video.get("report_url")
            video_url = video.get("video_url")
            poster_url = video.get("poster_url")
            if not all(isinstance(item, str) and item.startswith("/api/artifacts/") for item in (report_url, video_url, poster_url)):
                continue
            try:
                report_path = _under(self.root, str(report_url).removeprefix("/api/artifacts/"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, Mapping) or report.get("task") != DEMO_TASK_LABEL or report.get("policy_calls") != 0:
                    continue
                video_path = _under(self.root, str(video_url).removeprefix("/api/artifacts/"))
                poster_path = _under(self.root, str(poster_url).removeprefix("/api/artifacts/"))
                # The gallery poster is the first generated output. The judge's
                # scene reference must be the original conditioning image.
                scene_path = _under(self.root, "demo-judge-fixtures/scene-reference.png")
                if _sha256_file(scene_path) != _normalise_sha(report.get("source_png_sha256")):
                    continue
                recorded_sha = _normalise_sha(video.get("sha256"))
                if recorded_sha is None or _sha256_file(video_path) != recorded_sha:
                    continue
                candidates.append(
                    {
                        "id": str(video["id"]),
                        "title": str(video.get("title") or "Saved drawer example"),
                        "video_path": video_path,
                        "poster_path": poster_path,
                        "scene_path": scene_path,
                        "scene_url": "/api/artifacts/demo-judge-fixtures/scene-reference.png",
                        "video_url": str(video_url),
                        "poster_url": str(poster_url),
                        "report_url": str(report_url),
                        "video_sha256": recorded_sha,
                        "report_sha256": _sha256_file(report_path),
                        "task_id": DEMO_TASK_ID,
                        "task_label": DEMO_TASK_LABEL,
                        "action_source": "Recorded manual right action",
                        "controller_identity": "No policy controller was called",
                        "scene_reference_role": "scene reference (not a goal image)",
                    }
                )
            except (OSError, ValueError, DemoJudgeError):
                continue
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _latest_receipt(self) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT result_json FROM demo_judgments WHERE status IN ('completed', 'abstained') ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        if row is None or not row["result_json"]:
            return None
        try:
            payload = json.loads(row["result_json"])
        except ValueError:
            return None
        receipt = payload.get("adapter_receipt") if isinstance(payload, Mapping) else None
        return dict(receipt) if isinstance(receipt, Mapping) else None

    def readiness(self) -> Dict[str, Any]:
        clip = self._supported_clip()
        receipt = self._latest_receipt()
        base = {
            "profile": DEMO_PROFILE_ID,
            "experimental": True,
            "qualified": False,
            "base_model": {"id": QWEN_MODEL_ID, "revision": QWEN_BASE_REVISION},
            "adapter": {
                "id": DEMO_PROFILE_ID,
                "tree_sha256": SEMANTIC_ADAPTER_TREE_SHA256,
            },
            "sampling": {"sample_count": 5, "quorum": 3, "temperature": 0.7},
            "default_clip": self._public_clip(clip) if clip else None,
            "last_adapter_receipt": receipt,
        }
        if clip is None:
            return {**base, "configured": self.config is not None, "available": False, "reason": "No hash-bound saved Close the drawer clip with a scene reference is available."}
        if self.config is None:
            return {
                **base,
                "configured": False,
                "available": False,
                "reason": "The semantic epoch-02 judge deployment is not configured. The base model and formatting adapter are not substitutes.",
            }
        receipt_ok = isinstance(receipt, Mapping) and self._receipt_matches(receipt)
        return {
            **base,
            "configured": True,
            "available": True,
            "reason": "Configured; an explicit assessment request is required to load or invoke the private judge worker.",
            "deployment": ({"provider": "Trillium", "allocation_id": os.environ.get("PLUMB_DEMO_JUDGE_ALLOCATION_ID"), "max_replicas": 1} if self.config.worker_url else {"model_id": self.config.model_id, "deployment_id": self.config.deployment_id, "max_replicas": 1, "budget_cap_usd": self.config.budget_cap_usd}),
            "loaded_adapter_verified": receipt_ok,
        }

    def performance_observation(self) -> Dict[str, Any]:
        """Read-only management telemetry; never wakes the inference worker."""
        if self.config is None:
            return {"status": "unavailable"}
        if self.config.worker_url:
            try:
                with urlopen(self._worker_request("/health"), timeout=5) as response:
                    health = json.load(response)
                return {"status": "ready", "active_replicas": 1, "gpu_configuration": health.get("gpu_type"), "provider": "Trillium", "observed_at": utc_now()}
            except Exception:
                return {"status": "unavailable"}
        with self._lock:
            if time.time() - self._deployment_observed_at < 30:
                return self._deployment_observation
            try:
                call = subprocess.run(["baseten", "model", "deployment", "describe", "--profile", self.config.cli_profile,
                    "--model-id", self.config.model_id, "--deployment-id", self.config.deployment_id, "--output", "json"], capture_output=True, timeout=15, check=True)
                observed = json.loads(call.stdout)
                self._deployment_observation = {"status": observed.get("status"), "active_replicas": observed.get("active_replica_count"),
                    "autoscaling": observed.get("autoscaling_settings"), "gpu_configuration": observed.get("instance_type_name"), "observed_at": utc_now()}
            except (OSError, ValueError, subprocess.SubprocessError):
                self._deployment_observation = {"status": "unavailable", "active_replicas": None, "observed_at": utc_now()}
            self._deployment_observed_at = time.time()
            return self._deployment_observation

    @staticmethod
    def _public_clip(clip: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if clip is None:
            return None
        return {key: clip.get(key) for key in (
            "id", "title", "video_url", "poster_url", "report_url", "video_sha256", "task_id", "task_label",
            "action_source", "controller_identity", "scene_reference_role",
        )}

    def _prepare_input(self, judgment_id: str, clip: Mapping[str, Any]) -> Dict[str, Any]:
        preparation_started = time.perf_counter()
        from PIL import Image
        video = Path(str(clip["video_path"]))
        poster = Path(str(clip["scene_path"]))
        if _sha256_file(video) != clip["video_sha256"]:
            raise DemoJudgeInputError("Saved clip bytes no longer match the catalog hash.")
        frame_count, fps = _probe_video(video)
        indexes = _frame_indices(frame_count)
        artifact = self._artifact_dir(judgment_id)
        frames_dir = artifact / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        frames = _extract_frames(video, indexes, frames_dir)
        reference = artifact / "scene-reference.png"
        shutil.copyfile(poster, reference)
        if reference.stat().st_size > MAX_FRAME_BYTES:
            raise DemoJudgeInputError("Saved scene reference exceeds the image safety limit.")
        frame_records = [
            {"index": index, "file": frame.relative_to(artifact).as_posix(), "sha256": _sha256_file(frame), "bytes": frame.stat().st_size}
            for index, frame in zip(indexes, frames)
        ]
        for record, frame in zip(frame_records, frames):
            with Image.open(frame) as decoded:
                record["pixel_sha256"] = image_pixel_hash(decoded.convert("RGB"))["sha256"]
        reference_sha = _sha256_file(reference)
        seeds = _derive_seeds(str(clip["video_sha256"]) + ":" + str(clip["id"]))
        return {
            "schema_version": 1,
            "judgment_id": judgment_id,
            "clip": self._public_clip(clip),
            "task_id": clip["task_id"],
            **({"task_instruction": clip["task_label"], "task_rubric": custom_task_rubric(clip["task_label"])} if clip["task_id"] == CUSTOM_TASK_ID else {}),
            "rubric_hash": canonical_json_sha256({"diagnostic_rubric": custom_task_rubric(clip["task_label"])}) if clip["task_id"] == CUSTOM_TASK_ID else POT_RUBRIC_HASH if clip["task_id"] == POT_TASK_ID else BENCHMARK_TASK_REGISTRY.get(DEMO_TASK_ID).rubric_hash,
            "profile": DEMO_PROFILE_ID,
            "video_sha256": clip["video_sha256"],
            "video_frame_count": frame_count,
            "frame_indexes": list(indexes),
            "frame_timestamps": [index / fps for index in indexes],
            "frames": frame_records,
            "scene_reference": {
                "file": reference.name,
                "sha256": reference_sha,
                "source_uri": clip["scene_url"],
                "role": "scene_reference_not_goal",
            },
            "sampling": {"sample_count": 5, "quorum": 3, "temperature": 0.7, "top_p": 1.0, "max_new_tokens": 512, "retries_per_sample": 1},
            "seeds": list(seeds),
            "preparation_seconds": time.perf_counter() - preparation_started,
        }

    def submit(self, body: DemoJudgmentInput) -> Dict[str, Any]:
        admitted_at = time.time()
        if not _IDEMPOTENCY_RE.fullmatch(body.idempotency_key):
            raise DemoJudgeInputError("idempotency_key is invalid")
        clip = self._supported_clip(body.clip_id)
        if clip is None:
            raise DemoJudgeInputError("This saved clip is not supported by the Close the drawer judge profile.")
        if self.config is None:
            raise DemoJudgeUnavailable(self.readiness()["reason"])
        request_binding = {"clip_id": body.clip_id, "profile": body.profile}
        request_sha = "sha256:" + hashlib.sha256(canonical_json(request_binding).encode("utf-8")).hexdigest()
        with self._lock, self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM demo_judgments WHERE idempotency_key = ?", (body.idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_sha:
                    raise DemoJudgeInputError("idempotency_key already binds a different clip or judge profile")
                return self._public_row(existing)
        judgment_id = "judge-" + uuid.uuid4().hex
        try:
            prepared = self._prepare_input(judgment_id, clip)
            prepared["admitted_at_unix"] = admitted_at
            prepared["deployment"] = {"model_id": self.config.model_id, "deployment_id": self.config.deployment_id, "max_replicas": 1, "min_replicas": 0, "concurrency": 1, "team_id": "q8grpdw"}
            if self.config.worker_url:
                prepared["deployment"] = {"provider": "Trillium", "worker": "semantic-epoch02", "allocation_id": os.environ.get("PLUMB_DEMO_JUDGE_ALLOCATION_ID"), "max_replicas": 1, "min_replicas": 1, "concurrency": 1}
        except Exception:
            shutil.rmtree(self._artifact_dir(judgment_id), ignore_errors=True)
            raise
        input_sha = "sha256:" + hashlib.sha256(canonical_json(prepared).encode("utf-8")).hexdigest()
        prepared["input_sha256"] = input_sha
        self._write_json(self._artifact_dir(judgment_id) / "input.json", prepared)
        now = utc_now()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM demo_judgments WHERE idempotency_key = ?", (body.idempotency_key,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                shutil.rmtree(self._artifact_dir(judgment_id), ignore_errors=True)
                if existing["request_sha256"] != request_sha:
                    raise DemoJudgeInputError("idempotency_key already binds a different clip or judge profile")
                return self._public_row(existing)
            connection.execute(
                """INSERT INTO demo_judgments(id, idempotency_key, request_sha256, clip_id, profile, status, created_at, updated_at, input_json)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (judgment_id, body.idempotency_key, request_sha, body.clip_id, body.profile, now, now, canonical_json(prepared)),
            )
            self._event(connection, judgment_id, "sampling", "completed", {"frame_indexes": prepared["frame_indexes"], "video_sha256": prepared["video_sha256"]})
            self._event(connection, judgment_id, "queued", "queued", {"input_sha256": input_sha})
            connection.commit()
        self._pool.submit(self.execute, judgment_id)
        return self.get(judgment_id)

    def _set_status(self, judgment_id: str, status: str, stage: str, detail: Mapping[str, Any], *, started: bool = False, finished: bool = False, result: Optional[Mapping[str, Any]] = None, error: Optional[Mapping[str, Any]] = None) -> None:
        with self._lock, self._connection() as connection:
            now = utc_now()
            values: Dict[str, Any] = {"status": status, "updated_at": now, "id": judgment_id}
            assignments = ["status = :status", "updated_at = :updated_at"]
            if started:
                assignments.append("started_at = :started_at")
                values["started_at"] = now
            if finished:
                assignments.append("finished_at = :finished_at")
                values["finished_at"] = now
            if result is not None:
                assignments.append("result_json = :result_json")
                values["result_json"] = canonical_json(_safe_json(result))
            if error is not None:
                assignments.append("error_json = :error_json")
                values["error_json"] = canonical_json(_safe_json(error))
            connection.execute("UPDATE demo_judgments SET " + ", ".join(assignments) + " WHERE id = :id", values)
            self._event(connection, judgment_id, stage, status, detail)

    def _argv(self) -> List[str]:
        assert self.config is not None
        return [
            "baseten", "model", "predict", "--profile", self.config.cli_profile,
            "--model-id", self.config.model_id, "--deployment-id", self.config.deployment_id,
            "--file", "-", "--output", "json",
        ]

    def _remote_payload(self, prepared: Mapping[str, Any]) -> Dict[str, Any]:
        artifact = self._artifact_dir(str(prepared["judgment_id"]))
        frames = []
        for record in prepared["frames"]:
            path = artifact / str(record["file"])
            raw = path.read_bytes()
            if _sha256_file(path) != record["sha256"]:
                raise DemoJudgeError("persisted sampled frame hash changed before dispatch")
            frames.append({"png_base64": base64.b64encode(raw).decode("ascii"), "sha256": record["sha256"]})
        reference_info = prepared["scene_reference"]
        reference = artifact / str(reference_info["file"])
        if _sha256_file(reference) != reference_info["sha256"]:
            raise DemoJudgeError("persisted scene reference hash changed before dispatch")
        return {
            "schema_version": 1,
            "kind": "plumb_demo_semantic_judge_v1",
            "request_id": prepared["judgment_id"],
            "frames": frames,
            "frame_timestamps": prepared["frame_timestamps"],
            "reference_images": [{
                "png_base64": base64.b64encode(reference.read_bytes()).decode("ascii"),
                "sha256": reference_info["sha256"],
                "source_uri": reference_info["source_uri"],
                "role": reference_info["role"],
            }],
            "task_id": prepared["task_id"],
            **({"task_instruction": prepared["task_instruction"], "task_rubric": prepared["task_rubric"]} if prepared["task_id"] == CUSTOM_TASK_ID else {}),
            "seeds": prepared["seeds"],
            "provenance": {
                "clip_id": prepared["clip"]["id"],
                "video_sha256": prepared["video_sha256"],
                "protocol_id": "plumb-demo-judge-v1",
                "calibration_manifest_hash": None,
            },
        }

    @staticmethod
    def _receipt_matches(receipt: Mapping[str, Any]) -> bool:
        return (
            receipt.get("adapter_id") == DEMO_PROFILE_ID
            and _normalise_sha(receipt.get("adapter_tree_sha256")) == SEMANTIC_ADAPTER_TREE_SHA256
            and receipt.get("active_adapter") == DEMO_PROFILE_ID
            and receipt.get("adapter_enabled") is True
            and receipt.get("base_model_id") == QWEN_MODEL_ID
            and receipt.get("base_model_revision") == QWEN_BASE_REVISION
            and receipt.get("processor_revision") == QWEN_BASE_REVISION
        )

    def _validate_response(self, response: Any, judgment_id: str) -> Dict[str, Any]:
        if not isinstance(response, Mapping) or response.get("schema_version") != 1 or response.get("request_id") != judgment_id:
            raise DemoJudgeError("judge worker response does not bind this request")
        if response.get("status") != "completed":
            reason = response.get("reason") if isinstance(response.get("reason"), str) else "unknown worker failure"
            raise DemoJudgeError("judge worker did not complete: " + reason)
        report = response.get("report")
        receipt = response.get("adapter_receipt")
        if not isinstance(report, Mapping) or not isinstance(receipt, Mapping):
            raise DemoJudgeError("judge worker response lacks report or adapter receipt")
        if not self._receipt_matches(receipt):
            raise DemoJudgeError("judge worker did not prove the selected semantic epoch-02 adapter is loaded and enabled")
        with self._connection() as connection:
            stored = connection.execute("SELECT input_json FROM demo_judgments WHERE id = ?", (judgment_id,)).fetchone()
        prepared = json.loads(stored["input_json"])
        provenance = report.get("provenance") or {}
        evidence = provenance.get("evidence_hashes") or {}
        if evidence.get("video_hash") != prepared["video_sha256"] or evidence.get("frame_pixel_hashes") != [frame["pixel_sha256"] for frame in prepared["frames"]]:
            raise DemoJudgeError("judge response input hashes differ from the persisted frames")
        if provenance.get("rubric_hash") != prepared["rubric_hash"]:
            raise DemoJudgeError("judge response rubric differs from the requested task")
        samples = report.get("raw_judge_samples") or []
        if [sample.get("seed") for sample in samples] != prepared["seeds"]:
            raise DemoJudgeError("judge response seeds differ from the admitted request")
        for index, sample in enumerate(samples):
            for attempt_index, attempt in enumerate(sample.get("attempts") or []):
                if attempt.get("sample_index") != index or attempt.get("seed") != prepared["seeds"][index] or attempt.get("attempt_index") != attempt_index:
                    raise DemoJudgeError("judge attempt identity mismatch")
                if attempt.get("parsed") is not None and dict(parse_rubric_json(attempt.get("raw_output")).as_dict()) != attempt["parsed"]:
                    raise DemoJudgeError("judge attempt raw/parsed mismatch")
        assessment = _assessment_from_report(report)
        expected_success = None if assessment["status"] != "evaluable" else assessment["completion"] == "Completed"
        if report.get("binary_success") is not expected_success or report.get("progress") != assessment["progress"] or report.get("agreeing_samples") != assessment["quorum"]:
            raise DemoJudgeError("judge summary differs from local quorum computation")
        timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
        return {
            "report": _safe_json(report),
            "adapter_receipt": _safe_json(receipt),
            "assessment": assessment,
            "timing": _safe_json(timing),
            "deployment": prepared["deployment"],
            "generation_metrics": _safe_json(response.get("generation_metrics", [])),
            "experimental": True,
            "qualified": False,
        }

    def execute(self, judgment_id: str) -> Dict[str, Any]:
        try:
            record = self.get(judgment_id)
        except KeyError:
            raise
        if record["status"] != "queued":
            return record
        self._set_status(judgment_id, "warming", "warming", {"message": "Private judge worker invocation started."}, started=True)
        with self._connection() as connection:
            row = connection.execute("SELECT input_json FROM demo_judgments WHERE id = ?", (judgment_id,)).fetchone()
        if row is None:
            raise KeyError(judgment_id)
        prepared = json.loads(row["input_json"])
        try:
            payload = self._remote_payload(prepared)
            # A synchronous remote request does not reveal when warming ends.
            # Keep the observed state as waiting for the remote worker.
            started = time.perf_counter()
            assert self.config is not None
            environment = dict(os.environ)
            environment.pop("BASETEN_API_KEY", None)
            environment.pop("BASETEN_AUTH_TOKEN", None)
            try:
                call = self._cluster_call(payload, judgment_id) if self.config.worker_url else subprocess.run(
                    self._argv(), input=canonical_json(payload).encode("utf-8"), capture_output=True,
                    timeout=self.config.timeout_seconds, check=False, env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise DemoJudgeError("judge worker connection outcome is ambiguous; no automatic retry was issued") from error
            if len(call.stdout) + len(call.stderr) > MAX_CLI_RESPONSE_BYTES:
                raise DemoJudgeError("judge worker response exceeds the retained response limit")
            raw = {"returncode": call.returncode, "stdout": call.stdout.decode("utf-8", "replace"), "stderr": call.stderr.decode("utf-8", "replace")}
            self._write_json(self._artifact_dir(judgment_id) / "raw-response.json", raw)
            if call.returncode != 0:
                if "status 403" in raw["stderr"]:
                    raise DemoJudgeError("Baseten rejected authentication (HTTP 403). No automatic retry was issued.")
                raise DemoJudgeError("judge worker returned a nonzero CLI status")
            try:
                remote = json.loads(raw["stdout"])
            except ValueError as error:
                raise DemoJudgeError("judge worker returned unreadable JSON") from error
            remote_seconds = time.perf_counter() - started
            validation_started = time.perf_counter()
            result = self._validate_response(remote, judgment_id)
            result["timing"] = dict(result["timing"])
            result["timing"]["client_request_seconds"] = time.perf_counter() - started
            result["timing"]["remote_request_seconds"] = remote_seconds
            result["timing"]["preparation_seconds"] = prepared["preparation_seconds"]
            result["timing"]["outside_inference_seconds"] = max(0, remote_seconds - float(result["timing"].get("worker_inference_seconds", 0)))
            result["timing"]["outside_inference_scope"] = "Observed transport, serialization and request waiting combined; queue time is not separately observable."
            result["timing"]["client_request_scope"] = "CLI request, model wake/load, five-sample inference, response validation; excludes local journal write"
            if self.config.worker_url:
                result["timing"]["client_request_scope"] = "Private worker stream, five-sample inference and response validation; excludes local journal write"
            self._write_json(self._artifact_dir(judgment_id) / "result.json", result)
            terminal = "completed" if result["assessment"]["status"] == "evaluable" else "abstained"
            self._set_status(judgment_id, terminal, "result", {"assessment_status": result["assessment"]["status"]}, finished=True, result=result)
            result["timing"]["validation_and_persistence_seconds"] = time.perf_counter() - validation_started
            result["timing"]["action_to_persisted_seconds"] = time.time() - prepared["admitted_at_unix"]
            result["timing"]["total_scope"] = "Server admission through first durable result commit; browser network/render time excluded."
            self._write_json(self._artifact_dir(judgment_id) / "result.json", result)
            with self._connection() as connection:
                connection.execute("UPDATE demo_judgments SET result_json = ? WHERE id = ?", (canonical_json(result), judgment_id))
        except Exception as error:  # explicitly terminal: never resubmit this request
            failure = {"kind": type(error).__name__, "message": str(error), "automatic_retry_allowed": False}
            self._write_json(self._artifact_dir(judgment_id) / "failure.json", failure)
            status = "interrupted" if "ambiguous" in str(error).lower() else "failed"
            self._set_status(judgment_id, status, status, failure, finished=True, error=failure)
        return self.get(judgment_id)

    @staticmethod
    def _public_row(row: sqlite3.Row) -> Dict[str, Any]:
        input_value = json.loads(row["input_json"])
        result = json.loads(row["result_json"]) if row["result_json"] else None
        error = json.loads(row["error_json"]) if row["error_json"] else None
        judgment_id = row["id"]
        return {
            "id": judgment_id,
            "status": row["status"],
            "clip_id": row["clip_id"],
            "profile": row["profile"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "input": input_value,
            "result": result,
            "error": error,
            "experimental": True,
            "qualified": False,
            "result_url": "/api/artifacts/demo-judgments/%s/result.json" % judgment_id if result else None,
            "raw_response_url": None,
        }

    def get(self, judgment_id: str) -> Dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM demo_judgments WHERE id = ?", (judgment_id,)).fetchone()
        if row is None:
            raise KeyError(judgment_id)
        public = self._public_row(row)
        with self._connection() as connection:
            event = connection.execute("SELECT detail_json FROM demo_judgment_events WHERE judgment_id=? AND stage='sample' ORDER BY sequence DESC LIMIT 1", (judgment_id,)).fetchone()
        public["progress"] = json.loads(event["detail_json"]) if event else None
        artifact = self._artifact_dir(judgment_id)
        public["raw_response_url"] = "/api/artifacts/demo-judgments/%s/raw-response.json" % judgment_id if (artifact / "raw-response.json").is_file() else None
        public["failure_url"] = "/api/artifacts/demo-judgments/%s/failure.json" % judgment_id if (artifact / "failure.json").is_file() else None
        return public

    def _worker_request(self, suffix: str, payload=None):
        parsed = urlparse(self.config.worker_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise DemoJudgeError("Judge worker requires a loopback tunnel")
        token = Path(self.config.worker_token_file).read_text().strip()
        return Request(self.config.worker_url.rstrip("/") + suffix, data=None if payload is None else canonical_json(payload).encode(), headers={"Authorization":"Bearer " + token,"Content-Type":"application/json"})

    def _cluster_call(self, payload, judgment_id):
        result = None
        try:
            with urlopen(self._worker_request("/assess", payload), timeout=self.config.timeout_seconds) as response:
                while True:
                    line = response.readline(MAX_CLI_RESPONSE_BYTES + 1)
                    if not line: break
                    if len(line) > MAX_CLI_RESPONSE_BYTES: raise DemoJudgeError("Oversized judge event")
                    event = json.loads(line)
                    if event.get("kind") == "progress":
                        self._set_status(judgment_id,"assessing","sample",event)
                    elif event.get("kind") == "result":
                        result = event["response"]
        except (OSError, ValueError) as error:
            raise DemoJudgeError("Private judge stream interrupted; outcome is ambiguous; no retry") from error
        if result is None: raise DemoJudgeError("Private judge returned no result; outcome is ambiguous")
        return subprocess.CompletedProcess([],0,canonical_json(result).encode(),b"")

    def list(self, clip_id: Optional[str] = None) -> Dict[str, Any]:
        query = "SELECT * FROM demo_judgments"
        values: Tuple[Any, ...] = ()
        if clip_id is not None:
            query += " WHERE clip_id = ?"
            values = (clip_id,)
        query += " ORDER BY created_at DESC LIMIT 50"
        with self._connection() as connection:
            rows = connection.execute(query, values).fetchall()
        return {"judgments": [self.get(row["id"]) for row in rows], "experimental": True, "qualified": False}

    def events(self, judgment_id: str) -> Dict[str, Any]:
        self.get(judgment_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT sequence, stage, status, detail_json, created_at FROM demo_judgment_events WHERE judgment_id = ? ORDER BY sequence",
                (judgment_id,),
            ).fetchall()
        return {
            "judgment_id": judgment_id,
            "events": [
                {"sequence": row["sequence"], "stage": row["stage"], "status": row["status"], "detail": json.loads(row["detail_json"]), "created_at": row["created_at"]}
                for row in rows
            ],
        }


def register_demo_judge_routes(app: Any, service: DemoJudgeService) -> None:
    """Attach the additive saved-clip judge endpoints to the localhost app."""

    from fastapi import HTTPException

    @app.get("/api/demo/judge/readiness")
    def readiness() -> Dict[str, Any]:
        return service.readiness()

    @app.get("/api/demo/judge/performance")
    def performance() -> Dict[str, Any]:
        return service.performance_observation()

    @app.get("/api/demo/judgments")
    def list_judgments(clip_id: Optional[str] = None) -> Dict[str, Any]:
        if clip_id is not None and (not clip_id or len(clip_id) > 200):
            raise HTTPException(422, "Invalid clip ID")
        return service.list(clip_id)

    @app.post("/api/demo/judgments", status_code=202)
    def create_judgment(body: DemoJudgmentInput) -> Dict[str, Any]:
        try:
            return service.submit(body)
        except DemoJudgeUnavailable as error:
            raise HTTPException(409, {"reason": str(error), "readiness": service.readiness()})
        except DemoJudgeInputError as error:
            raise HTTPException(422, {"reason": str(error)})
        except DemoJudgeError as error:
            raise HTTPException(503, {"reason": str(error)})

    @app.get("/api/demo/judgments/{judgment_id}")
    def get_judgment(judgment_id: str) -> Dict[str, Any]:
        try:
            return service.get(judgment_id)
        except KeyError:
            raise HTTPException(404, "Judgment not found")

    @app.get("/api/demo/judgments/{judgment_id}/events")
    def judgment_events(judgment_id: str) -> Dict[str, Any]:
        try:
            return service.events(judgment_id)
        except KeyError:
            raise HTTPException(404, "Judgment not found")


__all__ = [
    "DEMO_PROFILE_ID",
    "DemoJudgeConfig",
    "DemoJudgeError",
    "DemoJudgeService",
    "DemoJudgmentInput",
    "register_demo_judge_routes",
]
