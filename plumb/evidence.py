"""Read-only presentation of imported diagnostics, never scored ledger rows."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return value
    return None


def _url(path: Path, root: Path) -> str:
    return "/api/artifacts/" + quote(path.relative_to(root).as_posix(), safe="/")


def _video(report_path: Path, metadata: dict, root: Path) -> Optional[str]:
    """Resolve only a same-directory media copy with the report's digest.

    Remote absolute paths are provenance, not permission to read that path.
    The displayed video must match the immutable report before it is linked.
    """
    name = Path(str(metadata.get("path", ""))).name
    expected = str(metadata.get("sha256", "")).removeprefix("sha256:")
    if not name.endswith(".mp4") or len(expected) != 64:
        return None
    path = report_path.parent / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
        return None
    if root not in path.resolve().parents:
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        return None
    return _url(path, root)


def experiments_payload(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    evidence = root / "cluster-evidence"
    reports = []
    # A report is either at the evidence root or in a named diagnostic bundle.
    # Do not recursively traverse large frame/state trees or symlinked imports.
    paths = [] if evidence.is_symlink() else sorted([*evidence.glob("*.json"), *evidence.glob("*/*.json")])
    for path in paths:
        try:
            if path.is_symlink() or root not in path.resolve().parents or evidence.resolve() not in path.resolve().parents or path.stat().st_size > 4 * 1024 * 1024:
                continue
            report = _mapping(json.loads(path.read_text()))
            if not isinstance(report.get("status"), str):
                continue  # Provenance-only sidecars are not experiment results.
            kind = report.get("kind")
            item = {"id": path.relative_to(evidence).as_posix().removesuffix(".json"),
                    "kind": kind, "status": str(report.get("status", "unknown")),
                    "qualification": "not_qualified_by_smoke", "qualified": False,
                    "report_url": _url(path, root), "video_url": None,
                    "frame_count": None, "action_dimensions": None,
                    "latency_seconds": None, "model_load_seconds": None,
                    "gpu_peak_memory_bytes": None, "total_seconds": None,
                    "ticks_completed": None, "ticks_requested": None,
                    "outcome": "unknown", "notes": []}
            timing = {}
            video = {}
            if kind == "cosmos3_nano_diffusers_smoke":
                result = _mapping(report.get("result"))
                timing = _mapping(result.get("timing"))
                item.update(model="nvidia/Cosmos3-Nano", stage="world",
                            frame_count=_number(result.get("returned_frame_count")),
                            timing_scope="world_inference_excludes_model_load",
                            notes=["Runtime conformance only; native one-step feedback is not qualified."])
                video = _mapping(_mapping(result.get("output_artifacts")).get("video"))
            elif kind == "irasim_original_one_step_smoke":
                timing = _mapping(report.get("timing"))
                video = _mapping(_mapping(report.get("artifacts")).get("video"))
                item.update(model="IRASim original Frame-Ada", stage="world",
                            frame_count=_number(report.get("frame_count")),
                            total_seconds=_number(report.get("total_seconds")),
                            timing_scope="world_inference_excludes_model_load",
                            notes=["Experimental two-frame call; not a physics or Gate B qualification."])
            elif kind == "plumb_irasim_openvla_closed_loop_diagnostic":
                total = _number(_mapping(report.get("timing")).get("total_seconds"))
                video = _mapping(report.get("video"))
                steps = report.get("step_records", [])
                peaks = [_number(_mapping(_mapping(s).get("world_timing")).get("gpu_peak_memory_bytes"))
                         for s in steps] if isinstance(steps, list) else []
                timing = {"gpu_peak_memory_bytes": max((n for n in peaks if n is not None), default=None)}
                item.update(model="OpenVLA → IRASim", stage="closed_loop", total_seconds=total,
                            frame_count=_number(report.get("frames_persisted")),
                            ticks_completed=_number(report.get("ticks_completed")),
                            ticks_requested=_number(report.get("ticks_requested")),
                            timing_scope="whole_diagnostic_including_load_and_artifacts",
                            notes=["Fresh image feedback per tick; runtime completion is not task success.",
                                   "State mode: " + str(report.get("world_state_mode", "unknown")),
                                   "Visual/physical integrity is unqualified; no task score is asserted."])
            elif kind == "plumb_irasim_native_open_loop_reference_diagnostic":
                artifacts = _mapping(report.get("artifacts"))
                video = _mapping(artifacts.get("mp4"))
                item.update(model="IRASim original Frame-Ada", stage="open_loop_reference",
                            frame_count=_number(artifacts.get("mp4_frame_count")),
                            total_seconds=_number(_mapping(report.get("timing")).get("total_seconds")),
                            timing_scope="whole_open_loop_reference_including_load_and_artifacts",
                            notes=["Released horizon: 15 supplied actions, 16 returned frames.",
                                   "First prediction sees 14 future action rows. This is not native OpenVLA feedback.",
                                   "Qualitative open-loop reference only; physical fidelity remains unqualified."])
            elif kind == "plumb_local_policy_smoke" and report.get("command") == "openvla":
                action = _mapping(report.get("action"))
                timing = action
                item.update(model="openvla/openvla-7b", stage="policy",
                            action_dimensions=_number(action.get("dimension")),
                            timing_scope="policy_inference_excludes_model_load",
                            notes=["One actual native action; no rollout outcome established."])
            elif kind == "plumb_local_policy_smoke" and report.get("command") == "judge":
                timing = {"wall_seconds": _mapping(report.get("timing")).get("judge_call_seconds")}
                raw_outcome = _mapping(report.get("outcome"))
                item.update(model="Qwen/Qwen2.5-VL-7B-Instruct", stage="judge",
                            total_seconds=_number(_mapping(report.get("timing")).get("total_seconds")),
                            timing_scope="five_sample_judge_call_including_lazy_load",
                            notes=["Uncalibrated diagnostic judge; scene reference is not a goal label.",
                                   "Raw judge status: " + str(raw_outcome.get("judge_status", "unknown"))])
            else:
                continue
            item.update(latency_seconds=_number(timing.get("wall_seconds")),
                        model_load_seconds=_number(timing.get("model_load_seconds")),
                        gpu_peak_memory_bytes=_number(timing.get("gpu_peak_memory_bytes")),
                        video_url=_video(path, video, root))
            reports.append(item)
        except (ValueError, OSError, TypeError):
            continue
    return {"experiments": reports, "source": "persisted_cluster_evidence", "qualified": False}
