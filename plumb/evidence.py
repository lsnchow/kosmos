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


def _replica_notes(report_path: Path, summary: dict) -> list:
    refs = summary.get("reports")
    if not isinstance(refs, list) or not 1 <= len(refs) <= 64:
        raise ValueError("Replica summary needs bounded raw report references")
    reports = []
    for ref in refs:
        candidate = report_path.parent / str(_mapping(ref).get("relative_path", ""))
        if candidate.is_symlink() or report_path.parent.resolve() not in candidate.resolve().parents:
            raise ValueError("Replica report escapes its evidence bundle")
        if not candidate.is_file() or candidate.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Replica report unavailable")
        raw = candidate.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != ref.get("sha256"):
            raise ValueError("Replica report digest mismatch")
        report = _mapping(json.loads(raw))
        if report.get("kind") != "plumb_octo_small_native_two_observation_diagnostic" or report.get("status") != "completed_unqualified":
            raise ValueError("Replica summary does not bind completed native reports")
        if report.get("source_release") != summary.get("source_release"):
            raise ValueError("Replica summary mixes source releases")
        reports.append(report)
    if len({r.get("slurm_process_id") for r in reports}) != len(reports):
        raise ValueError("Replica summary repeats a worker")
    calls = [_mapping(r.get("native_calls")) for r in reports]
    def finite_action(value: Any) -> bool:
        return isinstance(value, list) and len(value) == 7 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    for call in calls:
        if call.get("backend_calls") != 3:
            raise ValueError("Replica must record all three native calls")
        for case in ("first", "second", "after_reset_first"):
            entry = _mapping(call.get(case))
            proposal = entry.get("proposal")
            if not finite_action(entry.get("action")) or not isinstance(proposal, list) or len(proposal) != 4 or not all(finite_action(row) for row in proposal):
                raise ValueError("Replica has an incomplete native proposal")
    equal = all(len({json.dumps(_mapping(c.get(case)).get(field), sort_keys=True, allow_nan=False)
                     for c in calls}) == 1 for case in ("first", "second") for field in ("action", "proposal"))
    resets = all(c["first"][field] == c["after_reset_first"][field]
                 for c in calls for field in ("action", "proposal"))
    return ["%d hash-verified worker reports across %d nodes." % (len(reports), len({r.get("slurm_node") for r in reports})),
            "Fixed-input actions/proposals match: %s; every reset/repeat matches: %s." % (equal, resets),
            "Unqualified native-v0.1 profile; not a rollout, physics, cost or throughput result."]


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
            video_report_path = path
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
            elif kind == "plumb_irasim_causal_history_replay_diagnostic":
                outcome = _mapping(report.get("outcome"))
                rows = outcome.get("rows", [])
                video = _mapping(outcome.get("video"))
                video_report_path = path.parent / "history" / path.name
                item.update(model="IRASim original Frame-Ada", stage="causal_history_replay",
                            frame_count=_number(outcome.get("frame_count")),
                            ticks_completed=len(rows) if isinstance(rows, list) else None,
                            ticks_requested=16, total_seconds=_number(report.get("total_seconds")),
                            timing_scope="history_replay_and_repeat_including_load_and_artifacts",
                            notes=["Saved actions replayed with growing past-latent conditioning; no future action rows.",
                                   "Conditioning differs from training mask=1; policy was not queried.",
                                   "Runtime and repeatability only; no fidelity or task-success score."])
            elif kind == "plumb_local_policy_smoke" and report.get("command") == "openvla":
                action = _mapping(report.get("action"))
                timing = action
                item.update(model="openvla/openvla-7b", stage="policy",
                            action_dimensions=_number(action.get("dimension")),
                            timing_scope="policy_inference_excludes_model_load",
                            notes=["One actual native action; no rollout outcome established."])
            elif kind == "plumb_octo_small_native_two_observation_diagnostic":
                calls = _mapping(report.get("native_calls"))
                first = _mapping(calls.get("first"))
                action = first.get("action")
                runtime = _mapping(report.get("runtime"))
                timing = {"wall_seconds": first.get("wall_seconds"),
                          "gpu_peak_memory_bytes": _mapping(runtime.get("after_native_calls")).get(
                              "gpu_peak_memory_bytes", runtime.get("gpu_peak_memory_bytes"))}
                item.update(model="rail-berkeley/octo-small v1.0", stage="policy",
                            total_seconds=_number(report.get("total_seconds")),
                            action_dimensions=len(action) if isinstance(action, list) else None,
                            timing_scope="first_native_policy_call_excludes_model_load",
                            notes=["Two observed fixture frames and reset/repeat; four-action native proposals.",
                                   "No generated-image feedback, task outcome, or policy qualification is asserted."])
            elif kind == "plumb_octo_worker_reproducibility_diagnostic":
                item.update(model="Octo-Small v1.0 worker consistency", stage="policy_reproducibility",
                            timing_scope="no_episode_or_burst_timing",
                            notes=_replica_notes(path, report))
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
                        video_url=_video(video_report_path, video, root))
            reports.append(item)
        except (ValueError, OSError, TypeError):
            continue
    return {"experiments": reports, "source": "persisted_cluster_evidence", "qualified": False}
