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
    reports = []
    # A report is either at the evidence root or in a named diagnostic bundle.
    # Do not recursively traverse large frame/state trees or symlinked imports.
    evidence_roots = (root / "cluster-evidence", root / "live-integrated" / "cluster-evidence")
    paths = []
    for evidence in evidence_roots:
        if evidence.is_symlink() or not evidence.is_dir():
            continue
        paths.extend((evidence, path) for path in sorted([*evidence.glob("*.json"), *evidence.glob("*/*.json")]))
    for evidence, path in paths:
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
            elif kind == "plumb_octo_small_production_241fb_two_observation_smoke":
                calls = _mapping(report.get("native_calls"))
                first = _mapping(calls.get("first"))
                actions = first.get("actions")
                runtime = _mapping(report.get("runtime"))
                timing = {"wall_seconds": first.get("wall_seconds"),
                          "gpu_peak_memory_bytes": _mapping(runtime.get("after_native_calls")).get("gpu_peak_memory_bytes")}
                item.update(model="Octo-Small v1.0 production source 241fb", stage="policy",
                            total_seconds=_number(report.get("total_seconds")),
                            action_dimensions=len(actions[0]) if isinstance(actions, list) and actions and isinstance(actions[0], list) else None,
                            timing_scope="first_native_policy_call_excludes_model_load",
                            notes=["Production adapter with source 241fb; separate from the older v0.1 diagnostic.",
                                   "Static fixture inputs and reset/repeat only; no certified execute prefix or generated-image feedback.",
                                   "No physics fidelity, Gate A/B qualification, or task score is established."])
            elif kind == "plumb_octo_worker_reproducibility_diagnostic":
                item.update(model="Octo-Small v1.0 worker consistency", stage="policy_reproducibility",
                            timing_scope="no_episode_or_burst_timing",
                            notes=_replica_notes(path, report))
            elif kind == "plumb_susie_ll_gcbc_static_goal_fixture":
                source_release = report.get("source_release")
                error = _mapping(report.get("error"))
                item.update(
                    model="SuSIE_LL published gc_bc low-level component",
                    stage="policy_static_goal_fixture",
                    source_release=source_release if isinstance(source_release, str) and len(source_release) == 64 else None,
                    static_goal_fixture=True,
                    high_level_subgoal_assets_imported=report.get("high_level_subgoal_assets_imported") is True,
                    timing_scope="one_static_goal_fixture_policy_call_only",
                    notes=[
                        "Static current/goal vendor-video fixture only; not a rollout, task-success, physics, Gate, or primary-study result.",
                        "No high-level SuSIE subgoal or gated component is imported by this diagnostic.",
                    ],
                )
                if report.get("status") == "failed":
                    item.update(
                        error_type=error.get("type") if isinstance(error.get("type"), str) else "unknown",
                        error_message=error.get("message") if isinstance(error.get("message"), str) else "no error message recorded",
                    )
                    item["notes"].insert(0, "Fixture diagnostic failed before a policy action was established; inspect the raw report.")
                else:
                    actions = _mapping(report.get("actions"))
                    first = actions.get("transformed_physical")
                    repeat = actions.get("repeat_transformed_physical")
                    action_shape = actions.get("shape")
                    action_ok = (
                        isinstance(first, list)
                        and len(first) == 1
                        and isinstance(first[0], list)
                        and len(first[0]) == 7
                        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in first[0])
                    )
                    runtime = _mapping(report.get("runtime"))
                    module_versions = _mapping(runtime.get("module_versions"))
                    restore = _mapping(report.get("restore"))
                    fixture_timing = _mapping(report.get("timing"))
                    first_seconds = _number(
                        fixture_timing.get(
                            "first_propose_wall_seconds_including_model_load",
                            fixture_timing.get("first_call_wall_seconds"),
                        )
                    )
                    repeat_seconds = _number(fixture_timing.get("repeat_propose_wall_seconds_model_already_loaded"))
                    native_calls = _number(fixture_timing.get("total_native_calls"))
                    item.update(
                        action_shape=action_shape if action_shape == [1, 7] and action_ok else None,
                        finite_action=bool(actions.get("finite") is True and action_ok),
                        reset_repeat_matches=bool(action_ok and first == repeat),
                        restore_changed=restore.get("changed") if isinstance(restore.get("changed"), bool) else None,
                        jax_version=module_versions.get("jax") if isinstance(module_versions.get("jax"), str) else None,
                        first_propose_seconds_including_model_load=first_seconds,
                        repeat_propose_seconds_model_already_loaded=repeat_seconds,
                        total_native_calls=native_calls,
                        timing_scope=(
                            "two_native_proposals_first_includes_model_load_repeat_already_loaded"
                            if native_calls == 2 and first_seconds is not None and repeat_seconds is not None
                            else "static_goal_fixture_two_proposal_timing_partially_reported"
                        ),
                    )
                    item["notes"].append(
                        "A completed 1×7 finite action/reset-repeat check is runtime conformance only and does not establish success or fidelity."
                    )
            elif kind == "plumb_uncalibrated_judge_lora_pilot":
                timing = {"wall_seconds": report.get("wall_seconds"),
                          "gpu_peak_memory_bytes": report.get("gpu_peak_memory_bytes")}
                selected = _mapping(report.get("selected_checkpoint"))
                item.update(model="Qwen2.5-VL-7B experimental LoRA pilot", stage="judge_training_pilot",
                            timing_scope="training_and_validation_excludes_model_setup",
                            optimizer_steps=_number(report.get("optimizer_steps")),
                            development_loss_before=_number(report.get("validation_loss_before_training")),
                            development_loss_selected=_number(selected.get("development_validation_loss")),
                            notes=["Actual optimizer training on uncalibrated teacher-labelled development clips.",
                                   "Development label loss is not human accuracy, calibration, or a primary study result.",
                                   "Saved adapters stay on the cluster; no automatic scoring deployment."])
            elif kind == "plumb_unqualified_judge_lora_adapter_reload_comparison":
                item.update(model="Qwen2.5-VL saved-adapter comparison", stage="judge_adapter_reload",
                            timing_scope="development_inference_diagnostic_only",
                            notes=["Saved adapter reloaded and paired with the base model on development-validation clips.",
                                   "Agreement is against an uncalibrated teacher, not human ground truth or Gate-E evidence."])
            elif kind == "plumb_judge_lora_framework_preflight":
                timing = {"wall_seconds": report.get("wall_seconds"),
                          "gpu_peak_memory_bytes": report.get("gpu_peak_memory_bytes")}
                item.update(model="Qwen2.5-VL-7B LoRA framework preflight", stage="judge_training_preflight",
                            timing_scope="framework_preflight_collation_forward_backward_excludes_model_setup",
                            notes=["One real diagnostic input tested assistant-only loss and LoRA gradients.",
                                   "No optimizer step or trained adapter; not a completed fine-tune or judge-quality result.",
                                   "Original teacher disagreement/unknown outcome remains unchanged."])
                if report.get("status") not in ("completed_unqualified_preflight",):
                    item["notes"][0] = "Framework preflight did not complete: " + str(report.get("reason", "inspect raw report"))
            elif kind == "plumb_judge_json_structure_only_pilot":
                config = _mapping(report.get("config"))
                item.update(
                    model="Qwen2.5-VL-7B JSON-format structure-only pilot",
                    stage="judge_format_structure_pilot",
                    timing_scope="training_and_development_syntax_validation_only",
                    optimizer_steps=_number(report.get("optimizer_steps")),
                    development_syntax_loss_before=_number(report.get("dev_syntax_loss_before")),
                    development_syntax_loss_after=_number(report.get("dev_syntax_loss_after")),
                    direct_semantic_supervision_tokens=0,
                    notes=[
                        "Structure-only experimental fine-tune: JSON syntax tokens only; direct semantic supervision is forbidden.",
                        "Syntax loss is not human accuracy, semantic calibration, or a judge-quality improvement claim.",
                        "No deployment, production scoring, Gate D, or Gate E use is authorized from this pilot.",
                    ],
                )
                if config.get("semantic_supervision") != "forbidden":
                    item["notes"][0] = "Structure-only supervision audit is incomplete; inspect the raw report."
            elif kind == "plumb_judge_format_compare_v1":
                bare = _mapping(report.get("bare_json_counts"))
                item.update(
                    model="Qwen2.5-VL-7B base versus JSON-format adapter",
                    stage="judge_format_structure_comparison",
                    total_seconds=_number(report.get("wall_seconds")),
                    timing_scope="paired_greedy_generation_excludes_load",
                    base_bare_json_count=_number(bare.get("base")),
                    format_adapter_bare_json_count=_number(bare.get("format_adapter")),
                    semantic_comparable_count=_number(report.get("semantic_comparable_count")),
                    semantic_drift_count=_number(report.get("semantic_drift_count")),
                    notes=[
                        "Four development inputs changed from 0/4 bare JSON (base) to 4/4 bare JSON (format adapter).",
                        "All four semantically comparable decisions drifted; this is not an accuracy or semantic-preservation improvement.",
                        "Structure-only experimental comparison; do not deploy for scoring or claim human accuracy, calibration, or qualification.",
                    ],
                )
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
