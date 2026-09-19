#!/usr/bin/env python3
"""Optionally export a fresh, diverse, unqualified teacher-v2 LoRA pilot set.

The exporter reads frozen v1 candidate media and independently recomputes the
teacher-v2 unique-mode-at-least-three rule from v2 raw reports.  It writes a
new dataset only when the accepted development-only rows meet the minimum
counts *and* have at least two distinct accepted training semantic tuples.
Otherwise it writes a no-clobber audit report and creates no dataset/training
artifact.  It never repairs a teacher label or touches formal calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PURPOSE = "uncalibrated_judge_v2_distillation_pilot"
ROW_PURPOSE = "uncalibrated_judge_distillation_pilot"
LABEL_SOURCE = "uncalibrated_teacher_structured_sample"
INPUT_PROFILE = "qwen_rubric_serving_messages_v1"
MIN_TRAIN_ROWS = 2
MIN_VALIDATION_ROWS = 1
TOTAL_CANDIDATES = 16
_HEX = set("0123456789abcdef")


class ExportError(ValueError):
    """No source data may be copied until this error is resolved."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ExportError("%s must be a SHA-256 digest" % label)
    text = value.lower().strip()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ExportError("%s must be a SHA-256 digest" % label)
    return "sha256:" + text


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExportError("%s must be a regular JSON file" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError("unable to read %s" % label) from error
    if not isinstance(value, Mapping):
        raise ExportError("%s must be a JSON object" % label)
    return dict(value)


def _safe_source_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ExportError("%s must be a nonempty root-relative path" % label)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ExportError("%s escapes source root" % label)
    current = root
    for part in path.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ExportError("%s is missing" % label) from error
        if stat.S_ISLNK(mode):
            raise ExportError("%s contains a symlink" % label)
    if not current.is_file():
        raise ExportError("%s is not a regular file" % label)
    return current


def _copy_bound(source: Path, destination: Path, expected_sha: Any, label: str) -> str:
    """Copy one accepted source artifact into a fresh output path and rehash it."""

    expected = _digest(expected_sha, label)
    if _sha256_file(source) != expected:
        raise ExportError("%s changed before export" % label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ExportError("refusing to overwrite exported %s" % label)
    shutil.copyfile(source, destination)
    actual = _sha256_file(destination)
    if actual != expected:
        raise ExportError("copied %s does not retain its source hash" % label)
    os.chmod(destination, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return actual


def _write_exclusive(path: Path, payload: Any) -> str:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ExportError("refusing to overwrite %s" % path) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return _sha256_file(path)


def _write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    encoded = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ExportError("refusing to overwrite %s" % path) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return _sha256_file(path)


def _load_v1_candidates(source_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    """Use the original frozen candidate manifest without v1 teacher outputs."""

    from cluster import judge_teacher_pilot as v1

    candidate_path, _ = v1._regular_relative(source_root, "candidate-inputs.json", "candidate manifest")
    candidate = v1.validate_candidate_manifest(source_root, v1._json(candidate_path, "candidate manifest"))
    clips = candidate.get("clips")
    if not isinstance(clips, list) or len(clips) != TOTAL_CANDIDATES:
        raise ExportError("original candidate manifest must retain all 16 source clips")
    by_id = {str(clip["clip_id"]): dict(clip) for clip in clips}
    if len(by_id) != TOTAL_CANDIDATES:
        raise ExportError("original candidate manifest repeats a clip ID")
    return by_id, candidate_path


def _load_v2_freeze(v2_root: Path, candidate_path: Path) -> Dict[str, Any]:
    freeze_path = v2_root / "freeze.json"
    freeze = _load_object(freeze_path, "teacher-v2 freeze")
    if freeze.get("qualified") is not False or freeze.get("purpose") != "uncalibrated_teacher_v2_prompt_selection_diagnostic":
        raise ExportError("teacher-v2 freeze is not explicitly unqualified diagnostic evidence")
    profile = freeze.get("profile_config")
    if not isinstance(profile, Mapping) or not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        raise ExportError("teacher-v2 freeze lacks profile_config.profile_id")
    if _digest(freeze.get("profile_config_sha256"), "teacher-v2 profile_config_sha256") != _canonical_hash(profile):
        raise ExportError("teacher-v2 freeze profile config hash does not match its content")
    if _digest(freeze.get("candidate_manifest_sha256"), "teacher-v2 candidate_manifest_sha256") != _sha256_file(candidate_path):
        raise ExportError("teacher-v2 freeze does not bind the original candidate manifest")
    for key in ("model", "runtime"):
        if not isinstance(freeze.get(key), Mapping):
            raise ExportError("teacher-v2 freeze lacks %s binding" % key)
    return freeze


def _v2_report_paths(v2_root: Path) -> Iterable[Path]:
    raw_root = v2_root / "rawteacher-v2"
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ExportError("teacher-v2 rawteacher-v2 directory is missing or symlinked")
    for path in sorted(raw_root.rglob("attempt-*.json")):
        if path.is_symlink() or not path.is_file():
            raise ExportError("teacher-v2 raw report must be a regular file")
        yield path


def _select_v2(raw_report: Mapping[str, Any]) -> Dict[str, Any]:
    from cluster import judge_teacher_v2 as v2

    selected = dict(v2.select_v2(raw_report))
    if selected.get("status") == "accepted_unique_modal_tuple" and selected.get("modal_count") < 3:
        raise ExportError("teacher-v2 selection accepted a below-three mode")
    return selected


def _collect_reports(v2_root: Path, candidates: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any]) -> Dict[str, Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    """Index one raw v2 report per original candidate and recompute selection."""

    expected_profile_hash = _digest(freeze["profile_config_sha256"], "profile_config_sha256")
    result: Dict[str, Tuple[Path, Dict[str, Any], Dict[str, Any]]] = {}
    for path in _v2_report_paths(v2_root):
        report = _load_object(path, "teacher-v2 raw report")
        if report.get("schema") != "plumb-judge-teacher-v2-clip-v1" or report.get("qualified") is not False:
            raise ExportError("teacher-v2 raw report has an unrecognized schema/qualification status")
        if report.get("status") != "completed_unqualified":
            continue
        if _digest(report.get("profile_config_sha256"), "raw report profile_config_sha256") != expected_profile_hash:
            raise ExportError("teacher-v2 raw report profile hash differs from its freeze")
        if report.get("profile_config") != freeze.get("profile_config"):
            raise ExportError("teacher-v2 raw report profile config differs from its freeze")
        clip = report.get("clip")
        if not isinstance(clip, Mapping) or not isinstance(clip.get("clip_id"), str):
            raise ExportError("teacher-v2 raw report lacks a clip binding")
        clip_id = clip["clip_id"]
        candidate = candidates.get(clip_id)
        if candidate is None:
            raise ExportError("teacher-v2 raw report names a clip outside original frozen candidates")
        if report.get("input_binding_sha256") != candidate.get("input_binding_sha256") or clip.get("input_binding_sha256") != candidate.get("input_binding_sha256"):
            raise ExportError("teacher-v2 raw report does not bind the original candidate input")
        if clip.get("source_lineage_id") != candidate.get("source_lineage_id") or clip.get("cohort") != candidate.get("cohort"):
            raise ExportError("teacher-v2 raw report changes original lineage/cohort")
        raw = report.get("raw_judge_report")
        if not isinstance(raw, Mapping):
            raise ExportError("teacher-v2 raw report lacks raw_judge_report")
        recomputed = _select_v2(raw)
        if clip_id in result:
            raise ExportError("multiple completed teacher-v2 reports exist for one candidate; no post-hoc choice is allowed")
        result[clip_id] = (path, report, recomputed)
    return result


def eligibility(candidates: Mapping[str, Mapping[str, Any]], reports: Mapping[str, Tuple[Path, Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    """Assess counts/diversity without creating a v2 dataset."""

    accepted: Dict[str, Mapping[str, Any]] = {}
    exclusions: List[Dict[str, Any]] = []
    for clip_id, candidate in sorted(candidates.items()):
        record = reports.get(clip_id)
        if record is None:
            exclusions.append({"clip_id": clip_id, "source_lineage_id": candidate["source_lineage_id"], "split": candidate["cohort"], "reason": "no_completed_v2_raw_report"})
            continue
        _, _, selection = record
        if selection.get("status") != "accepted_unique_modal_tuple" or selection.get("modal_count") < 3:
            exclusions.append({"clip_id": clip_id, "source_lineage_id": candidate["source_lineage_id"], "split": candidate["cohort"], "reason": str(selection.get("status") or "selection_not_accepted"), "modal_count": selection.get("modal_count")})
            continue
        selected = selection.get("selected")
        if not isinstance(selected, Mapping) or not isinstance(selected.get("label"), Mapping) or not isinstance(selected.get("semantic_tuple"), list):
            raise ExportError("accepted v2 selection lacks an actual selected raw member")
        accepted[clip_id] = selected
    train = [selected for clip_id, selected in accepted.items() if candidates[clip_id]["cohort"] == "train"]
    validation = [selected for clip_id, selected in accepted.items() if candidates[clip_id]["cohort"] == "development_validation"]
    train_tuples = {tuple(selected["semantic_tuple"]) for selected in train}
    blockers = []
    if len(reports) != TOTAL_CANDIDATES:
        blockers.append("completed_v2_raw_reports_below_%d" % TOTAL_CANDIDATES)
    if len(train) < MIN_TRAIN_ROWS:
        blockers.append("accepted_train_below_%d" % MIN_TRAIN_ROWS)
    if len(validation) < MIN_VALIDATION_ROWS:
        blockers.append("accepted_development_validation_below_%d" % MIN_VALIDATION_ROWS)
    if len(train_tuples) < 2:
        blockers.append("accepted_train_semantic_tuple_diversity_below_2")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "accepted": accepted,
        "exclusions": exclusions,
        "counts": {"candidates": len(candidates), "accepted_train": len(train), "accepted_development_validation": len(validation), "accepted_train_distinct_semantic_tuples": len(train_tuples), "excluded": len(exclusions)},
    }


def _row_from_selection(source_root: Path, output_root: Path, candidate: Mapping[str, Any], raw_path: Path, report: Mapping[str, Any], selection: Mapping[str, Any], freeze: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy only an accepted candidate's media/raw report into a fresh root."""

    clip_id = str(candidate["clip_id"])
    target_dir = output_root / "clips" / clip_id
    frame_paths: List[str] = []
    frame_hashes: List[str] = []
    for index, frame in enumerate(candidate["frames"]):
        source = _safe_source_file(source_root, frame.get("path"), "accepted frame")
        target = target_dir / ("frame-%02d.png" % index)
        frame_hashes.append(_copy_bound(source, target, frame.get("sha256"), "accepted frame"))
        frame_paths.append(str(target.relative_to(output_root)))
    reference = candidate["reference"]
    reference_source = _safe_source_file(source_root, reference.get("path"), "accepted reference")
    reference_target = target_dir / "reference.png"
    reference_hash = _copy_bound(reference_source, reference_target, reference.get("sha256"), "accepted reference")
    video = candidate["video"]
    video_source = _safe_source_file(source_root, video.get("path"), "accepted video")
    video_target = target_dir / "source.mp4"
    video_hash = _copy_bound(video_source, video_target, video.get("sha256"), "accepted video")
    raw_target = output_root / "rawteacher-v2" / (clip_id + ".json")
    raw_hash = _copy_bound(raw_path, raw_target, _sha256_file(raw_path), "accepted raw teacher report")
    selected = selection["selected"]
    cohort = "uncalibrated_pilot_train" if candidate["cohort"] == "train" else "uncalibrated_pilot_validation"
    profile = freeze["profile_config"]
    return {
        "schema": "plumb-judge-v2-pilot-label-v1",
        "purpose": ROW_PURPOSE,
        "qualified": False,
        "label_source": LABEL_SOURCE,
        "teacher_profile": profile["profile_id"],
        "teacher_profile_config_sha256": _digest(freeze["profile_config_sha256"], "profile hash"),
        "input_profile": INPUT_PROFILE,
        "clip_id": clip_id,
        "source_lineage_id": candidate["source_lineage_id"],
        "cohort": cohort,
        "task": "close_drawer",
        "instruction": "Close the drawer",
        "rubric": profile["rubric"],
        "frames": frame_paths,
        "frame_sha256": frame_hashes,
        "frame_timestamps": [frame["timestamp"] for frame in candidate["frames"]],
        "timestamp_semantics": candidate.get("timestamp_semantics"),
        "reference_images": [{"path": str(reference_target.relative_to(output_root)), "sha256": reference_hash, "role": reference["role"], "provenance_uri": reference["source_uri"]}],
        "label": dict(selected["label"]),
        "teacher_selection": {"rule": freeze["profile_config"]["selection_rule"], "modal_count": selection["modal_count"], "semantic_tuple": list(selected["semantic_tuple"]), "sample_index": selected["sample_index"]},
        "raw_teacher_refs": [{"uri": str(raw_target.relative_to(output_root)), "sha256": raw_hash, "sample_index": selected["sample_index"]}],
        "video": {"path": str(video_target.relative_to(output_root)), "sha256": video_hash},
    }


def _audit_payload(source_root: Path, v2_root: Path, candidates: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any], decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {"schema_version": 1, "kind": "plumb_judge_v2_pilot_export_audit", "purpose": PURPOSE, "qualified": False, "status": "blocked_ineligible_no_dataset_created", "source_candidate_manifest": {"path": str(source_root / "candidate-inputs.json"), "sha256": _sha256_file(source_root / "candidate-inputs.json")}, "v2_freeze": {"path": str(v2_root / "freeze.json"), "sha256": _sha256_file(v2_root / "freeze.json"), "profile_config_sha256": freeze["profile_config_sha256"]}, "candidate_lineages": sorted(candidate["source_lineage_id"] for candidate in candidates.values()), "eligibility": {key: value for key, value in decision.items() if key != "accepted"}, "filled_gates": []}


def export(source_root: Path, v2_root: Path, output_root: Path, audit_output: Path) -> Dict[str, Any]:
    """Export only a fresh v2 set that passed pre-write eligibility checks."""

    if source_root.is_symlink() or v2_root.is_symlink() or not source_root.is_dir() or not v2_root.is_dir():
        raise ExportError("source and v2 roots must be existing non-symlink directories")
    source_root, v2_root = source_root.resolve(strict=True), v2_root.resolve(strict=True)
    candidates, candidate_path = _load_v1_candidates(source_root)
    freeze = _load_v2_freeze(v2_root, candidate_path)
    reports = _collect_reports(v2_root, candidates, freeze)
    decision = eligibility(candidates, reports)
    if not decision["eligible"]:
        if output_root.exists() or output_root.is_symlink():
            raise ExportError("ineligible v2 export must not reuse/create an output dataset root")
        _write_exclusive(audit_output, _audit_payload(source_root, v2_root, candidates, freeze, decision))
        return {"status": "blocked", "qualified": False, "dataset_created": False, "audit": str(audit_output), "eligibility": {key: value for key, value in decision.items() if key != "accepted"}}
    if output_root.exists() or output_root.is_symlink():
        raise ExportError("v2 output root already exists; fresh export is required")
    output_root.mkdir(mode=0o700, parents=True)
    try:
        rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "development_validation": []}
        teacher_hashes: Dict[str, Dict[str, str]] = {}
        candidate_states = []
        for clip_id, candidate in sorted(candidates.items()):
            record = reports.get(clip_id)
            selected = decision["accepted"].get(clip_id)
            if selected is None:
                reason = next(item["reason"] for item in decision["exclusions"] if item["clip_id"] == clip_id)
                candidate_states.append({"clip_id": clip_id, "source_lineage_id": candidate["source_lineage_id"], "split": candidate["cohort"], "status": "excluded", "exclusion_reason": reason})
                if record is not None:
                    teacher_hashes[clip_id] = {"uri": "source-v2/" + str(record[0].relative_to(v2_root)), "sha256": _sha256_file(record[0])}
                continue
            assert record is not None
            raw_path, report, recomputed = record
            row = _row_from_selection(source_root, output_root, candidate, raw_path, report, recomputed, freeze)
            split = candidate["cohort"]
            rows[split].append(row)
            teacher_hashes[clip_id] = {"uri": row["raw_teacher_refs"][0]["uri"], "sha256": row["raw_teacher_refs"][0]["sha256"]}
            candidate_states.append({"clip_id": clip_id, "source_lineage_id": candidate["source_lineage_id"], "split": split, "status": "accepted"})
        rows["train"].sort(key=lambda row: row["clip_id"])
        rows["development_validation"].sort(key=lambda row: row["clip_id"])
        labels_hash = _write_jsonl_exclusive(output_root / "teacher-v2-labels.jsonl", rows["train"] + rows["development_validation"])
        exclusions = [state for state in candidate_states if state["status"] == "excluded"]
        exclusions_hash = _write_jsonl_exclusive(output_root / "teacher-v2-exclusions.jsonl", exclusions)
        train_hash = _write_jsonl_exclusive(output_root / "train.jsonl", rows["train"])
        validation_hash = _write_jsonl_exclusive(output_root / "development_validation.jsonl", rows["development_validation"])
        lineages = sorted(candidate["source_lineage_id"] for candidate in candidates.values())
        _write_exclusive(output_root / "future_formal_exclusion_lineages.json", {"schema_version": 1, "purpose": "reserve_uncalibrated_teacher_v2_pilot_lineages", "qualified": False, "lineages": lineages, "sha256": _canonical_hash(lineages)})
        model, runtime = freeze["model"], freeze["runtime"]
        inputs = {
            "schema_version": 1, "schema": "plumb-judge-teacher-v2-pilot-inputs-v1", "purpose": ROW_PURPOSE, "qualified": False,
            "candidates": candidate_states,
            "raw_source_hashes": {
                "candidate_manifest": {"uri": "source-v1/candidate-inputs.json", "sha256": _sha256_file(candidate_path)},
                "model_manifest": {"uri": "source-v2/model-manifest", "sha256": _digest(model.get("manifest_sha256"), "v2 model manifest sha")},
                "runtime_lock": {"uri": "source-v2/runtime-lock", "sha256": _digest(runtime.get("sha256"), "v2 runtime lock sha")},
                "collection_config_sha256": _digest(freeze["profile_config_sha256"], "v2 profile config sha"),
                "teacher_reports": teacher_hashes,
                "labels": {"uri": "teacher-v2-labels.jsonl", "sha256": labels_hash},
                "exclusions": {"uri": "teacher-v2-exclusions.jsonl", "sha256": exclusions_hash},
                "splits": {"train": {"uri": "train.jsonl", "sha256": train_hash}, "development_validation": {"uri": "development_validation.jsonl", "sha256": validation_hash}},
            },
        }
        inputs_hash = _write_exclusive(output_root / "inputs.json", inputs)
        _write_exclusive(output_root / "export_freeze.json", {"schema_version": 1, "purpose": PURPOSE, "qualified": False, "inputs_sha256": inputs_hash, "profile_config": freeze["profile_config"], "profile_config_sha256": freeze["profile_config_sha256"], "source_candidate_manifest_sha256": _sha256_file(candidate_path), "source_lineages_sha256": _canonical_hash(lineages), "eligibility": {key: value for key, value in decision.items() if key != "accepted"}})
        for path in output_root.rglob("*"):
            if path.is_file(): os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(output_root, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return {"status": "exported_unqualified_v2_pilot_dataset", "qualified": False, "dataset_created": True, "output_root": str(output_root), "inputs_sha256": inputs_hash, "eligibility": {key: value for key, value in decision.items() if key != "accepted"}}
    except BaseException:
        # A partial fresh output remains visibly unusable; never delete source
        # evidence or silently retry into the same root.
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="original v1 prepared candidate dataset root")
    parser.add_argument("--v2-root", required=True, help="teacher-v2 diagnostic output root with freeze/raw reports")
    parser.add_argument("--output-root", required=True, help="new v2 dataset root; created only if eligible")
    parser.add_argument("--audit-output", required=True, help="new audit JSON path used only when eligibility fails")
    args = parser.parse_args(argv)
    try:
        result = export(Path(args.source_root), Path(args.v2_root), Path(args.output_root), Path(args.audit_output))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["dataset_created"] else 2
    except (ExportError, OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "qualified": False, "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
