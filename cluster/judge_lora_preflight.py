#!/usr/bin/env python3
"""Allocated-GPU compatibility preflight for the Qwen rubric-judge LoRA path.

This tool receives one hash-bound, explicitly unqualified diagnostic row made
from already recorded Qwen output and decoded source frames.  It executes one
assistant-only VLM forward/backward through the exact serving message layout.
It never creates a dataset split, adapter, optimizer step, training result,
calibration label, or qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


class PreflightError(ValueError):
    """Raised before a GPU call when the supplied diagnostic evidence is unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _digest_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PreflightError("%s must be a SHA-256 digest" % label)
    text = value.lower().strip()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PreflightError("%s must be a SHA-256 digest" % label)
    return "sha256:" + text


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError("unable to read %s" % label) from error
    if not isinstance(value, Mapping):
        raise PreflightError("%s must be a JSON object" % label)
    return dict(value)


def _frame_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreflightError("%s must be a non-empty relative frame path" % label)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PreflightError("%s escapes --row-root" % label)
    lexical = root / candidate
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise PreflightError("%s contains a symlink: %s" % (label, current))
        except OSError as error:
            raise PreflightError("unable to inspect %s" % label) from error
    path = lexical.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise PreflightError("%s resolves outside --row-root" % label) from error
    if not path.is_file():
        raise PreflightError("%s is not a regular source frame" % label)
    return path


def validate_row(row: Mapping[str, Any], row_root: Path) -> Dict[str, Any]:
    if row.get("purpose") != "framework_preflight_only" or row.get("qualified") is not False:
        raise PreflightError("row must explicitly be an unqualified framework_preflight_only diagnostic")
    if row.get("teacher_aggregation_status") != "unknown":
        raise PreflightError("diagnostic teacher aggregation must remain unknown")
    _digest_text(row.get("source_report_sha256"), "source_report_sha256")
    frames = row.get("frames")
    if not isinstance(frames, list) or len(frames) != 16:
        raise PreflightError("row must bind exactly 16 decoded source frames")
    # The row builder owns semantic pixel-hash verification against the source
    # report. Here we independently make sure every path remains a regular
    # source file before the VLM opens it.
    frame_hashes = row.get("frame_sha256")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != 16:
        raise PreflightError("row must retain a SHA-256 for every decoded frame")
    resolved = [_frame_path(row_root, value, "frames[%d]" % index) for index, value in enumerate(frames)]
    for index, path in enumerate(resolved):
        if _sha256(path) != _digest_text(frame_hashes[index], "frame_sha256[%d]" % index):
            raise PreflightError("decoded frame SHA-256 changed: frames[%d]" % index)
    references = row.get("reference_images")
    if not isinstance(references, list) or not references:
        raise PreflightError("row must bind at least one provenance-backed reference image")
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise PreflightError("reference_images[%d] must be an object" % index)
        path = _frame_path(row_root, reference.get("path"), "reference_images[%d].path" % index)
        if _sha256(path) != _digest_text(reference.get("sha256"), "reference_images[%d].sha256" % index):
            raise PreflightError("reference image SHA-256 changed: reference_images[%d]" % index)
    checked = dict(row)
    checked["frames"] = [str(path) for path in resolved]
    checked["reference_images"] = [
        {**dict(reference), "path": str(_frame_path(row_root, reference["path"], "reference image path"))}
        for reference in references
    ]
    return checked


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PreflightError("refusing to overwrite preflight report: %s" % path)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s-" % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PreflightError("preflight report was concurrently published") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise PreflightError("judge LoRA preflight requires an allocated Slurm GPU job")
    raw_row_path = Path(args.row)
    raw_row_root = Path(args.row_root)
    if raw_row_path.is_symlink() or raw_row_root.is_symlink() or not raw_row_root.is_dir():
        raise PreflightError("--row and --row-root must be existing non-symlink paths")
    row_path = raw_row_path.resolve(strict=True)
    row_root = raw_row_root.resolve(strict=True)
    row = validate_row(_load_object(row_path, "preflight row"), row_root)
    config = _load_object(Path(args.config).resolve(strict=True), "rendered one-arm config")
    from deploy.baseten.training.train_judge_lora import preflight_one_batch

    result = preflight_one_batch(
        config,
        row,
        model_root=str(Path(args.model_root).resolve(strict=True)),
        model_manifest=str(Path(args.model_manifest).resolve(strict=True)),
        seed=args.seed,
    )
    result.update(
        {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
            "source_row_sha256": _sha256(row_path),
            "source_frame_paths": row["frames"],
        }
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="one rendered preregistered arm JSON")
    parser.add_argument("--row", required=True, help="prepared unqualified diagnostic row JSON")
    parser.add_argument("--row-root", required=True, help="root under which row frame/reference paths resolve")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260919, help="recorded preflight RNG seed; not a training run")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        _write_no_clobber(Path(args.output), result)
        print(json.dumps({"output": args.output, "status": result["status"], "qualified": False}, sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "schema_version": 1,
            "kind": "plumb_judge_lora_framework_preflight",
            "qualified": False,
            "status": "blocked",
            "reason": str(error),
            "traceback": traceback.format_exc(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
        }
        try:
            _write_no_clobber(Path(args.output), failure)
        except (PreflightError, OSError) as write_error:
            failure["failure_receipt_error"] = str(write_error)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
