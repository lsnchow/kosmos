"""Training Job entrypoint: LoRA-distil the PLUMB rubric judge.

Importable without torch.  Every heavy import lives inside the function that
needs it, exactly like ``plumb/policies/*.py``, so this module can be checked by
pyflakes and unit-tested on a laptop with no GPU and no CUDA wheels.

Two modes:

``--check``
    Validate the rendered job config and the dataset on CPU, with no
    framework imports at all.  This is the mode that runs in CI and in
    ``docs/DISTILLATION_RUNBOOK.md`` step 4.  It refuses a config whose
    lineages are not provably disjoint, whose selection split is not
    development validation, or whose dataset rows carry an excluded lineage.
``--train`` (default inside the Training Job)
    Run the real LoRA fine-tune for one preregistered arm and write an arm
    result JSON.  If torch/transformers/peft/trl are absent, it exits nonzero
    with the missing package names.  It never substitutes a fabricated loss,
    accuracy, adapter digest, or GPU-second count for a real one, and it never
    writes an arm result for a run that did not finish.

What this file does not do: choose the arm (the preregistered search does),
decide whether the adapter may score anything (``plumb.distillation.
DistillationGate`` does, after a fresh held-out calibration and a paired
frozen-video comparison), or report a cost (the job's own billing does).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


#: Splits the job reads, and the cohort each must carry.  A row labelled with
#: any other cohort is refused rather than trained on.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "development_validation"
EXPECTED_COHORTS: Mapping[str, str] = {
    TRAIN_SPLIT: "distillation_train",
    VALIDATION_SPLIT: "distillation_dev_validation",
}

#: Fields every dataset row must carry.  ``frames`` is the 16 timestamped frame
#: references the judge protocol fixes; ``label`` is the teacher's structured
#: rubric output, not a free-text completion.
REQUIRED_ROW_FIELDS: Tuple[str, ...] = (
    "clip_id",
    "source_lineage_id",
    "cohort",
    "task",
    "instruction",
    "rubric",
    "frames",
    "frame_timestamps",
    "reference_images",
    "label",
)
REQUIRED_LABEL_FIELDS: Tuple[str, ...] = (
    "integrity",
    "collision",
    "progress",
    "completion_evidence",
    "evidence_frame_indices",
    "observable_reasons",
)
FRAMES_PER_CLIP = 16
SERVING_INPUT_PROFILE = "qwen_rubric_serving_messages_v1"
PREFLIGHT_INPUT_PROFILE = "qwen_rubric_preflight_video_reference_v1"

# These are the reviewed package versions in requirements.txt.  CUDA build
# suffixes are allowed for torch, but the resolved base version must match.
EXPECTED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "torch": "2.6.0",
    "transformers": "4.49.0",
    "peft": "0.14.0",
    "trl": "0.14.0",
}
DEFAULT_PER_DEVICE_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
MODEL_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$", re.IGNORECASE)

RESULT_FILENAME = "arm_result.json"
ADAPTER_DIRNAME = "adapter"


class TrainingConfigError(ValueError):
    """Raised for a job config or dataset this job refuses to train on."""


class TrainingRuntimeUnavailable(RuntimeError):
    """Raised when the training frameworks are not installed in this container."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrainingConfigError("%s must be a positive integer" % field)
    return value


def training_options(config: Mapping[str, Any]) -> Dict[str, int]:
    """Resolve the deliberately conservative one-example VLM batch contract.

    Sixteen images plus Qwen2.5-VL are not assumed to fit at a larger batch.
    Gradient accumulation is the only configurable effective-batch control and
    is recorded in every actual result/preflight report.
    """

    raw = config.get("training", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TrainingConfigError("job config training options must be an object")
    train_batch = _positive_int(
        raw.get("per_device_train_batch_size", DEFAULT_PER_DEVICE_BATCH_SIZE),
        "training.per_device_train_batch_size",
    )
    eval_batch = _positive_int(
        raw.get("per_device_eval_batch_size", DEFAULT_PER_DEVICE_BATCH_SIZE),
        "training.per_device_eval_batch_size",
    )
    accumulation = _positive_int(
        raw.get("gradient_accumulation_steps", DEFAULT_GRADIENT_ACCUMULATION_STEPS),
        "training.gradient_accumulation_steps",
    )
    if train_batch != DEFAULT_PER_DEVICE_BATCH_SIZE or eval_batch != DEFAULT_PER_DEVICE_BATCH_SIZE:
        raise TrainingConfigError(
            "only per-device batch size 1 is permitted until an allocated-GPU preflight records a larger fit"
        )
    return {
        "per_device_train_batch_size": train_batch,
        "per_device_eval_batch_size": eval_batch,
        "gradient_accumulation_steps": accumulation,
    }


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip()):
        raise TrainingConfigError("%s must be a SHA-256 digest" % field)
    return value.strip().lower()


def _bare_sha256(value: Any, field: str) -> str:
    return _sha256_text(value, field).split(":", 1)[-1]


def _regular_model_file(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TrainingConfigError("%s must be a non-empty relative path" % field)
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
        raise TrainingConfigError("%s escapes --model-root" % field)
    current = root
    for part in relative_path.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise TrainingConfigError("%s is missing: %s" % (field, current)) from error
        if stat.S_ISLNK(mode):
            raise TrainingConfigError("%s contains a symlink: %s" % (field, current))
    if not current.is_file():
        raise TrainingConfigError("%s is not a regular file: %s" % (field, current))
    return current


def _model_regular_files(root: Path, manifest_path: Path) -> List[Path]:
    """Enumerate every non-manifest file and reject links before model load."""

    files: List[Path] = []
    for parent, directories, names in os.walk(root, followlinks=False):
        directory = Path(parent)
        for name in list(directories) + list(names):
            path = directory / name
            if path.is_symlink():
                raise TrainingConfigError("--model-root contains a symlink: %s" % path)
        for name in names:
            path = directory / name
            if path.resolve() == manifest_path:
                continue
            if path.is_file():
                files.append(path)
    return sorted(files)


def _local_model_binding(
    model_root: str, model_manifest: str, student: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind the local-only model directory to the rendered student revision.

    This intentionally accepts no Hub/cache fallback.  The manifest must live
    under the supplied regular model root and name the same model ID and full
    revision as the job config.  It is a runtime compatibility/preflight
    binding, not an assertion that the model has passed Gate D.
    """

    root = Path(model_root)
    if not root.is_dir() or root.is_symlink():
        raise TrainingConfigError("--model-root must be an existing non-symlink directory")
    root = root.resolve(strict=True)
    manifest_path = Path(model_manifest)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TrainingConfigError("--model-manifest must be an existing regular file")
    manifest_path = manifest_path.resolve(strict=True)
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise TrainingConfigError("--model-manifest must be beneath --model-root") from error
    if manifest_path.stat().st_size > MODEL_MANIFEST_MAX_BYTES:
        raise TrainingConfigError("--model-manifest exceeds the 4 MiB safety limit")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingConfigError("--model-manifest is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise TrainingConfigError("--model-manifest must be a JSON object")
    declared_id = payload.get("model_id", payload.get("repo"))
    declared_revision = payload.get("model_revision", payload.get("revision"))
    if declared_id != student.get("model_id"):
        raise TrainingConfigError("model manifest model_id does not match the rendered student model")
    if declared_revision != student.get("model_revision"):
        raise TrainingConfigError("model manifest revision does not match the rendered student revision")
    files = payload.get("file_records", payload.get("files"))
    if not isinstance(files, list) or not files or not all(isinstance(record, Mapping) for record in files):
        raise TrainingConfigError("model manifest must retain non-empty file_records/files")
    listed: Dict[Path, Mapping[str, Any]] = {}
    for index, record in enumerate(files):
        path = _regular_model_file(root, record.get("path"), "model manifest file_records[%d].path" % index)
        relative = path.relative_to(root)
        if relative in listed:
            raise TrainingConfigError("model manifest repeats file path %s" % relative)
        expected_bytes = record.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise TrainingConfigError("model manifest file_records[%d].bytes must be a non-negative integer" % index)
        if path.stat().st_size != expected_bytes:
            raise TrainingConfigError("model file size does not match manifest: %s" % relative)
        expected_digest = _bare_sha256(record.get("sha256"), "model manifest file_records[%d].sha256" % index)
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise TrainingConfigError("model file SHA-256 does not match manifest: %s" % relative)
        listed[relative] = record
    actual_files = {path.relative_to(root) for path in _model_regular_files(root, manifest_path)}
    if actual_files != set(listed):
        missing = sorted(str(path) for path in actual_files - set(listed))
        extra = sorted(str(path) for path in set(listed) - actual_files)
        raise TrainingConfigError(
            "model manifest must bind every regular model file (unlisted=%s; absent=%s)"
            % (", ".join(missing) or "none", ", ".join(extra) or "none")
        )
    manifest_sha256 = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "model_root": str(root),
        "model_manifest": str(manifest_path),
        "model_manifest_sha256": manifest_sha256,
        "model_id": str(declared_id),
        "model_revision": str(declared_revision),
    }


# --------------------------------------------------------------------------
# Config and dataset validation.  No framework imports below this line.
# --------------------------------------------------------------------------


def load_job_config(path: str) -> Dict[str, Any]:
    """Load a payload rendered by ``deploy/baseten/training/job_config.py``."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        raise TrainingConfigError(
            "this config holds {0} payloads; a Training Job trains one arm, so pass one".format(len(payload))
        )
    if not isinstance(payload, dict):
        raise TrainingConfigError("job config must be a JSON object")
    return payload


def validate_job_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Refuse a config that has not cleared the distillation preconditions.

    The lineage decision itself was made by ``plumb.measurement.
    validate_source_lineage_leakage`` when the payload was rendered; this
    re-reads the recorded verdict inside the container so a hand-edited config
    cannot smuggle a leaky dataset past it.
    """

    hyperparameters = config.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise TrainingConfigError("job config lacks hyperparameters")
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping):
        raise TrainingConfigError("job config lacks dataset references")
    search = config.get("search")
    if not isinstance(search, Mapping) or not search.get("preregistration_hash"):
        raise TrainingConfigError(
            "job config lacks the preregistered search hash; an unpreregistered arm is not a search arm"
        )
    if datasets.get("lineage_disjointness") != "pass":
        raise TrainingConfigError(
            "dataset lineage disjointness is {0!r}, not 'pass'; training would cross a held-out or "
            "primary cohort".format(datasets.get("lineage_disjointness"))
        )
    if hyperparameters.get("selection_split") != VALIDATION_SPLIT:
        raise TrainingConfigError(
            "selection_split is {0!r}; an arm may only be selected on {1}".format(
                hyperparameters.get("selection_split"), VALIDATION_SPLIT
            )
        )
    early_stopping = hyperparameters.get("early_stopping")
    if not isinstance(early_stopping, Mapping):
        raise TrainingConfigError("job config lacks an early-stopping rule")
    if early_stopping.get("monitored_split") != VALIDATION_SPLIT:
        raise TrainingConfigError(
            "early stopping monitors {0!r}; it must monitor {1}".format(
                early_stopping.get("monitored_split"), VALIDATION_SPLIT
            )
        )
    for key in ("lora_r", "lora_alpha", "learning_rate", "num_train_epochs", "lora_target_modules"):
        if hyperparameters.get(key) in (None, ""):
            raise TrainingConfigError("job config lacks hyperparameter {0}".format(key))
    student = config.get("student_model")
    if not isinstance(student, Mapping) or not student.get("model_id") or not student.get("model_revision"):
        raise TrainingConfigError("job config lacks a revision-pinned student model")
    base_judge = config.get("base_judge")
    if not isinstance(base_judge, Mapping) or not base_judge.get("rubric_hash"):
        raise TrainingConfigError("job config lacks the base judge revision the rubric is frozen against")
    training_options(config)
    return dict(config)


def validate_preflight_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate only the frozen model/LoRA inputs for a no-training preflight.

    A framework preflight intentionally has no dataset, Gate-D base revision,
    or search-result authority.  Requiring those fields would invite callers
    to fabricate an otherwise absent formal dataset merely to exercise a
    tensor shape.  This route therefore cannot emit an arm result or metric.
    """

    if config.get("purpose") != "framework_preflight_only":
        raise TrainingConfigError("preflight config purpose must be framework_preflight_only")
    hyperparameters = config.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise TrainingConfigError("preflight config lacks hyperparameters")
    for key in ("lora_r", "lora_alpha", "learning_rate", "num_train_epochs", "lora_target_modules"):
        if hyperparameters.get(key) in (None, ""):
            raise TrainingConfigError("preflight config lacks hyperparameter %s" % key)
    student = config.get("student_model")
    if not isinstance(student, Mapping) or not student.get("model_id") or not student.get("model_revision"):
        raise TrainingConfigError("preflight config lacks a revision-pinned student model")
    training = config.get("training")
    if not isinstance(training, Mapping) or set(training) != {
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    }:
        raise TrainingConfigError("preflight config must explicitly freeze batch-size-1 and gradient accumulation options")
    training_options(config)
    return dict(config)


def validate_vlm_row(
    row: Mapping[str, Any],
    *,
    split: str,
    index: int,
    allow_preflight_profile: bool = False,
    require_dataset_identity: bool = True,
) -> Dict[str, Any]:
    """Validate the multimodal serving contract retained by a dataset row."""

    required = REQUIRED_ROW_FIELDS if require_dataset_identity else tuple(
        name for name in REQUIRED_ROW_FIELDS if name not in {"clip_id", "source_lineage_id", "cohort", "task"}
    )
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise TrainingConfigError("{0} row {1} lacks {2}".format(split, index, ", ".join(missing)))
    profile = row.get("input_profile")
    preflight = allow_preflight_profile and row.get("purpose") == "framework_preflight_only"
    if not preflight and profile != SERVING_INPUT_PROFILE:
        raise TrainingConfigError(
            "%s row %d input_profile %r is not the frozen serving VLM message layout" % (split, index, profile)
        )
    frames = row["frames"]
    if not isinstance(frames, list) or len(frames) != FRAMES_PER_CLIP or not all(isinstance(path, str) and path for path in frames):
        raise TrainingConfigError(
            "%s row %d has invalid frame refs; the judge protocol fixes exactly %d" % (split, index, FRAMES_PER_CLIP)
        )
    references = row.get("reference_images")
    if not isinstance(references, list) or not references or not all(isinstance(item, Mapping) for item in references):
        raise TrainingConfigError("%s row %d must retain one or more serving reference-image paths" % (split, index))
    for reference in references:
        if not isinstance(reference.get("path"), str) or not reference["path"]:
            raise TrainingConfigError("%s row %d reference image lacks a path" % (split, index))
        _sha256_text(reference.get("sha256"), "%s row %d reference image sha256" % (split, index))
    timestamps = row.get("frame_timestamps")
    if not isinstance(timestamps, list) or len(timestamps) != FRAMES_PER_CLIP:
        raise TrainingConfigError("%s row %d must retain 16 frame timestamps" % (split, index))
    previous: Optional[float] = None
    for timestamp in timestamps:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TrainingConfigError("%s row %d has a non-numeric frame timestamp" % (split, index))
        value = float(timestamp)
        if value != value or value in (float("inf"), float("-inf")) or (previous is not None and value <= previous):
            raise TrainingConfigError("%s row %d frame timestamps must be finite and strictly increasing" % (split, index))
        previous = value
    label = row["label"]
    if not isinstance(label, Mapping):
        raise TrainingConfigError("%s row %d label is not an object" % (split, index))
    label_missing = [name for name in REQUIRED_LABEL_FIELDS if name not in label]
    if label_missing:
        raise TrainingConfigError("%s row %d label lacks %s" % (split, index, ", ".join(label_missing)))
    return dict(row)


def load_split(dataset_dir: str, split: str, *, excluded_lineages: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Read and validate one JSONL split; refuse an excluded or malformed row."""

    if split not in EXPECTED_COHORTS:
        raise TrainingConfigError("unknown split {0!r}".format(split))
    source = Path(dataset_dir) / "{0}.jsonl".format(split)
    if not source.exists():
        raise TrainingConfigError("dataset split file is missing: {0}".format(source))
    excluded = {str(value) for value in excluded_lineages}
    rows: List[Dict[str, Any]] = []
    seen_clips = set()
    with source.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise TrainingConfigError("{0} row {1} is not an object".format(split, index))
            validated = validate_vlm_row(row, split=split, index=index)
            if row["cohort"] != EXPECTED_COHORTS[split]:
                raise TrainingConfigError(
                    "{0} row {1} carries cohort {2!r}; expected {3!r}".format(
                        split, index, row["cohort"], EXPECTED_COHORTS[split]
                    )
                )
            if str(row["source_lineage_id"]) in excluded:
                raise TrainingConfigError(
                    "{0} row {1} carries excluded lineage {2}; this is held-out or primary data".format(
                        split, index, row["source_lineage_id"]
                    )
                )
            if row["clip_id"] in seen_clips:
                raise TrainingConfigError("{0} repeats clip {1}".format(split, row["clip_id"]))
            seen_clips.add(row["clip_id"])
            rows.append(validated)
    if not rows:
        raise TrainingConfigError("{0} split is empty".format(split))
    return rows


def check(config_path: str, dataset_dir: Optional[str], excluded_path: Optional[str]) -> Dict[str, Any]:
    """Validate everything this job would train on, without importing torch."""

    config = validate_job_config(load_job_config(config_path))
    excluded: List[str] = []
    if excluded_path:
        with open(excluded_path, "r", encoding="utf-8") as handle:
            excluded = [str(value) for value in json.load(handle)]
    summary: Dict[str, Any] = {
        "status": "config_valid",
        "job_name": config.get("job_name"),
        "arm_id": (config.get("search") or {}).get("arm_id"),
        "preregistration_hash": (config.get("search") or {}).get("preregistration_hash"),
        "lineage_disjointness": (config.get("datasets") or {}).get("lineage_disjointness"),
        "excluded_lineages_checked": len(excluded),
        "splits": {},
    }
    if dataset_dir:
        for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
            rows = load_split(dataset_dir, split, excluded_lineages=excluded)
            summary["splits"][split] = {
                "rows": len(rows),
                "lineages": len({row["source_lineage_id"] for row in rows}),
            }
        train_lineages = {row["source_lineage_id"] for row in load_split(dataset_dir, TRAIN_SPLIT, excluded_lineages=excluded)}
        validation_lineages = {
            row["source_lineage_id"]
            for row in load_split(dataset_dir, VALIDATION_SPLIT, excluded_lineages=excluded)
        }
        shared = sorted(train_lineages & validation_lineages)
        if shared:
            raise TrainingConfigError(
                "train and development-validation share lineages, so early stopping would watch its own "
                "training data: {0}".format(", ".join(shared))
            )
        summary["status"] = "config_and_dataset_valid"
    else:
        summary["splits"] = {"status": "not_checked", "reason": "no --dataset-dir supplied"}
    return summary


# --------------------------------------------------------------------------
# Training.  Framework imports are lazy and their absence is reported by name.
# --------------------------------------------------------------------------


def _load_frameworks() -> Dict[str, Any]:
    """Import the training stack, or name exactly what is missing."""

    missing: List[str] = []
    modules: Dict[str, Any] = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            modules[name] = __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise TrainingRuntimeUnavailable(
            "training frameworks are not installed in this container: "
            + ", ".join(missing)
            + ". Install deploy/baseten/training/requirements.txt. Nothing was trained and no result was written."
        )
    observed = {name: str(getattr(module, "__version__", "")) for name, module in modules.items() if name in EXPECTED_RUNTIME_VERSIONS}
    mismatches = []
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = observed.get(name, "")
        if actual.split("+", 1)[0] != expected:
            mismatches.append("%s=%s (expected %s)" % (name, actual or "unknown", expected))
    if mismatches:
        raise TrainingRuntimeUnavailable(
            "training runtime versions do not match deploy/baseten/training/requirements.txt: "
            + ", ".join(mismatches)
        )
    return modules


def _digest_directory(directory: Path) -> str:
    """Hash the adapter files actually written, in a stable order."""

    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _claim_fresh_output_dir(output_dir: str, *, dataset_dir: str, model_root: str) -> Path:
    """Atomically reserve a fresh output directory without implicit resume.

    Training must not write beside or inside its immutable inputs, and a failed
    or partial prior arm must never be silently resumed under the same path.
    """

    dataset = Path(dataset_dir)
    model = Path(model_root)
    if not dataset.is_dir() or dataset.is_symlink() or not model.is_dir() or model.is_symlink():
        raise TrainingConfigError("dataset-dir and model-root must be existing non-symlink directories")
    dataset = dataset.resolve(strict=True)
    model = model.resolve(strict=True)
    candidate = Path(os.path.abspath(output_dir))
    if candidate.name in ("", ".", ".."):
        raise TrainingConfigError("output-dir must name a new child directory")
    # This lexical check prevents creating a parent below either immutable
    # input before the resolved containment check below.
    for input_path in (dataset, model):
        if _paths_overlap(candidate, input_path):
            raise TrainingConfigError("output-dir may not overlap dataset-dir or model-root")
    if candidate.exists():
        raise TrainingConfigError("output-dir already exists; implicit resume/overwrite is forbidden")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.parent.is_symlink():
        raise TrainingConfigError("output-dir parent may not be a symlink")
    parent = candidate.parent.resolve(strict=True)
    target = parent / candidate.name
    for input_path in (dataset, model):
        if _paths_overlap(target, input_path):
            raise TrainingConfigError("output-dir may not overlap dataset-dir or model-root")
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as error:
        raise TrainingConfigError("output-dir was concurrently claimed; no implicit resume is permitted") from error
    return target


def _serving_messages(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild the exact Qwen judge system/user message layout.

    Pixel paths stay out of the message, as in serving; references are counted
    in the template then supplied separately to the processor.  This prevents
    a text-and-16-images surrogate from being called judge distillation.
    """

    from plumb.policies.judge import JudgeRequest, QwenRubricJudge, ReferenceImage

    references = []
    for index, raw in enumerate(row["reference_images"]):
        digest = _sha256_text(raw["sha256"], "reference_images[%d].sha256" % index).split(":", 1)[-1]
        references.append(
            ReferenceImage(
                image=object(),
                source_uri=str(raw.get("provenance_uri", raw.get("source_uri", "preflight-reference-%d" % index))),
                sha256=digest,
            )
        )
    request = JudgeRequest(
        frames=tuple(object() for _ in row["frames"]),
        frame_timestamps=tuple(float(value) for value in row["frame_timestamps"]),
        reference_images=tuple(references),
        diagnostic_mode=True,
        task_instruction=str(row["instruction"]),
        task_rubric=str(row["rubric"]),
    )
    return [dict(message) for message in QwenRubricJudge._messages(request, format_retry=bool(row.get("prompt_format_retry", False)))]


def _build_collator(processor: Any) -> Any:
    """Collate VLM examples with loss only on assistant rubric tokens.

    Prompt, image placeholder, and chat-control tokens are all masked.  The
    prefix is independently template-rendered and must be an exact prefix of
    the full rendered conversation; if a processor changes that boundary this
    fails rather than training on a silently shifted target.
    """

    from PIL import Image

    def collate(batch: Sequence[Mapping[str, Any]]) -> Any:
        conversations = []
        prefixes = []
        images = []
        videos = []
        for row in batch:
            frames = [Image.open(ref).convert("RGB") for ref in row["frames"]]
            references = [Image.open(reference["path"]).convert("RGB") for reference in row["reference_images"]]
            prefix = _serving_messages(row)
            prefixes.append(prefix)
            conversations.append(prefix + [{"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}])
            # This exactly matches QwenRubricJudge: one 16-frame video plus
            # separate reference images, never sixteen unrelated images.
            videos.append(frames)
            images.append(references)
        texts = [
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            for conversation in conversations
        ]
        prefix_texts = [
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in prefixes
        ]
        inputs = processor(text=texts, images=images, videos=videos, return_tensors="pt", padding=True)
        prefix_inputs = processor(text=prefix_texts, images=images, videos=videos, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        full_mask = inputs.get("attention_mask")
        prefix_mask = prefix_inputs.get("attention_mask")
        if full_mask is None or prefix_mask is None:
            raise TrainingConfigError("processor must return attention_mask for assistant-only loss masking")
        # Attention-mask padding rather than token-ID padding is essential:
        # Qwen may use EOS as its pad token, and the assistant's terminating
        # EOS/end-of-turn token is part of the desired rubric completion.
        labels[full_mask == 0] = -100
        for index in range(len(batch)):
            prefix_length = int(prefix_mask[index].sum().item())
            full_length = int(full_mask[index].sum().item())
            if prefix_length < 1 or prefix_length >= full_length:
                raise TrainingConfigError("assistant target is empty or the processor did not preserve a prompt boundary")
            if not (inputs["input_ids"][index, :prefix_length] == prefix_inputs["input_ids"][index, :prefix_length]).all().item():
                raise TrainingConfigError("processor full conversation does not retain the exact rendered user-prefix boundary")
            labels[index, :prefix_length] = -100
        media_token_ids = set()
        for attribute in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "vision_token_id",
        ):
            value = getattr(processor, attribute, None)
            if isinstance(value, int):
                media_token_ids.add(value)
        for token_id in media_token_ids:
            labels[inputs["input_ids"] == token_id] = -100
        if any(not bool((labels[index] != -100).any().item()) for index in range(len(batch))):
            raise TrainingConfigError("assistant-only masking left no supervised rubric tokens in a batch row")
        inputs["labels"] = labels
        return inputs

    return collate


def _as_training_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Turn dataset rows into prompt/target pairs over the frozen rubric schema."""

    prepared = []
    for row in rows:
        prepared.append(
            {
                "frames": list(row["frames"]),
                "reference_images": [dict(reference) for reference in row["reference_images"]],
                "frame_timestamps": list(row["frame_timestamps"]),
                "instruction": row["instruction"],
                "rubric": row["rubric"],
                "prompt_format_retry": bool(row.get("prompt_format_retry", False)),
                "target": json.dumps(dict(row["label"]), sort_keys=True, ensure_ascii=False),
            }
        )
    return prepared


def preflight_one_batch(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    model_root: str,
    model_manifest: str,
    seed: int = 20260919,
) -> Dict[str, Any]:
    """Run one real-row LoRA forward/backward compatibility check on an GPU.

    This neither trains an arm nor emits an adapter/result.  A diagnostic raw
    judge vote may exercise tensor layout and gradient flow, but cannot become
    a training label, a calibration label, a selected metric, or a Gate claim.
    """

    config = validate_preflight_config(config)
    validated = validate_vlm_row(
        row,
        split="preflight",
        index=0,
        allow_preflight_profile=True,
        require_dataset_identity=False,
    )
    if validated.get("purpose") != "framework_preflight_only" or validated.get("qualified") is not False:
        raise TrainingConfigError("preflight requires an explicitly unqualified framework_preflight_only source row")
    if validated.get("teacher_aggregation_status") != "unknown":
        raise TrainingConfigError("preflight source must retain its unknown teacher aggregation; no selected label is accepted")
    modules = _load_frameworks()
    torch_module = modules["torch"]
    if not bool(torch_module.cuda.is_available()):
        raise TrainingRuntimeUnavailable("preflight requires an allocated CUDA GPU; CPU tensor shape checks are insufficient")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingConfigError("preflight seed must be an integer")
    # Seed before base/LoRA construction: adapter initialization itself is
    # random, so seeding only before the forward pass is not reproducible.
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    student = config["student_model"]
    binding = _local_model_binding(model_root, model_manifest, student)
    transformers, peft = modules["transformers"], modules["peft"]
    hyperparameters = config["hyperparameters"]
    load_kwargs: Dict[str, Any] = {
        "revision": student["model_revision"],
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch_module.bfloat16,
        "use_safetensors": True,
    }
    processor = transformers.AutoProcessor.from_pretrained(
        binding["model_root"], **{key: value for key, value in load_kwargs.items() if key != "torch_dtype"}
    )
    base_model = transformers.AutoModelForVision2Seq.from_pretrained(binding["model_root"], **load_kwargs)
    model = peft.get_peft_model(
        base_model,
        peft.LoraConfig(
            r=int(hyperparameters["lora_r"]),
            lora_alpha=int(hyperparameters["lora_alpha"]),
            lora_dropout=float(hyperparameters["lora_dropout"]),
            target_modules=list(hyperparameters["lora_target_modules"]),
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model = model.to("cuda")
    model.train()
    torch_module.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prepared = _as_training_rows([validated])
    inputs = _build_collator(processor)(prepared)
    inputs = {key: value.to("cuda") if callable(getattr(value, "to", None)) else value for key, value in inputs.items()}
    output = model(**inputs)
    loss = getattr(output, "loss", None)
    if loss is None or not bool(torch_module.isfinite(loss).item()):
        raise TrainingConfigError("preflight model returned no finite assistant-only loss")
    loss.backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable < 1:
        raise TrainingConfigError("preflight found no trainable LoRA parameters")
    gradient_parameters = 0
    gradient_square_sum = 0.0
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not bool(torch_module.isfinite(gradient).all().item()):
            raise TrainingConfigError("preflight found a non-finite LoRA gradient")
        norm = float(torch_module.linalg.vector_norm(gradient.float()).item())
        if norm > 0.0:
            gradient_parameters += 1
            gradient_square_sum += norm * norm
    if gradient_parameters < 1 or gradient_square_sum <= 0.0:
        raise TrainingConfigError("preflight found no finite non-zero gradient on a trainable LoRA parameter")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1:
        raise TrainingRuntimeUnavailable("preflight requires exactly one CUDA_VISIBLE_DEVICES allocation to bind its GPU UUID")
    try:
        identity = subprocess.run(
            ["nvidia-smi", "--id=" + visible[0], "--query-gpu=uuid,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        raise TrainingRuntimeUnavailable("preflight could not record allocated GPU UUID via nvidia-smi") from error
    if not identity:
        raise TrainingRuntimeUnavailable("preflight received an empty GPU UUID from nvidia-smi")
    return {
        "schema_version": 1,
        "kind": "plumb_judge_lora_framework_preflight",
        "qualified": False,
        "status": "completed_unqualified_preflight",
        "purpose": "framework_preflight_only",
        "reason": "One diagnostic source row verified VLM tensor layout and LoRA backward only; no training, calibration, or selection occurred.",
        "teacher_aggregation_status": "unknown",
        "input_profile": PREFLIGHT_INPUT_PROFILE,
        "preflight_seed": seed,
        "preflight_seed_scope": "set_before_base_model_and_lora_construction",
        "source_report_sha256": validated.get("source_report_sha256"),
        "model_binding": binding,
        "runtime_versions": {
            name: str(getattr(module, "__version__", ""))
            for name, module in modules.items()
            if name in EXPECTED_RUNTIME_VERSIONS
        },
        "loss": float(loss.detach().cpu().item()),
        "trainable_parameter_count": int(trainable),
        "trainable_parameters_with_nonzero_gradient": int(gradient_parameters),
        "lora_gradient_l2_norm": gradient_square_sum ** 0.5,
        "gpu_peak_memory_bytes": int(torch_module.cuda.max_memory_allocated()),
        "gpu_identity": identity,
        "wall_seconds": time.monotonic() - started,
        "timing_scope": "collation_plus_one_forward_backward; excludes model-manifest verification, imports, and model/LoRA construction",
        "training_options": training_options(config),
    }


def train(
    config: Mapping[str, Any],
    dataset_dir: str,
    output_dir: str,
    *,
    excluded_lineages: Sequence[str] = (),
    model_root: Optional[str] = None,
    model_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one arm and return its recorded result. Raises if it cannot finish."""

    modules = _load_frameworks()
    torch_module = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    trl = modules["trl"]

    hyperparameters = config["hyperparameters"]
    early_stopping = hyperparameters["early_stopping"]
    student = config["student_model"]
    arm_id = config["search"]["arm_id"]
    if not model_root or not model_manifest:
        raise TrainingConfigError("--model-root and --model-manifest are required; Hub/cache fallback is forbidden")
    model_binding = _local_model_binding(model_root, model_manifest, student)
    options = training_options(config)

    train_rows = _as_training_rows(load_split(dataset_dir, TRAIN_SPLIT, excluded_lineages=excluded_lineages))
    validation_rows = _as_training_rows(
        load_split(dataset_dir, VALIDATION_SPLIT, excluded_lineages=excluded_lineages)
    )

    load_kwargs: Dict[str, Any] = {
        "revision": student["model_revision"],
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch_module.bfloat16,
        "use_safetensors": True,
    }
    processor = transformers.AutoProcessor.from_pretrained(model_binding["model_root"], **{
        key: value for key, value in load_kwargs.items() if key != "torch_dtype"
    })
    model = transformers.AutoModelForVision2Seq.from_pretrained(model_binding["model_root"], **load_kwargs)

    lora_config = peft.LoraConfig(
        r=int(hyperparameters["lora_r"]),
        lora_alpha=int(hyperparameters["lora_alpha"]),
        lora_dropout=float(hyperparameters["lora_dropout"]),
        target_modules=list(hyperparameters["lora_target_modules"]),
        task_type="CAUSAL_LM",
        bias="none",
    )
    args = trl.SFTConfig(
        output_dir=output_dir,
        learning_rate=float(hyperparameters["learning_rate"]),
        num_train_epochs=float(hyperparameters["num_train_epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=early_stopping["mode"] == "max",
        logging_strategy="epoch",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        per_device_train_batch_size=options["per_device_train_batch_size"],
        per_device_eval_batch_size=options["per_device_eval_batch_size"],
        gradient_accumulation_steps=options["gradient_accumulation_steps"],
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=modules["datasets"].Dataset.from_list(train_rows),
        eval_dataset=modules["datasets"].Dataset.from_list(validation_rows),
        peft_config=lora_config,
        processing_class=processor,
        data_collator=_build_collator(processor),
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping["patience"]),
                early_stopping_threshold=float(early_stopping["min_delta"]),
            )
        ],
    )
    started = time.monotonic()
    train_output = trainer.train()
    metrics = trainer.evaluate()
    wall_seconds = time.monotonic() - started

    adapter_dir = Path(output_dir) / ADAPTER_DIRNAME
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    epochs_completed = train_output.metrics.get("epoch")
    result = {
        "schema_version": 1,
        "arm_id": arm_id,
        "status": "completed",
        "selection_metric": hyperparameters["selection_metric"],
        "selection_split": VALIDATION_SPLIT,
        "metric_value": metrics.get("eval_loss"),
        "metric_source": "transformers_trainer_evaluate_eval_loss",
        "epochs_completed": int(epochs_completed) if epochs_completed else None,
        "early_stopped_at_epoch": (
            int(epochs_completed)
            if epochs_completed and float(epochs_completed) < float(hyperparameters["num_train_epochs"])
            else None
        ),
        "adapter_uri": str(adapter_dir),
        "adapter_sha256": _digest_directory(adapter_dir),
        "train_rows": len(train_rows),
        "development_validation_rows": len(validation_rows),
        "wall_seconds": wall_seconds,
        "reported_gpu_seconds": None,
        "gpu_seconds_note": (
            "Wall seconds are measured here. Allocated GPU-seconds come from the Training Job's own "
            "resource ledger and are filled in from it; null means unknown, never zero."
        ),
        "preregistration_hash": config["search"]["preregistration_hash"],
        "base_judge_rubric_hash": config["base_judge"]["rubric_hash"],
        "model_binding": model_binding,
        "training_options": options,
        "runtime_versions": {
            name: str(getattr(module, "__version__", ""))
            for name, module in modules.items()
            if name in EXPECTED_RUNTIME_VERSIONS
        },
        "trainer_log_history": list(getattr(trainer.state, "log_history", ())),
    }
    if result["metric_value"] is None:
        raise TrainingConfigError(
            "the trainer reported no eval_loss, so this arm has no development-validation result; "
            "refusing to write an arm result without one"
        )
    return result


def write_result(output_dir: str, result: Mapping[str, Any]) -> str:
    destination = Path(output_dir) / RESULT_FILENAME
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise TrainingConfigError("claimed output directory is missing or a symlink")
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise TrainingConfigError("arm_result.json already exists; overwriting a training result is forbidden") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return str(destination)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_judge_lora.py",
        description="LoRA-distil the PLUMB rubric judge for one preregistered arm.",
    )
    parser.add_argument("--config", required=True, help="One rendered Training Jobs payload (JSON)")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("PLUMB_DISTILL_DATASET_DIR"),
        help="Directory holding train.jsonl and development_validation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("PLUMB_DISTILL_OUTPUT_DIR"),
        help="Where the adapter and arm_result.json are written",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("PLUMB_MODEL_ROOT"),
        help="immutable local Qwen model directory; Hub/cache fallback is forbidden",
    )
    parser.add_argument(
        "--model-manifest",
        default=os.environ.get("PLUMB_MODEL_MANIFEST"),
        help="model-root-contained manifest binding the configured model ID and revision",
    )
    parser.add_argument(
        "--excluded-lineages",
        help="JSON list of held-out and primary lineage IDs no row may carry",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the config and dataset on CPU and exit without importing any framework",
    )
    args = parser.parse_args(argv)

    try:
        if args.check:
            summary = check(args.config, args.dataset_dir, args.excluded_lineages)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if not args.dataset_dir or not args.output_dir:
            raise TrainingConfigError("--dataset-dir and --output-dir are required to train")
        config = validate_job_config(load_job_config(args.config))
        excluded: List[str] = []
        if args.excluded_lineages:
            with open(args.excluded_lineages, "r", encoding="utf-8") as handle:
                excluded = [str(value) for value in json.load(handle)]
        result = train(
            config,
            args.dataset_dir,
            args.output_dir,
            excluded_lineages=excluded,
            model_root=args.model_root,
            model_manifest=args.model_manifest,
        )
        path = write_result(args.output_dir, result)
        print("arm {0} finished; result written to {1}".format(result["arm_id"], path))
        return 0
    except TrainingRuntimeUnavailable as error:
        print("training runtime unavailable: {0}".format(error), file=sys.stderr)
        return 3
    except TrainingConfigError as error:
        print("refusing to train: {0}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
