#!/usr/bin/env python3
"""One bounded, explicitly unqualified Qwen rubric-judge LoRA pilot.

This is deliberately not ``train_judge_lora.py``'s formal distillation route.
It accepts only a frozen 12/4 development-only, teacher-labelled pilot panel,
reserves every source lineage against later formal use, and records losses as
engineering observations.  It does not create an ArmResult, select a formal
LoRA configuration, establish label quality, calibrate a judge, or update any
Gate record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PILOT_PURPOSE = "uncalibrated_judge_distillation_pilot"
PILOT_LABEL_SOURCE = "uncalibrated_teacher_structured_sample"
TRAIN_FILE = "train.jsonl"
VALIDATION_FILE = "development_validation.jsonl"
INPUTS_FILE = "inputs.json"
TRAIN_COHORT = "uncalibrated_pilot_train"
VALIDATION_COHORT = "uncalibrated_pilot_validation"
PILOT_TASK = "close_drawer"
PILOT_MAX_TRAIN_ROWS = 12
PILOT_MAX_VALIDATION_ROWS = 4
PILOT_MIN_TRAIN_ROWS = 2
PILOT_MIN_VALIDATION_ROWS = 1
PILOT_EPOCHS = 2
PILOT_SEED = 20260919
PILOT_LORA_R = 64
PILOT_LORA_ALPHA = 32
PILOT_LR = 1e-4
PILOT_TARGET_MODULES = ("q_proj", "v_proj")
ALLOWED_TIMESTAMP_SEMANTICS = ("unverified_nominal_dataset_times", "source_parquet_nominal_timestamp")
_SHA256_HEX = set("0123456789abcdef")


class PilotError(ValueError):
    """A pilot input or state is unsafe; no optimizer step may follow."""


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise PilotError("%s must be a SHA-256 digest" % label)
    text = value.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise PilotError("%s must be a SHA-256 digest" % label)
    return "sha256:" + text


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PilotError("%s must be a regular file" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotError("unable to read %s" % label) from error
    if not isinstance(value, Mapping):
        raise PilotError("%s must contain a JSON object" % label)
    return dict(value)


def _secure_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PilotError("%s must be a non-empty relative path" % label)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise PilotError("%s escapes dataset root" % label)
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise PilotError("%s is missing: %s" % (label, current)) from error
        if stat.S_ISLNK(mode):
            raise PilotError("%s contains a symlink: %s" % (label, current))
    if not current.is_file():
        raise PilotError("%s is not a regular file: %s" % (label, current))
    return current


def _finite_timestamp_sequence(values: Any, label: str) -> None:
    if not isinstance(values, list) or len(values) != 16:
        raise PilotError("%s must contain exactly 16 timestamps" % label)
    previous = None
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PilotError("%s[%d] must be finite numeric" % (label, index))
        if previous is not None and float(value) <= previous:
            raise PilotError("%s must be strictly increasing" % label)
        previous = float(value)


def _validate_row(raw: Mapping[str, Any], *, split: str, index: int, root: Path) -> Dict[str, Any]:
    """Validate one hash-bound pilot row and resolve its media only after hash checks."""

    from deploy.baseten.training import train_judge_lora as trainer

    expected_cohort = TRAIN_COHORT if split == "train" else VALIDATION_COHORT
    if raw.get("purpose") != PILOT_PURPOSE or raw.get("qualified") is not False:
        raise PilotError("%s row %d is not explicitly an unqualified pilot row" % (split, index))
    if raw.get("label_source") != PILOT_LABEL_SOURCE:
        raise PilotError("%s row %d does not declare the uncalibrated teacher label source" % (split, index))
    if raw.get("cohort") != expected_cohort:
        raise PilotError("%s row %d has the wrong pilot cohort" % (split, index))
    if raw.get("task") != PILOT_TASK:
        raise PilotError("pilot rows must be the one reviewed task %s" % PILOT_TASK)
    if raw.get("timestamp_semantics") not in ALLOWED_TIMESTAMP_SEMANTICS:
        raise PilotError("pilot rows must preserve an explicitly unverified nominal timestamp_semantics value")
    if raw.get("input_profile") != trainer.SERVING_INPUT_PROFILE:
        raise PilotError("pilot row does not preserve the serving Qwen video/reference message profile")
    if not isinstance(raw.get("clip_id"), str) or not raw["clip_id"].strip():
        raise PilotError("%s row %d lacks clip_id" % (split, index))
    if not isinstance(raw.get("source_lineage_id"), str) or not raw["source_lineage_id"].strip():
        raise PilotError("%s row %d lacks source_lineage_id" % (split, index))
    raw_refs = raw.get("raw_teacher_refs")
    if not isinstance(raw_refs, list) or not raw_refs or not all(isinstance(item, Mapping) for item in raw_refs):
        raise PilotError("%s row %d must retain raw_teacher_refs" % (split, index))
    for ref_index, reference in enumerate(raw_refs):
        if not isinstance(reference.get("uri"), str) or not reference["uri"].strip():
            raise PilotError("raw_teacher_refs[%d] lacks uri" % ref_index)
        path = _secure_file(root, reference["uri"], "raw_teacher_refs[%d].uri" % ref_index)
        if _sha256_file(path) != _digest(reference.get("sha256"), "raw_teacher_refs[%d].sha256" % ref_index):
            raise PilotError("%s row %d raw teacher report hash changed before load" % (split, index))
    trainer.validate_vlm_row(raw, split=split, index=index)
    _finite_timestamp_sequence(raw.get("frame_timestamps"), "%s row %d frame_timestamps" % (split, index))
    frame_hashes = raw.get("frame_sha256")
    if not isinstance(frame_hashes, list) or len(frame_hashes) != 16:
        raise PilotError("%s row %d must retain exactly 16 saved-frame SHA-256 values" % (split, index))
    frames: List[str] = []
    for frame_index, reference in enumerate(raw["frames"]):
        path = _secure_file(root, reference, "%s row %d frames[%d]" % (split, index, frame_index))
        if _sha256_file(path) != _digest(frame_hashes[frame_index], "frame_sha256[%d]" % frame_index):
            raise PilotError("%s row %d frame hash changed before load" % (split, index))
        frames.append(str(path))
    references = []
    for reference_index, reference in enumerate(raw["reference_images"]):
        if not isinstance(reference.get("role"), str) or not reference["role"].strip():
            raise PilotError("%s row %d reference[%d] lacks role" % (split, index, reference_index))
        if not isinstance(reference.get("provenance_uri"), str) or not reference["provenance_uri"].strip():
            raise PilotError("%s row %d reference[%d] lacks provenance_uri" % (split, index, reference_index))
        path = _secure_file(root, reference.get("path"), "%s row %d reference[%d]" % (split, index, reference_index))
        if _sha256_file(path) != _digest(reference.get("sha256"), "reference_images[%d].sha256" % reference_index):
            raise PilotError("%s row %d reference hash changed before load" % (split, index))
        references.append({**dict(reference), "path": str(path), "sha256": _digest(reference["sha256"], "reference sha256")})
    return {**dict(raw), "frames": frames, "reference_images": references}


def _load_final_inputs(root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], str, Dict[str, str]]:
    path = _secure_file(root, INPUTS_FILE, "final pilot inputs manifest")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotError("inputs.json is not strict JSON") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or payload.get("purpose") != PILOT_PURPOSE or payload.get("qualified") is not False:
        raise PilotError("inputs.json must be a schema-v1 object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16 or not all(isinstance(row, Mapping) for row in candidates):
        raise PilotError("inputs.json must retain exactly all 16 candidate rows")
    by_clip: Dict[str, Dict[str, Any]] = {}
    lineages = set()
    for index, candidate in enumerate(candidates):
        clip_id = candidate.get("clip_id")
        lineage = candidate.get("source_lineage_id")
        split = candidate.get("split")
        status = candidate.get("status")
        if not isinstance(clip_id, str) or not clip_id or not isinstance(lineage, str) or not lineage:
            raise PilotError("inputs.json candidate %d lacks clip/lineage identity" % index)
        if clip_id in by_clip or lineage in lineages:
            raise PilotError("inputs.json repeats a candidate clip or source lineage")
        if split not in {"train", "development_validation"} or status not in {"accepted", "excluded"}:
            raise PilotError("inputs.json candidate %s has invalid split/status" % clip_id)
        if status == "excluded" and (not isinstance(candidate.get("exclusion_reason"), str) or not candidate["exclusion_reason"].strip()):
            raise PilotError("inputs.json excluded candidate %s lacks an explicit reason" % clip_id)
        by_clip[clip_id] = dict(candidate)
        lineages.add(lineage)
    raw_source_hashes = payload.get("raw_source_hashes")
    if not isinstance(raw_source_hashes, Mapping) or not raw_source_hashes:
        raise PilotError("inputs.json must bind raw_source_hashes")
    for key in ("candidate_manifest", "model_manifest", "runtime_lock", "labels", "exclusions"):
        value = raw_source_hashes.get(key)
        if not isinstance(value, Mapping) or not isinstance(value.get("uri"), str) or not value["uri"]:
            raise PilotError("inputs.json raw_source_hashes.%s must bind uri and sha256" % key)
        _digest(value.get("sha256"), "raw_source_hashes.%s.sha256" % key)
    _digest(raw_source_hashes.get("collection_config_sha256"), "raw_source_hashes.collection_config_sha256")
    teacher_reports = raw_source_hashes.get("teacher_reports")
    if not isinstance(teacher_reports, Mapping) or set(teacher_reports) != set(by_clip):
        raise PilotError("inputs.json teacher_reports must bind every candidate clip")
    for clip_id, value in teacher_reports.items():
        if not isinstance(value, Mapping) or not isinstance(value.get("uri"), str) or not value["uri"]:
            raise PilotError("inputs.json teacher report %s lacks uri" % clip_id)
        _digest(value.get("sha256"), "inputs.json teacher report %s sha256" % clip_id)
    split_refs = raw_source_hashes.get("splits")
    if not isinstance(split_refs, Mapping) or set(split_refs) != {"train", "development_validation"}:
        raise PilotError("inputs.json must bind the two deterministic pilot split files")
    expected_names = {"train": TRAIN_FILE, "development_validation": VALIDATION_FILE}
    split_hashes: Dict[str, str] = {}
    for split, filename in expected_names.items():
        value = split_refs[split]
        if not isinstance(value, Mapping) or value.get("uri") != filename:
            raise PilotError("inputs.json split %s must bind %s without repartition" % (split, filename))
        split_hashes[filename] = _digest(value.get("sha256"), "inputs.json split %s sha256" % split)
    return by_clip, {INPUTS_FILE: _sha256_bytes(raw)}, _sha256_bytes(_canonical(sorted(lineages))), split_hashes


def load_pilot_dataset(dataset_root: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str], str, Dict[str, Dict[str, Any]]]:
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise PilotError("--dataset-root must be an existing non-symlink directory")
    root = dataset_root.resolve(strict=True)
    candidates, file_hashes, lineage_hash, split_hashes = _load_final_inputs(root)
    expected = (
        ("train", TRAIN_FILE, PILOT_MIN_TRAIN_ROWS, PILOT_MAX_TRAIN_ROWS),
        ("development_validation", VALIDATION_FILE, PILOT_MIN_VALIDATION_ROWS, PILOT_MAX_VALIDATION_ROWS),
    )
    splits: Dict[str, List[Dict[str, Any]]] = {}
    seen_clips = set()
    seen_lineages = set()
    for split, filename, minimum_count, maximum_count in expected:
        path = _secure_file(root, filename, split + " dataset")
        raw_bytes = path.read_bytes()
        file_hashes[filename] = _sha256_bytes(raw_bytes)
        if file_hashes[filename] != split_hashes[filename]:
            raise PilotError("%s bytes do not match the frozen inputs.json split hash" % filename)
        try:
            raw_rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PilotError("%s is not strict JSONL" % filename) from error
        if not minimum_count <= len(raw_rows) <= maximum_count or not all(isinstance(row, Mapping) for row in raw_rows):
            raise PilotError("%s must contain %d through %d accepted object rows" % (filename, minimum_count, maximum_count))
        rows = [_validate_row(row, split=split, index=index, root=root) for index, row in enumerate(raw_rows)]
        for row in rows:
            if row["clip_id"] in seen_clips or row["source_lineage_id"] in seen_lineages:
                raise PilotError("pilot splits may not share clip IDs or source lineages")
            candidate = candidates.get(row["clip_id"])
            if candidate is None or candidate.get("status") != "accepted":
                raise PilotError("pilot JSONL row is not accepted by frozen inputs.json")
            if candidate.get("split") != split or candidate.get("source_lineage_id") != row["source_lineage_id"]:
                raise PilotError("pilot JSONL row changes its frozen candidate split or lineage")
            seen_clips.add(row["clip_id"])
            seen_lineages.add(row["source_lineage_id"])
        splits[split] = rows
    accepted_ids = {row["clip_id"] for split_rows in splits.values() for row in split_rows}
    manifest_accepted = {clip_id for clip_id, candidate in candidates.items() if candidate["status"] == "accepted"}
    if accepted_ids != manifest_accepted:
        raise PilotError("pilot JSONL rows do not exactly match inputs.json accepted candidate set")
    return splits, file_hashes, lineage_hash, candidates


def load_pilot_config(path: Path) -> Tuple[Dict[str, Any], str]:
    payload = _load_object(path, "pilot config")
    source_hash = _sha256_file(path)
    if payload.get("schema_version") != 1 or payload.get("purpose") != PILOT_PURPOSE or payload.get("qualified") is not False:
        raise PilotError("pilot config must explicitly be schema-v1 unqualified judge distillation")
    for field in ("pilot_id",):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise PilotError("pilot config lacks %s" % field)
    student = payload.get("student_model")
    if not isinstance(student, Mapping) or not isinstance(student.get("model_id"), str) or not isinstance(student.get("model_revision"), str):
        raise PilotError("pilot config requires a pinned student_model")
    revision = student["model_revision"].lower()
    if len(revision) != 40 or any(character not in _SHA256_HEX for character in revision):
        raise PilotError("pilot student model_revision must be a 40-character immutable revision")
    lora = payload.get("lora")
    if not isinstance(lora, Mapping) or (
        lora.get("r"), lora.get("alpha"), lora.get("learning_rate"), lora.get("epochs"), tuple(lora.get("target_modules", ()))
    ) != (PILOT_LORA_R, PILOT_LORA_ALPHA, PILOT_LR, PILOT_EPOCHS, PILOT_TARGET_MODULES):
        raise PilotError("pilot config must retain r64/alpha32/lr1e-4/q_proj+v_proj/two-epoch exploratory bounds")
    training = payload.get("training")
    if not isinstance(training, Mapping) or (
        training.get("batch_size"), training.get("gradient_accumulation_steps"), training.get("seed")
    ) != (1, 1, PILOT_SEED):
        raise PilotError("pilot config must retain batch1/accum1/seed20260919")
    return payload, source_hash


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise PilotError("refusing to overwrite %s" % path) from error
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


def _freeze_output(
    output_dir: Path, config: Mapping[str, Any], config_hash: str, file_hashes: Mapping[str, str], lineage_hash: str,
    rows: Mapping[str, Sequence[Mapping[str, Any]]], candidates: Mapping[str, Mapping[str, Any]],
) -> None:
    lineages = sorted(str(candidate["source_lineage_id"]) for candidate in candidates.values())
    _write_json_exclusive(
        output_dir / "pilot_freeze.json",
        {
            "schema_version": 1,
            "purpose": PILOT_PURPOSE,
            "qualified": False,
            "pilot_config": config,
            "pilot_config_sha256": config_hash,
            "dataset_files_sha256": dict(file_hashes),
            "source_lineages_sha256": lineage_hash,
            "train_rows": len(rows["train"]),
            "development_validation_rows": len(rows["development_validation"]),
            "candidate_count": len(candidates),
            "accepted_candidate_count": sum(1 for candidate in candidates.values() if candidate["status"] == "accepted"),
            "excluded_candidate_count": sum(1 for candidate in candidates.values() if candidate["status"] == "excluded"),
            "formal_exclusion": "Every candidate lineage, including excluded teacher rows, is reserved and must be excluded from later formal distillation/calibration/primary datasets.",
        },
    )
    _write_json_exclusive(
        output_dir / "future_formal_exclusion_lineages.json",
        {"schema_version": 1, "purpose": "reserve_uncalibrated_pilot_lineages", "qualified": False, "lineages": lineages, "sha256": _sha256_bytes(_canonical(lineages))},
    )


def _gpu_identity() -> str:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 1:
        raise PilotError("pilot requires exactly one CUDA_VISIBLE_DEVICES allocation")
    try:
        output = subprocess.run(
            ["nvidia-smi", "--id=" + visible[0], "--query-gpu=uuid,name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        raise PilotError("unable to record allocated GPU UUID") from error
    if not output:
        raise PilotError("nvidia-smi returned an empty GPU identity")
    return output


def _freeze_tree(path: Path) -> str:
    for item in sorted(path.rglob("*"), reverse=True):
        if item.is_file():
            os.chmod(item, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif item.is_dir():
            os.chmod(item, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return _sha256_bytes(
        _canonical(
            [
                {"path": str(item.relative_to(path)), "sha256": _sha256_file(item), "bytes": item.stat().st_size}
                for item in sorted(path.rglob("*")) if item.is_file()
            ]
        )
    )


def _batch_loss(model: Any, collator: Any, row: Mapping[str, Any], torch_module: Any, *, backwards: bool) -> float:
    inputs = collator([row])
    inputs = {key: value.to("cuda") if callable(getattr(value, "to", None)) else value for key, value in inputs.items()}
    result = model(**inputs)
    loss = getattr(result, "loss", None)
    if loss is None or not bool(torch_module.isfinite(loss).item()):
        raise PilotError("model produced no finite assistant-only loss")
    if backwards:
        loss.backward()
    return float(loss.detach().cpu().item())


def _mean_validation_loss(model: Any, collator: Any, rows: Iterable[Mapping[str, Any]], torch_module: Any) -> float:
    model.eval()
    values = []
    with torch_module.no_grad():
        for row in rows:
            values.append(_batch_loss(model, collator, row, torch_module, backwards=False))
    model.train()
    if not values:
        raise PilotError("development validation is empty")
    return sum(values) / float(len(values))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise PilotError("pilot requires an allocated Slurm GPU job")
    from deploy.baseten.training import train_judge_lora as trainer

    raw_config_path = Path(args.pilot_config)
    raw_dataset_root = Path(args.dataset_root)
    if raw_config_path.is_symlink() or raw_dataset_root.is_symlink() or not raw_dataset_root.is_dir():
        raise PilotError("pilot config and dataset root must be existing non-symlink paths")
    config_path = raw_config_path.resolve(strict=True)
    dataset_root = raw_dataset_root.resolve(strict=True)
    config, config_hash = load_pilot_config(config_path)
    rows, file_hashes, lineage_hash, candidates = load_pilot_dataset(dataset_root)
    binding = trainer._local_model_binding(str(Path(args.model_root)), str(Path(args.model_manifest)), config["student_model"])
    output_dir = trainer._claim_fresh_output_dir(str(args.output_dir), dataset_dir=str(dataset_root), model_root=binding["model_root"])
    _freeze_output(output_dir, config, config_hash, file_hashes, lineage_hash, rows, candidates)

    modules = trainer._load_frameworks()
    torch_module, transformers, peft = modules["torch"], modules["transformers"], modules["peft"]
    if not bool(torch_module.cuda.is_available()):
        raise PilotError("pilot requires CUDA; no CPU substitute is permitted")
    torch_module.manual_seed(PILOT_SEED)
    torch_module.cuda.manual_seed_all(PILOT_SEED)
    load_kwargs = {
        "revision": config["student_model"]["model_revision"], "local_files_only": True, "trust_remote_code": False,
        "torch_dtype": torch_module.bfloat16, "use_safetensors": True,
    }
    processor = transformers.AutoProcessor.from_pretrained(binding["model_root"], **{key: value for key, value in load_kwargs.items() if key != "torch_dtype"})
    base_model = transformers.AutoModelForVision2Seq.from_pretrained(binding["model_root"], **load_kwargs)
    enable_checkpointing = getattr(base_model, "gradient_checkpointing_enable", None)
    if not callable(enable_checkpointing):
        raise PilotError("base model does not support gradient checkpointing")
    enable_checkpointing()
    enable_input_grads = getattr(base_model, "enable_input_require_grads", None)
    if callable(enable_input_grads):
        enable_input_grads()
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False
    lora = config["lora"]
    model = peft.get_peft_model(base_model, peft.LoraConfig(r=PILOT_LORA_R, lora_alpha=PILOT_LORA_ALPHA, lora_dropout=0.0, target_modules=list(PILOT_TARGET_MODULES), task_type="CAUSAL_LM", bias="none"))
    model = model.to("cuda")
    model.train()
    optimizer = torch_module.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=PILOT_LR)
    prepared = {split: trainer._as_training_rows(split_rows) for split, split_rows in rows.items()}
    collator = trainer._build_collator(processor)
    torch_module.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    validation_before = _mean_validation_loss(model, collator, prepared["development_validation"], torch_module)
    epochs = []
    best = None
    optimizer_steps = 0
    adapters_root = output_dir / "adapters"
    adapters_root.mkdir()
    for epoch in range(1, PILOT_EPOCHS + 1):
        train_losses = []
        for row in prepared["train"]:
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(_batch_loss(model, collator, row, torch_module, backwards=True))
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
            if not gradients or any(not bool(torch_module.isfinite(gradient).all().item()) for gradient in gradients):
                raise PilotError("pilot found no finite LoRA gradients before optimizer step")
            if not any(float(torch_module.linalg.vector_norm(gradient.float()).item()) > 0.0 for gradient in gradients):
                raise PilotError("pilot found no non-zero LoRA gradients before optimizer step")
            optimizer.step()
            optimizer_steps += 1
        validation_loss = _mean_validation_loss(model, collator, prepared["development_validation"], torch_module)
        checkpoint = adapters_root / ("epoch-%02d" % epoch)
        checkpoint.mkdir()
        model.save_pretrained(str(checkpoint), safe_serialization=True)
        processor.save_pretrained(str(checkpoint))
        checkpoint_hash = _freeze_tree(checkpoint)
        epoch_record = {"epoch": epoch, "mean_train_loss": sum(train_losses) / float(len(train_losses)), "development_validation_loss": validation_loss, "adapter_path": str(checkpoint), "adapter_tree_sha256": checkpoint_hash}
        epochs.append(epoch_record)
        print(json.dumps({"event": "unqualified_pilot_epoch", "qualified": False, **epoch_record}, sort_keys=True), flush=True)
        if best is None or validation_loss < best["development_validation_loss"]:
            best = dict(epoch_record)
    assert best is not None
    result = {
        "schema_version": 1, "kind": "plumb_uncalibrated_judge_lora_pilot", "purpose": PILOT_PURPOSE, "qualified": False,
        "status": "completed_unqualified_exploratory_pilot",
        "reason": "Teacher-labelled development-only engineering pilot; losses are not human-quality, calibration, formal-search, or scoring evidence.",
        "pilot_config_sha256": config_hash, "dataset_files_sha256": file_hashes, "source_lineages_sha256": lineage_hash,
        "future_formal_exclusion_lineages_ref": str(output_dir / "future_formal_exclusion_lineages.json"),
        "student_model": dict(config["student_model"]), "model_binding": binding, "lora": dict(lora),
        "training": dict(config["training"]), "seed_scope": "set_before_base_model_and_lora_construction", "runtime_versions": {name: str(getattr(module, "__version__", "")) for name, module in modules.items() if name in trainer.EXPECTED_RUNTIME_VERSIONS},
        "validation_loss_before_training": validation_before, "epochs": epochs, "selected_checkpoint_by": "lowest_development_validation_loss_only", "selected_checkpoint": best,
        "optimizer_steps": optimizer_steps, "gpu_peak_memory_bytes": int(torch_module.cuda.max_memory_allocated()), "gpu_identity": _gpu_identity(),
        "wall_seconds": time.monotonic() - started, "timing_scope": "validation_plus_two_%d-row manual-AdamW epochs plus final validation; excludes manifest verification and model construction" % len(prepared["train"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "source_release": os.environ.get("PLUMB_SOURCE_RELEASE"),
    }
    _write_json_exclusive(output_dir / "pilot_result.json", result)
    _freeze_tree(output_dir)
    os.chmod(output_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps({"status": result["status"], "qualified": False, "output_dir": args.output_dir}, sort_keys=True))
        return 0
    except (PilotError, ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "qualified": False, "reason": str(error), "traceback": traceback.format_exc()}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
