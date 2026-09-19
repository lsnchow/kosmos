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

RESULT_FILENAME = "arm_result.json"
ADAPTER_DIRNAME = "adapter"


class TrainingConfigError(ValueError):
    """Raised for a job config or dataset this job refuses to train on."""


class TrainingRuntimeUnavailable(RuntimeError):
    """Raised when the training frameworks are not installed in this container."""


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
    return dict(config)


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
            missing = [name for name in REQUIRED_ROW_FIELDS if not row.get(name)]
            if missing:
                raise TrainingConfigError(
                    "{0} row {1} lacks {2}".format(split, index, ", ".join(missing))
                )
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
            frames = row["frames"]
            if not isinstance(frames, list) or len(frames) != FRAMES_PER_CLIP:
                raise TrainingConfigError(
                    "{0} row {1} has {2} frame refs; the judge protocol fixes exactly {3}".format(
                        split, index, len(frames) if isinstance(frames, list) else "non-list", FRAMES_PER_CLIP
                    )
                )
            label = row["label"]
            if not isinstance(label, Mapping):
                raise TrainingConfigError("{0} row {1} label is not an object".format(split, index))
            label_missing = [name for name in REQUIRED_LABEL_FIELDS if name not in label]
            if label_missing:
                raise TrainingConfigError(
                    "{0} row {1} label lacks {2}".format(split, index, ", ".join(label_missing))
                )
            rows.append(dict(row))
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
    for name in ("torch", "transformers", "peft", "trl"):
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
    return modules


def _digest_directory(directory: Path) -> str:
    """Hash the adapter files actually written, in a stable order."""

    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _build_collator(processor: Any) -> Any:
    """Collate 16 frames plus the structured rubric target for one clip.

    Whether this collator is what TRL's ``SFTTrainer`` wants for Qwen2.5-VL at
    the pinned Transformers version is the compatibility question the training
    container answers; see the header of requirements.txt.
    """

    from PIL import Image

    def collate(batch: Sequence[Mapping[str, Any]]) -> Any:
        conversations = []
        images = []
        for row in batch:
            frames = [Image.open(ref).convert("RGB") for ref in row["frames"]]
            images.append(frames)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image"} for _ in frames]
                        + [{"type": "text", "text": row["prompt"]}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]},
                ]
            )
        texts = [
            processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            for conversation in conversations
        ]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
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
                "prompt": "{0}\n\n{1}".format(row["instruction"], row["rubric"]),
                "target": json.dumps(dict(row["label"]), sort_keys=True, ensure_ascii=False),
            }
        )
    return prepared


def train(
    config: Mapping[str, Any],
    dataset_dir: str,
    output_dir: str,
    *,
    excluded_lineages: Sequence[str] = (),
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

    train_rows = _as_training_rows(load_split(dataset_dir, TRAIN_SPLIT, excluded_lineages=excluded_lineages))
    validation_rows = _as_training_rows(
        load_split(dataset_dir, VALIDATION_SPLIT, excluded_lineages=excluded_lineages)
    )

    cache_root = os.environ.get("PLUMB_MODEL_CACHE_ROOT")
    load_kwargs: Dict[str, Any] = {
        "revision": student["model_revision"],
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": torch_module.bfloat16,
    }
    if cache_root:
        load_kwargs["cache_dir"] = cache_root
    processor = transformers.AutoProcessor.from_pretrained(student["model_id"], **{
        key: value for key, value in load_kwargs.items() if key != "torch_dtype"
    })
    model = transformers.AutoModelForVision2Seq.from_pretrained(student["model_id"], **load_kwargs)

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
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=validation_rows,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
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
        result = train(config, args.dataset_dir, args.output_dir, excluded_lineages=excluded)
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
