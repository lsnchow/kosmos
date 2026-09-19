"""Baseten Training Jobs configuration for PLUMB judge distillation.

This module is the declarative counterpart to ``deploy/baseten/chain.py``: it
holds the concrete configuration for the distillation jobs and renders their
payloads, and it never calls Baseten.  ``plumb.distillation`` owns every
refusal; this file only supplies the values and the container identity.

Run it to see exactly what would be submitted:

    .venv/bin/python deploy/baseten/training/job_config.py --render

Nothing here has been submitted.  Each rendered payload carries
``"status": "rendered_not_submitted"`` and a ``submission_blockers`` list, and
``"route_status": "unconfirmed_against_training_jobs_reference"``: the exact
Training Jobs route and accepted body must be confirmed against
https://docs.baseten.co/training/overview at deploy time, the same discipline
``chain.py``'s ``autoscaling_patch_payloads()`` uses for its management-API
bodies.

The dataset URI, the Gate-D held-out lineages and the primary-study lineages are
*inputs*, not defaults.  They are supplied on the command line because guessing
them would defeat the disjointness check that is the point of the exercise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = TRAINING_DIR / "requirements.txt"
ENTRYPOINT_PATH = TRAINING_DIR / "train_judge_lora.py"

#: Training container. Unverified: no image built from this tag plus
#: requirements.txt has been resolved, and AGENT-BUILD-SPEC.md section 7
#: requires framework/model compatibility to be checked in this container
#: before an arm's result may be trusted.
TRAINING_IMAGE = "baseten/training-py311-cu124:2026-09-19"
TRAINING_PYTHON_VERSION = "3.11"
TRAINING_FRAMEWORK = "trl"

#: One H100 per arm, one node. Chosen to match the world-model benchmarking
#: starting point in section 7 (one H100, measure before scaling), not because
#: it has been shown to fit. A configured accelerator count is a request, not
#: reserved capacity.
TRAINING_ACCELERATOR = "H100"
TRAINING_ACCELERATOR_COUNT = 1
TRAINING_NODE_COUNT = 1

#: Read by train_judge_lora.py. Names only: no value is ever recorded here.
ENV_DISTILL_CONFIG = "PLUMB_DISTILL_CONFIG"
ENV_DISTILL_DATASET_DIR = "PLUMB_DISTILL_DATASET_DIR"
ENV_DISTILL_OUTPUT_DIR = "PLUMB_DISTILL_OUTPUT_DIR"
ENV_MODEL_CACHE_ROOT = "PLUMB_MODEL_CACHE_ROOT"
TRAINING_ENV_VAR_NAMES = (
    ENV_DISTILL_CONFIG,
    ENV_DISTILL_DATASET_DIR,
    ENV_DISTILL_OUTPUT_DIR,
    ENV_MODEL_CACHE_ROOT,
)

#: The Chain uses this same secret name for the base weights (chain.py
#: HF_SECRET_KEY). Training reads the same pinned revision from the cache.
HF_SECRET_KEY = "hf_access_token"

PROJECT_NAME = "plumb-judge-distillation"
JOB_NAME_PREFIX = "distill-judge-lora"
START_COMMAND = "python train_judge_lora.py --config ${PLUMB_DISTILL_CONFIG}"


def requirements_digest() -> str:
    """Digest the pinned requirements so a rendered payload names its image inputs."""

    return "sha256:" + hashlib.sha256(REQUIREMENTS_PATH.read_bytes()).hexdigest()


def training_runtime(*, framework_compatibility_checked: bool = False) -> Any:
    """The training container, with compatibility unchecked until it builds."""

    from plumb.distillation import TrainingRuntime

    return TrainingRuntime(
        image=TRAINING_IMAGE,
        python_version=TRAINING_PYTHON_VERSION,
        framework=TRAINING_FRAMEWORK,
        requirements_ref=str(REQUIREMENTS_PATH.relative_to(REPO_ROOT)),
        requirements_sha256=requirements_digest(),
        start_command=START_COMMAND,
        accelerator=TRAINING_ACCELERATOR,
        accelerator_count=TRAINING_ACCELERATOR_COUNT,
        node_count=TRAINING_NODE_COUNT,
        environment_variable_names=TRAINING_ENV_VAR_NAMES,
        secret_names=(HF_SECRET_KEY,),
        framework_compatibility_checked=framework_compatibility_checked,
    )


def build_payloads(
    *,
    calibration_manifest: Sequence[Any],
    gate_d_heldout_lineages: Sequence[str],
    primary_lineages: Sequence[str],
    dataset_uri: str,
    dataset_id: str,
    frozen_at: str,
    preregistered_at: str,
    search_id: str,
    base_judge_config: Dict[str, Any],
    student_config: Dict[str, Any],
    label_source: str,
    label_count: int,
    usd_unavailable_reason: str,
    framework_compatibility_checked: bool = False,
) -> List[Dict[str, Any]]:
    """Render one Training Jobs payload per preregistered arm.

    Every refusal happens inside ``plumb.distillation``: this function will
    raise ``DistillationError`` if the dataset touches a held-out or
    primary-study lineage, if the search is missing a named arm, or if the cost
    line item is recorded as zero.
    """

    from plumb.distillation import (
        BaseJudgeRevision,
        StudentModel,
        TrainingBudgetItem,
        dataset_from_calibration_manifest,
        default_search_space,
        render_search_payloads,
    )

    exclusions = _lineage_exclusions(gate_d_heldout_lineages, primary_lineages)
    budget = TrainingBudgetItem(
        label_count=label_count,
        label_source=label_source,
        usd_unavailable_reason=usd_unavailable_reason,
    )
    dataset = dataset_from_calibration_manifest(
        calibration_manifest,
        dataset_id=dataset_id,
        exclusions=exclusions,
        label_source=label_source,
        budget=budget,
        storage_uri=dataset_uri,
        frozen_at=frozen_at,
    )
    search = default_search_space(search_id=search_id, preregistered_at=preregistered_at)
    return render_search_payloads(
        project_name=PROJECT_NAME,
        job_name_prefix=JOB_NAME_PREFIX,
        search=search,
        dataset=dataset,
        base_judge=BaseJudgeRevision(**base_judge_config),
        student=StudentModel(**student_config),
        runtime=training_runtime(framework_compatibility_checked=framework_compatibility_checked),
    )


def _lineage_exclusions(gate_d_heldout: Sequence[str], primary: Sequence[str]) -> Any:
    from plumb.distillation import LineageExclusions

    return LineageExclusions(
        gate_d_heldout=tuple(gate_d_heldout),
        primary_study=tuple(primary),
    )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest_rows(path: str) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = _load_json(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("clips"), list):
        return [dict(row) for row in payload["clips"]]
    raise SystemExit("calibration manifest must be a JSON list, a JSONL file, or {'clips': [...]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="job_config.py",
        description="Render the PLUMB judge-distillation Training Jobs payloads. Never submits.",
    )
    parser.add_argument("--render", action="store_true", help="Print the rendered payloads as JSON")
    parser.add_argument("--output", type=Path, default=None, help="Write the payloads to this path")
    parser.add_argument("--calibration-manifest", help="Gate-D calibration manifest (JSON or JSONL)")
    parser.add_argument("--heldout-lineages", help="JSON list of the Gate-D held-out source lineage IDs")
    parser.add_argument("--primary-lineages", help="JSON list of the primary-study source lineage IDs")
    parser.add_argument("--dataset-uri", help="Object-store URI the training job reads")
    parser.add_argument("--dataset-id", default="distill-dataset-v1")
    parser.add_argument("--frozen-at", help="ISO-8601 timestamp with offset: when the dataset was frozen")
    parser.add_argument("--preregistered-at", help="ISO-8601 timestamp with offset: when the search was frozen")
    parser.add_argument("--search-id", default="lora-search-v1")
    parser.add_argument("--base-judge", help="JSON file describing the Gate-D judge revision")
    parser.add_argument("--student", help="JSON file describing the student model")
    parser.add_argument("--label-source", default="original_judge_aggregate")
    parser.add_argument("--label-count", type=int, help="How many training labels were acquired")
    parser.add_argument(
        "--usd-unavailable-reason",
        default="no verified price basis for the training accelerator has been established",
    )
    parser.add_argument(
        "--framework-compatibility-checked",
        action="store_true",
        help="Set only after the training container has actually built and loaded the model",
    )
    parser.add_argument("--digest", action="store_true", help="Print the requirements digest and exit")
    args = parser.parse_args(argv)

    if args.digest:
        print("{0}  {1}".format(REQUIREMENTS_PATH.relative_to(REPO_ROOT), requirements_digest()))
        return 0

    if not args.render and args.output is None:
        parser.print_help()
        print("\nNothing rendered. Pass --render or --output; there is no submit mode.")
        return 0

    if not ENTRYPOINT_PATH.exists():
        raise SystemExit(
            "training entrypoint is missing: {0}. The rendered start_command would fail in the "
            "container.".format(ENTRYPOINT_PATH)
        )

    required = (
        "calibration_manifest",
        "heldout_lineages",
        "primary_lineages",
        "dataset_uri",
        "frozen_at",
        "preregistered_at",
        "base_judge",
        "student",
        "label_count",
    )
    missing = [name.replace("_", "-") for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise SystemExit(
            "cannot render without: --" + ", --".join(missing) + "\n"
            "These are inputs on purpose. Guessing the held-out or primary lineages would defeat the "
            "disjointness check that gates this job."
        )

    payloads = build_payloads(
        calibration_manifest=_load_manifest_rows(args.calibration_manifest),
        gate_d_heldout_lineages=_load_json(args.heldout_lineages),
        primary_lineages=_load_json(args.primary_lineages),
        dataset_uri=args.dataset_uri,
        dataset_id=args.dataset_id,
        frozen_at=args.frozen_at,
        preregistered_at=args.preregistered_at,
        search_id=args.search_id,
        base_judge_config=_load_json(args.base_judge),
        student_config=_load_json(args.student),
        label_source=args.label_source,
        label_count=int(args.label_count),
        usd_unavailable_reason=args.usd_unavailable_reason,
        framework_compatibility_checked=bool(args.framework_compatibility_checked),
    )
    encoded = json.dumps(payloads, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print("wrote {0} payload(s) to {1}".format(len(payloads), args.output))
    if args.render:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main(sys.argv[1:]))
