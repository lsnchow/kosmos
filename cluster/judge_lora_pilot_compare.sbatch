#!/bin/bash
# Reload only the saved unqualified pilot adapter for paired dev-validation inference.
#
# Usage:
#   sbatch cluster/judge_lora_pilot_compare.sbatch <release-sha-dir> <pilot-result.json> \
#     <pilot-dataset-root> <qwen-model-root> <model-manifest> <new-comparison.json>
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-lora-pilot-compare-%j.log

set -euo pipefail

COMPARE_RELEASE=${1:?supply immutable release directory}
COMPARE_RESULT=${2:?supply completed unqualified pilot_result.json}
COMPARE_DATASET=${3:?supply pilot dataset root}
COMPARE_MODEL_ROOT=${4:?supply local Qwen model root}
COMPARE_MODEL_MANIFEST=${5:?supply local Qwen download manifest}
COMPARE_OUTPUT=${6:?supply a new comparison report path}
COMPARE_ROOT=/scratch/lchow432/plumb
COMPARE_VENV="$COMPARE_ROOT/venv-judge-lora-tf449/bin/python"

module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29

test -x "$COMPARE_VENV"
test -d "$COMPARE_RELEASE"
test -f "$COMPARE_RESULT"
test -d "$COMPARE_DATASET"
test -d "$COMPARE_MODEL_ROOT"
test -f "$COMPARE_MODEL_MANIFEST"
test ! -e "$COMPARE_OUTPUT"

COMPARE_RELEASE_SHA=$(basename "$COMPARE_RELEASE")
if [[ ! "$COMPARE_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory must be named by a 64-character source SHA" >&2
  exit 2
fi
"$COMPARE_VENV" - "$COMPARE_RELEASE" "$COMPARE_RELEASE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release = Path(sys.argv[1])
expected = sys.argv[2]
manifest = release / "RELEASE.json"
raw = manifest.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("RELEASE.json hash does not match release directory SHA")
for record in json.loads(raw).get("files", []):
    path = release / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise SystemExit("release file missing/size mismatch: " + record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("release file hash mismatch: " + record["path"])
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$COMPARE_RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$COMPARE_RELEASE_SHA"

"$COMPARE_VENV" "$COMPARE_RELEASE/cluster/judge_lora_pilot_compare.py" \
  --pilot-result "$COMPARE_RESULT" \
  --dataset-root "$COMPARE_DATASET" \
  --model-root "$COMPARE_MODEL_ROOT" \
  --model-manifest "$COMPARE_MODEL_MANIFEST" \
  --output "$COMPARE_OUTPUT"
