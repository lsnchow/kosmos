#!/bin/bash
# One explicitly unqualified teacher-labelled development LoRA pilot.
#
# Usage:
#   sbatch cluster/judge_lora_pilot.sbatch <release-sha-dir> <pilot-config.json> \
#     <pilot-dataset-root> <qwen-model-root> <model-manifest> <new-output-dir>
#
# The pilot is deliberately separate from the formal Training Jobs search. It
# never creates a Gate-D/E record, an ArmResult, or a formal candidate result.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-lora-pilot-%j.log

set -euo pipefail

PILOT_RELEASE=${1:?supply immutable release directory}
PILOT_CONFIG=${2:?supply frozen pilot config JSON}
PILOT_DATASET=${3:?supply prepared teacher-labelled dataset root}
PILOT_MODEL_ROOT=${4:?supply local Qwen model root}
PILOT_MODEL_MANIFEST=${5:?supply model-root-contained download manifest}
PILOT_OUTPUT=${6:?supply fresh output directory}
PILOT_ROOT=/scratch/lchow432/plumb
PILOT_VENV="$PILOT_ROOT/venv-judge-lora-tf449/bin/python"

module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29

test -x "$PILOT_VENV"
test -d "$PILOT_RELEASE"
test -f "$PILOT_CONFIG"
test -d "$PILOT_DATASET"
test -d "$PILOT_MODEL_ROOT"
test -f "$PILOT_MODEL_MANIFEST"
test ! -e "$PILOT_OUTPUT"

PILOT_RELEASE_SHA=$(basename "$PILOT_RELEASE")
if [[ ! "$PILOT_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory must be named by a 64-character source SHA" >&2
  exit 2
fi
"$PILOT_VENV" - "$PILOT_RELEASE" "$PILOT_RELEASE_SHA" <<'PY'
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
export PYTHONPATH="$PILOT_RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$PILOT_RELEASE_SHA"

"$PILOT_VENV" "$PILOT_RELEASE/cluster/judge_lora_pilot.py" \
  --pilot-config "$PILOT_CONFIG" \
  --dataset-root "$PILOT_DATASET" \
  --model-root "$PILOT_MODEL_ROOT" \
  --model-manifest "$PILOT_MODEL_MANIFEST" \
  --output-dir "$PILOT_OUTPUT"
