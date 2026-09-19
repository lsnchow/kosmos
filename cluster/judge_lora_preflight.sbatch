#!/bin/bash
# One allocated-GPU, no-update compatibility preflight for judge LoRA.
#
# Usage:
#   sbatch cluster/judge_lora_preflight.sbatch \
#     /scratch/lchow432/plumb/releases/<release-sha> \
#     /scratch/lchow432/plumb/evidence/judge-preflight-row.json \
#     /scratch/lchow432/plumb \
#     /scratch/lchow432/plumb/evidence/one-arm.json \
#     /scratch/lchow432/plumb/models/Qwen--Qwen2.5-VL-7B-Instruct \
#     /scratch/lchow432/plumb/models/Qwen--Qwen2.5-VL-7B-Instruct/PLUMB-TRAINING-MANIFEST.json
#
# This runs one forward/backward only. It does not train, step an optimizer,
# save an adapter, select an arm, or claim calibration/qualification.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-lora-preflight-%j.log

set -euo pipefail

JUDGE_RELEASE=${1:?supply immutable release directory}
JUDGE_ROW=${2:?supply prepared diagnostic preflight row JSON}
JUDGE_ROW_ROOT=${3:?supply row-root containing the decoded PNGs}
JUDGE_CONFIG=${4:?supply one rendered preregistered arm JSON}
JUDGE_MODEL_ROOT=${5:?supply local Qwen model directory}
JUDGE_MODEL_MANIFEST=${6:?supply model-root-contained revision manifest}
JUDGE_ROOT=/scratch/lchow432/plumb
JUDGE_OUTPUT="$JUDGE_ROOT/evidence/judge-lora-preflight-${SLURM_JOB_ID}.json"
JUDGE_VENV="$JUDGE_ROOT/venv-judge-lora-tf449/bin/python"

module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29

test -x "$JUDGE_VENV"
test -d "$JUDGE_RELEASE"
test -f "$JUDGE_ROW"
test -d "$JUDGE_ROW_ROOT"
test -f "$JUDGE_CONFIG"
test -d "$JUDGE_MODEL_ROOT"
test -f "$JUDGE_MODEL_MANIFEST"
test ! -e "$JUDGE_OUTPUT"

JUDGE_RELEASE_SHA=$(basename "$JUDGE_RELEASE")
if [[ ! "$JUDGE_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory must be named by a 64-character source SHA" >&2
  exit 2
fi
"$JUDGE_VENV" - "$JUDGE_RELEASE" "$JUDGE_RELEASE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release = Path(sys.argv[1])
expected = sys.argv[2]
manifest = release / "RELEASE.json"
raw = manifest.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("RELEASE.json hash does not match source release directory SHA")
payload = json.loads(raw)
for record in payload.get("files", []):
    path = release / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise SystemExit("release file missing/size mismatch: " + record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("release file hash mismatch: " + record["path"])
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JUDGE_RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$JUDGE_RELEASE_SHA"

"$JUDGE_VENV" "$JUDGE_RELEASE/cluster/judge_lora_preflight.py" \
  --config "$JUDGE_CONFIG" \
  --row "$JUDGE_ROW" \
  --row-root "$JUDGE_ROW_ROOT" \
  --model-root "$JUDGE_MODEL_ROOT" \
  --model-manifest "$JUDGE_MODEL_MANIFEST" \
  --output "$JUDGE_OUTPUT"
