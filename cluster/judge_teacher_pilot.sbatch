#!/bin/bash
# One-H100 unqualified Qwen teacher-label collector for the frozen 16-clip pilot.
#
# Usage (all candidate media stays under DATASET_ROOT; nothing is copied):
#   sbatch cluster/judge_teacher_pilot.sbatch \
#     /scratch/lchow432/plumb/releases/<release-sha> \
#     /scratch/lchow432/plumb/experiments/judge-lora-pilot-v1 \
#     /scratch/lchow432/plumb/models/qwen-judge-training-view-cc594-v1 \
#     /scratch/lchow432/plumb/models/qwen-judge-training-view-cc594-v1/PLUMB-TRAINING-MANIFEST.json \
#     /scratch/lchow432/plumb/evidence/judge-training-runtime-tf449-v1.json \
#     16
#
# Lower --max-clips bounds a shard. Completed hash-matched per-clip reports are
# resumed without re-querying; failed reports are preserved and retried as a new
# attempt. This does not train, calibrate, score a study, or fill any gate.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-teacher-pilot-%j.log

set -euo pipefail
umask 077

JUDGE_RELEASE=${1:?supply immutable release directory}
JUDGE_DATASET_ROOT=${2:?supply dataset root with candidate-inputs.json and media}
JUDGE_MODEL_ROOT=${3:?supply local Qwen model view}
JUDGE_MODEL_MANIFEST=${4:?supply local Qwen model manifest}
JUDGE_RUNTIME_LOCK=${5:?supply judge training runtime lock}
JUDGE_MAX_CLIPS=${6:-16}
JUDGE_SHARD_SUMMARY=${7:-}
JUDGE_ROOT=/scratch/lchow432/plumb
JUDGE_VENV="$JUDGE_ROOT/venv-judge-lora-tf449/bin/python"
JUDGE_CANDIDATES="$JUDGE_DATASET_ROOT/candidate-inputs.json"

module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29

test -x "$JUDGE_VENV"
test -d "$JUDGE_RELEASE"
test -d "$JUDGE_DATASET_ROOT"
test -f "$JUDGE_CANDIDATES"
test -d "$JUDGE_MODEL_ROOT"
test -f "$JUDGE_MODEL_MANIFEST"
test -f "$JUDGE_RUNTIME_LOCK"

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
if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
    raise SystemExit("RELEASE.json hash does not match release directory name")
payload = json.loads(manifest.read_text(encoding="utf-8"))
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

JUDGE_ARGS=(
  --dataset-root "$JUDGE_DATASET_ROOT"
  --input-manifest candidate-inputs.json
  --model-root "$JUDGE_MODEL_ROOT"
  --model-manifest "$JUDGE_MODEL_MANIFEST"
  --runtime-lock "$JUDGE_RUNTIME_LOCK"
  --max-clips "$JUDGE_MAX_CLIPS"
)
if [[ -n "$JUDGE_SHARD_SUMMARY" ]]; then
  JUDGE_ARGS+=(--shard-summary "$JUDGE_SHARD_SUMMARY")
fi

"$JUDGE_VENV" "$JUDGE_RELEASE/cluster/judge_teacher_pilot.py" "${JUDGE_ARGS[@]}"
