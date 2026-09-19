#!/bin/bash
# Experimental v2 teacher diagnostic only: no training or adapter creation.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-teacher-v2-%j.log
set -euo pipefail
RELEASE=${1:?release}; DATASET=${2:?dataset-root}; OUTPUT=${3:?fresh-output-dir}; MODEL=${4:?model-root}; MANIFEST=${5:?model-manifest}; LOCK=${6:?runtime-lock}; MAX=${7:-16}
ROOT=/scratch/lchow432/plumb; PY="$ROOT/venv-judge-lora-tf449/bin/python"
module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29
test -x "$PY"; test -d "$RELEASE"; test -f "$DATASET/candidate-inputs.json"; test -d "$MODEL"; test -f "$MANIFEST"; test -f "$LOCK"
JUDGE_V2_RELEASE_SHA=$(basename "$RELEASE")
[[ "$JUDGE_V2_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]] || exit 2
"$PY" - "$RELEASE" "$JUDGE_V2_RELEASE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
raw = (root / "RELEASE.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
    raise SystemExit("source release manifest hash mismatch")
for row in json.loads(raw)["files"]:
    path = root / row["path"]
    if not path.is_file() or path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit("source release file mismatch: " + row["path"])
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$JUDGE_V2_RELEASE_SHA"
"$PY" "$RELEASE/cluster/judge_teacher_v2.py" --dataset-root "$DATASET" --output-dir "$OUTPUT" --model-root "$MODEL" --model-manifest "$MANIFEST" --runtime-lock "$LOCK" --max-clips "$MAX"
