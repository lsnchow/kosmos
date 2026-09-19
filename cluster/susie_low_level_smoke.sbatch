#!/usr/bin/env bash
# One H100 static-goal diagnostic for the released AutoEval gc_bc low-level
# component. All positional paths must already be staged under cluster scratch.
#
# sbatch cluster/susie_low_level_smoke.sbatch RELEASE CHECKPOINT README BRIDGE_SOURCE INPUT_MANIFEST FRAME0 GOAL RUNTIME_LOCK REPORT
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/susie-low-level-smoke-%j.log
set -euo pipefail
umask 077
module load StdEnv/2023 python/3.10 cuda/12.2 cudnn/8.9.5.29

SUSIE_RELEASE=${1:?source release directory}
SUSIE_CHECKPOINT=${2:?direct checkpoint/checkpoint file}
SUSIE_README=${3:?immutable artifact README.md}
SUSIE_BRIDGE_SOURCE=${4:?clean SOAR gc_bc source checkout}
SUSIE_INPUT_MANIFEST=${5:?hash-bound two-frame input manifest}
SUSIE_FRAME_ZERO=${6:?current frame PNG}
SUSIE_GOAL_FRAME=${7:?goal frame PNG}
SUSIE_RUNTIME_LOCK=${8:?Octo runtime lock JSON}
SUSIE_REPORT=${9:?new evidence report path}

if [[ ! -f "$SUSIE_RELEASE/RELEASE.json" ]]; then
  echo "missing immutable RELEASE.json: $SUSIE_RELEASE" >&2
  exit 2
fi
if [[ ! -n "${SLURM_JOB_ID:-}" ]]; then
  echo "must run under Slurm" >&2
  exit 2
fi

SUSIE_ROOT=$(cd "$(dirname "$SUSIE_RELEASE")/.." && pwd)
SUSIE_PYTHON=${SUSIE_PYTHON:-"$SUSIE_ROOT/venv-octo/bin/python"}
if [[ ! -x "$SUSIE_PYTHON" ]]; then
  echo "missing isolated Python runtime: $SUSIE_PYTHON" >&2
  exit 2
fi

SUSIE_RELEASE_SHA=$(basename "$SUSIE_RELEASE")
if [[ ! "$SUSIE_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory name must be the 64-character RELEASE.json SHA" >&2
  exit 2
fi
"$SUSIE_PYTHON" - "$SUSIE_RELEASE" "$SUSIE_RELEASE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release = Path(sys.argv[1])
expected = sys.argv[2]
manifest = release / "RELEASE.json"
raw = manifest.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("RELEASE.json hash does not match release directory name")
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$SUSIE_RELEASE:$SUSIE_BRIDGE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$SUSIE_RELEASE_SHA"

mkdir -p "$(dirname "$SUSIE_REPORT")"
"$SUSIE_PYTHON" "$SUSIE_RELEASE/cluster/susie_low_level_smoke.py" \
  --root "$SUSIE_ROOT" \
  --checkpoint "$SUSIE_CHECKPOINT" \
  --checkpoint-readme "$SUSIE_README" \
  --bridge-source "$SUSIE_BRIDGE_SOURCE" \
  --input-manifest "$SUSIE_INPUT_MANIFEST" \
  --frame-0 "$SUSIE_FRAME_ZERO" \
  --goal-frame "$SUSIE_GOAL_FRAME" \
  --runtime-lock "$SUSIE_RUNTIME_LOCK" \
  --report "$SUSIE_REPORT" \
  --source-release "$PLUMB_SOURCE_RELEASE" \
  --profile-id "susie-ll-autoeval-gc-bc-jax0420-compatibility-deviation" \
  --jax-version "0.4.20"
