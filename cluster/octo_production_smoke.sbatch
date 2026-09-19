#!/bin/bash
# One-H100 unqualified smoke of PLUMB's *production* Octo 241fb profile.
#
# Submit only after root has created the separate clean 241fb checkout:
#   sbatch cluster/octo_production_smoke.sbatch \
#     /scratch/lchow432/plumb/releases/<RELEASE_JSON_SHA256> \
#     /scratch/lchow432/plumb/fixtures/octo-small-bridge-cv2-linear-rgb-256-v3/frame-00.png \
#     /scratch/lchow432/plumb/fixtures/octo-small-bridge-cv2-linear-rgb-256-v3/frame-01.png \
#     /scratch/lchow432/plumb/fixtures/octo-small-bridge-cv2-linear-rgb-256-v3/manifest.json \
#     close_drawer
#
# It never clones, installs, downloads, mutates a checkout, or runs a world
# rollout.  Python imports Octo from source-octo-autoeval241fb under the new
# copied venv and requires its own immutable 241fb runtime lock.  The existing
# asset manifest retains its original staging-code provenance and is verified
# as data input, never relabelled as a 241fb runtime lock.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/octo-production-smoke-%j.log

set -euo pipefail
umask 077

OCTO_RELEASE=${1:?supply immutable release directory as argument 1}
OCTO_FRAME_ZERO=${2:?supply canonical frame-00 PNG as argument 2}
OCTO_FRAME_ONE=${3:?supply canonical frame-01 PNG as argument 3}
OCTO_INPUT_MANIFEST=${4:?supply canonical input manifest as argument 4}
OCTO_TASK=${5:-close_drawer}
OCTO_ROOT=/scratch/lchow432/plumb
OCTO_PRODUCTION_SOURCE="$OCTO_ROOT/source-octo-autoeval241fb"
OCTO_VENV="$OCTO_ROOT/venv-octo-autoeval241fb"
OCTO_RUNTIME_LOCK="$OCTO_ROOT/evidence/octo-runtime-autoeval241fb-v1.json"
OCTO_REPORT="$OCTO_ROOT/evidence/octo-small-production-241fb-smoke-${SLURM_JOB_ID}-${SLURM_STEP_ID:-batch}-${SLURM_PROCID:-0}.json"

module load StdEnv/2023 python/3.10 cuda/12.2 cudnn/8.9.5.29

test -x "$OCTO_VENV/bin/python"
test -d "$OCTO_RELEASE"
test -d "$OCTO_PRODUCTION_SOURCE/.git"
test -f "$OCTO_FRAME_ZERO"
test -f "$OCTO_FRAME_ONE"
test -f "$OCTO_INPUT_MANIFEST"
test -f "$OCTO_RUNTIME_LOCK"

OCTO_RELEASE_SHA=$(basename "$OCTO_RELEASE")
if [[ ! "$OCTO_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory name must be a 64-character RELEASE.json SHA" >&2
  exit 2
fi

"$OCTO_VENV/bin/python" - "$OCTO_RELEASE" "$OCTO_RELEASE_SHA" "$OCTO_PRODUCTION_SOURCE" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

release = Path(sys.argv[1]).resolve()
expected_release = sys.argv[2]
source = Path(sys.argv[3]).resolve()
manifest = release / "RELEASE.json"
if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_release:
    raise SystemExit("RELEASE.json hash does not match source release directory SHA")
payload = json.loads(manifest.read_text(encoding="utf-8"))
for record in payload.get("files", []):
    path = release / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise SystemExit("release file missing/size mismatch: " + record["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("release file hash mismatch: " + record["path"])
head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.strip()
if head != "241fb3514b7c40957a86d869fecb7c7fc353f540" or dirty:
    raise SystemExit("production Octo source must be clean at 241fb3514b7c40957a86d869fecb7c7fc353f540")
PY

export HF_HOME="$OCTO_ROOT/models/.hf-octo"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
# Keep JAX as GPU owner; the harness disables TensorFlow GPU visibility before
# importing JAX/Octo.  The report records this explicit allocation mode.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# This ordering prevents the existing editable 37951 package from winning the
# import.  The harness inspect-checks the loaded module path as a second guard.
export PYTHONPATH="$OCTO_PRODUCTION_SOURCE:$OCTO_RELEASE${PYTHONPATH:+:$PYTHONPATH}"

"$OCTO_VENV/bin/python" "$OCTO_RELEASE/cluster/octo_production_smoke.py" \
  --root "$OCTO_ROOT" \
  --release-dir "$OCTO_RELEASE" \
  --production-source "$OCTO_PRODUCTION_SOURCE" \
  --runtime-lock "$OCTO_RUNTIME_LOCK" \
  --frame-0 "$OCTO_FRAME_ZERO" \
  --frame-1 "$OCTO_FRAME_ONE" \
  --input-manifest "$OCTO_INPUT_MANIFEST" \
  --task-id "$OCTO_TASK" \
  --seed 0 \
  --report "$OCTO_REPORT"
