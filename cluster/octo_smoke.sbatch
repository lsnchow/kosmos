#!/bin/bash
# One-H100 source-pinned Octo-Small diagnostic.  Submit a frozen release only:
#   sbatch cluster/octo_smoke.sbatch /scratch/lchow432/plumb/releases/<SHA> /path/frame-00.png /path/frame-01.png /path/manifest.json close_drawer
#
# This script does not qualify a policy cell.  It records three native calls
# (two fixture frames plus reset/repeat) and exits without world-model rollout.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/octo-smoke-%j.log

set -euo pipefail

OCTO_RELEASE=${1:?supply immutable release directory as argument 1}
OCTO_FRAME_ZERO=${2:?supply canonical frame-00 PNG as argument 2}
OCTO_FRAME_ONE=${3:?supply canonical frame-01 PNG as argument 3}
OCTO_INPUT_MANIFEST=${4:?supply canonical input manifest as argument 4}
OCTO_TASK=${5:-close_drawer}
OCTO_ROOT=/scratch/lchow432/plumb
OCTO_REPORT="$OCTO_ROOT/evidence/octo-small-native-smoke-${SLURM_JOB_ID}-${SLURM_STEP_ID:-batch}-${SLURM_PROCID:-0}.json"
OCTO_RUNTIME_LOCK="$OCTO_ROOT/evidence/octo-runtime-cuda12-jax0420-v2.json"

module load StdEnv/2023 python/3.10 cuda/12.2 cudnn/8.9.5.29

test -x "$OCTO_ROOT/venv-octo/bin/python"
test -d "$OCTO_ROOT/source-octo/.git"
test -d "$OCTO_RELEASE"
test -f "$OCTO_FRAME_ZERO"
test -f "$OCTO_FRAME_ONE"
test -f "$OCTO_INPUT_MANIFEST"
test -f "$OCTO_RUNTIME_LOCK"

OCTO_RELEASE_SHA=$(basename "$OCTO_RELEASE")
if [[ ! "$OCTO_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]]; then
  echo "release directory name must be a 64-character hexadecimal source SHA" >&2
  exit 2
fi
"$OCTO_ROOT/venv-octo/bin/python" - "$OCTO_RELEASE" "$OCTO_RELEASE_SHA" <<'PY'
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
OCTO_RUNTIME_LOCK_SHA=$(sha256sum "$OCTO_RUNTIME_LOCK" | awk '{print $1}')
"$OCTO_ROOT/venv-octo/bin/python" - "$OCTO_RUNTIME_LOCK" "$OCTO_RUNTIME_LOCK_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise SystemExit("Octo runtime lock hash changed while preparing the job")
lock = json.loads(path.read_text(encoding="utf-8"))
if lock.get("schema") != "plumb-octo-runtime-lock-v1":
    raise SystemExit("unexpected Octo runtime lock schema")
if lock.get("source_revision") != "37951e4e6d708fd76374f6e09e716763fe2673b1":
    raise SystemExit("runtime lock does not bind the reviewed Octo source")
PY

export HF_HOME="$OCTO_ROOT/models/.hf-octo"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
# Avoid JAX reserving all visible VRAM; the report records this explicit mode.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PLUMB_SOURCE_RELEASE="$OCTO_RELEASE_SHA"
export PYTHONPATH="$OCTO_RELEASE${PYTHONPATH:+:$PYTHONPATH}"

"$OCTO_ROOT/venv-octo/bin/python" "$OCTO_RELEASE/cluster/octo_smoke.py" \
  --root "$OCTO_ROOT" \
  --frame-0 "$OCTO_FRAME_ZERO" \
  --frame-1 "$OCTO_FRAME_ONE" \
  --input-manifest "$OCTO_INPUT_MANIFEST" \
  --task-id "$OCTO_TASK" \
  --report "$OCTO_REPORT" \
  --runtime-lock-id octo-runtime-cuda12-jax0420-v2 \
  --runtime-lock-sha256 "$OCTO_RUNTIME_LOCK_SHA"
