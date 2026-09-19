#!/bin/bash
# Offline saved-adapter JSON-format and semantic-drift check, never qualification.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-format-compare-%j.log
set -euo pipefail
FORMAT_RELEASE=${1:?immutable release}
FORMAT_RESULT=${2:?format training result}
FORMAT_MODEL=${3:?local model root}
FORMAT_MANIFEST=${4:?model manifest}
FORMAT_OUTPUT=${5:?new comparison JSON}
FORMAT_PY=/scratch/lchow432/plumb/venv-judge-lora-tf449/bin/python
module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29
test -x "$FORMAT_PY"
test -f "$FORMAT_RESULT"
test -f "$FORMAT_MANIFEST"
test ! -e "$FORMAT_OUTPUT"
FORMAT_RELEASE_SHA=$(basename "$FORMAT_RELEASE")
[[ "$FORMAT_RELEASE_SHA" =~ ^[0-9a-f]{64}$ ]] || exit 2
"$FORMAT_PY" - "$FORMAT_RELEASE" "$FORMAT_RELEASE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
raw = (root / "RELEASE.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
    raise SystemExit("source release manifest mismatch")
for record in json.loads(raw)["files"]:
    path = root / record["path"]
    if not path.is_file() or path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit("source release file mismatch: " + record["path"])
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$FORMAT_RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$FORMAT_RELEASE_SHA"
"$FORMAT_PY" "$FORMAT_RELEASE/cluster/judge_format_compare.py" \
  --training-result "$FORMAT_RESULT" --model-root "$FORMAT_MODEL" \
  --model-manifest "$FORMAT_MANIFEST" --output "$FORMAT_OUTPUT"
