#!/bin/bash
# One-H100 format-only LoRA diagnostic; no semantic supervision or qualification.
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-format-pilot-%j.log
set -euo pipefail
RELEASE=${1:?release}; DATASET=${2:?dataset}; TEACHER=${3:?teacher-v2-root}; MODEL=${4:?model}; MANIFEST=${5:?manifest}; OUTPUT=${6:?fresh-output}
ROOT=/scratch/lchow432/plumb; PY="$ROOT/venv-judge-lora-tf449/bin/python"
module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29
test -x "$PY"; test -d "$RELEASE"; test -f "$DATASET/candidate-inputs.json"; test -d "$TEACHER"; test -d "$MODEL"; test -f "$MANIFEST"; test ! -e "$OUTPUT"
SHA=$(basename "$RELEASE")
[[ "$SHA" =~ ^[0-9a-f]{64}$ ]]
"$PY" - "$RELEASE" "$SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
release=Path(sys.argv[1]); expected=sys.argv[2]; manifest=release/'RELEASE.json'
if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected: raise SystemExit('RELEASE.json hash mismatch')
for record in json.loads(manifest.read_text()).get('files',[]):
 path=release/record['path']
 if not path.is_file() or path.stat().st_size!=record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']: raise SystemExit('release file mismatch: '+record['path'])
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$RELEASE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$SHA"
"$PY" "$RELEASE/cluster/judge_format_pilot.py" --dataset-root "$DATASET" --teacher-root "$TEACHER" --model-root "$MODEL" --model-manifest "$MANIFEST" --output-dir "$OUTPUT"
