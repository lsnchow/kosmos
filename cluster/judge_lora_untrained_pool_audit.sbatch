#!/usr/bin/env bash
#SBATCH --account=def-ikarlin
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/judge-lora-untrained-audit-%j.log
set -euo pipefail
release="$1"; result="$2"; dataset="$3"; model="$4"; manifest="$5"; output="$6"
runtime=/scratch/lchow432/plumb/venv-judge-lora-tf449/bin/python
module load StdEnv/2023 python/3.11 arrow/19.0.1 cuda/12.2 cudnn/8.9.5.29
test -x "$runtime"; test -d "$release"; test -f "$result"; test -d "$dataset"; test -d "$model"; test -f "$manifest"; test ! -e "$output"
release_hash=$(basename "$release")
[[ "$release_hash" =~ ^[0-9a-f]{64}$ ]]
"$runtime" - "$release" "$release_hash" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); expected=sys.argv[2]
manifest=root/'RELEASE.json'; raw=manifest.read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected: raise SystemExit('release manifest hash mismatch')
for row in json.loads(raw)['files']:
 path=root/row['path']
 if not path.is_file() or path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
  raise SystemExit('release file mismatch: '+row['path'])
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$release${PYTHONPATH:+:$PYTHONPATH}" PLUMB_SOURCE_RELEASE="$release_hash"
"$runtime" "$release/cluster/judge_lora_untrained_pool_audit.py" --pilot-result "$result" --dataset-root "$dataset" --model-root "$model" --model-manifest "$manifest" --output "$output" --max-clips 9
