#!/usr/bin/env bash
#SBATCH --job-name=plumb-openvla
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/lchow432/plumb/logs/openvla-smoke-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
export PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
"$PLUMB_ROOT/venv-openvla-tf440/bin/python" "$PLUMB_SOURCE/cluster/policy_smoke.py" \
  openvla --vision "$PLUMB_ROOT/fixtures/bridge_video.mp4" \
  --task 'Put the pot to the left of the purple item.' \
  --report "$PLUMB_ROOT/evidence/openvla-smoke-$SLURM_JOB_ID.json"
