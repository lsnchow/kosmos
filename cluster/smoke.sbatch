#!/usr/bin/env bash
#SBATCH --job-name=plumb-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lchow432/plumb/logs/smoke-%j.log

set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
"$PLUMB_ROOT/venv-model-std2023/bin/python" -m plumb.adapters.smoke cosmos \
  --model-path "$PLUMB_ROOT/models/nvidia--Cosmos3-Nano" \
  --actions "$PLUMB_ROOT/fixtures/bridge_actions.json" \
  --vision "$PLUMB_ROOT/fixtures/bridge_video.mp4" \
  --resolution-tier 480 --fps 5 --steps 30 --guidance-scale 1 \
  --prompt 'Put the pot to the left of the purple item.' \
  --model-manifest "$PLUMB_ROOT/evidence/nvidia--Cosmos3-Nano-download.json" \
  --output-video "$PLUMB_ROOT/evidence/cosmos-smoke-$SLURM_JOB_ID.mp4" \
  --output-frame-dir "$PLUMB_ROOT/evidence/cosmos-smoke-$SLURM_JOB_ID-frames" \
  --frame-selection all \
  --report "$PLUMB_ROOT/evidence/cosmos-smoke-$SLURM_JOB_ID.json"
