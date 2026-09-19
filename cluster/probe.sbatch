#!/usr/bin/env bash
#SBATCH --job-name=plumb-probe
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lchow432/plumb/logs/probe-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
"$PLUMB_ROOT/venv-model-std2023/bin/python" -m plumb.adapters.probe_suite \
  --model-path "$PLUMB_ROOT/models/nvidia--Cosmos3-Nano" \
  --actions "$PLUMB_ROOT/fixtures/bridge_actions.json" \
  --vision "$PLUMB_ROOT/fixtures/bridge_video.mp4" \
  --model-manifest "$PLUMB_ROOT/evidence/nvidia--Cosmos3-Nano-download.json" \
  --output-dir "$PLUMB_ROOT/evidence/cosmos-probe-$SLURM_JOB_ID" \
  --resolution-tier 256 --fps 5 --steps 30 --guidance-scale 1 \
  --include-suffix-perturbation
