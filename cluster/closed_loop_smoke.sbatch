#!/usr/bin/env bash
#SBATCH --job-name=plumb-loop
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lchow432/plumb/logs/closed-loop-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
export PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
"$PLUMB_ROOT/venv-irasim/bin/python" "$PLUMB_SOURCE/cluster/closed_loop_smoke.py" \
  --root "$PLUMB_ROOT" --ticks 16 --seed 0 \
  --output-dir "$PLUMB_ROOT/evidence/closed-loop-$SLURM_JOB_ID"
