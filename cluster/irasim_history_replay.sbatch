#!/usr/bin/env bash
#SBATCH --job-name=plumb-irasim-history
#SBATCH --partition=debug
#SBATCH --account=def-ikarlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lchow432/plumb/logs/irasim-history-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
export PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:?frozen source release required}
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
cd "$PLUMB_SOURCE"
"$PLUMB_ROOT/venv-irasim/bin/python" -m unittest discover -s tests -p test_irasim_runtime.py
"$PLUMB_ROOT/venv-irasim/bin/python" -m cluster.irasim_history_replay \
  --root "$PLUMB_ROOT" --output-dir "$PLUMB_ROOT/evidence/irasim-history-$SLURM_JOB_ID" \
  --same-seed-repeat
