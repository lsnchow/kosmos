#!/usr/bin/env bash
#SBATCH --job-name=plumb-irasim-replay
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lchow432/plumb/logs/irasim-state-replay-%j.log

set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
export PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"

"$PLUMB_ROOT/venv-irasim/bin/python" -m unittest discover \
  -s "$PLUMB_SOURCE/tests" -p "test_irasim_runtime.py"

"$PLUMB_ROOT/venv-irasim/bin/python" "$PLUMB_SOURCE/cluster/irasim_state_replay.py" \
  --root "$PLUMB_ROOT" \
  --source-report "$PLUMB_ROOT/evidence/closed-loop-937704/report.json" \
  --output-dir "$PLUMB_ROOT/evidence/irasim-state-replay-$SLURM_JOB_ID" \
  --same-seed-repeat
