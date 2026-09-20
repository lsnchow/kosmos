#!/usr/bin/env bash
# One-H100, unqualified interactive OpenVLA + original-IRASim live-demo worker.
# Usage: sbatch cluster/live_demo_server.sbatch [PLUMB_SOURCE] TOKEN_FILE [PORT]
# Root can override the bounded default with sbatch --partition/--time options.
#SBATCH --job-name=plumb-live-demo
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lchow432/plumb/logs/live-demo-%j.log
set -euo pipefail

PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
TOKEN_FILE=${2:?Usage: sbatch live_demo_server.sbatch [PLUMB_SOURCE] TOKEN_FILE [PORT]}
PORT=${3:-8917}

if [ ! -r "$TOKEN_FILE" ]; then
  echo "live-demo token file is missing or unreadable" >&2
  exit 2
fi

module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"

# Bind to the compute-node private interface. The operator should use an SSH
# tunnel from the login node; bearer auth remains required on every endpoint.
exec "$PLUMB_ROOT/venv-irasim/bin/python" "$PLUMB_SOURCE/cluster/live_demo_server.py" \
  --root "$PLUMB_ROOT" --token-file "$TOKEN_FILE" --host 0.0.0.0 --port "$PORT"
