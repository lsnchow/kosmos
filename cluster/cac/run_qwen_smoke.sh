#!/usr/bin/env bash
# Run inside the persistent allocated CAC Slurm shell, never from login1.
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  printf 'Run this only inside an allocated Slurm GPU shell.\n' >&2
  exit 2
fi
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
PLUMB_ROOT=${PLUMB_ROOT:-/global/scratch/${USER}/plumb}
case "$PLUMB_ROOT" in /global/scratch/*/plumb) ;; *) exit 2;; esac
RUNTIME="$PLUMB_ROOT/venv-qwen-tf517"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

exec "$RUNTIME/bin/python" "$PLUMB_ROOT/source/cluster/policy_smoke.py" \
  --root "$PLUMB_ROOT" judge \
  --transformers-version 5.17.0 "$@"
