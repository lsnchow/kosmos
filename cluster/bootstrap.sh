#!/usr/bin/env bash
# Run only on the allocated cluster. Model caches never touch the laptop.
set -euo pipefail
PLUMB_ROOT=${PLUMB_ROOT:-/scratch/lchow432/plumb}
case "$PLUMB_ROOT" in /scratch/*/plumb|/global/scratch/*/plumb) ;; *) printf 'Refusing unexpected cluster root: %s\n' "$PLUMB_ROOT" >&2; exit 2;; esac
mkdir -p "$PLUMB_ROOT"/{logs,models,cache,tmp,evidence}
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PLUMB_ROOT/tmp"
export PIP_CACHE_DIR="$PLUMB_ROOT/cache/pip"
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
python3 -m venv "$PLUMB_ROOT/venv-tools"
"$PLUMB_ROOT/venv-tools/bin/python" -m pip install --upgrade pip huggingface_hub
printf 'Cluster download environment ready at %s\n' "$PLUMB_ROOT/venv-tools"
