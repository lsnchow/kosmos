#!/usr/bin/env bash
# Download the immutable Qwen checkpoint directly to Frontenac shared scratch.
# Invoke only after a GPU allocation exists; this script never moves weights
# through a workstation and leaves a verified upstream-manifest in evidence/.
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
PLUMB_ROOT=${PLUMB_ROOT:-/global/scratch/${USER}/plumb}
case "$PLUMB_ROOT" in
  /global/scratch/*/plumb) ;;
  *)
    printf 'Refusing unexpected Frontenac PLUMB_ROOT: %s\n' "$PLUMB_ROOT" >&2
    exit 2
    ;;
esac

RUNTIME="$PLUMB_ROOT/venv-qwen-tf517"
SOURCE="$PLUMB_ROOT/source/cluster/download_assets.py"
if [ ! -x "$RUNTIME/bin/python" ]; then
  printf 'Qwen runtime is absent; run cluster/cac/setup_qwen_runtime.sh first.\n' >&2
  exit 2
fi
if [ ! -f "$SOURCE" ]; then
  printf 'Cluster source is absent: %s\n' "$SOURCE" >&2
  exit 2
fi
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PLUMB_ROOT/tmp"
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"

exec "$RUNTIME/bin/python" "$SOURCE" \
  Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --root "$PLUMB_ROOT" --max-gb 20 --execute
