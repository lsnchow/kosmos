#!/usr/bin/env bash
# Install on the network-enabled login node; run inference only on compute nodes.
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
PLUMB_ROOT=${PLUMB_ROOT:-/scratch/lchow432/plumb}
PLUMB_RUNTIME="$PLUMB_ROOT/venv-model-std2023"
case "$PLUMB_ROOT" in /scratch/*/plumb|/global/scratch/*/plumb) ;; *) exit 2;; esac
mkdir -p "$PLUMB_ROOT"/{cache/pip,cache/huggingface,tmp,logs,evidence,fixtures}
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$PLUMB_ROOT/cache/pip"
export TMPDIR="$PLUMB_ROOT/tmp"
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
if [ ! -x "$PLUMB_RUNTIME/bin/python" ]; then
  virtualenv --no-download "$PLUMB_RUNTIME"
fi
"$PLUMB_RUNTIME/bin/python" -m pip install --upgrade pip
"$PLUMB_RUNTIME/bin/python" -m pip install \
  'torch==2.6.0+computecanada' 'torchvision==0.21.0+computecanada' --no-index
# The pinned Diffusers setup requires hub>=1.31, incompatible with Transformers4.
# Its example requirements still say <5; the resolver/import probe is authoritative.
"$PLUMB_RUNTIME/bin/python" -m pip install \
  'transformers>=5,<6' 'accelerate>=0.31.0' \
  safetensors huggingface_hub sentencepiece numpy pillow imageio imageio-ffmpeg av \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@a3e0b8ec235c27a6c17a21976daf7fd32d819d05'
"$PLUMB_RUNTIME/bin/python" -m pip freeze > "$PLUMB_ROOT/evidence/runtime-freeze.txt"
printf 'Runtime installed. GPU execution must use allocated compute nodes.\n'
