#!/usr/bin/env bash
# Cluster-only experimental IRASim runtime. No weights or VAE assets are downloaded here.
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
PLUMB_ROOT=${PLUMB_ROOT:-/scratch/lchow432/plumb}
IRASIM_REPO=${IRASIM_REPO:?Set IRASIM_REPO to the pre-cloned c72b6dade6fcd65971e0aa8ab49ea39b15108c90 checkout}
IRASIM_VENV=${IRASIM_VENV:-$PLUMB_ROOT/venv-irasim}
test -d "$IRASIM_REPO/.git" || { echo "missing checked-out IRASim repo" >&2; exit 2; }
test "$(git -C "$IRASIM_REPO" rev-parse HEAD)" = c72b6dade6fcd65971e0aa8ab49ea39b15108c90 || { echo "wrong IRASim revision" >&2; exit 2; }
if [ ! -x "$IRASIM_VENV/bin/python" ]; then virtualenv --no-download "$IRASIM_VENV"; fi
"$IRASIM_VENV/bin/pip" install --upgrade pip
# Alliance Torch wheels are selected explicitly; upstream uses CUDA 11.8 but
# this cluster runtime is CUDA 12.6. Do not install flash-attn/xformers.
"$IRASIM_VENV/bin/pip" install 'torch==2.6.0+computecanada' 'torchvision==0.21.0+computecanada' --no-index
# diffusers 0.24 imports cached_download, so hub must stay below 0.26.
"$IRASIM_VENV/bin/pip" install 'diffusers==0.24.0' 'huggingface_hub==0.25.2' \
  'transformers==4.40.1' 'accelerate==0.24.1' 'timm==0.9.10' \
  'omegaconf' 'einops' 'einops-exts' 'rotary-embedding-torch<0.9' \
  'imageio' 'imageio-ffmpeg' 'scikit-image' 'safetensors' 'sentencepiece'
"$IRASIM_VENV/bin/python" -m pip freeze > "$PLUMB_ROOT/evidence/irasim-runtime-freeze.txt"
echo "Runtime only. Set explicit local checkpoint/VAE/scheduler paths; this script never downloads them."
