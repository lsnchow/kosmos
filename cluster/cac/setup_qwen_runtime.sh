#!/usr/bin/env bash
# Frontenac-only Qwen judge runtime.  Run its networked package install from a
# login node; run GPU imports/inference only from the allocated Slurm shell.
set -euo pipefail

# Transformers 5.17 currently requires tokenizers>=0.23.1, while CAC's
# wheelhouse tops out at 0.22.2.  Loading the site Rust toolchain lets the
# pinned newer tokenizer build from its small source distribution on login1.
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 rust/1.95.0

# Frontenac currently exports a stale /scratch/$USER value.  The mounted,
# shared scratch filesystem on both login1 and frnt191 is /global/scratch.
PLUMB_ROOT=${PLUMB_ROOT:-/global/scratch/${USER}/plumb}
case "$PLUMB_ROOT" in
  /global/scratch/*/plumb) ;;
  *)
    printf 'Refusing unexpected Frontenac PLUMB_ROOT: %s\n' "$PLUMB_ROOT" >&2
    exit 2
    ;;
esac

RUNTIME="$PLUMB_ROOT/venv-qwen-tf517"
mkdir -p "$PLUMB_ROOT"/{cache/pip,cache/huggingface,tmp,logs,evidence,fixtures,models,source}
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$PLUMB_ROOT/cache/pip"
export TMPDIR="$PLUMB_ROOT/tmp"
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
export PYTHONNOUSERSITE=1

if [ ! -x "$RUNTIME/bin/python" ]; then
  virtualenv --no-download "$RUNTIME"
fi

"$RUNTIME/bin/python" -m pip install --upgrade pip
"$RUNTIME/bin/python" -m pip install --no-index \
  'torch==2.6.0+computecanada' 'torchvision==0.21.0+computecanada'
# Qwen needs the native Transformers implementation only.  Keep it separate
# from the Diffusers/Cosmos environment and pin the module version that must
# be supplied to QwenJudgeProfile.
"$RUNTIME/bin/python" -m pip install \
  'transformers==5.17.0' 'accelerate>=0.31.0' \
  'huggingface_hub>=1.31,<2' safetensors sentencepiece numpy pillow \
  imageio imageio-ffmpeg av
"$RUNTIME/bin/python" -m pip check
"$RUNTIME/bin/python" - <<'PY'
import importlib.metadata
import json
import platform
import sys

import torch
import torchvision
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

actual = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch_module_version": torch.__version__,
    "torch_distribution_version": importlib.metadata.version("torch"),
    "torchvision_module_version": torchvision.__version__,
    "transformers_module_version": transformers.__version__,
    "transformers_distribution_version": importlib.metadata.version("transformers"),
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "qwen_native_loader": Qwen2_5_VLForConditionalGeneration.__name__,
    "processor_loader": AutoProcessor.__name__,
}
if actual["transformers_module_version"] != "5.17.0":
    raise SystemExit("Pinned Qwen Transformers module mismatch: " + json.dumps(actual, sort_keys=True))
print(json.dumps(actual, sort_keys=True))
PY
"$RUNTIME/bin/python" -m pip freeze > "$PLUMB_ROOT/evidence/qwen-runtime-freeze.txt"
printf 'Qwen runtime ready at %s. Profile transformers_version is 5.17.0.\n' "$RUNTIME"
