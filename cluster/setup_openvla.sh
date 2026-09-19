#!/usr/bin/env bash
# Separate local-only OpenVLA runtime.  Do not install this into the Cosmos
# Diffusers/Transformers-5 environment: OpenVLA's audited remote code requires
# Transformers 4.40.1 and tokenizers 0.19.1.
set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17

PLUMB_ROOT=${PLUMB_ROOT:-/scratch/lchow432/plumb}
case "$PLUMB_ROOT" in
  /scratch/*/plumb|/global/scratch/*/plumb) ;;
  *)
    printf 'Refusing unexpected PLUMB_ROOT: %s\n' "$PLUMB_ROOT" >&2
    exit 2
    ;;
esac

OPENVLA_RUNTIME="$PLUMB_ROOT/venv-openvla-tf440"
mkdir -p "$PLUMB_ROOT"/{cache/pip,cache/huggingface,tmp,logs,evidence,fixtures}
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$PLUMB_ROOT/cache/pip"
export TMPDIR="$PLUMB_ROOT/tmp"
export XDG_CACHE_HOME="$PLUMB_ROOT/cache"
export PYTHONNOUSERSITE=1

if [ ! -x "$OPENVLA_RUNTIME/bin/python" ]; then
  virtualenv --no-download "$OPENVLA_RUNTIME"
fi

# Compute Canada supplies these exact CUDA-matched wheels locally.  Do not let
# pip resolve a public Torch build or consume packages from the Cosmos venv.
"$OPENVLA_RUNTIME/bin/python" -m pip install --no-index \
  'torch==2.6.0+computecanada' 'torchvision==0.21.0+computecanada'

# This environment intentionally has no Diffusers/Cosmos dependency.  These
# packages must be resolved into this venv only, then frozen as evidence.
"$OPENVLA_RUNTIME/bin/python" -m pip install \
  'transformers==4.40.1' 'tokenizers==0.19.1' 'timm==0.9.10' \
  'safetensors>=0.4.3' 'huggingface_hub<1' 'accelerate>=0.31,<1' \
  numpy pillow imageio imageio-ffmpeg av

"$OPENVLA_RUNTIME/bin/python" -m pip check
"$OPENVLA_RUNTIME/bin/python" - <<'PY'
import json
import importlib.metadata
import platform
import sys
import timm
import tokenizers
import torch
import torchvision
import transformers

expected = {
    "transformers": "4.40.1",
    "tokenizers": "0.19.1",
    "timm": "0.9.10",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
}
actual = {
    "python": sys.version,
    "platform": platform.platform(),
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "timm": timm.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "reviewed_openvla_remote_code_revision": "47a0ec7fc4ec123775a391911046cf33cf9ed83f",
}
actual["distribution_versions"] = {name: importlib.metadata.version(name) for name in expected}
wrong = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if str(actual.get(key)).split("+", 1)[0] != value}
if wrong:
    raise SystemExit("Pinned OpenVLA runtime mismatch: " + json.dumps(wrong, sort_keys=True))
print(json.dumps(actual, sort_keys=True))
PY
"$OPENVLA_RUNTIME/bin/python" -m pip freeze > "$PLUMB_ROOT/evidence/openvla-runtime-freeze.txt"
printf 'OpenVLA runtime installed at %s (kept separate from venv-model-std2023).\n' "$OPENVLA_RUNTIME"
