#!/usr/bin/env bash
# Prepare or verify the isolated Octo v1 source/runtime on a cluster login node.
#
# This helper never downloads model weights or public CUDA/JAX wheels.  Stage
# assets first with prepare_octo_assets.py, and install the pinned JAX/Flax/
# TensorFlow dependencies through reviewed site-supported means before --verify.
set -euo pipefail

OCTO_SETUP_ROOT=${1:-/scratch/lchow432/plumb}
OCTO_SETUP_MODE=${2:---plan}
OCTO_SETUP_SOURCE="$OCTO_SETUP_ROOT/source-octo"
OCTO_SETUP_VENV="$OCTO_SETUP_ROOT/venv-octo"
OCTO_SETUP_HF_HOME="$OCTO_SETUP_ROOT/models/.hf-octo"
OCTO_SETUP_COMMIT=37951e4e6d708fd76374f6e09e716763fe2673b1
OCTO_SETUP_JAX_BASE=${OCTO_JAX_VERSION:-0.4.20}
OCTO_SETUP_TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OCTO_SETUP_PYTHON=${OCTO_PYTHON:-python3.10}

usage() {
  cat <<'EOF'
Usage: cluster/setup_octo_runtime.sh [PLUMB_ROOT] [--plan|--execute|--verify]

  --plan     Print exact source/runtime expectations; no writes (default).
  --execute  Clone the immutable Octo source and make an isolated venv. It uses
             `pip install --no-deps -e`; install no package dependencies.
  --verify   Make no writes. Verify source HEAD/cleanliness, staged small Octo
             assets, offline T5 tokenizer lookup, and installed package pins.
EOF
}

if [[ "$OCTO_SETUP_MODE" != "--plan" && "$OCTO_SETUP_MODE" != "--execute" && "$OCTO_SETUP_MODE" != "--verify" ]]; then
  usage >&2
  exit 2
fi

case "$OCTO_SETUP_MODE" in
  --plan)
    cat <<EOF
Octo source: https://github.com/octo-models/octo @ $OCTO_SETUP_COMMIT
Source target: $OCTO_SETUP_SOURCE
Venv target: $OCTO_SETUP_VENV
Requested interpreter: $OCTO_SETUP_PYTHON (override with OCTO_PYTHON only after recording the compatibility change)
Assets command (prints plan unless --execute is supplied):
  python cluster/prepare_octo_assets.py --root $OCTO_SETUP_ROOT --model small

Required runtime base versions: JAX $OCTO_SETUP_JAX_BASE, Flax 0.7.5, TensorFlow 2.15,
NumPy 1.24.3, Orbax/Optax compatible with Octo v0.1.0. A site package version
with a '+computecanada' suffix is accepted only when its base JAX version is
$OCTO_SETUP_JAX_BASE. If OCTO_JAX_VERSION overrides 0.4.20, create an immutable
compatibility profile and rerun golden action/Gate-A/B fixtures; do not pool it
with the released replication profile. Do not install public CUDA wheels here.

Offline variables for every runtime/smoke process:
  HF_HOME=$OCTO_SETUP_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EOF
    ;;
  --execute)
    if [[ -e "$OCTO_SETUP_SOURCE" ]]; then
      OCTO_SETUP_HEAD=$(git -C "$OCTO_SETUP_SOURCE" rev-parse HEAD)
      OCTO_SETUP_DIRTY=$(git -C "$OCTO_SETUP_SOURCE" status --porcelain --untracked-files=no)
      if [[ "$OCTO_SETUP_HEAD" != "$OCTO_SETUP_COMMIT" || -n "$OCTO_SETUP_DIRTY" ]]; then
        echo "Refusing to reuse source-octo: expected clean $OCTO_SETUP_COMMIT, found $OCTO_SETUP_HEAD" >&2
        exit 2
      fi
    else
      git clone --filter=blob:none --no-checkout https://github.com/octo-models/octo.git "$OCTO_SETUP_SOURCE"
      git -C "$OCTO_SETUP_SOURCE" checkout --detach "$OCTO_SETUP_COMMIT"
    fi
    if [[ ! -x "$OCTO_SETUP_VENV/bin/python" ]]; then
      "$OCTO_SETUP_PYTHON" -m venv "$OCTO_SETUP_VENV"
    fi
    "$OCTO_SETUP_VENV/bin/python" -m pip install --no-deps -e "$OCTO_SETUP_SOURCE"
    echo "Source and empty dependency-isolated venv prepared. Install reviewed, site-compatible pinned dependencies separately, then run --verify."
    ;;
  --verify)
    if [[ ! -x "$OCTO_SETUP_VENV/bin/python" ]]; then
      echo "Missing isolated Octo Python: $OCTO_SETUP_VENV/bin/python" >&2
      exit 2
    fi
    git -C "$OCTO_SETUP_SOURCE" diff --quiet
    git -C "$OCTO_SETUP_SOURCE" diff --cached --quiet
    [[ "$(git -C "$OCTO_SETUP_SOURCE" rev-parse HEAD)" == "$OCTO_SETUP_COMMIT" ]]
    "$OCTO_SETUP_VENV/bin/python" "$OCTO_SETUP_TOOL_DIR/prepare_octo_assets.py" --root "$OCTO_SETUP_ROOT" --model small --verify
    HF_HOME="$OCTO_SETUP_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
      "$OCTO_SETUP_VENV/bin/python" - "$OCTO_SETUP_SOURCE" "$OCTO_SETUP_JAX_BASE" <<'PY'
import importlib.metadata
import inspect
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
expected_jax = sys.argv[2]
import jax
import octo.model.octo_model as octo_model
from transformers import AutoConfig, AutoTokenizer

actual_jax = jax.__version__
if actual_jax.split("+", 1)[0] != expected_jax:
    raise SystemExit("JAX mismatch: expected base %s, got %s" % (expected_jax, actual_jax))
module_file = Path(inspect.getfile(octo_model)).resolve()
module_file.relative_to(source)
if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
    raise SystemExit("offline flags were not preserved")
AutoConfig.from_pretrained("t5-base", local_files_only=True)
AutoTokenizer.from_pretrained("t5-base", local_files_only=True)
print({"jax_version_full": actual_jax, "octo_module": str(module_file), "transformers": importlib.metadata.version("transformers")})
PY
    ;;
esac
