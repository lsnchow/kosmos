#!/usr/bin/env bash
#SBATCH --job-name=plumb-irasim
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lchow432/plumb/logs/irasim-smoke-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
IRASIM_PYTHON="$PLUMB_ROOT/venv-irasim/bin/python"
CHECKPOINT="$PLUMB_ROOT/models/irasim/frame_ada_0300000.safetensors"
CONFIG="$PLUMB_ROOT/irasim-source/configs/evaluation/bridge/frame_ada.yaml"
if [ ! -f "$CHECKPOINT" ]; then
  "$IRASIM_PYTHON" "$PLUMB_SOURCE/cluster/convert_irasim_checkpoint.py" \
    --checkpoint "$PLUMB_ROOT/models/irasim/frame_ada_0300000.pt" \
    --output "$CHECKPOINT" --repo "$PLUMB_ROOT/irasim-source" --config "$CONFIG" \
    --source-sha256 ff740a0faf7cfeabfdd598562276afb88e9423dd3cb0c10eecfe005c621094ef \
    --report "$PLUMB_ROOT/evidence/irasim-safe-conversion-$SLURM_JOB_ID.json"
fi
"$IRASIM_PYTHON" "$PLUMB_SOURCE/cluster/prepare_irasim_fixture.py" \
  --root "$PLUMB_ROOT" --policy-report "$PLUMB_ROOT/evidence/openvla-smoke-937500.json"
"$IRASIM_PYTHON" "$PLUMB_SOURCE/cluster/irasim_smoke.py" \
  --repo "$PLUMB_ROOT/irasim-source" --checkpoint "$CHECKPOINT" \
  --vae "$PLUMB_ROOT/models/stabilityai--stable-diffusion-xl-base-1.0" \
  --scheduler "$PLUMB_ROOT/irasim-source/pretrained_models/scheduler" --config "$CONFIG" \
  --image "$PLUMB_ROOT/fixtures/irasim-openvla-first-frame.png" \
  --action "$PLUMB_ROOT/fixtures/irasim-openvla-action.json" \
  --action-provenance "$PLUMB_ROOT/fixtures/irasim-openvla-action-provenance.json" \
  --output "$PLUMB_ROOT/evidence/irasim-one-step-$SLURM_JOB_ID.mp4" \
  --report "$PLUMB_ROOT/evidence/irasim-one-step-$SLURM_JOB_ID.json"
