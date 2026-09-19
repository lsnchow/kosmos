#!/usr/bin/env bash
#SBATCH --job-name=plumb-irasim-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lchow432/plumb/logs/irasim-probe-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17 opencv/4.11.0
PLUMB_ROOT=/scratch/lchow432/plumb
PLUMB_SOURCE=${1:-$PLUMB_ROOT/source}
export PYTHONPATH="$PLUMB_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PLUMB_SOURCE_RELEASE="$(basename "$PLUMB_SOURCE")"
export HF_HOME="$PLUMB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="${SLURM_TMPDIR:-$PLUMB_ROOT/tmp}"
"$PLUMB_ROOT/venv-irasim/bin/python" "$PLUMB_SOURCE/cluster/irasim_probe_suite.py" \
  --repo "$PLUMB_ROOT/irasim-source" \
  --checkpoint "$PLUMB_ROOT/models/irasim/frame_ada_0300000.safetensors" \
  --checkpoint-sha256 d0ea8e8bec50818e414a278fe58fb8d187f577cfe8b260cadc880cfa6107778d \
  --vae "$PLUMB_ROOT/models/stabilityai--stable-diffusion-xl-base-1.0" \
  --scheduler "$PLUMB_ROOT/irasim-source/pretrained_models/scheduler" \
  --config "$PLUMB_ROOT/irasim-source/configs/evaluation/bridge/frame_ada.yaml" \
  --image "$PLUMB_ROOT/fixtures/irasim-openvla-first-frame.png" \
  --action "$PLUMB_ROOT/fixtures/irasim-openvla-action.json" \
  --action-provenance "$PLUMB_ROOT/fixtures/irasim-openvla-action-provenance.json" \
  --output-dir "$PLUMB_ROOT/evidence/irasim-probe-$SLURM_JOB_ID" --seeds 0,1 --axis-delta .005
