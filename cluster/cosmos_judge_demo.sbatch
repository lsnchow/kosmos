#!/usr/bin/env bash
#SBATCH --job-name=plumb-cosmos-judge
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/lchow432/plumb/logs/cosmos-judge-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
export PYTHONPATH="$1"
export HF_HOME=/scratch/lchow432/plumb/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/scratch/lchow432/plumb/venv-model-std2023/bin/python "$1/cluster/cosmos_demo_server.py" --token-file "$2" --port 8919 &
cosmos_pid=$!
/scratch/lchow432/plumb/venv-judge-lora-tf449/bin/python "$1/cluster/judge_demo_server.py" --token-file "$2" --port 8921 &
judge_pid=$!
trap 'kill "$cosmos_pid" "$judge_pid" 2>/dev/null || true' EXIT
wait -n "$cosmos_pid" "$judge_pid"
