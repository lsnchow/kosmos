#!/usr/bin/env bash
#SBATCH --job-name=kosmos-presentation
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/lchow432/plumb/logs/presentation-%j.log
set -euo pipefail
module load StdEnv/2023 python/3.11 cuda/12.6 cudnn/9.5.1.17
export PYTHONPATH="$1"
export HF_HOME=/scratch/lchow432/plumb/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cosmos_pid=""
judge_pid=""
cleanup() {
    if [[ -n "$cosmos_pid" ]]; then kill "$cosmos_pid" 2>/dev/null || true; fi
    if [[ -n "$judge_pid" ]]; then kill "$judge_pid" 2>/dev/null || true; fi
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' TERM INT
# Recover process crashes inside this one bounded allocation. No new GPU job
# or inference request is automatically submitted by this loop.
for attempt in 1 2 3 4; do
    printf 'Starting model workers, attempt %s of 4\n' "$attempt"
    /scratch/lchow432/plumb/venv-model-std2023/bin/python "$1/cluster/cosmos_demo_server.py" --token-file "$2" --port 8919 &
    cosmos_pid=$!
    /scratch/lchow432/plumb/venv-judge-lora-tf449/bin/python "$1/cluster/judge_demo_server.py" --token-file "$2" --port 8921 &
    judge_pid=$!
    wait -n "$cosmos_pid" "$judge_pid" || true
    cleanup
    cosmos_pid=""
    judge_pid=""
    sleep 5
done
exit 1
