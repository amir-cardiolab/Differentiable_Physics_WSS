#!/bin/bash
# Run a script of this folder on one GPU node. Account/partition: pass them to sbatch,
#   e.g. sbatch -A <account> -p <gpu-partition> --gres=gpu:1 cluster/run_gpu.sbatch
#   sbatch cluster/run_gpu.sbatch                                   # full run (200 epochs, N=4000)
#   sbatch --export=ALL,EPOCHS=1,FINAL_IDX=200 cluster/run_gpu.sbatch # quick benchmark
#   sbatch --export=ALL,SCRIPT=verify_gradient.py,N=20 cluster/run_gpu.sbatch
#SBATCH --job-name=bfs_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
SCRIPT=${SCRIPT:-Diff_Physics_GPU.py}
source cluster/activate_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v python "$SCRIPT" 2>&1 | grep -v -E "JIT|FutureWarning|active\["
