#!/bin/bash
# Figures + metrics for finished runs (run from 2D-BFS-GPU/):
#   sbatch cluster/postprocess.sbatch <OUT_DIR> [<OUT_DIR_2> ...]
#SBATCH --job-name=bfs_post
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source cluster/activate_env.sh
python postprocess.py "$@" 2>&1 | grep -v -E "JIT|FutureWarning|libibverbs|HDF5-DIAG|^\s+#|major:|minor:"
