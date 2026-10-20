#!/bin/bash
# Run a script of this folder on a CPU node with the fenics_gpu env (sbatch -A <account> -p <partition> ...).
#   sbatch --export=ALL,SCRIPT=<script.py>[,VAR=...] cluster/run_cpu.sbatch
#SBATCH --job-name=bfs_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
source cluster/activate_env.sh
/usr/bin/time -v python "$SCRIPT" 2>&1 | grep -v -E "JIT|FutureWarning|active\["
