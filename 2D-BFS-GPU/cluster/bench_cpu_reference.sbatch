#!/bin/bash
# Time the ORIGINAL CPU script (pure FEniCS + dolfin-adjoint).
#   sbatch --export=ALL,N=4000,EPOCHS=10 cluster/bench_cpu_reference.sbatch   (from 2D-BFS-GPU/)
#SBATCH --job-name=bfs_cpu_ref
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail
N=${N:-200}
EPOCHS=${EPOCHS:-1}
ENV=${ENV:-fenics2018}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
export OMP_NUM_THREADS=1
export DIJITSO_CACHE_DIR=${BFS_SCRATCH:-$HOME}/.cache/dijitso

B=logs/bench_cpu_ref_N${N}_ep${EPOCHS}_${SLURM_JOB_ID}.py
sed -e "s/^epochs = 200/epochs = ${EPOCHS}/" \
    -e "s/^final_idx = 4000/final_idx = ${N}/" \
    -e 's/^inverse_dir = (/inverse_dir = "bench_" + (/' \
    -e 's/^    grad_u = rf.derivative()/    _tg = time.time(); grad_u = rf.derivative(); print("ADJOINT_TIME %.2f" % (time.time() - _tg))/' \
    Diff_Physics_CPU_reference.py > "$B"
python -c "import dolfin, fenics_adjoint; print('dolfin', dolfin.__version__)"
/usr/bin/time -v python "$B" 2>&1 | grep -v -E "JIT|FutureWarning|active\["
