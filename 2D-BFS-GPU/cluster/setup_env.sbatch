#!/bin/bash
# Build the fenics_gpu env and pack it into ONE tarball:
#     ${BFS_SCRATCH:-$HOME}/envs/fenics_gpu.tar.gz      (BFS_SCRATCH: e.g. your scratch directory)
#
# Why a tarball: scratch file systems that purge files by age delete an unpacked conda
# env piece by piece (conda keeps the packages' old timestamps). Jobs unpack the tarball
# to node-local disk (cluster/activate_env.sh), which is also much faster than importing
# thousands of small files over NFS. Needs conda/miniconda at ~/miniconda3 (or edit below).
#
#   sbatch cluster/setup_env.sbatch        (run from 2D-BFS-GPU/, ~30-45 min)
#SBATCH --job-name=fenics_gpu_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -eo pipefail

SCR=${BFS_SCRATCH:-$HOME}
TARBALL=$SCR/envs/fenics_gpu.tar.gz
LOCAL=${SLURM_TMPDIR:-/scratch/local/$USER/$SLURM_JOB_ID}
[ -d /scratch/local ] || LOCAL=/tmp/$USER/$SLURM_JOB_ID
ENV_PREFIX=$LOCAL/fenics_gpu
export CONDA_PKGS_DIRS=$LOCAL/pkgs
export PIP_CACHE_DIR=$LOCAL/pip_cache
mkdir -p "$LOCAL" "$SCR/envs"
trap 'rm -rf "$LOCAL"' EXIT

source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels --solver=libmamba \
    python=3.9 fenics=2019.1.0 "numpy<1.24" scipy matplotlib vtk conda-pack
conda activate "$ENV_PREFIX"

# dolfin-adjoint matching FEniCS 2019.1
pip install --no-deps "git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0"
# PyTorch with CUDA 11.8 wheels (bundles its own CUDA runtime; sm_60..sm_90)
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118
# NVIDIA cuDSS (GPU sparse direct solver, used by gpu_cudss.py); pulls cuBLAS 12,
# which coexists with torch's CUDA-11 cuBLAS (driver must support CUDA 12)
pip install nvidia-cudss-cu12

python - <<'PY'
import dolfin, pyadjoint, fenics_adjoint, torch, scipy, vtk, numpy
print("dolfin", dolfin.__version__, "| pyadjoint", pyadjoint.__version__,
      "| torch", torch.__version__, "| numpy", numpy.__version__)
PY

conda-pack -p "$ENV_PREFIX" -o "${TARBALL%.tar.gz}.tmp.tar.gz" --n-threads 8 --force
mv "${TARBALL%.tar.gz}.tmp.tar.gz" "$TARBALL"
ls -lh "$TARBALL"
echo "ENV READY: $TARBALL"
