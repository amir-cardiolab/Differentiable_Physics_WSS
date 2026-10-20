# Source from a job script:  source cluster/activate_env.sh
# Unpacks the fenics_gpu tarball (cluster/setup_env.sbatch) to node-local disk
# and activates it. Removed automatically when the job ends.
SCR=${BFS_SCRATCH:-$HOME}        # where setup_env.sbatch put envs/fenics_gpu.tar.gz
TARBALL=$SCR/envs/fenics_gpu.tar.gz
[ -f "$TARBALL" ] || { echo "missing $TARBALL - run cluster/setup_env.sbatch" >&2; exit 1; }
ENV_LOCAL=${SLURM_TMPDIR:-/scratch/local/$USER/${SLURM_JOB_ID:-$$}}
[ -d /scratch/local ] || ENV_LOCAL=/tmp/$USER/${SLURM_JOB_ID:-$$}
ENV_LOCAL=$ENV_LOCAL/fenics_gpu
mkdir -p "$ENV_LOCAL"
trap 'rm -rf "$ENV_LOCAL" "$DIJITSO_CACHE_DIR"' EXIT
t0=$(date +%s)
tar -xzf "$TARBALL" -C "$ENV_LOCAL"
source "$ENV_LOCAL/bin/activate"
conda-unpack
echo "[env] unpacked to $ENV_LOCAL in $(( $(date +%s) - t0 ))s"
export OMP_NUM_THREADS=1
export DIJITSO_CACHE_DIR=$(dirname "$ENV_LOCAL")/dijitso   # FEniCS JIT cache, per job
