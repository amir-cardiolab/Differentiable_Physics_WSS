#!/bin/bash
# Build cuDOLFINx 0.10 with the adjoint patch, inside the official cuDOLFINx container:
#
#   docker run --gpus all -it -v $PWD:/work -w /work benpachev/cudolfinx:v0.10.0-cuda12.6
#   (or: apptainer exec --nv docker://benpachev/cudolfinx:v0.10.0-cuda12.6 bash)
#   bash cudolfinx/build.sh [PREFIX]                 # run from 2D-BFS-GPU/, default PREFIX=./cudolfinx-build
#
# then, before running the scripts:  source PREFIX/env.sh
# CUDA_ARCH (default "80;86;89;90") selects the GPU architectures to compile for.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$(mkdir -p "${1:-$PWD/cudolfinx-build}" && cd "${1:-$PWD/cudolfinx-build}" && pwd)
ARCH=${CUDA_ARCH:-80;86;89;90}
UPSTREAM=https://github.com/bpachev/cuda-dolfinx.git
BASE=258274d                       # cuDOLFINx v0.10.0.post2 + 4, the version the patch applies to
export PETSC_DIR=${PETSC_DIR:-/usr/local/petsc} PETSC_ARCH=${PETSC_ARCH:-linux-gnu-real64-32-cuda}
export CMAKE_PREFIX_PATH=/usr/local/dolfinx-real:$PREFIX/install:$CMAKE_PREFIX_PATH
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH
CUDA_LIB=$(dirname "$(dirname "$(command -v nvcc)")")/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CUDA_LIB:$LD_LIBRARY_PATH CUDACXX=$(command -v nvcc)

echo "## source: $UPSTREAM @ $BASE + cuda-dolfinx-adjoint.patch"
rm -rf "$PREFIX/src" && git clone -q "$UPSTREAM" "$PREFIX/src"
git -C "$PREFIX/src" checkout -q "$BASE"
git -C "$PREFIX/src" apply "$HERE/cuda-dolfinx-adjoint.patch"

echo "## C++ core"
cmake -G Ninja -S "$PREFIX/src/cpp" -B "$PREFIX/cpp" -DCMAKE_INSTALL_PREFIX="$PREFIX/install" -DCMAKE_BUILD_TYPE=Release \
  -DCUDOLFINX_SKIP_BUILD_TESTS=YES -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DHDF5_ENABLE_PARALLEL=ON \
  -DCMAKE_CXX_FLAGS="-I$PREFIX/src/cpp" > "$PREFIX/cmake.log"
ninja -C "$PREFIX/cpp" install > "$PREFIX/ninja.log"

echo "## Python package"
rm -rf "$PREFIX/site" && mkdir -p "$PREFIX/site"
(cd "$PREFIX/src/python" && CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-I$PREFIX/install/include -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DHDF5_ENABLE_PARALLEL=ON" \
  pip install --no-build-isolation --no-deps --no-cache-dir --config-settings=cmake.build-type=Release \
  --target "$PREFIX/site" . > "$PREFIX/pip.log")
pip install --no-deps --target "$PREFIX/site" pyadjoint-ad checkpoint_schedules sympy mpmath h5py nvidia-cudss-cu12 > "$PREFIX/deps.log"

cat > "$PREFIX/env.sh" <<ENV
export PYTHONPATH=$PREFIX/site:\$PYTHONPATH
export LD_LIBRARY_PATH=$PREFIX/install/lib:$CUDA_LIB:\$LD_LIBRARY_PATH
export CUDSS_LIB_DIR=$PREFIX/site/nvidia/cu12/lib
ENV
echo "## done: source $PREFIX/env.sh"
