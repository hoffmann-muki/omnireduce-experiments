#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load gcc/11.3.0 cuda/12.3.0/gcc/11.3.0/icelake nccl/2.18.1-1/gcc/11.3.0/icelake openmpi/4.1.5 2>/dev/null || true

CUDA_HOME=${CUDA_HOME:-$(which nvcc | xargs dirname | xargs dirname)}
NCCL_HOME=${NCCL_HOME:-/usr/local/nccl2}
[[ -f "$NCCL_HOME/include/nccl.h" ]] || { echo "ERROR: nccl.h not found in $NCCL_HOME/include"; exit 1; }

MPICXX=$(which mpicxx || which mpic++ || echo "not found")
[[ "$MPICXX" != "not found" ]] || { echo "ERROR: MPI compiler not found"; exit 1; }

echo "Building nccl_test..."
$MPICXX -O3 "$SCRIPT_DIR/nccl_test.cpp" -o "$SCRIPT_DIR/nccl_test" \
    -I$CUDA_HOME/include -I$NCCL_HOME/include \
    -L$CUDA_HOME/lib64 -L$NCCL_HOME/lib \
    -lcudart -lnccl
echo "✓ $SCRIPT_DIR/nccl_test"
