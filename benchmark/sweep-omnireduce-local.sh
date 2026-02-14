#!/bin/bash
#
# Local OmniReduce benchmark sweep (single machine, mpirun)
# Useful for testing on a single compute node with multiple processes
#
# Usage:
#   export LD_LIBRARY_PATH=/path/to/omnireduce/build:$LD_LIBRARY_PATH
#   ./sweep-omnireduce-local.sh
#

set -e

# Configuration
MESSAGE_SIZES=(268435456 536870912 1073741824)  # 256 MiB, 512 MiB, 1024 MiB (in bytes)
NODE_COUNTS=(2 4 8)  # worker processes on this machine
DENSITY="1.0"  # dense data
BLOCK_SIZE=256
BACKEND="gloo"

# Paths (adjust to your environment)
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIREDUCE_BUILD="${OMNIREDUCE_BUILD:-./../omnireduce-RDMA/build}"

# Verify environment
check_env() {
    if [[ ! -f "$BENCHMARK_DIR/benchmark.py" ]]; then
        echo "ERROR: benchmark.py not found in $BENCHMARK_DIR"
        exit 1
    fi
    
    if [[ -z "$LD_LIBRARY_PATH" ]] || [[ ! "$LD_LIBRARY_PATH" =~ "$OMNIREDUCE_BUILD" ]]; then
        echo "WARNING: LD_LIBRARY_PATH may not include $OMNIREDUCE_BUILD"
        echo "  Run: export LD_LIBRARY_PATH=$OMNIREDUCE_BUILD:\$LD_LIBRARY_PATH"
    fi
    
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
    
    if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
        export GLOO_SOCKET_IFNAME=eth0
        echo "Set GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME (adjust if needed)"
    fi
}

# Run benchmark locally with mpirun
run_benchmark_local() {
    local num_workers=$1
    local msg_size=$2
    local result_dir=$3
    
    mkdir -p "$result_dir"
    
    echo "Running: workers=$num_workers, msgsize=$((msg_size/1024/1024))MiB, density=$DENSITY"
    
    # Run mpirun-based benchmark
    # Note: requires aggregator running elsewhere or compiled with --enable-shared-memory
    cd "$BENCHMARK_DIR"
    
    mpirun -n "$num_workers" python benchmark.py \
        --backend "$BACKEND" \
        --tensor-size "$msg_size" \
        --block-size "$BLOCK_SIZE" \
        --density "$DENSITY" \
        --ip 127.0.0.1 \
        2>&1 | tee "$result_dir/benchmark_${num_workers}w_${msg_size}b.log"
    
    echo "  ✓ Completed"
}

main() {
    echo "OmniReduce Local Benchmark Sweep (mpirun)"
    echo "=========================================="
    echo "Message sizes: ${MESSAGE_SIZES[@]} bytes (256/512/1024 MiB)"
    echo "Worker processes: ${NODE_COUNTS[@]}"
    echo "Density: $DENSITY"
    echo ""
    
    check_env
    
    local results_root="./100G-results/local-sweep-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$results_root"
    
    echo "Results: $results_root"
    echo ""
    
    for num_workers in "${NODE_COUNTS[@]}"; do
        for msg_size in "${MESSAGE_SIZES[@]}"; do
            local result_dir="$results_root/workers_${num_workers}_msgsize_$((msg_size/1024/1024))MiB"
            run_benchmark_local "$num_workers" "$msg_size" "$result_dir"
        done
    done
    
    echo ""
    echo "=========================================="
    echo "Sweep completed!"
    echo "Results: $results_root"
}

main "$@"
