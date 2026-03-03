#!/bin/bash
#
# Minimal NCCL benchmark: pure NCCL allreduce, no aggregators
# Based on nccl-benchmark.sh by the original authors, adapted for Perlmutter/SLURM
#
# Usage:
#   ./minimal-nccl-benchmark.sh [OPTIONS]
#
# Options:
#   --msg-size MiB    Message size: 256, 512, or 1024 (default: 256)
#   --density FLOAT   Tensor density 0.0-1.0 (default: 0.05)
#   --backend NAME    PyTorch backend: nccl, gloo (default: nccl)
#   --warmup ITERS    Warmup iterations (default: 5)
#   --measure ITERS   Measurement iterations (default: 20)
#   --help            Show this help message
#
# Prerequisites:
#   - Running inside a SLURM allocation (salloc or sbatch)
#   - Allocation with --gpus-per-node and --ntasks-per-node set
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark.py"
CONDA_PYTHON=${CONDA_PYTHON:-python}

# Defaults
MSG_SIZE_MIB=256
DENSITY=0.05
BACKEND=nccl
WARMUP_ITERS=5
MEASURE_ITERS=20
BLOCK_SIZE=256

while [[ $# -gt 0 ]]; do
    case $1 in
        --msg-size)  MSG_SIZE_MIB="$2";  shift 2 ;;
        --density)   DENSITY="$2";       shift 2 ;;
        --backend)   BACKEND="$2";       shift 2 ;;
        --warmup)    WARMUP_ITERS="$2";  shift 2 ;;
        --measure)   MEASURE_ITERS="$2"; shift 2 ;;
        --help)      grep "^#" "$0" | head -20; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# Convert MiB to bytes
case "$MSG_SIZE_MIB" in
    256)  TENSOR_SIZE=268435456  ;;
    512)  TENSOR_SIZE=536870912  ;;
    1024) TENSOR_SIZE=1073741824 ;;
    *) echo "ERROR: --msg-size must be 256, 512, or 1024"; exit 1 ;;
esac

# ── Auto-detect SLURM allocation ──────────────────────────────────────────────
if [[ -z "$SLURM_NODELIST" ]]; then
    echo "ERROR: Not inside a SLURM allocation. Run salloc or sbatch first."
    exit 1
fi

HOSTS=$(scontrol show hostnames "$SLURM_NODELIST")
NUM_NODES=$(echo "$HOSTS" | wc -l)
readarray -t NODE_ARR <<< "$HOSTS"
COORD_IP="${NODE_ARR[0]}"

# GPUs per node
if [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
    GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"
else
    GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
fi

TOTAL_WORKERS=$(( NUM_NODES * GPUS_PER_NODE ))

# ── Auto-detect network interface ─────────────────────────────────────────────
if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
    NCCL_SOCKET_IFNAME=$(ip -o -4 addr show | grep -v "127.0.0.1" | awk '{print $2; exit}')
    [[ -z "$NCCL_SOCKET_IFNAME" ]] && NCCL_SOCKET_IFNAME=nmn0
    export NCCL_SOCKET_IFNAME
fi
if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
    export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
fi

# ── Result directory ───────────────────────────────────────────────────────────
RESULT_DIR="${SCRIPT_DIR}/results/${BACKEND}/node_${NUM_NODES}/msgsize_${MSG_SIZE_MIB}MiB/density_${DENSITY}"
mkdir -p "$RESULT_DIR"
CSV_FILE="${RESULT_DIR}/summary.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg" > "$CSV_FILE"
fi

# ── Print config ───────────────────────────────────────────────────────────────
echo "Minimal NCCL Benchmark"
echo "======================"
echo "  Nodes          : $NUM_NODES  (${NODE_ARR[*]})"
echo "  GPUs/node       : $GPUS_PER_NODE"
echo "  Total workers   : $TOTAL_WORKERS"
echo "  Coordinator IP  : $COORD_IP"
echo "  Backend         : $BACKEND"
echo "  Message size    : ${MSG_SIZE_MIB} MiB  ($TENSOR_SIZE floats)"
echo "  Density         : $DENSITY"
echo "  Warmup / measure: $WARMUP_ITERS / $MEASURE_ITERS"
echo "  NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "  Results dir     : $RESULT_DIR"
echo ""

# ── Run 3 times, pool timing, compute stats ───────────────────────────────────
for run_num in 1 2 3; do
    RUN_DIR="${RESULT_DIR}/run_${run_num}"
    mkdir -p "$RUN_DIR"
    echo "---- Run $run_num/3 ----"

    # Kill stale python processes on all nodes
    for node in "${NODE_ARR[@]}"; do
        ssh "$node" "pkill -9 python" 2>/dev/null || true &
    done
    wait
    sleep 1

    # Launch one worker per GPU across all nodes
    global_rank=0
    for ((node_idx=0; node_idx<NUM_NODES; node_idx++)); do
        node="${NODE_ARR[$node_idx]}"
        for ((local_gpu=0; local_gpu<GPUS_PER_NODE; local_gpu++)); do
            echo "  worker rank=$global_rank  node=$node  gpu=$local_gpu"
            ssh "$node" "
                export CUDA_VISIBLE_DEVICES=$local_gpu
                export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
                export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
                export NCCL_DEBUG=WARN
                cd $SCRIPT_DIR
                $CONDA_PYTHON benchmark.py \
                    --backend $BACKEND \
                    --tensor-size $TENSOR_SIZE \
                    --block-size $BLOCK_SIZE \
                    --density $DENSITY \
                    --rank $global_rank \
                    --size $TOTAL_WORKERS \
                    --ip $COORD_IP \
                    --warmup-iters $WARMUP_ITERS \
                    --measure-iters $MEASURE_ITERS
            " > "${RUN_DIR}/worker_${global_rank}.log" 2>&1 &
            global_rank=$(( global_rank + 1 ))
        done
    done

    echo "  Waiting for all workers..."
    wait
    echo "  Run $run_num complete."
done

# ── Pool timings and compute stats ────────────────────────────────────────────
echo ""
echo "Computing statistics..."
all_time_only=()
all_time_with_barrier=()

for run_num in 1 2 3; do
    RUN_DIR="${RESULT_DIR}/run_${run_num}"
    for logfile in "${RUN_DIR}"/worker_*.log; do
        [[ -f "$logfile" ]] || continue
        while IFS= read -r line; do
            if [[ $line =~ time_only:([0-9.e+\-]+)\;time_with_barrier:([0-9.e+\-]+)\; ]]; then
                all_time_only+=("${BASH_REMATCH[1]}")
                all_time_with_barrier+=("${BASH_REMATCH[2]}")
            fi
        done < "$logfile"
    done
done

if [[ ${#all_time_only[@]} -eq 0 ]]; then
    echo "WARNING: No timing data found. Check worker logs in $RESULT_DIR/run_*/"
    exit 1
fi

time_only_stats=$(printf "%s\n" "${all_time_only[@]}" | awk '{
    if (NR==1||$1<min) min=$1; if (NR==1||$1>max) max=$1; sum+=$1; count++
} END { if (count>0) printf "%.1f,%.1f,%.1f", min, max, sum/count }')

barrier_stats=$(printf "%s\n" "${all_time_with_barrier[@]}" | awk '{
    if (NR==1||$1<min) min=$1; if (NR==1||$1>max) max=$1; sum+=$1; count++
} END { if (count>0) printf "%.1f,%.1f,%.1f", min, max, sum/count }')

echo "$NUM_NODES,$MSG_SIZE_MIB,$time_only_stats,$barrier_stats" >> "$CSV_FILE"

echo "✓ Results appended to: $CSV_FILE"
echo ""
echo "node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg"
echo "$NUM_NODES,$MSG_SIZE_MIB,$time_only_stats,$barrier_stats"
