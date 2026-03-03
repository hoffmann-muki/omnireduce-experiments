#!/bin/bash
#
# Minimal OmniReduce benchmark: allreduce via OmniReduce aggregators + gloo backend
# Based on omni-benchmark.sh by the original authors, adapted for Perlmutter/SLURM
#
# Usage:
#   ./minimal-omni-benchmark.sh [OPTIONS]
#
# Options:
#   --msg-size MiB    Message size: 256, 512, or 1024 (default: 256)
#   --density FLOAT   Tensor density 0.0-1.0 (default: 0.05)
#   --warmup ITERS    Warmup iterations (default: 5)
#   --measure ITERS   Measurement iterations (default: 20)
#   --help            Show this help message
#
# Prerequisites:
#   - Running inside a SLURM allocation (salloc or sbatch)
#   - omnireduce.cfg present in the same directory as this script
#   - OmniReduce built at OMNIREDUCE_BUILD / OMNIREDUCE_AGG paths below
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PYTHON=${CONDA_PYTHON:-python}

# OmniReduce paths (override via env vars if needed)
OMNIREDUCE_BUILD=${OMNIREDUCE_BUILD:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/omnireduce}
OMNIREDUCE_AGG=${OMNIREDUCE_AGG:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/example/aggregator}
OMNIREDUCE_AGG_LD="$OMNIREDUCE_BUILD:/usr/lib/shifter/mpich-1.1/dep:/usr/lib64:/opt/cray/libfabric/1.22.0/lib64"

# OmniReduce always uses gloo backend
BACKEND=gloo
BLOCK_SIZE=256

# Defaults
MSG_SIZE_MIB=256
DENSITY=0.05
WARMUP_ITERS=5
MEASURE_ITERS=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --msg-size)  MSG_SIZE_MIB="$2";  shift 2 ;;
        --density)   DENSITY="$2";       shift 2 ;;
        --warmup)    WARMUP_ITERS="$2";  shift 2 ;;
        --measure)   MEASURE_ITERS="$2"; shift 2 ;;
        --help)      grep "^#" "$0" | head -22; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# Convert MiB to bytes (float count, float = 4 bytes)
case "$MSG_SIZE_MIB" in
    256)  TENSOR_SIZE=67108864   ;;   # 256 MiB / 4 bytes = 67108864 floats
    512)  TENSOR_SIZE=134217728  ;;
    1024) TENSOR_SIZE=268435456  ;;
    *) echo "ERROR: --msg-size must be 256, 512, or 1024"; exit 1 ;;
esac

# ── Validate omnireduce.cfg ────────────────────────────────────────────────────
if [[ ! -f "${SCRIPT_DIR}/omnireduce.cfg" ]]; then
    echo "ERROR: omnireduce.cfg not found in $SCRIPT_DIR"
    exit 1
fi

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
if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
    GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | grep -v "127.0.0.1" | awk '{print $2; exit}')
    [[ -z "$GLOO_SOCKET_IFNAME" ]] && GLOO_SOCKET_IFNAME=nmn0
    export GLOO_SOCKET_IFNAME
fi

# ── Distribute omnireduce.cfg to all nodes ────────────────────────────────────
echo "Distributing omnireduce.cfg to all nodes..."
for node in "${NODE_ARR[@]}"; do
    scp "${SCRIPT_DIR}/omnireduce.cfg" "${node}:${SCRIPT_DIR}/omnireduce.cfg" 2>/dev/null || true &
done
wait

# ── Result directory ───────────────────────────────────────────────────────────
RESULT_DIR="${SCRIPT_DIR}/results/omnireduce/node_${NUM_NODES}/msgsize_${MSG_SIZE_MIB}MiB/density_${DENSITY}"
mkdir -p "$RESULT_DIR"
CSV_FILE="${RESULT_DIR}/summary.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg" > "$CSV_FILE"
fi

# ── Print config ───────────────────────────────────────────────────────────────
echo "Minimal OmniReduce Benchmark"
echo "============================"
echo "  Nodes           : $NUM_NODES  (${NODE_ARR[*]})"
echo "  GPUs/node        : $GPUS_PER_NODE"
echo "  Total workers    : $TOTAL_WORKERS"
echo "  Coordinator IP   : $COORD_IP"
echo "  Backend          : $BACKEND (gloo via OmniReduce)"
echo "  Message size     : ${MSG_SIZE_MIB} MiB  ($TENSOR_SIZE floats)"
echo "  Density          : $DENSITY"
echo "  Warmup / measure : $WARMUP_ITERS / $MEASURE_ITERS"
echo "  GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "  Aggregator binary: $OMNIREDUCE_AGG"
echo "  Results dir      : $RESULT_DIR"
echo ""

# ── Helper: start aggregators on all nodes ────────────────────────────────────
start_aggregators() {
    echo "  Starting ${NUM_NODES} aggregators (one per node)..."
    for node in "${NODE_ARR[@]}"; do
        ssh "$node" "
            export LD_LIBRARY_PATH=${OMNIREDUCE_AGG_LD}:\$LD_LIBRARY_PATH
            export CUDA_VISIBLE_DEVICES=''
            pkill -9 aggregator 2>/dev/null || true
            $OMNIREDUCE_AGG
        " > "${RESULT_DIR}/aggregator_${node}.log" 2>&1 &
    done
    wait
    sleep 3   # give aggregators time to initialize
}

# ── Helper: stop aggregators on all nodes ─────────────────────────────────────
stop_aggregators() {
    echo "  Stopping aggregators..."
    for node in "${NODE_ARR[@]}"; do
        ssh "$node" "pkill -9 aggregator" 2>/dev/null || true &
    done
    wait
    sleep 1
}

# ── Run 3 times, pool timings, compute stats ──────────────────────────────────
for run_num in 1 2 3; do
    RUN_DIR="${RESULT_DIR}/run_${run_num}"
    mkdir -p "$RUN_DIR"
    echo "---- Run $run_num/3 ----"

    # Kill stale python processes
    for node in "${NODE_ARR[@]}"; do
        ssh "$node" "pkill -9 python" 2>/dev/null || true &
    done
    wait
    sleep 1

    start_aggregators

    # Launch one worker per GPU across all nodes
    global_rank=0
    for ((node_idx=0; node_idx<NUM_NODES; node_idx++)); do
        node="${NODE_ARR[$node_idx]}"
        for ((local_gpu=0; local_gpu<GPUS_PER_NODE; local_gpu++)); do
            echo "  worker rank=$global_rank  node=$node  gpu=$local_gpu"
            ssh "$node" "
                export CUDA_VISIBLE_DEVICES=$local_gpu
                export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
                export LD_LIBRARY_PATH=${OMNIREDUCE_BUILD}:\$LD_LIBRARY_PATH
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
    stop_aggregators
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
    echo "  Also check aggregator logs: $RESULT_DIR/aggregator_*.log"
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
