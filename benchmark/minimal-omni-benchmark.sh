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
OMNIREDUCE_BUILD=${OMNIREDUCE_BUILD:-/home/hoffmuki/scratch/omnireduce/omnireduce-RDMA/omnireduce/build}
OMNIREDUCE_AGG=${OMNIREDUCE_AGG:-/home/hoffmuki/scratch/omnireduce/omnireduce-RDMA/example/aggregator}
# GCC lib must come first for correct libstdc++, then omnireduce and system libs
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so)")
OMNIREDUCE_AGG_LD="$GCC_LIBDIR:$OMNIREDUCE_BUILD:/lib64"

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
    # Prefer high-speed fabric (ib*, cxi*, mlx*) over management networks
    GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | grep -v "127.0.0.1" | awk '{print $2}' | grep -E "^(ib|cxi|mlx)" | head -1)
    # Fall back to any non-loopback interface if no fabric interface found
    [[ -z "$GLOO_SOCKET_IFNAME" ]] && GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | grep -v "127.0.0.1" | awk '{print $2; exit}')
    export GLOO_SOCKET_IFNAME
fi

# ── Collect InfiniBand IPs from all nodes and generate dynamic omnireduce.cfg ──
echo "Collecting InfiniBand IPs from all nodes..."
# Use the detected fabric interface; if none found, try ib0 (InfiniBand default)
FABRIC_IF=${GLOO_SOCKET_IFNAME:-ib0}
declare -a NODE_IPS
declare -a WORKER_IP_LIST
for node in "${NODE_ARR[@]}"; do
    # Get the first (primary) IP on the fabric interface
    node_ip=$(srun --overlap --nodes=1 --nodelist="$node" bash -c "ip -o -4 addr show $FABRIC_IF 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1" 2>/dev/null)
    if [[ -z "$node_ip" ]]; then
        echo "ERROR: Could not get IP for node $node on interface $FABRIC_IF"
        echo "  Available interfaces on $node:"
        srun --overlap --nodes=1 --nodelist="$node" bash -c "ip -o -4 addr show | grep -v 127.0.0.1" 2>/dev/null | sed 's/^/    /'
        exit 1
    fi
    NODE_IPS+=("$node_ip")
    # Each node has GPUS_PER_NODE workers, all with the same IP
    for ((i=0; i<GPUS_PER_NODE; i++)); do
        WORKER_IP_LIST+=("$node_ip")
    done
    echo "  Node $node: $node_ip"
done

# Build comma-separated lists
# Use printf to join arrays more robustly than IFS trick
AGGREGATOR_IPS=""
for ip in "${NODE_IPS[@]}"; do
    AGGREGATOR_IPS="${AGGREGATOR_IPS}${ip},"
done
AGGREGATOR_IPS="${AGGREGATOR_IPS%,}"  # Remove trailing comma

WORKER_IPS=""
for ip in "${WORKER_IP_LIST[@]}"; do
    WORKER_IPS="${WORKER_IPS}${ip},"
done
WORKER_IPS="${WORKER_IPS%,}"  # Remove trailing comma

echo "  Aggregator IPs: $AGGREGATOR_IPS"
echo "  Worker IPs: $WORKER_IPS"

# Generate dynamic omnireduce.cfg based on SLURM allocation
OMNIREDUCE_CFG="${SCRIPT_DIR}/omnireduce.cfg"
cat > "$OMNIREDUCE_CFG" <<EOF
[omnireduce]
num_workers = $TOTAL_WORKERS
num_aggregators = $NUM_NODES
num_threads = 8
worker_cores = -1,-1,-1,-1,-1,-1,-1,-1
aggregator_cores = -1,-1,-1,-1,-1,-1,-1,-1
threshold = 0.0
buffer_size = 1024
chunk_size = 1048576
bitmap_chunk_size = 16777216
message_size = 256
block_size = 256
ib_hca = mlx5_0
ib_port = 1
gid_idx = 2
sl = 2
gpu_devId = 0
direct_memory = 1
adaptive_blocksize = 0
tcp_port = 19875
worker_ips = $WORKER_IPS
aggregator_ips = $AGGREGATOR_IPS
EOF

echo "  Generated omnireduce.cfg"

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
        # Run aggregator as a daemon via srun.
        # nohup + background so srun returns immediately, leaving aggregator alive.
        srun --overlap --nodes=1 --nodelist="$node" bash -c "
            export LD_LIBRARY_PATH=${OMNIREDUCE_AGG_LD}:\$LD_LIBRARY_PATH
            export CUDA_VISIBLE_DEVICES=''
            pkill -9 aggregator 2>/dev/null || true
            cd $SCRIPT_DIR
            nohup $OMNIREDUCE_AGG >> ${RESULT_DIR}/aggregator_${node}.log 2>&1 &
            echo \"aggregator PID: \$!\"
        " &
    done
    wait   # Wait for all srun aggregator jobs to launch
    sleep 2   # give aggregators time to bind ports and initialize
}

# ── Helper: stop aggregators on all nodes ─────────────────────────────────────
stop_aggregators() {
    echo "  Stopping aggregators..."
    for node in "${NODE_ARR[@]}"; do
        srun --overlap --nodes=1 --nodelist="$node" bash -c "pkill -9 aggregator" 2>/dev/null || true &
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
        srun --overlap --nodes=1 --nodelist="$node" bash -c "pkill -9 python" 2>/dev/null || true &
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
            srun --overlap --nodes=1 --nodelist="$node" bash -c "
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
