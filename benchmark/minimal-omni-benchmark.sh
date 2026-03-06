#!/bin/bash
#
# Minimal OmniReduce benchmark: allreduce via OmniReduce aggregators + gloo backend
# Based on omni-benchmark.sh by the original authors, adapted for Perlmutter/SLURM
#
# Usage:
#   ./minimal-omni-benchmark.sh [OPTIONS]
#
# Options:
#   --msg-size MiB      Message size: 256, 512, or 1024 (default: 256)
#   --density FLOAT     Tensor density 0.0-1.0 (default: 0.05)
#   --warmup ITERS      Warmup iterations (default: 5)
#   --measure ITERS     Measurement iterations (default: 20)
#   --backend BACKEND   Collective backend: gloo (sparse via OmniReduce) or nccl (all-GPU) (default: gloo)
#   --help              Show this help message
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
OMNIREDUCE_BUILD=${OMNIREDUCE_BUILD:-/home/hoffmuki/scratch/omnireduce/omnireduce-RDMA/omnireduce}
OMNIREDUCE_AGG=${OMNIREDUCE_AGG:-/home/hoffmuki/scratch/omnireduce/omnireduce-RDMA/example/aggregator}
# GCC lib must come first for correct libstdc++, then omnireduce and system libs
GCC_LIBDIR=$(dirname "$(gcc -print-file-name=libstdc++.so)")
OMNIREDUCE_AGG_LD="$GCC_LIBDIR:$OMNIREDUCE_BUILD:/lib64"

# Defaults
BACKEND=gloo
BLOCK_SIZE=256
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
        --backend)   BACKEND="$2";       shift 2 ;;
        --help)      grep "^#" "$0" | head -24; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate backend choice
if [[ "$BACKEND" != "gloo" && "$BACKEND" != "nccl" ]]; then
    echo "ERROR: --backend must be 'gloo' or 'nccl', got: $BACKEND"
    exit 1
fi

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

# GPUs per node
if [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
    GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"
else
    GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
fi

TOTAL_WORKERS=$(( NUM_NODES * GPUS_PER_NODE ))

# ── Collect interface IPs from all nodes for both backends ──────────
echo "Collecting IPs from all nodes on interface: ${GLOO_SOCKET_IFNAME:-auto-detected}..."
# Use the detected fabric interface; if none found, try ib0 (InfiniBand default)
FABRIC_IF=${GLOO_SOCKET_IFNAME:-ib0}
declare -a NODE_IPS
declare -a WORKER_IP_LIST
for node in "${NODE_ARR[@]}"; do
    # Get the first (primary) IP on the fabric interface via srun
    # Use --ntasks=1 to force single execution (--overlap alone uses all GPU slots)
    node_ip=$(srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "ip -o -4 addr show $FABRIC_IF 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1" 2>/dev/null | tr -d '\n')
    if [[ -z "$node_ip" ]]; then
        echo "ERROR: Could not get IP for node $node on interface $FABRIC_IF"
        echo "  Available interfaces on $node:"
        srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "ip -o -4 addr show | grep -v 127.0.0.1" 2>/dev/null | sed 's/^/    /'
        exit 1
    fi
    NODE_IPS+=("$node_ip")
    # Each node has GPUS_PER_NODE workers, all with the same IP
    for ((i=0; i<GPUS_PER_NODE; i++)); do
        WORKER_IP_LIST+=("$node_ip")
    done
    echo "  Node $node: $node_ip"
done

# Set coordinator IP to first node's IP as both gloo and nccl need this for initialization
COORD_IP="${NODE_IPS[0]}"

# Build comma-separated lists for omnireduce.cfg
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

echo "  Coordinator IP: $COORD_IP"

# Generate dynamic omnireduce.cfg if using gloo backend since nccl doesn't need aggregators
if [[ "$BACKEND" == "gloo" ]]; then
    echo "  Aggregator IPs: $AGGREGATOR_IPS"
    echo "  Worker IPs: $WORKER_IPS"
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
fi

# ── Result directory ───────────────────────────────────────────────────────────
RESULT_DIR="${SCRIPT_DIR}/results/${BACKEND}/node_${NUM_NODES}/msgsize_${MSG_SIZE_MIB}MiB/density_${DENSITY}"
mkdir -p "$RESULT_DIR"
CSV_FILE="${RESULT_DIR}/summary.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg" > "$CSV_FILE"
fi

# ── Print config ───────────────────────────────────────────────────────────────
echo "Collective Benchmark"
echo "===================="
echo "  Nodes           : $NUM_NODES  (${NODE_ARR[*]})"
echo "  GPUs/node        : $GPUS_PER_NODE"
echo "  Total workers    : $TOTAL_WORKERS"
echo "  Coordinator IP   : $COORD_IP"
echo "  Backend          : $BACKEND"
if [[ "$BACKEND" == "gloo" ]]; then
    echo "  Backend mode     : Sparse collective via OmniReduce aggregators"
else
    echo "  Backend mode     : All-GPU collective with no aggregators"
fi
echo "  Message size     : ${MSG_SIZE_MIB} MiB  ($TENSOR_SIZE floats)"
echo "  Density          : $DENSITY"
echo "  Warmup / measure : $WARMUP_ITERS / $MEASURE_ITERS"
echo "  GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
if [[ "$BACKEND" == "gloo" ]]; then
    echo "  Aggregator binary: $OMNIREDUCE_AGG"
fi
echo "  Results dir      : $RESULT_DIR"
echo ""

# ── Helper: start aggregators on all nodes ────────────────────────────────────
start_aggregators() {
    echo "  Starting ${NUM_NODES} aggregators (one per node)..."
    for node in "${NODE_ARR[@]}"; do
        # Run aggregator as a daemon via srun.
        # nohup + background so srun returns immediately, leaving aggregator alive.
        # Use --ntasks=1 to force single execution (without it, spawns on all GPU slots)
        srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "
            module unload boost 2>/dev/null || true
            module load boost/gcc/11.3.0
            export GCC_LIBDIR=\$(dirname \$(gcc -print-file-name=libstdc++.so))
            export LD_LIBRARY_PATH=\$GCC_LIBDIR:${OMNIREDUCE_BUILD}:/lib64:/usr/lib64:\$LD_LIBRARY_PATH
            export CUDA_VISIBLE_DEVICES=''
            pkill -9 aggregator 2>/dev/null || true
            cd $SCRIPT_DIR
            nohup stdbuf -oL $OMNIREDUCE_AGG >> ${RESULT_DIR}/aggregator_${node}.log 2>&1 &
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
        srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "pkill -9 aggregator" 2>/dev/null || true &
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
        srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "pkill -9 python" 2>/dev/null || true &
    done
    wait
    sleep 1

    if [[ "$BACKEND" == "gloo" ]]; then
        start_aggregators
    fi

    # Launch one worker per GPU across all nodes
    global_rank=0
    
    # Build the worker command (same for both backends)
    build_worker_cmd() {
        local rank=$1
        echo "
            module unload boost 2>/dev/null || true
            module load boost/gcc/11.3.0
            export CUDA_VISIBLE_DEVICES=$2
            export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
            [[ -n '$GLOO_SOCKET_IFNAME' ]] && export NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
            # For NCCL: allow extra time for initialization and handle errors gracefully
            export NCCL_INIT_TIMEOUT=300
            export NCCL_ASYNC_ERROR_HANDLING=1
            export PYTHONUNBUFFERED=1
            export GCC_LIBDIR=\$(dirname \$(gcc -print-file-name=libstdc++.so))
            export LD_LIBRARY_PATH=\$GCC_LIBDIR:${OMNIREDUCE_BUILD}:/lib64:/usr/lib64:\$LD_LIBRARY_PATH
            cd $SCRIPT_DIR
            $CONDA_PYTHON -u benchmark.py \
                --backend $BACKEND \
                --tensor-size $TENSOR_SIZE \
                --block-size $BLOCK_SIZE \
                --density $DENSITY \
                --rank $rank \
                --size $TOTAL_WORKERS \
                --ip $COORD_IP \
                --warmup-iters $WARMUP_ITERS \
                --measure-iters $MEASURE_ITERS \
                --sparsity-type elementwise
        "
    }
    
    for ((node_idx=0; node_idx<NUM_NODES; node_idx++)); do
        node="${NODE_ARR[$node_idx]}"
        for ((local_gpu=0; local_gpu<GPUS_PER_NODE; local_gpu++)); do
            echo "  worker rank=$global_rank  node=$node  gpu=$local_gpu"
            
            worker_cmd=$(build_worker_cmd $global_rank $local_gpu)
            
            # For NCCL: launch rank 0 first, wait for it to bind listener, then launch others
            # For gloo: launch all concurrently (OmniReduce aggregators handle coordination)
            if [[ "$BACKEND" == "nccl" && $global_rank -eq 0 ]]; then
                # Rank 0: launch in background, then sleep to let it bind the TCP listener
                echo "    [NCCL rank 0] Launching with head start for socket listener..."
                srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "$worker_cmd" \
                    > "${RUN_DIR}/worker_${global_rank}.log" 2>&1 &
                sleep 5  # Give rank 0 time to start Python and bind the TCP store
            else
                # All other ranks (or all ranks for gloo): launch concurrently in background
                srun --overlap --ntasks=1 --nodes=1 --nodelist="$node" bash -c "$worker_cmd" \
                    > "${RUN_DIR}/worker_${global_rank}.log" 2>&1 &
            fi
            
            global_rank=$(( global_rank + 1 ))
        done
    done

    echo "  Waiting for all workers..."
    wait
    if [[ "$BACKEND" == "gloo" ]]; then
        stop_aggregators
    fi
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
