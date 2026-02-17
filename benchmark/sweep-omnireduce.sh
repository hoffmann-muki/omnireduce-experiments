#!/bin/bash
#
# OmniReduce benchmark sweep: message sizes × node counts × dense data
# Runs allreduce benchmarks with OmniReduce across multiple topologies
#
# Usage:
#   ./sweep-omnireduce.sh [--msg-size MiB]
#
# Options:
#   --msg-size MiB    Benchmark only specified message size (256, 512, or 1024 MiB)
#   --warmup ITERS    Number of warmup iterations (default 10)
#   --measure ITERS   Number of measurement iterations (default 100)
#   --help            Show this help message
#
# Prerequisites:
#   - Running within a SLURM allocation (salloc or sbatch)
#   - LD_LIBRARY_PATH includes omnireduce/build directory
#   - CUDA_VISIBLE_DEVICES set (optional; defaults to 0)
#

set -e

# Parse command-line arguments
MSG_SIZE_MIB=""
WARMUP_ITERS=10
MEASURE_ITERS=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --msg-size)
            MSG_SIZE_MIB="$2"
            shift 2
            ;;
        --warmup)
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --measure)
            MEASURE_ITERS="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | head -25
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# Configuration
if [[ -n "$MSG_SIZE_MIB" ]]; then
    # Validate and convert specified message size
    case "$MSG_SIZE_MIB" in
        256) MESSAGE_SIZES=(268435456) ;;
        512) MESSAGE_SIZES=(536870912) ;;
        1024) MESSAGE_SIZES=(1073741824) ;;
        *)
            echo "ERROR: Invalid message size. Choose from: 256, 512, or 1024 MiB"
            exit 1
            ;;
    esac
else
    # Default: all message sizes
    MESSAGE_SIZES=(268435456 536870912 1073741824)  # 256 MiB, 512 MiB, 1024 MiB (in bytes)
fi
NODE_COUNTS=()  # will be auto-detected from SLURM allocation
DENSITY="1.0"  # dense data (no sparsity)
BLOCK_SIZE=256
BACKEND="gloo"

# Path configuration (adjust if your install is in a different location)
OMNIREDUCE_BUILD=${OMNIREDUCE_BUILD:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/omnireduce/build}
OMNIREDUCE_AGG=${OMNIREDUCE_AGG:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/example/aggregator}
BENCHMARK_SCRIPT=${BENCHMARK_SCRIPT:-$(pwd)/benchmark.py}

# Calculate aggregator count using log2 heuristic: floor(log2(num_nodes)), min 1
calc_aggregators() {
    local num_nodes=$1
    
    if [[ $num_nodes -le 1 ]]; then
        echo 1
        return
    fi
    
    local aggs=1
    local power=2  # 2^1
    
    while [[ $((power * 2)) -le $num_nodes ]]; do
        power=$((power * 2))
        ((aggs++))
    done
    
    echo $aggs
}

# Auto-detect node count and populate omnireduce.cfg from SLURM allocation
detect_node_count() {
    if [[ -z "$SLURM_NODELIST" ]]; then
        echo "ERROR: SLURM_NODELIST not set. Please run within a SLURM allocation (salloc/sbatch)"
        exit 1
    fi
    
    # Get hostnames from allocation
    local hosts=$(scontrol show hostnames "$SLURM_NODELIST")
    local num_nodes=$(echo "$hosts" | wc -l)
    if [[ $num_nodes -lt 1 ]]; then
        echo "ERROR: Could not determine node count from SLURM_NODELIST=$SLURM_NODELIST"
        exit 1
    fi
    
    # Auto-update omnireduce.cfg with SLURM allocation info
    local hosts_csv=$(echo "$hosts" | paste -sd, -)

    # Build an array of hosts (readarray reads all lines correctly)
    readarray -t hosts_arr <<< "$hosts"

    # Decide number of aggregators using the log2 heuristic
    local num_aggs=$(calc_aggregators "$num_nodes")

    # Choose first `num_aggs` hosts as aggregators (can be changed to a different placement heuristic)
    local aggregator_arr=()
    for ((i=0; i<num_aggs; i++)); do
        aggregator_arr+=("${hosts_arr[$i]}")
    done
    local aggregator_csv=$(IFS=,; echo "${aggregator_arr[*]}")

    echo "Auto-configuring omnireduce.cfg from SLURM allocation:"
    echo "  Detected nodes: $hosts_csv"
    echo "  Aggregators: $aggregator_csv"

    # Update omnireduce.cfg in place (create if missing)
    if [[ ! -f omnireduce.cfg ]]; then
        echo "[omnireduce]" > omnireduce.cfg
    fi

    # Update worker and aggregator IPs, counts
    sed -i.bak 's/^worker_ips.*/worker_ips = '"$hosts_csv"'/' omnireduce.cfg
    sed -i.bak 's/^aggregator_ips.*/aggregator_ips = '"$aggregator_csv"'/' omnireduce.cfg
    sed -i.bak 's/^num_workers.*/num_workers = '"$num_nodes"'/' omnireduce.cfg
    sed -i.bak 's/^num_aggregators.*/num_aggregators = '"$num_aggs"'/' omnireduce.cfg

    # If keys didn't exist, append them
    grep -q "^worker_ips" omnireduce.cfg || echo "worker_ips = $hosts_csv" >> omnireduce.cfg
    grep -q "^aggregator_ips" omnireduce.cfg || echo "aggregator_ips = $aggregator_csv" >> omnireduce.cfg
    grep -q "^num_workers" omnireduce.cfg || echo "num_workers = $num_nodes" >> omnireduce.cfg
    grep -q "^num_aggregators" omnireduce.cfg || echo "num_aggregators = $num_aggs" >> omnireduce.cfg
    
    rm -f omnireduce.cfg.bak
    
    echo "Detected $num_nodes nodes in allocation (will use $num_aggs aggregators)"
    NODE_COUNTS=($num_nodes)
}

# Parse omnireduce.cfg
read_config() {
    local wnum=0 anum=0
    local worker_arr=() aggregator_arr=()
    
    while read line; do
        if [[ $line =~ "num_workers" ]]; then
            wnum=$((${line: 14}))
        fi
        if [[ $line =~ "num_aggregators" ]]; then
            anum=$((${line: 18}))
        fi
        if [[ $line =~ "direct_memory" ]]; then
            local mode_num=$((${line: 16}))
            if [[ "$mode_num" -eq "1" ]]; then
                MODE="GDR"
            fi
        fi
        if [[ $line =~ "worker_ips" ]]; then
            line=${line: 13}
            worker_arr=(${line//,/ })
        fi
        if [[ $line =~ "aggregator_ips" ]]; then
            line=${line: 17}
            aggregator_arr=(${line//,/ })
        fi
    done < omnireduce.cfg
    
    echo "Loaded from omnireduce.cfg:"
    echo "  Workers: $wnum, Aggregators: $anum, Mode: $MODE"
    
    # Return as associative array (bash 4+)
    # For now, export for use in functions
    export MAX_WORKERS=$wnum
    export MAX_AGGREGATORS=$anum
    export WORKER_IPS="${worker_arr[*]}"
    export AGGREGATOR_IPS="${aggregator_arr[*]}"
}

# Start aggregators on specified count
start_aggregators() {
    local anum=$1
    local aggregator_ips=($AGGREGATOR_IPS)
    
    # ensure we don't iterate past the provided aggregator_ips
    local provided=${#aggregator_ips[@]}
    if [[ $anum -gt $provided ]]; then
        echo "Warning: requested $anum aggregators but only $provided aggregator_ips provided; limiting to $provided"
        anum=$provided
    fi

    echo "Starting $anum aggregators using srun..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        echo "  Starting aggregator on $agg_host"
        srun --nodes=1 -w "$agg_host" --exclusive \
            bash -c "pkill -9 aggregator; $OMNIREDUCE_AGG" > aggregator_${i}.log 2>&1 &
    done
    wait
    sleep 2  # give aggregators time to initialize
}

# Stop aggregators
stop_aggregators() {
    local anum=$1
    local aggregator_ips=($AGGREGATOR_IPS)
    
    # ensure we don't iterate past the provided aggregator_ips
    local provided=${#aggregator_ips[@]}
    if [[ $anum -gt $provided ]]; then
        anum=$provided
    fi
    
    echo "Stopping $anum aggregators..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        srun --nodes=1 -w "$agg_host" --exclusive bash -c "pkill -9 aggregator" &
    done
    wait
    sleep 1
}

# Run benchmark for given node count and message size
run_benchmark() {
    local wnum=$1
    local msg_size=$2
    local result_dir=$3
    
    local worker_ips=($WORKER_IPS)
    
    # Create result directory
    mkdir -p "$result_dir"
    
    echo "Running benchmark: nodes=$wnum, msg_size=$((msg_size/1024/1024))MiB, density=$DENSITY"
    
    # Start aggregators
    start_aggregators "$MAX_AGGREGATORS"
    
    # Clean up any stale python processes that might be holding ports
    echo "  Cleaning up stale python processes..."
    local aggregator_ips=($AGGREGATOR_IPS)
    for ((i=0; i<MAX_AGGREGATORS; i++)); do
        srun --nodes=1 -w "${aggregator_ips[$i]}" --exclusive bash -c "pkill -9 python" 2>/dev/null || true
    done
    for ((i=0; i<wnum; i++)); do
        srun --nodes=1 -w "${worker_ips[$i]}" --exclusive bash -c "pkill -9 python" 2>/dev/null || true
    done
    sleep 1
    
    # Start workers using srun (no SSH needed; runs within SLURM allocation)
    # Use first aggregator's IP as distributed rendezvous coordinator (stable, dedicated node)
    local coord_ip="${aggregator_ips[0]}"
    for ((i=0; i<wnum; i++)); do
        local worker_host="${worker_ips[$i]}"
        echo "  Starting worker $i on $worker_host (coordinator: $coord_ip)"
        
        srun --nodes=1 --ntasks=1 --exclusive -w "$worker_host" \
            bash -c "cd $(dirname $BENCHMARK_SCRIPT) && \
                     export LD_LIBRARY_PATH=${OMNIREDUCE_BUILD}:\$LD_LIBRARY_PATH && \
                     export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} && \
                     export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} && \
                     python $(basename $BENCHMARK_SCRIPT) \
                       --backend $BACKEND \
                       --tensor-size $msg_size \
                       --block-size $BLOCK_SIZE \
                       --density $DENSITY \
                       --rank $i \
                       --size $wnum \
                       --ip $coord_ip \
                       --warmup-iters $WARMUP_ITERS \
                       --measure-iters $MEASURE_ITERS" \
            > "$result_dir/worker_${i}.log" 2>&1 &
    done
    
    # Wait for workers to complete
    echo "  Waiting for workers to complete..."
    wait
    
    # Stop aggregators
    stop_aggregators "$MAX_AGGREGATORS"
    
    # Aggregate results from local log files (shared pscratch filesystem)
    echo "  Collecting results..."
    local result_file="$result_dir/summary.txt"
    echo "Nodes: $wnum, MessageSize: $((msg_size/1024/1024))MiB, Density: $DENSITY, BlockSize: $BLOCK_SIZE" > "$result_file"
    for ((i=0; i<wnum; i++)); do
        # Extract last few lines from worker log
        if [[ -f "$result_dir/worker_${i}.log" ]]; then
            echo "--- Worker $i ---" >> "$result_file"
            tail -10 "$result_dir/worker_${i}.log" >> "$result_file" 2>/dev/null || true
        fi
    done
    
    echo "  ✓ Benchmark completed"
}

# Main sweep loop
main() {
    echo "OmniReduce Benchmark Sweep"
    echo "============================"
    echo "Message sizes: ${MESSAGE_SIZES[@]} bytes"
    echo ""
    
    # Detect available nodes from SLURM
    detect_node_count
    echo "Node counts to test: ${NODE_COUNTS[@]}"
    echo ""
    
    # Read config
    if [[ ! -f omnireduce.cfg ]]; then
        echo "ERROR: omnireduce.cfg not found in current directory"
        exit 1
    fi
    read_config
    
    # Check environment variables
    if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "WARNING: CUDA_VISIBLE_DEVICES not set, defaulting to 0"
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    # Auto-detect network interface
    if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
        GLOO_SOCKET_IFNAME=$(ip -o -4 addr show | grep -v "127.0.0.1" | awk '{print $2; exit}')
        if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
            for iface in eth0 eno1 en0 hsn0 wlan0; do
                if ip addr show "$iface" &>/dev/null; then
                    GLOO_SOCKET_IFNAME=$iface
                    break
                fi
            done
        fi
        if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
            GLOO_SOCKET_IFNAME=lo
        fi
        export GLOO_SOCKET_IFNAME
        echo "Auto-detected GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    fi
    
    # Warn about LD_LIBRARY_PATH
    if [[ -z "$LD_LIBRARY_PATH" ]]; then
        echo "WARNING: LD_LIBRARY_PATH not set. Make sure omnireduce/build is in LD_LIBRARY_PATH"
    fi
    
    # Create root results directory
    local results_root="./100G-results/sweep-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$results_root"
    
    echo "Results will be saved to: $results_root"
    echo ""
    
    # Sweep over node counts and message sizes
    for node_count in "${NODE_COUNTS[@]}"; do
        # Skip node counts larger than available workers
        if [[ $node_count -gt $MAX_WORKERS ]]; then
            echo "Skipping node_count=$node_count (exceeds max_workers=$MAX_WORKERS)"
            continue
        fi
        
        for msg_size in "${MESSAGE_SIZES[@]}"; do
            local result_dir="$results_root/nodes_${node_count}_msgsize_$((msg_size/1024/1024))MiB"
            run_benchmark "$node_count" "$msg_size" "$result_dir"
        done
    done
    
    echo ""
    echo "============================"
    echo "Sweep completed!"
    echo "Results saved to: $results_root"
}

main "$@"
