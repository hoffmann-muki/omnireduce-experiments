#!/bin/bash
#
# OmniReduce benchmark sweep: message sizes × node counts × dense data
# Runs allreduce benchmarks with OmniReduce across multiple topologies
#
# Usage:
#   ./sweep-omnireduce.sh [--msg-size MiB] [--density FLOAT] [--backend NAME]
#
# Options:
#   --msg-size MiB    Benchmark only specified message size (256, 512, or 1024 MiB)
#   --density FLOAT   Data density 0.0-1.0 (default 1.0)
#   --backend NAME    PyTorch backend: nccl, gloo, mpi (default gloo)
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

# Get script directory for locate extract_stats.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command-line arguments
MSG_SIZE_MIB=""
DENSITY="1.0"
BACKEND="nccl"
WARMUP_ITERS=10
MEASURE_ITERS=100
while [[ $# -gt 0 ]]; do
    case $1 in
        --msg-size)
            MSG_SIZE_MIB="$2"
            shift 2
            ;;
        --density)
            DENSITY="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
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
            grep "^#" "$0" | head -29
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
BLOCK_SIZE=256

# Path configuration (adjust if your install is in a different location)
OMNIREDUCE_BUILD=${OMNIREDUCE_BUILD:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/omnireduce/build}
OMNIREDUCE_AGG=${OMNIREDUCE_AGG:-/pscratch/sd/h/hmuki/omnireduce/omnireduce-RDMA/example/aggregator}
BENCHMARK_SCRIPT=${BENCHMARK_SCRIPT:-$(pwd)/benchmark.py}

# Detect GPUs per node from SLURM allocation
detect_gpus_per_node() {
    # Try SLURM_GPUS_PER_NODE first (set if --gpus-per-node was used in sbatch/salloc)
    if [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
        # SLURM_GPUS_PER_NODE can be "4" or "4(IDX:0-1)" etc; extract the number
        echo "${SLURM_GPUS_PER_NODE%%(*}"
    else
        # Fallback: try to count available GPUs on the current node
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --list-gpus 2>/dev/null | wc -l
        else
            # Default to 1 if no GPU detection available
            echo 1
        fi
    fi
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

    # One aggregator per node
    local num_aggs=$num_nodes

    # Choose first `num_aggs` hosts as aggregators (will be all nodes in this case)
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
    
    echo "Detected $num_nodes nodes in allocation (will use $num_nodes aggregators - one per node)"
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
        srun --nodes=1 -w "$agg_host" --exclusive --gpus=0 bash -c "pkill -9 aggregator" &
    done
    wait
    sleep 1
}

# Run benchmark for given configuration
# Args: num_nodes, gpus_per_node, msg_size, result_dir, worker_ips_str, aggregator_ips_str, coord_ip
run_benchmark() {
    local num_nodes=$1
    local gpus_per_node=$2
    local msg_size=$3
    local result_dir=$4
    local worker_ips_str=$5
    local aggregator_ips_str=$6
    local coord_ip=$7
    
    # Convert space-separated strings to arrays
    read -ra worker_ips <<< "$worker_ips_str"
    read -ra agg_ips <<< "$aggregator_ips_str"
    
    # Create result directory
    mkdir -p "$result_dir"
    
    local total_workers=$((num_nodes * gpus_per_node))
    echo "Running benchmark: nodes=$num_nodes, gpus_per_node=$gpus_per_node, total_workers=$total_workers, msg_size=$((msg_size/1024/1024))MiB, density=$DENSITY"
    
    # Start aggregators
    local num_aggs=${#agg_ips[@]}
    echo "Starting $num_aggs aggregators using srun..."
    for ((i=0; i<num_aggs; i++)); do
        local agg_host="${agg_ips[$i]}"
        echo "  Starting aggregator on $agg_host"
        srun --nodes=1 -w "$agg_host" --exclusive --gpus=0 \
            bash -c "pkill -9 aggregator; $OMNIREDUCE_AGG" > aggregator_${i}.log 2>&1 &
    done
    wait
    sleep 2
    
    # Clean up any stale python processes
    echo "  Cleaning up stale python processes..."
    for ((i=0; i<num_aggs; i++)); do
        srun --nodes=1 -w "${agg_ips[$i]}" --exclusive --gpus=0 bash -c "pkill -9 python" 2>/dev/null || true
    done
    for ((i=0; i<num_nodes; i++)); do
        srun --nodes=1 -w "${worker_ips[$i]}" --exclusive --gpus=0 bash -c "pkill -9 python" 2>/dev/null || true
    done
    sleep 1
    
    # Launch workers: one srun per GPU across all nodes
    local global_rank=0
    for ((node_idx=0; node_idx<num_nodes; node_idx++)); do
        local worker_host="${worker_ips[$node_idx]}"
        for ((local_rank=0; local_rank<gpus_per_node; local_rank++)); do
            echo "  Starting worker rank=$global_rank (node=$node_idx, local_gpu=$local_rank) on $worker_host"
            
            srun --nodes=1 --ntasks=1 --exclusive -w "$worker_host" \
                bash -c "cd $(dirname $BENCHMARK_SCRIPT) && \
                         export LD_LIBRARY_PATH=${OMNIREDUCE_BUILD}:\$LD_LIBRARY_PATH && \
                         export CUDA_VISIBLE_DEVICES=$local_rank && \
                         export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} && \
                         python $(basename $BENCHMARK_SCRIPT) \
                           --backend $BACKEND \
                           --tensor-size $msg_size \
                           --block-size $BLOCK_SIZE \
                           --density $DENSITY \
                           --rank $global_rank \
                           --size $total_workers \
                           --ip $coord_ip \
                           --warmup-iters $WARMUP_ITERS \
                           --measure-iters $MEASURE_ITERS" \
                > "$result_dir/worker_${global_rank}.log" 2>&1 &
            
            global_rank=$((global_rank + 1))
        done
    done
    
    # Wait for all workers to complete
    echo "  Waiting for workers to complete..."
    wait
    
    # Stop aggregators
    echo "Stopping $num_aggs aggregators..."
    for ((i=0; i<num_aggs; i++)); do
        srun --nodes=1 -w "${agg_ips[$i]}" --exclusive bash -c "pkill -9 aggregator" &
    done
    wait
    sleep 1
    
    # Aggregate results
    echo "  Collecting results..."
    local result_file="$result_dir/summary.txt"
    echo "Nodes: $num_nodes, GPUs/Node: $gpus_per_node, TotalWorkers: $total_workers, MessageSize: $((msg_size/1024/1024))MiB, Density: $DENSITY, BlockSize: $BLOCK_SIZE" > "$result_file"
    for ((i=0; i<total_workers; i++)); do
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
    
    # Detect GPUs per node
    local gpus_per_node=$(detect_gpus_per_node)
    echo "Detected $gpus_per_node GPUs per node"
    
    # Create root results directory
    local results_root="./results/${BACKEND}"
    mkdir -p "$results_root"
    
    echo "Results will be saved to: $results_root"
    echo ""
    
    # Sweep over node counts and message sizes, running each config 3 times
    for node_count in "${NODE_COUNTS[@]}"; do
        # Skip node counts larger than available workers
        if [[ $node_count -gt $MAX_WORKERS ]]; then
            echo "Skipping node_count=$node_count (exceeds max_workers=$MAX_WORKERS)"
            continue
        fi
        
        for msg_size in "${MESSAGE_SIZES[@]}"; do
            local msgsize_mib=$((msg_size/1024/1024))
            local base_result_dir="$results_root/node_${node_count}/msgsize_${msgsize_mib}MiB/density_${DENSITY}"
            local csv_file="$base_result_dir/summary.csv"
            
            # Ensure base directory exists
            mkdir -p "$base_result_dir"
            
            # Initialize CSV header if file doesn't exist
            # Format now tracks both metrics (time_only and time_with_barrier)
            if [[ ! -f "$csv_file" ]]; then
                echo "node_count,msgsize,time_only_min,time_only_max,time_only_avg,time_with_barrier_min,time_with_barrier_max,time_with_barrier_avg" > "$csv_file"
            fi
            
            # Run benchmark 3 times and collect stats
            for run_num in 1 2 3; do
                local run_result_dir="${base_result_dir}/run_${run_num}"
                echo "Running node_count=$node_count, msgsize=${msgsize_mib}MiB, gpus_per_node=$gpus_per_node, run=$run_num/3"
                
                # Get coordinate IP (first aggregator)
                local aggregator_ips=($AGGREGATOR_IPS)
                local coord_ip="${aggregator_ips[0]}"
                
                # Call run_benchmark with all parameters
                run_benchmark "$node_count" "$gpus_per_node" "$msg_size" "$run_result_dir" \
                    "$WORKER_IPS" "$AGGREGATOR_IPS" "$coord_ip"
            done
            
            # Pool all raw timings across all 3 runs for both metrics
            local all_time_only=()
            local all_time_with_barrier=()
            
            for run_num in 1 2 3; do
                local run_result_dir="${base_result_dir}/run_${run_num}"
                for logfile in "$run_result_dir"/worker_*.log; do
                    if [[ -f "$logfile" ]]; then
                        while IFS= read -r line; do
                            # Parse: time_only:VALUE;time_with_barrier:VALUE;
                            if [[ $line =~ time_only:([0-9.e+\-]+)\;time_with_barrier:([0-9.e+\-]+)\; ]]; then
                                all_time_only+=("${BASH_REMATCH[1]}")
                                all_time_with_barrier+=("${BASH_REMATCH[2]}")
                            fi
                        done < "$logfile"
                    fi
                done
            done
            
            # Compute overall stats for both metrics using awk
            if [[ ${#all_time_only[@]} -gt 0 ]]; then
                local time_only_stats=$(printf "%s\n" "${all_time_only[@]}" | awk '{
                    if (NR==1 || $1<min) min=$1
                    if (NR==1 || $1>max) max=$1
                    sum+=$1
                    count++
                } END {
                    if (count > 0) {
                        avg=sum/count
                        printf "%.1f,%.1f,%.1f", min, max, avg
                    }
                }')
                
                local barrier_stats=$(printf "%s\n" "${all_time_with_barrier[@]}" | awk '{
                    if (NR==1 || $1<min) min=$1
                    if (NR==1 || $1>max) max=$1
                    sum+=$1
                    count++
                } END {
                    if (count > 0) {
                        avg=sum/count
                        printf "%.1f,%.1f,%.1f", min, max, avg
                    }
                }')
                
                # Write CSV row with node_count, msgsize, time_only stats, time_with_barrier stats
                echo "$node_count,$msgsize_mib,$time_only_stats,$barrier_stats" >> "$csv_file"
                echo "  ✓ Results appended to: $csv_file"
            fi
        done
    done
    
    echo ""
    echo "============================"
    echo "Sweep completed!"
    echo "Results saved to: $results_root"
}

main "$@"
