#!/bin/bash
#
# OmniReduce benchmark sweep: message sizes × node counts × dense data
# Runs allreduce benchmarks with OmniReduce across multiple topologies
#
# Usage:
#   ./sweep-omnireduce.sh
#
# Prerequisites:
#   - omnireduce.cfg in current directory (configure worker/aggregator IPs first)
#   - CUDA_VISIBLE_DEVICES and GLOO_SOCKET_IFNAME env vars set
#   - aggregator and worker binaries built and accessible via SSH
#

set -e

# Configuration
MESSAGE_SIZES=(268435456 536870912 1073741824)  # 256 MiB, 512 MiB, 1024 MiB (in bytes)
NODE_COUNTS=(2 4 8 16 32)
DENSITY="1.0"  # dense data (no sparsity)
BLOCK_SIZE=256
BACKEND="gloo"
WARMUP_ITERATIONS=10
MEASURE_ITERATIONS=100
MODE="RDMA"

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
    
    echo "Starting $anum aggregators..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        echo "  Starting aggregator on $agg_host"
        ssh -p 2222 "$agg_host" "pkill -9 aggregator; cd /usr/local/omnireduce/example; nohup ./aggregator > aggregator_${i}.log 2>&1 &" &
    done
    wait
    sleep 2  # give aggregators time to initialize
}

# Stop aggregators
stop_aggregators() {
    local anum=$1
    local aggregator_ips=($AGGREGATOR_IPS)
    
    echo "Stopping $anum aggregators..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        ssh -p 2222 "$agg_host" "pkill -9 aggregator" &
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
    start_aggregators "$wnum"
    
    # Start workers
    for ((i=0; i<wnum; i++)); do
        local worker_host="${worker_ips[$i]}"
        echo "  Starting worker $i on $worker_host"
        ssh -p 2222 "$worker_host" \
            "cd /home/exps/benchmark && \
             export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} && \
             export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} && \
             /usr/local/conda/bin/python benchmark.py \
               --backend $BACKEND \
               --tensor-size $msg_size \
               --block-size $BLOCK_SIZE \
               --density $DENSITY \
               --rank $i \
               --size $wnum \
               --ip ${worker_ips[0]} \
               > $result_dir/worker_${i}.log 2>&1" &
    done
    
    # Wait for workers to complete
    echo "  Waiting for workers to complete..."
    wait
    
    # Stop aggregators
    stop_aggregators "$wnum"
    
    # Aggregate results (simple: grab latencies from logs)
    echo "  Collecting results..."
    local result_file="$result_dir/summary.txt"
    if [[ -f "${worker_ips[0]}:$result_dir/worker_0.log" ]]; then
        # Collect timing data from worker logs
        echo "Nodes: $wnum, MessageSize: $((msg_size/1024/1024))MiB, Density: $DENSITY" > "$result_file"
        for ((i=0; i<wnum; i++)); do
            local worker_host="${worker_ips[$i]}"
            scp -P 2222 "$worker_host:$result_dir/worker_${i}.log" "$result_dir/worker_${i}.log" 2>/dev/null || true
            # Extract timing from log (example: grep "time:" and average)
            grep "time:" "$result_dir/worker_${i}.log" | awk '{sum+=$NF} END {if(NR>0) print "Worker '$i' avg: " sum/NR " us"}' >> "$result_file" || true
        done
    fi
    
    echo "  ✓ Benchmark completed"
}

# Main sweep loop
main() {
    echo "OmniReduce Benchmark Sweep"
    echo "============================"
    echo "Message sizes: ${MESSAGE_SIZES[@]} bytes"
    echo "Node counts: ${NODE_COUNTS[@]}"
    echo "Density: $DENSITY"
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
    if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
        echo "WARNING: GLOO_SOCKET_IFNAME not set, defaulting to eth0"
        export GLOO_SOCKET_IFNAME=eth0
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
