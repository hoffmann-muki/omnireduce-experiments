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
NODE_COUNTS=()  # will be auto-detected from SLURM allocation
DENSITY="1.0"  # dense data (no sparsity)
BLOCK_SIZE=256
BACKEND="gloo"
WARMUP_ITERATIONS=10
MEASURE_ITERATIONS=100
MODE="RDMA"
SSH_PORT=${SSH_PORT:-22}

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
    local first_host=$(echo "$hosts" | head -n1)
    
    echo "Auto-configuring omnireduce.cfg from SLURM allocation:"
    echo "  Detected nodes: $hosts_csv"
    echo "  Aggregator: $first_host"
    
    # Update omnireduce.cfg in place (create if missing)
    if [[ ! -f omnireduce.cfg ]]; then
        echo "[omnireduce]" > omnireduce.cfg
    fi
    
    # Update worker and aggregator IPs, counts
    sed -i.bak 's/^worker_ips.*/worker_ips = '"$hosts_csv"'/' omnireduce.cfg
    sed -i.bak 's/^aggregator_ips.*/aggregator_ips = '"$first_host"'/' omnireduce.cfg
    sed -i.bak 's/^num_workers.*/num_workers = '"$num_nodes"'/' omnireduce.cfg
    sed -i.bak 's/^num_aggregators.*/num_aggregators = 1/' omnireduce.cfg
    
    # If keys didn't exist, append them
    grep -q "^worker_ips" omnireduce.cfg || echo "worker_ips = $hosts_csv" >> omnireduce.cfg
    grep -q "^aggregator_ips" omnireduce.cfg || echo "aggregator_ips = $first_host" >> omnireduce.cfg
    grep -q "^num_workers" omnireduce.cfg || echo "num_workers = $num_nodes" >> omnireduce.cfg
    grep -q "^num_aggregators" omnireduce.cfg || echo "num_aggregators = 1" >> omnireduce.cfg
    
    rm -f omnireduce.cfg.bak
    
    echo "Detected $num_nodes nodes in allocation"
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

# Deploy omnireduce.cfg to nodes
deploy_config() {
    local wnum=$1
    local anum=$2
    local worker_ips=($WORKER_IPS)
    local aggregator_ips=($AGGREGATOR_IPS)
    
    echo "Deploying omnireduce.cfg to aggregator and worker nodes..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        ssh -p $SSH_PORT "$agg_host" "mkdir -p /usr/local/omnireduce/example" 2>/dev/null || true
        scp -P $SSH_PORT omnireduce.cfg "$agg_host":/usr/local/omnireduce/example/ 2>/dev/null || true
    done
    for ((i=0; i<wnum; i++)); do
        local worker_host="${worker_ips[$i]}"
        ssh -p $SSH_PORT "$worker_host" "mkdir -p /home/exps/benchmark" 2>/dev/null || true
        scp -P $SSH_PORT omnireduce.cfg "$worker_host":/home/exps/benchmark/ 2>/dev/null || true
    done
}

# Start aggregators on specified count
start_aggregators() {
    local anum=$1
    local aggregator_ips=($AGGREGATOR_IPS)
    
    echo "Starting $anum aggregators..."
    for ((i=0; i<anum; i++)); do
        local agg_host="${aggregator_ips[$i]}"
        echo "  Starting aggregator on $agg_host"
        ssh -p $SSH_PORT "$agg_host" "pkill -9 aggregator; cd /usr/local/omnireduce/example; nohup ./aggregator > aggregator_${i}.log 2>&1 &" &
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
    start_aggregators "$MAX_AGGREGATORS"
    
    # Start workers
    for ((i=0; i<wnum; i++)); do
        local worker_host="${worker_ips[$i]}"
        echo "  Starting worker $i on $worker_host"
        ssh -p $SSH_PORT "$worker_host" \
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
    stop_aggregators "$MAX_AGGREGATORS"
    
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
        echo "WARNING: LD_LIBRARY_PATH not set. Make sure omnireduce/build is in LD_LIBRARY_PATH on worker nodes"
    fi
    
    # Deploy config files to nodes
    deploy_config "$MAX_WORKERS" "$MAX_AGGREGATORS"
    
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
