#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -f "$SCRIPT_DIR/nccl_test" ]] || { echo "ERROR: nccl_test not found. Run: bash build_nccl_test.sh"; exit 1; }
[[ -n "$SLURM_NODELIST" ]] || { echo "ERROR: Not in SLURM allocation"; exit 1; }

NUM_NODES=$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE%%(*}
TOTAL_WORKERS=$(( NUM_NODES * GPUS_PER_NODE ))

echo "NCCL Test: $NUM_NODES nodes × $GPUS_PER_NODE GPUs = $TOTAL_WORKERS tasks (TCP)"

export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO

srun --ntasks=$TOTAL_WORKERS --ntasks-per-node=$GPUS_PER_NODE --overlap "$SCRIPT_DIR/nccl_test"
echo "✓ Complete"
