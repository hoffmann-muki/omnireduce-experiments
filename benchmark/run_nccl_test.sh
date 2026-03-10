#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load gcc/11.3.0 cuda/12.3.0/gcc/11.3.0/icelake nccl/2.18.1-1/gcc/11.3.0/icelake openmpi/4.1.5 2>/dev/null || true

[[ -f "$SCRIPT_DIR/nccl_test" ]] || { echo "ERROR: nccl_test not found. Run: bash build_nccl_test.sh"; exit 1; }
[[ -n "$SLURM_NODELIST" ]] || { echo "ERROR: Not in SLURM allocation. Try: salloc --nodes=2 --gpus-per-node=4 --time=00:30:00 bash"; exit 1; }

NUM_NODES=$(scontrol show hostnames "$SLURM_NODELIST" | wc -l)
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}  # Default to 4 if not set
TOTAL_WORKERS=$(( NUM_NODES * GPUS_PER_NODE ))

echo "NCCL Test: $NUM_NODES nodes × $GPUS_PER_NODE GPUs = $TOTAL_WORKERS tasks (RoCe v1 RDMA tuning)"

# Zaratan has RoCe v1 only with GID Index 0; no RoCe v2 support
# NCCL_IB_ROCE_VERSION_NUM=1 tells NCCL to use RoCe v1 explicitly
# NCCL_IB_GID_INDEX=0 selects the only available GID
# NCCL_IB_AR_DISABLE=1 disables Adaptive Routing (unsupported on RoCe v1)
# NCCL_NET_GDR_LEVEL=3 maximizes GPU Direct RDMA throughput
export NCCL_IB_DISABLE=0 \
       NCCL_IB_ROCE_VERSION_NUM=1 \
       NCCL_IB_GID_INDEX=0 \
       NCCL_IB_AR_DISABLE=1 \
       NCCL_NET_GDR_LEVEL=3 \
       NCCL_P2P_DISABLE=0 \
       NCCL_SOCKET_IFNAME=ib0 \
       NCCL_DEBUG=INFO \
       NCCL_INIT_TIMEOUT=120

srun --ntasks=$TOTAL_WORKERS --ntasks-per-node=$GPUS_PER_NODE --overlap "$SCRIPT_DIR/nccl_test"
echo "✓ Complete"
