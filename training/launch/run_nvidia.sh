#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-node NVIDIA GPU training launcher
#
# Requirements:
#   - passwordless SSH between nodes listed in HOSTFILE
#   - NCCL and CUDA installed on all nodes
#   - Shared filesystem mount visible at same path on all nodes
#
# Usage (single-node, 8 GPUs):
#   bash training/launch/run_nvidia.sh
#
# Usage (multi-node):
#   MASTER_ADDR=10.0.0.1 NUM_NODES=2 GPUS_PER_NODE=8 \
#     bash training/launch/run_nvidia.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable environment (override via export before calling this script)
# ---------------------------------------------------------------------------
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29500}"
: "${NUM_NODES:=1}"
: "${GPUS_PER_NODE:=$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
: "${NODE_RANK:=0}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/training/configs/lora_config.yaml"
DS_CONFIG="${PROJECT_ROOT}/training/configs/ds_zero3_nvidia.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/llama-pt-debugger"

echo "===================================================================="
echo " LLaMA-3.1-8B Fine-tuning — NVIDIA"
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  NUM_NODES     : ${NUM_NODES}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  NODE_RANK     : ${NODE_RANK}"
echo "  PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "===================================================================="

# NCCL tuning for high-bandwidth interconnects (NVLink / InfiniBand)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0         # Adjust to your network interface
export NCCL_ASYNC_ERROR_HANDLING=1

# WandB — set outside this script or in your environment
# export WANDB_PROJECT=llama-pt-debugger
# export WANDB_API_KEY=...

cd "${PROJECT_ROOT}"

torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  training/train.py \
    --config "${CONFIG}" \
    --device cuda \
    --deepspeed_config "${DS_CONFIG}"

echo "Training complete. Adapter saved to ${OUTPUT_DIR}"
