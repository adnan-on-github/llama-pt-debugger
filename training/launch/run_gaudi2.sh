#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-node Gaudi2 HPU training launcher
#
# Requirements:
#   - SynapseAI + habana_frameworks installed on all nodes
#   - optimum-habana installed
#   - Shared filesystem at same path on all nodes
#   - MPI / passwordless SSH configured between nodes
#
# Usage (single-node, 8 Gaudi2 cards):
#   bash training/launch/run_gaudi2.sh
#
# Usage (multi-node via MPI):
#   NUM_NODES=2 GAUDI_PER_NODE=8 \
#     bash training/launch/run_gaudi2.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable environment
# ---------------------------------------------------------------------------
: "${NUM_NODES:=1}"
: "${GAUDI_PER_NODE:=8}"
: "${MASTER_ADDR:=localhost}"
: "${MASTER_PORT:=29500}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/training/configs/lora_config.yaml"
DS_CONFIG="${PROJECT_ROOT}/training/configs/ds_zero3_gaudi2.json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/llama-pt-debugger"
TOTAL_RANKS=$(( NUM_NODES * GAUDI_PER_NODE ))

echo "===================================================================="
echo " LLaMA-3.1-8B Fine-tuning — Gaudi2 HPU"
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  NUM_NODES     : ${NUM_NODES}"
echo "  GAUDI_PER_NODE: ${GAUDI_PER_NODE}"
echo "  TOTAL_RANKS   : ${TOTAL_RANKS}"
echo "  PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "===================================================================="

# Habana environment
export HABANA_LOGS="${PROJECT_ROOT}/logs/habana"
export LOG_LEVEL_all=4
export ENABLE_CONSOLE=true
export PT_HPU_LAZY_MODE=1        # Enable lazy execution
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

cd "${PROJECT_ROOT}"
mkdir -p "${HABANA_LOGS}"

if [ "${NUM_NODES}" -eq 1 ]; then
  # Single-node: use gaudi_spawn from optimum-habana
  python -m optimum.habana.distributed.gaudi_spawn \
    --use_mpi \
    --world_size "${GAUDI_PER_NODE}" \
    training/train.py \
      --config "${CONFIG}" \
      --device hpu \
      --deepspeed_config "${DS_CONFIG}"
else
  # Multi-node: use mpirun with a hostfile
  # Create hostfile: one line per node (hostname slots=N)
  HOSTFILE="${PROJECT_ROOT}/training/launch/hostfile"
  if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: ${HOSTFILE} not found."
    echo "Create it with lines like:"
    echo "  node01 slots=${GAUDI_PER_NODE}"
    echo "  node02 slots=${GAUDI_PER_NODE}"
    exit 1
  fi

  mpirun \
    --allow-run-as-root \
    -n "${TOTAL_RANKS}" \
    --hostfile "${HOSTFILE}" \
    -x MASTER_ADDR="${MASTER_ADDR}" \
    -x MASTER_PORT="${MASTER_PORT}" \
    -x PT_HPU_LAZY_MODE \
    -x PT_HPU_ENABLE_LAZY_COLLECTIVES \
    -x HABANA_LOGS \
    -x LOG_LEVEL_all \
    python training/train.py \
      --config "${CONFIG}" \
      --device hpu \
      --deepspeed_config "${DS_CONFIG}"
fi

echo "Training complete. Adapter saved to ${OUTPUT_DIR}"
