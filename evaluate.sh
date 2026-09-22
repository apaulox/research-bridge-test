#!/bin/bash
# Usage: ./evaluate.sh <CHECKPOINT_NAME>

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./evaluate.sh <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1
PROJECT_ROOT="/home/huskypaul/DenseSSL"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${CHECKPOINT_NAME}"

echo "======================================"
echo "📊 Evaluation Started"
echo "📂 Target outputs dir: ${OUTPUT_DIR}"
echo "======================================"

python evaluate.py \
    --results_root "${OUTPUT_DIR}" \
    --normalization True
