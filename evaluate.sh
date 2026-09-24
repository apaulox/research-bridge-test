#!/bin/bash
# Usage: ./evaluate.sh <CHECKPOINT_NAME>
set -eo pipefail
source /home/huskypaul/miniforge3/etc/profile.d/conda.sh
conda activate densessl

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./evaluate.sh <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${CHECKPOINT_NAME}"

echo "======================================"
echo "📊 Evaluation Started"
echo "📂 Target outputs dir: ${OUTPUT_DIR}"
echo "======================================"

python evaluate.py \
    --results_root "${OUTPUT_DIR}" \
    --normalization True
