#!/bin/bash
# Usage: ./test.sh <CHECKPOINT_NAME>
# 예시: ./test.sh my_experiment

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./test.sh <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1
PROJECT_ROOT=".../DenseSSL"
WEIGHTS_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}/mono2binaural"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${CHECKPOINT_NAME}"
SPLIT_FILE="${PROJECT_ROOT}/splits/unseen1.json"

echo "======================================"
echo "🧪 Testing (Inference) Started"
echo "📂 Loading weights from: ${WEIGHTS_DIR}"
echo "💾 Outputs will be saved to: ${OUTPUT_DIR}"
echo "======================================"

python demo_batch.py \
    --split_file "${SPLIT_FILE}" \
    --weights_visual "${WEIGHTS_DIR}/visual_best.pth" \
    --weights_audio "${WEIGHTS_DIR}/audio_best.pth" \
    --output_dir_root "${OUTPUT_DIR}" \
    --hop_size 0.05
