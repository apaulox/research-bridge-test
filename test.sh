#!/bin/bash
# Usage: ./test.sh NAME [BACKBONE]
set -eo pipefail
source /home/huskypaul/miniforge3/etc/profile.d/conda.sh
conda activate densessl

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./test.sh <CHECKPOINT_NAME> [dinov2_vitb14_reg|dinov3_vitb16]"
  exit 1
fi

CHECKPOINT_NAME=$1
VISUAL_BACKBONE=${2:-dinov3_vitb16}
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
WEIGHTS_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}/mono2binaural"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${CHECKPOINT_NAME}"
SPLIT_FILE="${PROJECT_ROOT}/splits/unseen1.json"

echo "======================================"
echo "🧪 Testing (Inference) Started"
echo "📂 Loading weights from: ${WEIGHTS_DIR}"
echo "💾 Outputs will be saved to: ${OUTPUT_DIR}"
echo "👁️ Visual backbone: ${VISUAL_BACKBONE}"
echo "======================================"

python -u demo_batch.py \
    --visual_backbone "${VISUAL_BACKBONE}" \
    --split_file "${SPLIT_FILE}" \
    --weights_visual "${WEIGHTS_DIR}/visual_best.pth" \
    --weights_audio "${WEIGHTS_DIR}/audio_best.pth" \
    --output_dir_root "${OUTPUT_DIR}" \
    --hop_size 0.05
