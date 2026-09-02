#!/bin/bash
# Usage: ./train.sh <CHECKPOINT_NAME>

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./train.sh <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1
PROJECT_ROOT=".../DenseSSL"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}"
SPLIT_FILE="${PROJECT_ROOT}/splits/unseen1.json"

echo "======================================"
echo "🚀 Training Started"
echo "📂 Checkpoints dir: ${CHECKPOINT_DIR}"
echo "📄 Split file: ${SPLIT_FILE}"
echo "======================================"

python train.py \
    --norm_semantic \
    --norm_spatial \
    --split_file "${SPLIT_FILE}" \
    --name mono2binaural \
    --model audioVisual \
    --checkpoints_dir "${CHECKPOINT_DIR}" \
    --save_epoch_freq 50 \
    --display_freq 10 \
    --save_latest_freq 100 \
    --batchSize 16 \
    --spatial_num_samples 5 \
    --learning_rate_decrease_itr 10 \
    --niter 1000 \
    --lr_visual 0.000025 \
    --lr_audio 0.00025 \
    --nThreads 4 \
    --gpu_ids 0 \
    --validation_on \
    --validation_freq 100 \
    --validation_batches 50 \
    --tensorboard True |& tee -a "${PROJECT_ROOT}/train_${CHECKPOINT_NAME}.log"
