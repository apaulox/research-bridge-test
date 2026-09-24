#!/bin/bash
# Usage: ./train.sh <NAME> [BACKBONE] [POOL] [SEED] [single|multi] [AGGREGATION_ORDER]

set -eo pipefail

CONDA_SH="/home/huskypaul/miniforge3/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_SH}" ]; then
  echo "Error: Miniforge initialization script not found: ${CONDA_SH}"
  exit 1
fi

# Scripts run in non-interactive shells (including tmux) do not inherit the
# interactive conda function. Initialize it here and select the project env.
source "${CONDA_SH}"
conda activate densessl

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./train.sh <CHECKPOINT_NAME> [dinov2_vitb14_reg|dinov3_vitb16]"
  exit 1
fi

CHECKPOINT_NAME=$1
VISUAL_BACKBONE=${2:-dinov2_vitb14_reg}
SEMANTIC_POOL=${3:-avg}
EXPERIMENT_SEED=${4:-42}
HEAD_LAYOUT=${5:-multi}
AGGREGATION_ORDER=${6:-product_then_pool}
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}"
SPLIT_FILE="${PROJECT_ROOT}/splits/unseen1.json"
cd "${PROJECT_ROOT}"

echo "======================================"
echo "🚀 Training Started"
echo "📂 Checkpoints dir: ${CHECKPOINT_DIR}"
echo "📄 Split file: ${SPLIT_FILE}"
echo "👁️ Visual backbone: ${VISUAL_BACKBONE}"
echo "Semantic pooling: ${SEMANTIC_POOL}"
echo "Seed: ${EXPERIMENT_SEED}"
echo "Contrastive heads: ${HEAD_LAYOUT}"
echo "Aggregation order: ${AGGREGATION_ORDER}"
echo "🐍 Python: $(command -v python)"
echo "======================================"

python -u train.py \
    --head_layout "${HEAD_LAYOUT}" \
    --aggregation_order "${AGGREGATION_ORDER}" \
    --semantic_pool "${SEMANTIC_POOL}" \
    --seed "${EXPERIMENT_SEED}" \
    --visual_backbone "${VISUAL_BACKBONE}" \
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
    --resume \
    --validation_batches 50 \
    --tensorboard True |& tee -a "${PROJECT_ROOT}/train_${CHECKPOINT_NAME}.log"

# Reached only after successful training; checkpoints remain safe if upload fails.
if [ "${WANDB_AUTO_UPLOAD:-1}" = "1" ]; then
    if ! python -u publish_experiment.py "${CHECKPOINT_NAME}"; then
        echo "Training completed, but W&B upload failed. Retry:"
        echo "python publish_experiment.py ${CHECKPOINT_NAME}"
        exit 1
    fi
fi
