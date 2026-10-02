#!/bin/bash
# Usage: ./train.sh <CHECKPOINT_NAME> [VISUAL_BACKBONE] [SEED]

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
  echo "Usage: ./train.sh <CHECKPOINT_NAME> [dinov2_vitb14_reg|dinov3_vitb16] [SEED]"
  exit 1
fi

if [ "$#" -gt 3 ]; then
  echo "Usage: ./train.sh <CHECKPOINT_NAME> [dinov2_vitb14_reg|dinov3_vitb16] [SEED]"
  exit 1
fi

CHECKPOINT_NAME=$1
VISUAL_BACKBONE=${2:-dinov2_vitb14_reg}
PROJECT_ROOT="/home/huskypaul/DenseSSL"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}"
SPLIT_FILE="${PROJECT_ROOT}/splits/unseen1.json"
TRAINING_STATE="${CHECKPOINT_DIR}/mono2binaural/training_latest.pth"
SEED_RECORD="${CHECKPOINT_DIR}/mono2binaural/opt.txt"
if [ -f "${TRAINING_STATE}" ]; then
  if [ ! -f "${SEED_RECORD}" ]; then
    echo "Error: training state exists but its seed record is missing: ${SEED_RECORD}"
    exit 1
  fi
  EXPERIMENT_SEED=$(awk -F ': ' '$1 == "seed" {print $2; exit}' "${SEED_RECORD}")
  if [ -z "${EXPERIMENT_SEED}" ]; then
    echo "Error: training state exists but no seed was recorded in ${SEED_RECORD}"
    exit 1
  fi
  if [ "$#" -ge 3 ] && [ "$3" != "${EXPERIMENT_SEED}" ]; then
    echo "Error: requested seed $3 differs from the saved seed ${EXPERIMENT_SEED}"
    exit 1
  fi
elif [ "$#" -ge 3 ]; then
  EXPERIMENT_SEED=$3
else
  EXPERIMENT_SEED=$(python -c 'import secrets; print(secrets.randbits(63))')
fi
cd "${PROJECT_ROOT}"

echo "======================================"
echo "🚀 Training Started"
echo "📂 Checkpoints dir: ${CHECKPOINT_DIR}"
echo "📄 Split file: ${SPLIT_FILE}"
echo "👁️ Visual backbone: ${VISUAL_BACKBONE}"
echo "Seed: ${EXPERIMENT_SEED} (saved in the experiment's opt.txt)"
echo "🐍 Python: $(command -v python)"
echo "======================================"

python train.py \
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
