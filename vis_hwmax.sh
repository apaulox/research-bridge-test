#!/bin/bash
# Usage: ./vis_hwmax.sh <CHECKPOINT_NAME>

# piano:      123  225  817  922  207  230  380
# keyboard:    56  233  344 1106 1144    6  413
# guitar:      50  725  366  641  717 1034 1292
# ukulele:   1595 1462  823   47   67 1694 1769
# cello_fin:        83  804  161 1661  385 1153
# harp:       811  952  264   77  352  248  716
# cello_bow:  373  147 1363  254  160  904  114
# trumpet:   1235  268  106    7  372  317  968
# drum:       616   24  407   41  143  135  851
# all:        123  233  366   47  352  904  968

if [ -z "$1" ]; then
  echo "Error: Checkpoint name is required."
  echo "Usage: ./vis_hwmax.sh <CHECKPOINT_NAME>"
  exit 1
fi

CHECKPOINT_NAME=$1
PROJECT_ROOT=".../DenseSSL"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${CHECKPOINT_NAME}"

echo "======================================"
echo "🖼️  Visualize HW Max Started"
echo "📂 Target checkpoints dir: ${CHECKPOINT_DIR}"
echo "======================================"

python visualize_hw_max.py \
    --checkpoints_dir "${CHECKPOINT_DIR}" \
    --sample_ids 123 225 817 922 207 230 380
