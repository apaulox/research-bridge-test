#!/bin/bash
set -euo pipefail

WEIGHTS_DIR="/home/huskypaul/dinov3_weights"
WEIGHTS_PATH="${WEIGHTS_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

echo "Paste the official ViT-B/16 LVD-1689M URL from Meta's access e-mail."
read -rsp "DINOv3 weights URL: " DINOV3_URL
echo

if [[ "${DINOV3_URL}" != https://* ]]; then
  echo "Error: expected an https:// URL from Meta."
  exit 1
fi

mkdir -p "${WEIGHTS_DIR}"
wget -c "${DINOV3_URL}" -O "${WEIGHTS_PATH}"

echo "Saved DINOv3 weights to ${WEIGHTS_PATH}"
