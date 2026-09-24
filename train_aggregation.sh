#!/bin/bash
# ./train_aggregation.sh EXPERIMENT single|multi pool_then_product|product_then_pool [SEED]
set -eo pipefail
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 EXPERIMENT single|multi pool_then_product|product_then_pool [SEED]"
    exit 2
fi
case "$2" in single|multi) ;; *) echo "Invalid head layout: $2"; exit 2;; esac
case "$3" in pool_then_product|product_then_pool) ;; *) echo "Invalid aggregation: $3"; exit 2;; esac
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/train.sh" "$1" dinov3_vitb16 avg "${4:-16655445212791328023}" "$2" "$3"
