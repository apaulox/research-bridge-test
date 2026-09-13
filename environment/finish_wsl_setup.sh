#!/usr/bin/env bash
set -euo pipefail
project=/home/huskypaul/dacon
environment=/home/huskypaul/miniforge3/envs/dacon-cu128
source_folder=/mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON
export PATH="$environment/bin:$PATH"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$project/setup"
cp "$source_folder/environment/requirements-wsl-cu128.txt" "$project/setup/"
cp "$source_folder/environment/verify_wsl_cuda.py" "$project/setup/"
python -m pip install -r "$project/setup/requirements-wsl-cu128.txt" --progress-bar off 2>&1 | tee "$project/logs/pip-install.log"
python -m pip check
python -m ipykernel install --sys-prefix --name dacon-cu128 --display-name 'Python (DACON CUDA 12.8)'
python -m pip freeze > "$project/setup/requirements-lock.txt"
python "$project/setup/verify_wsl_cuda.py" 2>&1 | tee "$project/logs/cuda-check.log"
python "$project/musicdet/server_check.py" --baseline-dir "$project/baseline" > "$project/logs/environment.json"
cd "$project/baseline"
python -u script.py 2>&1 | tee "$project/logs/baseline-smoke.log"
