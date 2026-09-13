# DACON workspace transfer

This repository contains experiment code, environment specifications, dataset manifests,
local evaluation records, and submission packaging tools.

## Runtime

- Windows workspace: `C:\Users\husky_gp7j99y\Documents\ChatGPT\DACON`
- Training/inference: WSL Ubuntu-24.04, conda environment `dacon-cu128`
- Python: `/home/huskypaul/miniforge3/envs/dacon-cu128/bin/python`
- Dependency snapshots: `environment/requirements-wsl-cu128.txt`, `environment/wsl_environment/requirements-lock.txt`
- Environment checks: `environment/WSL_READY.md`, `environment/wsl_environment/environment.json`

Scripts currently contain machine-specific Windows and WSL paths. Adjust these paths
and obtain the model/data assets before running on a different computer. Cloning this
repository alone does not recreate a runnable training environment.

## Latest completed experiment

HTDemucs vocals feed the frozen XLS-R encoder and trained Linear probe.
The sum of drums, bass and other feeds frozen EAT and trained AASIST+Linear.
Both runs retain the previous data selection, channel augmentation, seed and 2 epochs.

- Results: `aggressive_audio/stem_experiment_medium_v1/RESULTS.md`
- Resume/checkpoint instructions: `aggressive_audio/RESUME_DEMUCS_MEDIUM.md`
- Submission builder: `aggressive_audio/package_demucs.py`
- Actual ZIP verification: `aggressive_audio/records/demucs_submission_zip_verification.json`

## Assets stored separately

Git excludes source audio, separated stem caches, pretrained weights, training checkpoints,
assembled submission directories, and ZIP files. They remain on the original computer.
Training resume requires the official base models, dataset assets and the corresponding
`aggressive_audio/runs/*/checkpoint_latest.pt`, including optimizer and RNG state.

The latest local submission is `submit_eat_xlsr_demucs_resnet38_mean_v1.zip` (5.10 GiB).
Its SHA-256 sidecar is included in Git under `results/`. The ZIP itself must be transferred separately.
