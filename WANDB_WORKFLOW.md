# Completed training and evaluation

Run training from `/home/huskypaul/DenseSSL_new`:

```bash
./train.sh EXPERIMENT dinov3_vitb16 42
```

After successful training, the script uploads TensorBoard history and the best
model artifact to the existing W&B DenseSSL project. Set `WANDB_AUTO_UPLOAD=0`
to disable this step. Upload errors preserve local checkpoints.

```bash
python publish_experiment.py EXPERIMENT --check-only
python publish_experiment.py EXPERIMENT --evaluation EVALUATION_LOG.txt
```

Artifacts include best weights, split, options, source, and package versions.
They omit datasets, credentials, and optimizer state. DINOv2 artifacts also
include cached backbone source; the portable runner supports DINOv2 only.
Use this workspace and its local backbone dependencies for DINOv3 evaluation.
