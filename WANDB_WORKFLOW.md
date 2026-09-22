# Completed training and portable evaluation

Run attention training from DenseSSL_attention:

```bash
./train.sh densessl_unseen1_dinov2reg_semsimpool_vcap_s42 dinov2_vitb14_reg visual_attention 42
```

The third argument selects pooling. `avg` selects the average control;
`visual_attention` enables learned semantic similarity pooling. Training prints
the selected pooling and seed. Different configurations use different run names.

After successful training, train.sh uploads TensorBoard history, the best
validation marker, and a model artifact to the existing W&B DenseSSL project.
This happens at completion, not continuously. Keep the training host awake and
connected until upload finishes. Set WANDB_AUTO_UPLOAD=0 to disable this step.
Upload errors preserve checkpoints and print a retry command:

```bash
python publish_experiment.py densessl_unseen1_dinov2reg_semsimpool_vcap_s42
```

After running inference and evaluation, include the evaluation log to publish
STFT and envelope metrics. This does not run evaluation automatically:

```bash
python publish_experiment.py EXPERIMENT --evaluation EVALUATION_LOG.txt
```

Artifacts include best visual/audio/criterion weights, split, options, current
Python source, package versions, and cached DINOv2 source. They omit datasets,
credentials and optimizer state. They support inference, not training resume.
The portable runner currently supports DINOv2+register only.

On another CUDA laptop with DenseSSL dependencies and FAIR-Play audio/frames:

```bash
wandb login
wandb artifact get stringwoo22-hanyang-university/DenseSSL/densessl_unseen1_dinov2reg_semsimpool_vcap_s42-best:best --root ./attention_best
python -u ./attention_best/source/evaluate_bundle.py --audio-dir /path/to/FAIR-Play/audios --video-dir /path/to/FAIR-Play/frames --output-dir ./attention_outputs
```

The runner generates binaural audio and then computes evaluation metrics.
Package versions are recorded in environment.txt; install a compatible CUDA
PyTorch environment rather than assuming the package list is portable as-is.

Verified control: avg, seed 42, 1000 completed epochs. Best validation is
0.563196192185084 at epoch 150 / step 124800. Test STFT mean is
0.8489192323938298; envelope mean is 0.13105849905584294.
Artifact: densessl_unseen1_dinov2reg_semsimpool_avg_s42-best:v0.

Validation completed: pooling/gradient/resume tests, checkpoint metadata and
history consistency, remote W&B summary, artifact download, one-clip inference.
The one-clip evaluation reached the standard-deviation calculation, which needs
at least two clips. A two-clip end-to-end check has not been completed.
