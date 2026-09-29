# DenseSSL single head with 2.5D-style decoder input

## Architecture

- Frozen DINOv3 ViT-B/16 by default. For 224×448 images, use its native 768×14×28 patch map without interpolation.
- A trainable 768→768 visual layer is shared before the paths split. The contrastive-only head then projects 768→384→384 and pools over HW.
- The decoder uses the shared 768-channel map: adaptive average pool to 7×14, 1×1 Conv 768→8, flatten to 784 channels, then concatenate with the 512-channel mono bottleneck. Its first upconvolution receives 1296 channels.
- Mono encoder: complex spectrogram of L+R → 512 channels, shared with the U-Net decoder.
- Audio contrastive projection: 512→512→384. No separate stereo encoder.
- Token L2 normalization → channelwise visual HW max / audio FT mean → all-pair dot products.
- One symmetric image–mono InfoNCE (temperature 0.07, fixed) plus complex L−R spectrogram MSE.
- Total loss = MSE + contrastive_weight × InfoNCE; default weight 1.0.
- The 384-channel contrastive projection is not fed to the decoder. Both losses update the shared 768-channel visual layer and mono encoder; each branch also has its own downstream layers.

The old head/aggregation switches and category-based extra sampling are removed.
Training and validation use shuffled, distinct dataset indices with full batches.
Validation follows the historical Ours policy: with 104 clips and batch size 16,
each pass evaluates 96 clips and drops the final 8, then averages the equal-size batch MSEs.
Validation/best checkpoint selection remains reconstruction MSE only.

## Run

```bash
cd /home/huskypaul/DenseSSL_new
./train.sh densessl_unseen1_dinov3_mono384_poolprod_2p5d784_s42 dinov3_vitb16 42
./test.sh densessl_unseen1_dinov3_mono384_poolprod_2p5d784_s42 dinov3_vitb16
./evaluate.sh densessl_unseen1_dinov3_mono384_poolprod_2p5d784_s42
```

The existing `densessl` conda environment, FAIR-Play data, and pretrained backbone are reused.
Training is not started by preparing this workspace. `WANDB_AUTO_UPLOAD=0` disables completion upload.
Use fresh experiment names: older model/optimizer checkpoints are incompatible with this architecture.
Training resumes only when the saved `training_config` matches.

## Checks

```bash
python test_single_mono.py
python test_training_loop.py
python smoke_single_mono.py
```

The smoke check uses the real frozen DINOv3 on CUDA with a synthetic batch and writes no training checkpoints.
Legacy ISTFT behavior is unchanged; `reEncodeAudio.py` is an unused Python 2 utility.
