# DenseSSL_new: image–mono single contrastive head

## Architecture

- Frozen DINOv3 ViT-B/16 by default. Patch map is resized to the existing 16×32 decoder interface for 224×448 images.
- Visual projection: 768→384→384 (1×1 Conv, BN, ReLU, 1×1 Conv, BN).
- The **same projected visual map** feeds the decoder and the contrastive loss.
- Mono encoder: complex spectrogram of L+R → 512 channels, shared with the U-Net decoder.
- Audio contrastive projection: 512→512→384. No separate stereo encoder.
- Token L2 normalization → channelwise visual HW max / audio FT mean → all-pair dot products.
- One symmetric image–mono InfoNCE (temperature 0.07, fixed) plus complex L−R spectrogram MSE.
- Total loss = MSE + contrastive_weight × InfoNCE; default weight 1.0.
- Decoder receives the unpooled, unnormalized 384-channel visual map. Mono decoder features stay 512 channels.

The old head/aggregation switches and category-based extra sampling are removed.
Training and validation use shuffled, distinct dataset indices with full batches.
Validation follows the historical Ours policy: with 104 clips and batch size 16,
each pass evaluates 96 clips and drops the final 8, then averages the equal-size batch MSEs.
Validation/best checkpoint selection remains reconstruction MSE only.

## Run

```bash
cd /home/huskypaul/DenseSSL_new
./train.sh mono384_v1 dinov3_vitb16 16655445212791328023
./test.sh mono384_v1 dinov3_vitb16
./evaluate.sh mono384_v1
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
