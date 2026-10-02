# Ours: DINOv3 native patch map and 784-channel decoder input

Recorded: 2026-09-30. This change applies to `/home/huskypaul/DenseSSL`.

For a 224×448 video frame, frozen DINOv3 ViT-B/16 produces `[B, 768, 14, 28]`. The model now keeps that native patch map instead of interpolating it to 16×32. The existing trainable `shared_proj` remains before the semantic/spatial split. The two 384-channel visual contrastive heads and both audio contrastive paths are unchanged.

The binaural decoder continues to receive only the spatial slice `spa`, `[B, 384, 14, 28]`, before its contrastive projection. It applies adaptive average pooling to `[B, 384, 7, 14]`, the existing 1×1 convolution from 384 to 8 channels, and flattening to `[B, 784]`. The 784-channel visual vector is broadcast over the mono audio bottleneck and concatenated with its 512 channels, making the first decoder up-convolution input 1296 channels. The former `4096→512` visual projection is removed.

The existing shared gradient connection and `return out, spa` are preserved: the contrastive losses and binaural MSE can both update `shared_proj`. No extra common layer is added, and the semantic slice is not added to the decoder input.

The semantic/spatial contrastive losses, audio encoders, reconstruction target, dataset, and training hyperparameters have not been changed. The DINOv2 option still produces a 16×32 patch map, which the same adaptive decoder path pools to 7×14. Old decoder checkpoints are architecture-incompatible; use a fresh experiment name for this structure.

Training accepts an optional seed as the third `train.sh` argument. For a new experiment with no seed argument, a 63-bit seed is drawn from the operating system, printed, and saved in the experiment's `opt.txt`. Python, NumPy, PyTorch, and CUDA RNGs are initialized before model construction and reset before data sampling. On resume, `train.sh` reuses the recorded seed and `train.py` restores the RNG states saved in `training_latest.pth`. A conflicting explicit seed is rejected. Example with a fixed seed: `./train.sh densessl_unseen1_dinov3_multi_784_s42 dinov3_vitb16 42`.
