# DINOv3 backbone ablation

The DINOv2 baseline remains the default. The ablation changes only the frozen
visual backbone from `dinov2_vitb14_reg` to `dinov3_vitb16`.

Both backbones output 768 channels and exclude register/storage tokens by using
`x_norm_patchtokens`. With a 224x448 frame, DINOv2/14 produces a 16x32 map and
DINOv3/16 produces a 14x28 map. The DINOv3 adapter bilinearly resizes its map to
16x32 so all trainable DenseSSL layers and loss settings remain unchanged.

The local DINOv3 source checkout was validated at commit:

`6876159a11b4df116f30f667f8c9888617df0751`

## Weights

Request access at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/.
Meta sends the official checkpoint URLs by e-mail after approval. Then run:

```bash
cd /home/huskypaul/DenseSSL
./setup_dinov3_weights.sh
```

The prompt is hidden so the signed URL is not stored in shell history.

## Commands

Reproduce the existing DINOv2 + registers configuration:

```bash
./train.sh densessl_unseen1_dinov2_reg dinov2_vitb14_reg
```

Run the DINOv3-B/16 ablation under the same training configuration:

```bash
./train.sh densessl_unseen1_dinov3_vitb16 dinov3_vitb16
```

Inference:

```bash
./test.sh densessl_unseen1_dinov3_vitb16 dinov3_vitb16
```

Evaluation is unchanged:

```bash
./evaluate.sh densessl_unseen1_dinov3_vitb16
```
