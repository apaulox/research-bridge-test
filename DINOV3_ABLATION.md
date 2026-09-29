# Visual backbone configuration

DINOv3 ViT-B/16 is the default frozen backbone. DINOv2 with registers is still
available through --visual_backbone. Both output 768-channel patch maps.
For 224x448 images the native DINOv3 map is 14x28; it is not interpolated.
The shared trainable 768-channel layer reads the patch map before the two
paths split. The contrastive-only projection reads its output. A separate
decoder path average-pools it to 7x14, projects 768 to 8 channels, and flattens
to 784 channels before joining the mono audio bottleneck. The 384-channel
contrastive projection itself is not used as decoder input.

Local source: /home/huskypaul/dinov3.
Pretrained weights: /home/huskypaul/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth.
See README.md for the image-mono architecture and execution commands.
