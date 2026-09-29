# Visual backbone configuration

DINOv3 ViT-B/16 is the default frozen backbone. DINOv2 with registers is still
available through --visual_backbone. Both output 768-channel patch maps.
For 224x448 images the DINOv3 map is resized from 14x28 to 16x32 before the
shared 384-channel projection, matching the decoder interface.

Local source: /home/huskypaul/dinov3.
Pretrained weights: /home/huskypaul/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth.
See README.md for the image-mono architecture and execution commands.
