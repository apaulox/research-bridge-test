"""One real DINOv3 forward/backward, with no training checkpoint written."""
import random
from types import SimpleNamespace
import numpy as np
import torch
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models.criterion import DenseAVContrastiveLoss

seed = 16655445212791328023
random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
builder = ModelBuilder()
visual = builder.build_visual(backbone='dinov3_vitb16', dinov3_repo='/home/huskypaul/dinov3',
    dinov3_weights='/home/huskypaul/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', head_layout='single')
audio = builder.build_audio(head_layout='single')
model = AudioVisualModel((visual, audio), SimpleNamespace()).cuda().train()
criterion = DenseAVContrastiveLoss(feature_dim=512, aggregation_order='pool_then_product').cuda()
criterion.log_temp_sem.requires_grad_(False); criterion.log_temp_spa.requires_grad_(False)
data = {'frame': torch.randn(3, 3, 224, 448, device='cuda'),
        'audio_mix_spec': torch.randn(3, 2, 257, 64, device='cuda'),
        'audio_diff_spec': torch.randn(3, 2, 257, 64, device='cuda')}
output = model(data)
losses = criterion(output['semantic_visual_feat'][:2], output['semantic_audio_feat'][:2],
    output['spatial_visual_feat'], output['spatial_audio_feat'], s=1, norm_semantic=True, norm_spatial=True)
loss = torch.nn.functional.mse_loss(output['binaural_spectrogram'][:2], output['audio_gt'][:2]) + losses['loss']
loss.backward()
for param in (visual.single_proj[0].weight, audio.single_proj[0].weight, audio.audionet_upconvlayer5[0].weight):
    assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
assert all(p.grad is None for p in visual.feature_extraction.parameters())
print('PASS: real pretrained DINOv3 single + pool_then_product forward/backward; loss=', loss.item())
print('head shape:', tuple(output['semantic_visual_feat'].shape), 'prediction:', tuple(output['binaural_spectrogram'].shape))
