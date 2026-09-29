"""Real pretrained DINOv3 CUDA forward/backward; no training checkpoint writes."""
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models.criterion import AudioVisualContrastiveLoss


def main():
    torch.manual_seed(42)
    builder = ModelBuilder()
    visual = builder.build_visual(backbone='dinov3_vitb16', dinov3_repo='/home/huskypaul/dinov3',
        dinov3_weights='/home/huskypaul/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    audio = builder.build_audio()
    model = AudioVisualModel((visual, audio), SimpleNamespace()).cuda().train()
    criterion = AudioVisualContrastiveLoss().cuda()
    data = {'frame': torch.randn(2, 3, 224, 448, device='cuda'),
            'audio_mix_spec': torch.randn(2, 2, 257, 64, device='cuda'),
            'audio_diff_spec': torch.randn(2, 2, 257, 64, device='cuda')}
    out = model(data)
    mse = F.mse_loss(out['binaural_spectrogram'], out['audio_gt'])
    contrastive = criterion(out['visual_feat'], out['audio_feat'])['loss']
    (mse + contrastive).backward()
    for p in (visual.shared_proj[0].weight, visual.contrastive_projection[0].weight,
              audio.conv1x1[0].weight,
              audio.projection[-2].weight,
              audio.audionet_convlayer1[0].weight, audio.audionet_upconvlayer5[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(p.grad is None for p in visual.feature_extraction.parameters())
    print('PASS: real DINOv3 CUDA; visual', tuple(out['visual_feat'].shape),
          'audio', tuple(out['audio_feat'].shape), 'prediction', tuple(out['binaural_spectrogram'].shape),
          'MSE', mse.item(), 'contrastive', contrastive.item())


if __name__ == '__main__':
    main()
