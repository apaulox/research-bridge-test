"""Image-mono InfoNCE with fixed 384-channel pool-then-product features."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AudioVisualContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self.register_buffer('temperature', torch.tensor(float(temperature)))

    def compute_sim_matrix(self, visual, audio):
        if visual.ndim != 4 or audio.ndim != 4 or visual.shape[1] != 384 or audio.shape[1] != 384:
            raise ValueError('Expected visual [B,384,H,W] and audio [B,384,F,T]')
        # Normalize tokens before pooling; no post-pooling normalization.
        visual = F.normalize(visual.flatten(2), p=2, dim=1).max(dim=2).values
        audio = F.normalize(audio.flatten(2), p=2, dim=1).mean(dim=2)
        return visual @ audio.T

    def forward(self, visual, audio):
        if visual.shape[0] != audio.shape[0] or visual.shape[0] < 2:
            raise ValueError('Contrastive training needs at least two matched image-mono pairs')
        similarity = self.compute_sim_matrix(visual, audio)
        logits = similarity / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return {'loss': loss, 'similarity': similarity}
