"""Exercise the actual training entrypoint with tiny synthetic modules and data."""
import json
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch
import torch
from torch import nn


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 384, 1)

    def forward(self, x):
        return self.projection(x)


class Audio(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(2, 384, 1)
        self.mask = nn.Conv2d(2, 2, 1)
        self.visual = nn.Linear(384, 2)

    def forward(self, mono, visual):
        bias = self.visual(visual.mean((2, 3)))[:, :, None, None]
        return (self.mask(mono[:, :, :-1]) + bias).tanh(), self.projection(mono)


class Loader:
    def __init__(self):
        self.batch = {'frame': torch.randn(2, 3, 2, 3),
                      'audio_mix_spec': torch.randn(2, 2, 9, 4),
                      'audio_diff_spec': torch.randn(2, 2, 9, 4)}

    def load_data(self): return self
    def __len__(self): return 2
    def __iter__(self): return iter([self.batch])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Training entrypoint integration check needs CUDA')
    with tempfile.TemporaryDirectory(prefix='single-mono-loop-') as directory:
        root = Path(directory)
        split = root / 'split.json'
        split.write_text(json.dumps({'train': [], 'val': []}))
        common = ['train.py', '--visual_backbone', 'dinov3_vitb16', '--gpu_ids', '0',
                  '--batchSize', '2', '--nThreads', '0', '--checkpoints_dir', directory,
                  '--name', 'smoke', '--split_file', str(split), '--display_freq', '1',
                  '--validation_on', '--validation_freq', '1', '--validation_batches', '1',
                  '--save_epoch_freq', '1', '--save_latest_freq', '1', '--resume', '--tensorboard', 'True']
        with patch('data.data_loader.CreateDataLoader', side_effect=lambda opt: Loader()), \
             patch('models.models.ModelBuilder.build_visual', side_effect=lambda **kwargs: Visual()), \
             patch('models.models.ModelBuilder.build_audio', side_effect=lambda **kwargs: Audio()):
            for epochs in (1, 2):
                with patch.object(sys, 'argv', common + ['--niter', str(epochs)]):
                    runpy.run_path('train.py', run_name='__main__')
                state = torch.load(root / 'smoke/training_latest.pth', map_location='cpu', weights_only=False)
                assert state['next_epoch'] == epochs + 1 and state['total_steps'] == epochs * 2
                assert state['training_config']['architecture'] == 'image_mono_384_v1'
                assert state['contrastive_criterion'].keys() == {'temperature'}
        best = torch.load(root / 'smoke/criterion_best.pth', map_location='cpu', weights_only=False)
        assert best['training_config'] == state['training_config']
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(root / 'smoke/tensorboard')).Reload()
        assert set(events.Tags()['scalars']) == {'data/loss', 'data/loss_mse', 'data/loss_contrastive', 'data/val_loss'}
        assert len(events.Scalars('data/loss')) == 2
    print('PASS: real train.py new run + resume, validation, best/latest weights and TensorBoard')


if __name__ == '__main__':
    main()
