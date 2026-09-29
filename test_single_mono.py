"""Checks formulas, gradient routing, target independence, batching and resume."""
import ast
import copy
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.networks import VisualNet, AudioNet
from models.audioVisual_model import AudioVisualModel
from models.criterion import AudioVisualContrastiveLoss
from data.custom_dataset_data_loader import CustomDatasetDataLoader


class FakeDino(nn.Module):
    def __init__(self, patch_size=14):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.patch_size = patch_size

    def forward_features(self, images):
        tokens = F.avg_pool2d(images, self.patch_size, self.patch_size).flatten(2).transpose(1, 2)
        return {'x_norm_patchtokens': tokens.repeat(1, 1, 256) * self.scale}


def check_dinov3_decoder_shapes():
    dinov3 = ModuleType('dinov3')
    hub = ModuleType('dinov3.hub')
    backbones = ModuleType('dinov3.hub.backbones')
    backbones.dinov3_vitb16 = lambda **kwargs: FakeDino(patch_size=16)
    with patch.dict(sys.modules, {'dinov3': dinov3, 'dinov3.hub': hub,
                                  'dinov3.hub.backbones': backbones}):
        visual = VisualNet(backbone='dinov3_vitb16', dinov3_repo='.', pretrained=False)
    audio = AudioNet()
    visual.eval()
    audio.eval()
    with torch.no_grad():
        contrastive, decoder_source = visual(torch.randn(1, 3, 224, 448))
        pooled = audio.visual_pool(decoder_source)
        reduced = audio.conv1x1(pooled)
    assert decoder_source.shape == (1, 768, 14, 28)
    assert contrastive.shape == (1, 384, 14, 28)
    assert pooled.shape == (1, 768, 7, 14)
    assert reduced.shape == (1, 8, 7, 14)
    assert reduced.flatten(1).shape == (1, 784)
    assert audio.audionet_upconvlayer1[0].in_channels == 1296
    print('PASS: DINOv3 native 14x28, decoder 7x14 -> 8 -> 784, no interpolation')


def nonzero(grads):
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def check_formula():
    criterion = AudioVisualContrastiveLoss()
    visual = torch.randn(3, 384, 2, 3, requires_grad=True)
    audio = torch.randn(3, 384, 2, 2, requires_grad=True)
    v = F.normalize(visual.flatten(2), dim=1)
    a = F.normalize(audio.flatten(2), dim=1)
    expected = torch.stack([torch.stack([
        torch.dot(vi.max(dim=1).values, aj.mean(dim=1)) for aj in a
    ]) for vi in v])
    result = criterion(visual, audio)
    torch.testing.assert_close(result['similarity'], expected)
    labels = torch.arange(3)
    expected_loss = (F.cross_entropy(expected / 0.07, labels) +
                     F.cross_entropy(expected.T / 0.07, labels)) / 2
    torch.testing.assert_close(result['loss'], expected_loss)
    nonzero(torch.autograd.grad(result['loss'], (visual, audio)))
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(criterion.compute_sim_matrix(visual, audio[perm]), expected[:, perm])
    # This input distinguishes pooling-before-product from the old ordering.
    v = torch.zeros(1, 384, 1, 2); a = torch.zeros(1, 384, 1, 1)
    v[0, 0, 0, 0] = v[0, 1, 0, 1] = 1
    a[0, :2] = 1
    torch.testing.assert_close(criterion.compute_sim_matrix(v, a), torch.tensor([[2.0 ** 0.5]]))
    for temp in (0, -1, float('nan')):
        try:
            AudioVisualContrastiveLoss(temp)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid temperature accepted')
    print('PASS: pool-then-product formula, symmetric InfoNCE and feature gradients')


def training_functions():
    tree = ast.parse(Path('train.py').read_text())
    names = {'create_optimizer', 'save_training_state', 'load_training_state'}
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name in names], type_ignores=[])
    scope = dict(torch=torch, random=random, np=np, os=os)
    exec(compile(functions, 'train.py', 'exec'), scope)
    return scope


def check_model_and_resume():
    with patch('torch.hub.load', side_effect=lambda *args, **kwargs: FakeDino()):
        visual = VisualNet(backbone='dinov2_vitb14_reg', pretrained=False)
        audio = AudioNet()
        model = AudioVisualModel((visual, audio), SimpleNamespace())
        criterion = AudioVisualContrastiveLoss()
        criterion.training_config = {'architecture': 'image_mono_2p5d_784_shared768_v3'}
        inputs = {'frame': torch.randn(2, 3, 224, 448),
                  'audio_mix_spec': torch.randn(2, 2, 257, 64),
                  'audio_diff_spec': torch.randn(2, 2, 257, 64)}
        decoder_inputs = []
        hook = audio.conv1x1.register_forward_pre_hook(lambda module, args: decoder_inputs.append(args[0]))
        output = model(inputs)
        hook.remove()
        assert decoder_inputs[0].shape == (2, 768, 7, 14)
        assert decoder_inputs[0].requires_grad is True
        assert output['visual_feat'].shape == (2, 384, 16, 32)
        assert output['audio_feat'].shape == (2, 384, 8, 2)
        assert output['binaural_spectrogram'].shape == (2, 2, 256, 64)
        mse = F.mse_loss(output['binaural_spectrogram'], output['audio_gt'])
        contrastive = criterion(output['visual_feat'], output['audio_feat'])['loss']
        # Both objectives update the 768-channel visual trunk and mono encoder.
        # The contrastive head and decoder reduction stay branch-specific.
        mono_weight = audio.audionet_convlayer1[0].weight
        shared_weight = visual.shared_proj[0].weight
        visual_weight = visual.contrastive_projection[0].weight
        decoder_weight = audio.conv1x1[0].weight
        nonzero(torch.autograd.grad(mse, (shared_weight, decoder_weight, mono_weight), retain_graph=True))
        nonzero(torch.autograd.grad(contrastive, (shared_weight, visual_weight, mono_weight,
                                                 audio.projection[-2].weight), retain_graph=True))
        assert torch.autograd.grad(mse, visual_weight, allow_unused=True, retain_graph=True)[0] is None
        assert torch.autograd.grad(contrastive, decoder_weight, allow_unused=True,
                                   retain_graph=True)[0] is None
        assert torch.autograd.grad(mse, audio.projection[-2].weight, allow_unused=True, retain_graph=True)[0] is None
        assert torch.autograd.grad(contrastive, audio.audionet_upconvlayer5[0].weight,
                                   allow_unused=True, retain_graph=True)[0] is None
        scope = training_functions()
        opt = SimpleNamespace(lr_visual=2.5e-5, lr_audio=2.5e-4,
                              optimizer='adam', beta1=0.9, weight_decay=0.0005)
        optimizer = scope['create_optimizer']((visual, audio), opt)
        (mse + contrastive).backward()
        assert all(p.grad is None for p in visual.feature_extraction.parameters())
        optimizer.step()
        del output, mse, contrastive, decoder_inputs
        model.eval()
        with torch.no_grad():
            reference = model(inputs)
            no_target = model({k: v for k, v in inputs.items() if k != 'audio_diff_spec'})
            swapped_target = model({**inputs, 'audio_diff_spec': -inputs['audio_diff_spec']})
            for key in ('binaural_spectrogram', 'audio_feat', 'visual_feat'):
                torch.testing.assert_close(reference[key], no_target[key], rtol=0, atol=0)
                torch.testing.assert_close(reference[key], swapped_target[key], rtol=0, atol=0)
            assert 'audio_gt' not in no_target
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / 'state.pth')
            scope['save_training_state'](checkpoint, 2, 2, 0.5, visual, audio, optimizer, criterion)
            expected_random = (random.random(), np.random.rand(), torch.rand(1))
            saved = visual.contrastive_projection[0].weight.detach().clone()
            saved_moment = optimizer.state[visual.contrastive_projection[0].weight]['exp_avg'].clone()
            with torch.no_grad():
                visual.contrastive_projection[0].weight.zero_()
                optimizer.state[visual.contrastive_projection[0].weight]['exp_avg'].zero_()
            result = scope['load_training_state'](checkpoint, visual, audio, optimizer, criterion, 'cpu')
            assert result == (2, 2, 0.5)
            torch.testing.assert_close(visual.contrastive_projection[0].weight, saved)
            torch.testing.assert_close(optimizer.state[visual.contrastive_projection[0].weight]['exp_avg'], saved_moment)
            assert random.random() == expected_random[0] and np.random.rand() == expected_random[1]
            torch.testing.assert_close(torch.rand(1), expected_random[2])
            restored_visual = VisualNet(backbone='dinov2_vitb14_reg', pretrained=False)
            restored_audio = AudioNet()
            restored_visual.load_state_dict(visual.state_dict(), strict=True)
            restored_audio.load_state_dict(audio.state_dict(), strict=True)
            restored = AudioVisualModel((restored_visual, restored_audio), SimpleNamespace()).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs)['binaural_spectrogram'], reference['binaural_spectrogram'])
            criterion.training_config = {'architecture': 'incompatible'}
            try:
                scope['load_training_state'](checkpoint, visual, audio, optimizer, criterion, 'cpu')
            except ValueError:
                pass
            else:
                raise AssertionError('Incompatible checkpoint accepted')
        print('PASS: shared visual trunk and mono encoder; separate heads; target-free inference, optimizer/RNG resume, strict reload')


def check_batches():
    dataset = torch.utils.data.TensorDataset(torch.arange(7))
    with patch('data.custom_dataset_data_loader.CreateDataset', return_value=dataset):
        loader = CustomDatasetDataLoader()
        loader.initialize(SimpleNamespace(mode='train', batchSize=3, nThreads=0))
        batches = [b[0] for b in loader]
        assert len(batches) == 2 and all(len(b) == 3 for b in batches)
        assert len(torch.cat(batches).unique()) == 6
        loader.initialize(SimpleNamespace(mode='val', batchSize=3, nThreads=0))
        assert isinstance(loader.dataloader.sampler, torch.utils.data.RandomSampler)
        batches = [b[0] for b in loader]
        assert len(batches) == 2 and all(len(b) == 3 for b in batches)
        assert len(torch.cat(batches).unique()) == 6
        loader.initialize(SimpleNamespace(mode='test', batchSize=3, nThreads=0))
        assert torch.equal(torch.cat([b[0] for b in loader]), torch.arange(7))
    dataset = torch.utils.data.TensorDataset(torch.arange(104))
    with patch('data.custom_dataset_data_loader.CreateDataset', return_value=dataset):
        loader.initialize(SimpleNamespace(mode='val', batchSize=16, nThreads=0))
        batches = [b[0] for b in loader]
        assert len(batches) == 6 and all(len(b) == 16 for b in batches)
        assert len(torch.cat(batches).unique()) == 96
    print('PASS: unique full batches; validation shuffles and uses 96/104; test keeps all samples')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(7)
    check_dinov3_decoder_shapes()
    check_formula()
    check_model_and_resume()
    check_batches()
