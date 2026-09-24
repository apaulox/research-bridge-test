"""Formula, shared-head gradient, legacy equivalence and checkpoint tests."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from models.networks import VisualNet, AudioNet
from models.criterion import DenseAVContrastiveLoss


class FakeDino(nn.Module):
    def forward_features(self, x):
        tokens = F.avg_pool2d(x, 14, 14).flatten(2).transpose(1, 2)
        return {'x_norm_patchtokens': tokens.repeat(1, 1, 256)}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    v, a = torch.randn(3, 8, 2, 3), torch.randn(3, 8, 2, 2)
    for norm in (False, True):
        for mode in ('semantic', 'spatial'):
            x, y = v.flatten(2), a.flatten(2)
            if norm:
                x, y = F.normalize(x, dim=1), F.normalize(y, dim=1)
            for order in ('pool_then_product', 'product_then_pool'):
                expected = torch.empty(3, 3)
                for i in range(3):
                    for j in range(3):
                        if order == 'pool_then_product':
                            expected[i, j] = torch.dot(x[i].max(1).values, y[j].mean(1))
                        else:
                            expected[i, j] = (x[i].T @ y[j]).max(0).values.mean()
                loss = DenseAVContrastiveLoss(aggregation_order=order)
                actual = loss.compute_sim_matrix(v, a, norm, mode)
                torch.testing.assert_close(actual, expected)
                perm = torch.tensor([2, 0, 1])
                torch.testing.assert_close(loss.compute_sim_matrix(v, a[perm], norm, mode), actual[:, perm])
    # Explicit counterexample proves the two switches perform distinct operations.
    x = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
    y = torch.ones(1, 2, 1, 1)
    assert DenseAVContrastiveLoss(aggregation_order='pool_then_product').compute_sim_matrix(x, y).item() == 2
    assert DenseAVContrastiveLoss().compute_sim_matrix(x, y).item() == 1
    try:
        DenseAVContrastiveLoss(semantic_pool='visual_attention', aggregation_order='pool_then_product')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid combination accepted')

    with patch('torch.hub.load', side_effect=lambda *args, **kwargs: FakeDino()):
        visual = VisualNet(pretrained=False, head_layout='single')
        audio = AudioNet(head_layout='single')
        images = torch.randn(3, 3, 224, 448)
        heads, decoder = visual(images)
        assert heads.shape == (3, 2, 512, 16, 32)
        assert decoder.shape == (3, 384, 16, 32)
        torch.testing.assert_close(heads[:, 0], heads[:, 1])
        assert not hasattr(visual, 'semantic_proj') and not hasattr(audio, 'spatial_proj')
        mix, stereo = torch.randn(3, 2, 257, 64), torch.randn(3, 4, 257, 64)
        pred, _, asem, aspa = audio(mix, stereo, decoder)
        assert pred.shape == (3, 2, 256, 64) and asem.shape[1] == aspa.shape[1] == 512
        for order in ('pool_then_product', 'product_then_pool'):
            criterion = DenseAVContrastiveLoss(feature_dim=512, aggregation_order=order)
            losses = criterion(heads[:2, 0], asem[:2], heads[:, 1], aspa, s=1,
                               norm_semantic=True, norm_spatial=True)
            for key in ('loss_semantic', 'loss_spatial'):
                grads = torch.autograd.grad(losses[key],
                    (visual.single_proj[0].weight, audio.single_proj[0].weight), retain_graph=True)
                assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), key
        restored = VisualNet(pretrained=False, head_layout='single')
        restored.load_state_dict(visual.state_dict(), strict=True)
        visual.eval(); restored.eval(); audio.eval()
        torch.testing.assert_close(visual(images)[0], restored(images)[0])
        torch.testing.assert_close(audio(mix, stereo, decoder)[0], audio(mix, torch.zeros_like(stereo), decoder)[0])

        # Multi retains the exact original architecture, RNG use and outputs.
        path = Path('/home/huskypaul/DenseSSL_attention/models/networks.py')
        spec = importlib.util.spec_from_file_location('legacy_networks', path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for cls, old_cls, args in ((VisualNet, legacy.VisualNet, {'pretrained': False}),
                                   (AudioNet, legacy.AudioNet, {})):
            torch.manual_seed(42); new = cls(**args, head_layout='multi').eval()
            torch.manual_seed(42); old = old_cls(**args).eval()
            assert new.state_dict().keys() == old.state_dict().keys()
            for key in new.state_dict():
                torch.testing.assert_close(new.state_dict()[key], old.state_dict()[key], rtol=0, atol=0)
            values = (images,) if cls is VisualNet else (mix, stereo, decoder)
            for new_value, old_value in zip(new(*values), old(*values)):
                torch.testing.assert_close(new_value, old_value, rtol=0, atol=0)
    print('PASS: both formulas, candidate symmetry, shared gradients from both losses, shapes, checkpoint reload, inference independence, legacy multi equivalence')


if __name__ == '__main__':
    main()
