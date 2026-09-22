import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import random
import torch
from models.criterion import DenseAVContrastiveLoss


def main():
    # Read only function definitions: never launch train.py's training loop.
    tree = ast.parse(Path('train.py').read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ('create_optimizer', 'save_training_state', 'load_training_state')], type_ignores=[])
    import os
    scope = dict(torch=torch, random=random, np=np, os=os)
    exec(compile(functions, 'train.py', 'exec'), scope)
    torch.manual_seed(42)
    v, a = torch.randn(3, 8, 3, 4), torch.randn(3, 8, 2, 3)
    avg = DenseAVContrastiveLoss(feature_dim=8)
    att = DenseAVContrastiveLoss(semantic_pool='visual_attention', feature_dim=8)
    for norm in (False, True):
        baseline = avg.compute_sim_matrix(v, a, norm)
        actual = att.compute_sim_matrix(v, a, norm)
        torch.testing.assert_close(actual, baseline)
    # Independent historical formula and spatial-path invariance.
    vn = torch.nn.functional.normalize(v.flatten(2), dim=1)
    an = torch.nn.functional.normalize(a.flatten(2), dim=1)
    expected = torch.einsum('bdp,cdq->bcpq', vn, an).max(2).values.mean(2)
    torch.testing.assert_close(avg.compute_sim_matrix(v, a, True), expected)
    att.attention_weight.data.normal_(std=0.1)
    torch.testing.assert_close(att.compute_sim_matrix(v, a, True, 'spatial'),
                               avg.compute_sim_matrix(v, a, True, 'spatial'))
    # Every candidate audio uses the same rule: permuting candidates permutes columns.
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(att.compute_sim_matrix(v, a[order], True),
                               att.compute_sim_matrix(v, a, True)[:, order])
    nets = (torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    opt = SimpleNamespace(lr_visual=2.5e-5, lr_audio=2.5e-4,
        lr_attention=2.5e-5, optimizer='adam', beta1=0.9, weight_decay=0.0005)
    for p in (att.log_temp_sem, att.log_temp_spa):
        p.requires_grad_(False)
    att.ablation_config = {'semantic_pool': 'visual_attention', 'seed': 42}
    optimizer = scope['create_optimizer'](nets, opt, att)
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert any(p is att.attention_weight for p in params)
    assert all(p is not att.log_temp_sem and p is not att.log_temp_spa for p in params)
    optimizer.zero_grad()
    loss = att.contrast_loss(att.compute_sim_matrix(v, a, True), att.temp_sem)
    loss.backward()
    assert torch.isfinite(att.attention_weight.grad).all()
    assert att.attention_weight.grad.abs().sum() > 0
    optimizer.step()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'state.pth')
        scope['save_training_state'](path, 2, 48, 0.5, *nets, optimizer, att)
        saved = att.attention_weight.detach().clone()
        att.attention_weight.data.zero_()
        result = scope['load_training_state'](path, *nets, optimizer, att, 'cpu')
        assert result == (2, 48, 0.5)
        torch.testing.assert_close(att.attention_weight, saved)
        att.ablation_config = {'semantic_pool': 'avg'}
        try:
            scope['load_training_state'](path, *nets, optimizer, att, 'cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched resume was accepted')
    print('PASS: historical avg, uniform attention, spatial invariance, candidate symmetry, gradients, optimizer, resume and mismatch guard')


if __name__ == '__main__':
    main()
