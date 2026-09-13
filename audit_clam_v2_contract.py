"""Audit baseline parity and the two possible CLAM batch interpretations."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import torch

workspace = Path(__file__).resolve().parent
dacon = Path('/home/huskypaul/dacon')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    baseline = ast.parse((dacon / 'baseline/script.py').read_text())
    v1 = ast.parse((workspace / 'script_clam_official.py').read_text())
    old = {n.name: ast.dump(n) for n in baseline.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n) for n in v1.body if isinstance(n, ast.FunctionDef)}
    intentional = {'parse_arguments', 'predict_fake_scores_for_all_files', 'main'}
    baseline_diff = [name for name in old if old[name] != new.get(name)]
    assert set(baseline_diff) == intentional, baseline_diff
    official = load('official', dacon / 'clam/MoM-CLAM/models/clam.py')
    submission = load('submission', workspace / 'script_clam_official.py')
    model = official.CLAM(13, 13, 768, 768).eval()
    checkpoint = dacon / 'submissions/clam_official_v1/model/clam/best_model_triplet_loss_margin_0.2.pth'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    independent = copy.deepcopy(model)
    independent.crossAttention1.batch_first = True
    independent.crossAttention2.batch_first = True
    port = submission.CLAM().eval()
    port.load_state_dict(model.state_dict())
    torch.manual_seed(42)
    mert, wav = torch.randn(16, 13, 768), torch.randn(16, 13, 768)
    with torch.inference_mode():
        old_single = torch.cat([model(mert[i:i+1], wav[i:i+1]) for i in range(16)])
        fixed_single = torch.cat([independent(mert[i:i+1], wav[i:i+1]) for i in range(16)])
        fixed_batch = independent(mert, wav)
        official_batch = model(mert, wav)
        port_batch = port(mert, wav)
        training_batch = model.forward_training(mert, wav)[0]
    torch.testing.assert_close(old_single, fixed_single, rtol=0, atol=0)
    torch.testing.assert_close(old_single, fixed_batch, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(official_batch, port_batch, rtol=0, atol=0)
    torch.testing.assert_close(official_batch, training_batch, rtol=0, atol=0)
    result = {
        'baseline_identical_functions': [name for name in old if name not in baseline_diff],
        'baseline_intentional_differences': baseline_diff,
        'axis_fix_single_vs_v1_max_logit_delta': (old_single-fixed_single).abs().max().item(),
        'axis_fix_batch16_vs_v1_single_max_logit_delta': (old_single-fixed_batch).abs().max().item(),
        'official_batch16_vs_v1_single_max_logit_delta': (official_batch-old_single).abs().max().item(),
        'port_vs_official_batch16_max_logit_delta': (official_batch-port_batch).abs().max().item(),
        'official_forward_vs_training_max_logit_delta': (official_batch-training_batch).abs().max().item(),
        'interpretation': 'Axis correction preserves v1 per-file predictions; batch16 reproduces published head behavior but retains peer dependence.'
    }
    (workspace / 'clam_v2_contract_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
