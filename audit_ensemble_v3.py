"""CPU contract checks and accurate replacement metadata before packaging."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

D=Path('/home/huskypaul/dacon')
S=D/'submissions/eat75_clam25_fmc_v3'
V=D/'clam/validation/ensemble_v3'
spec=importlib.util.spec_from_file_location('v3',S/'script.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
head=m.load_clam_head(torch.device('cpu'))
results={}
for count in [1,3,16,17,33]:
    generator=torch.Generator().manual_seed(123)
    files=[Path(f'{i:03d}.wav') for i in range(count)]
    a={p.stem:torch.randn(13,768,generator=generator) for p in files}
    b={p.stem:torch.randn(13,768,generator=generator) for p in files}
    state=torch.get_rng_state().clone()
    actual=m.predict_clam_batch_logits(head,files,a,b,torch.device('cpu'))
    assert torch.equal(state,torch.get_rng_state())
    reordered=m.predict_clam_batch_logits(head,list(reversed(files)),a,b,torch.device('cpu'))
    assert actual==reordered
    permutation=torch.randperm(count,generator=torch.Generator().manual_seed(42)).tolist()
    expected={}
    with torch.inference_mode():
        for start in range(0,count,16):
            batch=[files[i] for i in permutation[start:start+16]]
            values=head(torch.stack([a[p.stem] for p in batch]),torch.stack([b[p.stem] for p in batch])).view(-1).tolist()
            expected.update({p.stem:v for p,v in zip(batch,values)})
    assert actual==expected
    results[str(count)]='exact raw head match; reorder invariant; global RNG unchanged'
for path in (S/'model').rglob('*.py'):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
        modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(name=='timm' or name.startswith('timm.') for name in modules),path
asset_count=0
for folder in ['df_arena_1b','htdemucs','panns','eat_fmc']:
    for path in (S/'model'/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            original=D/'submissions/eat75_clam25_fmc_v2'/path.relative_to(S)
            assert os.path.samefile(path,original),path
            asset_count+=1
info='''EAT75 + CLAM25 FMC v3
Source: eat75_clam25_fmc_v2; EAT weights and vendored runtime unchanged (no timm dependency).
CLAM official frozen MoM checkpoint; sequence-first attention unchanged.
Head inference: batch 16, sorted IDs then local torch randperm seed 42, retain partial final batch.
This reproduces official batching and retains cross-file attention; it is not a batch-axis repair.
Input remains v2: original mix decoded to 16k, MERT resampled to 24k, W2V 16k, first 90s / zero padding.
This differs from official native-rate extraction and is intentional to preserve the v2 preprocessing comparison.
Calibration: model/clam/fmc_batch16_calibration.json, logistic C=1 lbfgs fit ONLY on TTM01 dev 100 real + 100 fake.
Separate TTM01-TTM05 eval is used for reporting only; grouping seed and 75:25 weights fixed in advance.
Music output: sigmoid(0.75*logit(EAT) + 0.25*logit(calibrated CLAM)), probabilities clipped at 1e-5.
Voice, presence, Demucs, FILE=max(VP*VF,MP*MF), submission interface and requirements preserved.
FILE values may change because Music changes. No DACON score is available for v3.
See ENSEMBLE_V3_INFO.json for calibration and local evaluation.
'''
# Replace the directory entry rather than mutate a file hardlinked to v2.
temporary=S/'model/CLAM_EAT_INFO.v3.tmp'
temporary.write_text(info)
temporary.replace(S/'model/CLAM_EAT_INFO.txt')
report=dict(batch_contract=results,unchanged_asset_files=asset_count,source_timm_imports=0)
(V/'contract_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
