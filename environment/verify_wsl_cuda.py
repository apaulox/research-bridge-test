"""Verify real CUDA operations and MusicDET training mechanics, not accuracy."""
import json
from pathlib import Path
import sys
import torch
import torchaudio

assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.version.cuda == '12.8', torch.version.cuda
torch.manual_seed(688)
torch.cuda.reset_peak_memory_stats()
matrix = torch.randn(512, 512, device='cuda')
product = matrix @ matrix.T
assert torch.isfinite(product).all()
waveform = torch.randn(2, 64600, device='cuda')
resampled = torchaudio.functional.resample(waveform, 16000, 32000)
assert resampled.shape == (2, 129200)
sys.path.insert(0, str(Path.home() / 'dacon/musicdet'))
from vendor.MusicDET.model import SpecNF
model = SpecNF(K=2, L=1, R=5).cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
labels = torch.zeros(2, dtype=torch.long, device='cuda')
_, nll = model(waveform, labels)
assert torch.isfinite(nll).all()
nll.mean().backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
model.eval()
with torch.inference_mode():
    _, evaluation = model(waveform, labels)
assert torch.isfinite(evaluation).all()
torch.cuda.synchronize()
report = dict(python=sys.version, torch=torch.__version__, torchaudio=torchaudio.__version__,
    cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
    capability=torch.cuda.get_device_capability(), arch_list=torch.cuda.get_arch_list(),
    cuda_matmul=True, cuda_resample=True, musicdet_forward_backward=True,
    musicdet_eval=True, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
    note='MusicDET used random weights and synthetic audio only. No trained detector or performance claim.')
print(json.dumps(report, indent=2))
(Path.home() / 'dacon/logs/cuda_verification.json').write_text(json.dumps(report, indent=2))
