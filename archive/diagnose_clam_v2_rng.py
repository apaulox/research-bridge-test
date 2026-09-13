"""Locate baseline runtime variability without modifying submitted inference."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import importlib.util
import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch

root = Path('/home/huskypaul/dacon')
spec = importlib.util.spec_from_file_location('baseline', root / 'baseline/script.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda')
audio = root / 'clam/validation/official_v2/test/AUDIT_0000.wav'
def states():
    return {'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'torch_gpu': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
separator = baseline.load_htdemucs_model()
before = states()
first, _ = baseline.separate_voice_and_music(audio, separator, device)
middle = states()
second, _ = baseline.separate_voice_and_music(audio, separator, device)
after = states()
report = {'separation_max_delta': float(np.max(np.abs(first-second))),
          'separation_changed_rng': [key for key in before if before[key] != middle[key]],
          'separation_second_changed_rng': [key for key in middle if middle[key] != after[key]]}
del separator
torch.cuda.empty_cache()
model, index = baseline.load_df_arena_model(device)
before = states()
score1 = baseline.predict_fake(model, index, first, device)
middle = states()
score2 = baseline.predict_fake(model, index, first, device)
after = states()
report.update(same_model_same_audio_scores=[score1, score2],
              df_changed_rng=[key for key in before if before[key] != middle[key]],
              df_second_changed_rng=[key for key in middle if middle[key] != after[key]])
print(json.dumps(report, indent=2), flush=True)
(Path(__file__).resolve().parents[1] / 'clam_v2_rng_diagnostic.json').write_text(json.dumps(report, indent=2))
