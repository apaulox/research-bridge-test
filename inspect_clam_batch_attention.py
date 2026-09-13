"""Read-only check of whether official CLAM predictions depend on batch peers."""
import importlib.util
import json
from pathlib import Path
import torch

root = Path('/home/huskypaul/dacon/clam/MoM-CLAM')
spec = importlib.util.spec_from_file_location('official_clam', root / 'models/clam.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.CLAM(13, 13, 768, 768).eval()
checkpoint = next(root.rglob('best_model_triplet_loss_margin_0.2.pth'))
model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
torch.manual_seed(42)
mert = torch.randn(16, 13, 768)
wav = torch.randn(16, 13, 768)
with torch.inference_mode():
    alone = model(mert[:1], wav[:1]).item()
    batched = model(mert, wav)[0].item()
report = dict(batch_first=model.crossAttention1.batch_first,
              attention_input_shape=[16, 1, 768],
              same_sample_logit_alone=alone,
              same_sample_logit_in_batch16=batched,
              absolute_difference=abs(alone-batched),
              caveat='Synthetic embeddings; proves batch dependence, not its DACON score impact')
print(json.dumps(report, indent=2))
(Path(__file__).resolve().parent / 'clam_batch_attention_check.json').write_text(json.dumps(report, indent=2))
