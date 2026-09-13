"""Export the selected inference weights; verify fairseq to torchaudio conversion."""
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

from models import load_speech, DEFAULT_EAT
from infer import apply_adaptation
from run import crop, read_audio

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "submission_eat_xlsr_aug_mean_v1"


def main():
    STAGE.mkdir(exist_ok=True)
    modeldir = STAGE / "model"
    modeldir.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(688)
    original = apply_adaptation(load_speech(), ROOT / "runs/speech_probe_channel_v1/best_adaptation.pt", "speech")
    from torchaudio.models.wav2vec2.utils import import_fairseq_model
    from torchaudio.models.wav2vec2.utils.import_fairseq import _parse_config
    cfg = _parse_config(original.m_ssl.model)
    converted = import_fairseq_model(original.m_ssl.model).eval()
    original.cuda()
    converted.cuda()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    rows = json.loads((ROOT / "fusion_diagnostic/labels.json").read_text())
    samples = [crop(read_audio(r["path"]), samples=64600) for r in rows[32:36]]
    samples += [np.random.default_rng(688).normal(0, .03, n).astype("float32") for n in [64000, 96000]]
    errors = []
    with torch.inference_mode():
        for audio in samples:
            x = torch.from_numpy(audio).unsqueeze(0).cuda()
            ref = original(x)
            features, _ = converted(F.layer_norm(x, (x.shape[-1],)))
            actual = original.proj_fc(features.mean(1))
            print(json.dumps({"conversion_probe":len(audio), "reference":ref.tolist(), "converted":actual.tolist()}), flush=True)
            torch.testing.assert_close(actual, ref, atol=2e-3, rtol=1e-4)
            torch.testing.assert_close(actual.softmax(-1), ref.softmax(-1), atol=1e-4, rtol=1e-4)
            errors.append({"samples": len(audio), "max_logit_error": float((ref-actual).abs().max()),
                           "max_probability_error": float((ref.softmax(-1)-actual.softmax(-1)).abs().max())})
    speechdir = modeldir / "speech_xlsr"
    speechdir.mkdir()
    torch.save({"encoder": {k:v.cpu() for k,v in converted.state_dict().items()},
                "probe": {k:v.cpu() for k,v in original.proj_fc.state_dict().items()}}, speechdir / "model.pt")
    (speechdir / "config.json").write_text(json.dumps(cfg, indent=2))
    (speechdir / "conversion_check.json").write_text(json.dumps(errors, indent=2))
    shutil.copy2(ROOT / "assets/speech_xlsr1b/README.md", speechdir / "OFFICIAL_MODEL_CARD.md")
    del original, converted
    gc.collect()
    torch.cuda.empty_cache()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(DEFAULT_EAT, modeldir / "eat_fmc", ignore=ignore)
    delta = torch.load(ROOT / "runs/music_head_channel_1024_v1/best_adaptation.pt", map_location="cpu", weights_only=True)["state"]
    statepath = modeldir / "eat_fmc/model_state.pt"
    state = torch.load(statepath, map_location="cpu", weights_only=True)
    assert all(k in state and state[k].shape == v.shape for k,v in delta.items())
    state.update(delta)
    torch.save(state, statepath)
    for name in ["panns", "htdemucs"]:
        shutil.copytree(Path("/home/huskypaul/dacon/baseline/model") / name, modeldir / name, ignore=ignore)
    print(json.dumps({"stage":str(STAGE), "conversion_errors":errors}), flush=True)


if __name__ == "__main__":
    main()
