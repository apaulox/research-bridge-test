"""EAT-only music + adapted XLS-R probe; presence-gated mean FILE fusion."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model/runtime"))

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model
import audio_helpers as base
from eat_loader import load_eat


def probability(model, audio, device, speech=False):
    if np.sqrt(np.mean(audio.astype(float) ** 2)) < 1e-5:
        return 0.0
    x = torch.from_numpy(audio.copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        if speech:
            encoder, probe = model
            features, _ = encoder(F.layer_norm(x, (x.shape[-1],)))
            logits = probe(features.mean(1))
        else:
            logits = model(x)["logits"]
        return float(logits.float().softmax(-1)[0, 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, default=Path("data/test"))
    p.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    p.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(688)
    # The fused SDPA kernel changes this deep encoder's scores; match the verified author computation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = base.select_device(args.device)
    base.PANNS_DIR = ROOT / "model/panns"
    base.HTDEMUCS_DIR = ROOT / "model/htdemucs"
    columns, rows = base.read_sample_submission(args.sample_submission)
    files = base.order_audio_files(base.find_audio_files(args.test_dir), rows)
    started = time.monotonic()
    presence = base.predict_presence_for_all_files(files, device)
    music = load_eat(ROOT / "model/eat_fmc").to(device).eval()
    scores = {}
    for path in files:
        audio = base.load_audio(path)
        if len(audio) < 64000:
            audio = np.tile(audio, (64000 + len(audio) - 1) // len(audio))
        scores[path.stem] = {"MUSIC_FAKE_PROB": probability(music, audio[:64000], device)}
    del music
    if device.type == "cuda":
        torch.cuda.empty_cache()
    folder = ROOT / "model/speech_xlsr"
    encoder = wav2vec2_model(**json.loads((folder / "config.json").read_text()), aux_num_out=None)
    probe = torch.nn.Linear(1280, 2)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["encoder"], strict=True)
    probe.load_state_dict(state["probe"], strict=True)
    del state
    speech = (encoder.to(device).eval(), probe.to(device).eval())
    demucs = base.load_htdemucs_model()
    for index, path in enumerate(files, 1):
        voice, _ = base.separate_voice_and_music(path, demucs, device)
        vf = max(probability(speech, base.extract_segment(voice, start), device, True)
                 for start in base.get_segment_starts(len(voice)))
        vp, mp = presence[path.stem]
        mf = scores[path.stem]["MUSIC_FAKE_PROB"]
        scores[path.stem].update(VOICE_FAKE_PROB=vf, VOICE_PRESENT_PROB=vp, MUSIC_PRESENT_PROB=mp,
                                 FILE_FAKE_PROB=(vp * vf + mp * mf) / 2)
        if index % 16 == 0:
            print(f"Completed {index}/{len(files)}", flush=True)
    for row in rows:
        row.update(scores[row["ID"]])
        values = np.array([row[c] for c in base.PREDICTION_COLUMNS], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Invalid output probabilities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"files":len(files), "seconds":time.monotonic()-started, "output":str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
