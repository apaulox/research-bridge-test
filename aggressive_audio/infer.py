"""Local EAT-only music + official wav2vec2 probe inference; optional adaptation deltas."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from contracts import fuse
from models import load_eat, load_speech, logits
from run import crop

ROOT = Path(__file__).resolve().parent


def baseline_helpers():
    # Reuse the audited presence/separation operations to isolate model/fusion changes.
    path = ROOT.parent / "script_eat75_musicdet25_fmc.py"
    spec = importlib.util.spec_from_file_location("dacon_base_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = Path("/home/huskypaul/dacon/baseline/model")
    module.PANNS_DIR = base / "panns"
    module.HTDEMUCS_DIR = base / "htdemucs"
    return module


def apply_adaptation(model, path, branch):
    if path:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if obj["branch"] != branch:
            raise ValueError("Adaptation branch mismatch")
        state = model.state_dict()
        delta = obj["state"]
        if not delta or any(k not in state or state[k].shape != v.shape for k, v in delta.items()):
            raise ValueError("Invalid adaptation delta")
        state.update(delta)
        model.load_state_dict(state, strict=True)
    return model.eval()


def probability(model, x, branch, device):
    if np.sqrt(np.mean(x.astype(float)**2)) < 1e-5:
        return 0.0
    with torch.inference_mode():
        return float(logits(model, torch.from_numpy(x.copy()).unsqueeze(0).to(device), branch).float().softmax(-1)[0, 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--music-adaptation", type=Path)
    p.add_argument("--speech-adaptation", type=Path)
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.manual_seed(688)
    device = torch.device("cuda")
    started = time.monotonic()
    base = baseline_helpers()
    files = base.find_audio_files(args.input_dir)
    if not files:
        raise ValueError("No audio inputs")
    if len({f.stem for f in files}) != len(files):
        raise ValueError("Duplicate file IDs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    presence = base.predict_presence_for_all_files(files, device)
    # Process branches sequentially to limit VRAM. There is no cross-file statistic or attention.
    music = apply_adaptation(load_eat(), args.music_adaptation, "music").to(device)
    scores = {}
    for index, path in enumerate(files, 1):
        audio = base.load_audio(path)
        scores[path.stem] = {"ID": path.stem, "MUSIC_FAKE_PROB": probability(music, crop(audio), "music", device)}
        if index % 16 == 0:
            print(json.dumps({"stage": "music", "files_done": index, "total": len(files)}), flush=True)
    del music
    torch.cuda.empty_cache()
    speech = apply_adaptation(load_speech(), args.speech_adaptation, "speech").to(device)
    demucs = base.load_htdemucs_model()
    for index, path in enumerate(files, 1):
        voice, _ = base.separate_voice_and_music(path, demucs, device)
        segment_scores = [probability(speech, base.extract_segment(voice, start), "speech", device)
                          for start in base.get_segment_starts(len(voice))]
        vp, mp = presence[path.stem]
        scores[path.stem].update({"VOICE_FAKE_PROB": max(segment_scores),
                                 "VOICE_PRESENT_PROB": vp, "MUSIC_PRESENT_PROB": mp})
        if index % 16 == 0:
            print(json.dumps({"stage": "speech", "files_done": index, "total": len(files)}), flush=True)
    rows = [scores[path.stem] for path in files]
    (args.output_dir / "component_scores.json").write_text(json.dumps(rows, indent=2))
    for method in ["mean", "max"]:
        with (args.output_dir / f"submission_{method}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID"] + base.PREDICTION_COLUMNS)
            writer.writeheader()
            for row in rows:
                file_probability = fuse(row["VOICE_FAKE_PROB"], row["MUSIC_FAKE_PROB"],
                                        row["VOICE_PRESENT_PROB"], row["MUSIC_PRESENT_PROB"], method)
                writer.writerow(dict(row, FILE_FAKE_PROB=file_probability))
    (args.output_dir / "inference_contract.json").write_text(json.dumps({
        "music": "EAT+AASIST only; original mix first 64000 samples",
        "speech": "AntiDeepfake XLS-R1B+linear probe; Demucs vocals; 64600 samples; segment max",
        "mean_formula": "(VOICE_PRESENT_PROB*VOICE_FAKE_PROB + MUSIC_PRESENT_PROB*MUSIC_FAKE_PROB)/2",
        "comparison": "Identical cached component predictions for mean and max",
        "note": "Local experimental runner; not yet a portable, server-validated submission ZIP",
        "music_adaptation": str(args.music_adaptation), "speech_adaptation": str(args.speech_adaptation),
        "elapsed_seconds": time.monotonic() - started,
        "max_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3
    }, indent=2))
    print(json.dumps({"files": len(rows), "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
