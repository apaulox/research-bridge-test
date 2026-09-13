"""Construct a small, labelled external FILE diagnostic; never uses DACON inputs."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from augment import overlay_voice
from contracts import load_manifest, stable_seed
from run import crop, read_audio, subset

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / "fusion_diagnostic"
    folder.mkdir(exist_ok=False)
    audio_dir = folder / "audio"
    audio_dir.mkdir()
    music_rows = load_manifest(ROOT / "manifests/music.jsonl")
    music = {label: sorted([r for r in music_rows if r["split"] == "holdout" and r["label_fake"] == label],
                           key=lambda r: stable_seed("fusion-v1", r["id"]))[:8] for label in [0, 1]}
    speech_rows = load_manifest(ROOT / "manifests/speech.jsonl")
    used_dev = {r["id"] for r in subset(speech_rows, "dev", 128, 688)}
    speech = {label: sorted([r for r in speech_rows if r["split"] == "dev" and r["label_fake"] == label and r["id"] not in used_dev],
                            key=lambda r: stable_seed("fusion-v1", r["id"]))[:8] for label in [0, 1]}
    if any(len(pool) != 8 for pool in list(music.values()) + list(speech.values())):
        raise ValueError("Insufficient external source examples")
    records = []

    def save(x, label, scenario, sources):
        file_id = f"FUS_{len(records):04d}"
        path = audio_dir / (file_id + ".wav")
        sf.write(path, x, 16000, subtype="FLOAT")
        records.append({"ID": file_id, "path": str(path), "label_file_fake": label,
                        "scenario": scenario, "sources": sources})

    for voice_label in [0, 1]:
        for music_label in [0, 1]:
            for i in range(8):
                vr, mr = speech[voice_label][i], music[music_label][i]
                x, _ = overlay_voice(crop(read_audio(mr["path"])), crop(read_audio(vr["path"])),
                                     np.random.default_rng(stable_seed("fusion-mix", i)))
                save(x, int(voice_label or music_label), f"voice{voice_label}_music{music_label}", [vr["id"], mr["id"]])
    for label in [0, 1]:
        for row in speech[label]:
            save(crop(read_audio(row["path"])), label, "speech_only", [row["id"]])
        for row in music[label]:
            # An original music clip can contain vocals. Do NOT assert voice absence.
            save(crop(read_audio(row["path"])), label, "original_music_clip", [row["id"]])
    (folder / "labels.json").write_text(json.dumps(records, indent=2))
    (folder / "scope.json").write_text(json.dumps({
        "purpose": "FILE mean/max diagnostic, not DACON or SOTA validation",
        "files": len(records), "source_reuse": "Sources reused in paired mixtures; not 64 independent original recordings",
        "split": "FMC source holdout; ASV dev utterances outside branch model-selection subset",
        "limits": ["Upstream checkpoint overlap unknown", "ASV speakers may match model-selection dev speakers",
                   "No component presence ground truth inferred from original music clips",
                   "CoSG/cloning generalization is not evaluated by this set"]}, indent=2))
    print(json.dumps({"files": len(records), "folder": str(folder)}), flush=True)


if __name__ == "__main__":
    main()
