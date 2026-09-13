"""Train vocal and accompaniment branches with Demucs-prepared external data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

from contracts import load_manifest
from infer import baseline_helpers
from run import subset

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "stem_experiment_medium_v1"
BRANCHES = {"music":("music_head_channel_1024_v1", "music.jsonl", 64000),
            "speech":("speech_probe_channel_v1", "speech_extended.jsonl", 64000)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare():
    WORK.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(688)
    helper = baseline_helpers()
    model = helper.load_htdemucs_model()
    checkpoint = helper.HTDEMUCS_DIR / "955717e8-8726e21a.th"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    started = time.monotonic()
    for branch, (old, manifest_name, samples) in BRANCHES.items():
        raw_rows = load_manifest(ROOT / "manifests" / manifest_name)
        rows = subset(raw_rows, "train", 512, 688, True) + subset(raw_rows, "dev", 128, 688)
        assert len(rows) == 1280
        previous = json.loads((ROOT / "runs" / old / "config.json").read_text())
        for split, key in [("train", "selected_training_ids"), ("dev", "selected_dev_ids")]:
            assert sorted(r["id"] for r in rows if r["split"] == split) == sorted(previous[key]), key
        assert previous["samples"] == samples == 64000
        write_json(WORK / (branch + "_controlled_settings.json"), {
            "reference_run":old, "same_train_dev_ids":True, "same_initial_official_model":True,
            "same_augmentation":"channel, 25% clean, 1-3 transforms",
            "same_training":{"seed":688,"epochs":2,"batch_size":2,"accumulate":8,"lr":1e-4,
                             "last_layers":0,"samples":64000,"voice_overlay":False},
            "only_intended_change":"Demucs accompaniment input" if branch == "music" else "Demucs vocal input"})
        folder = ROOT / "stem_experiment_v1" / branch
        folder.mkdir(parents=True, exist_ok=True)
        prepared = []
        for i, row in enumerate(rows, 1):
            if (WORK / "STOP").exists():
                print("Stopped preprocessing; completed per-file caches are reusable.", flush=True)
                return False
            path = Path(row["path"])
            key = hashlib.sha256(row["id"].encode()).hexdigest()
            audio = folder / (key + ".wav")
            metadata = folder / (key + ".json")
            signature = {"source":str(path), "source_size":path.stat().st_size,
                         "source_mtime_ns":path.stat().st_mtime_ns, "demucs_sha256":checkpoint_sha,
                         "branch":branch, "stem":"vocals" if branch == "speech" else "drums+bass+other",
                         "shifts":0, "overlap":0.25, "normalize":"mono mean/std, reverse on each source",
                         "sample_rate":16000, "channels":1}
            cached = json.loads(metadata.read_text()) if metadata.exists() else None
            if not audio.exists() or cached is None or cached["signature"] != signature:
                voice, music = helper.separate_voice_and_music(path, model, torch.device("cuda"))
                x = voice if branch == "speech" else music
                if not len(x) or not np.isfinite(x).all():
                    raise ValueError(f"Invalid Demucs output: {row['id']}")
                temp = audio.with_suffix(".wav.tmp")
                sf.write(temp, x, 16000, subtype="FLOAT", format="WAV")
                temp.replace(audio)
                write_json(metadata, {"signature":signature,"samples":len(x),
                                      "rms":float(np.sqrt(np.mean(x.astype(float)**2)))})
            prepared.append(dict(row, path=str(audio), original_audio_path=str(path),
                                 processing=signature, label_scope="original source authenticity retained on its relevant separated stem"))
            if i % 32 == 0 or i == len(rows):
                progress = {"phase":"separate", "branch":branch,"done":i,"total":len(rows),
                            "seconds":time.monotonic()-started}
                write_json(WORK / "progress.json", progress)
                print(json.dumps(progress),flush=True)
        destination = WORK / (branch + ".jsonl")
        temp = destination.with_suffix(".jsonl.tmp")
        temp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in prepared),encoding="utf-8")
        temp.replace(destination)
        load_manifest(destination)
    del model
    torch.cuda.empty_cache()
    return True


def main():
    if not prepare():
        return
    write_json(WORK / "contract.json", {
        "voice":"Demucs vocals -> XLS-R fixed -> temporal mean -> trainable linear probe",
        "music":"Demucs drums+bass+other -> EAT fixed -> trainable AASIST+Linear",
        "training":"Same official initialization, frozen encoders, seed/lr/batch/accumulation/2 epochs and 64000-sample crops as previous selected runs",
        "augmentation":"Original channel augmentation: 25% clean, otherwise 1-3 transforms, applied after separation; no voice overlay",
        "stress_scope":"Post-separation channel stress, not a test of separator robustness to noisy mixtures",
        "cohorts":"Same source-group splitting and sample selection as earlier runs: 1024 train, 256 dev per branch, 2 epochs",
        "limitations":"Small external adaptation diagnostic; upstream overlap unresolved; not a DACON score",
        "fusion":"mean retained; CPS ResNet38 unchanged; no new ZIP requested at this stage"})
    for branch, (old, _, samples) in BRANCHES.items():
        name = f"{branch}_demucs_medium_v1"
        run = ROOT / "runs" / name
        if (run / "complete.json").exists():
            continue
        if (WORK / "STOP").exists():
            return
        reference_cfg = json.loads((ROOT / "runs" / old / "config.json").read_text())
        expected = {"seed":688,"epochs":2,"batch_size":2,"accumulate":8,"lr":1e-4,
                    "last_layers":0,"samples":64000,"voice_overlay":False,"aug":"channel"}
        assert all(reference_cfg[k] == v for k,v in expected.items()), "Reference settings differ"
        cmd = [sys.executable,"-B",str(ROOT/"run.py"),"train","--branch",branch,"--name",name,
               "--manifest",str(WORK/(branch+".jsonl")),"--epochs","2","--aug","channel",
               "--train-per-class","512","--dev-per-class","128","--balance-generators",
               "--samples",str(samples),"--seed","688","--lr","0.0001",
               "--batch-size","2","--accumulate","8","--last-layers","0"]
        if run.exists():
            cmd.append("--resume")
        write_json(WORK / "progress.json", {"phase":"train","branch":branch,"run":name})
        print(json.dumps({"training":name}),flush=True)
        with (WORK/(name+".log")).open("a") as log:
            result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print((WORK/(name+".log")).read_text()[-3000:],flush=True)
            raise RuntimeError(f"Training failed: {name}")
        if not (run/"complete.json").exists():
            return
    results = {b:json.loads((ROOT/"runs"/f"{b}_demucs_medium_v1"/"history.json").read_text()) for b in BRANCHES}
    write_json(WORK/"training_results.json",results)
    write_json(WORK/"progress.json",{"phase":"training_complete"})
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()
