"""Build source-grouped manifests from public corpora, separate from DACON inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from contracts import load_manifest, stable_seed

ROOT = Path(__file__).resolve().parent


def write(rows, name, note):
    folder = ROOT / "manifests"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    audited = load_manifest(path)
    report = {"manifest": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "counts": dict(Counter(f"{r['split']}/{r['label_fake']}/{r['generator']}" for r in audited)),
              "unique_sources": len({r['source_id'] for r in audited}), "notes": note}
    path.with_suffix(".audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def music():
    audio = Path("/home/huskypaul/dacon/musicdet/vendor/MusicDET/datasets/fakemusiccaps/all_audio_wav")
    generators = {"TTM01": "MusicGen_medium", "TTM02": "MusicLDM", "TTM03": "AudioLDM2",
                  "TTM04": "StableAudioOpen", "TTM05": "Mustango"}
    rows = []
    for path in sorted(audio.glob("*.wav")):
        prefix = path.stem[:5]
        fake = prefix in generators and path.stem[5:6] == "_"
        source_id = path.stem[6:] if fake else path.stem
        split_number = stable_seed(688, source_id) % 100
        split = "train" if split_number < 80 else "dev" if split_number < 90 else "holdout"
        rows.append({"id": path.stem, "path": str(path), "source_id": source_id, "split": split,
                     "label_fake": int(fake), "domain": "music", "generator": generators[prefix] if fake else "human",
                     "origin": "public_external", "label_status": "confirmed",
                     "label_scope": "original_FMC_music_authenticity_not_voice_presence_annotation",
                     "source_url": "https://zenodo.org/records/15063698" if fake else "https://huggingface.co/datasets/google/MusicCaps",
                     "license": "FMC CC-BY-NC-4.0 (official Zenodo metadata)" if fake else "MusicCaps metadata CC-BY-SA-4.0; audio source terms retained"})
    write(rows, "music.jsonl", ["All variants of a MusicCaps source share a split.",
                                "Reconstructed local FMC/MusicCaps corpus; NOT exact author EAT training list.",
                                "Upstream checkpoint training overlap is unknown; holdout is only unseen by our adaptation.",
                                "MusicCaps metadata license does not itself license YouTube audio."])


def speech(per_class=0):
    base = ROOT / "assets/asv2019"
    archive_path = base / "LA.zip"
    official_index = None
    if not archive_path.exists():
        audit = json.loads((ROOT / "records/audit-mirror.json").read_text())
        if audit["mismatches"] or audit["common_members"] != audit["official_members"]:
            raise ValueError("Mirror members are not equivalent to official archive")
        official_index = audit["official_index"]
        archive_path = ROOT / "assets/asv2019_mirror/LA.zip"
    with zipfile.ZipFile(archive_path) as zf:
        selected_ids = None
        if per_class:
            selected_ids = set()
            for official_split in ["train", "dev"]:
                name = next(n for n in zf.namelist() if not "/._" in n and
                            Path(n).name.startswith(f"ASVspoof2019.LA.cm.{official_split}.") and n.endswith(".txt"))
                protocol_rows = [line.split() for line in zf.read(name).decode().splitlines()]
                cap = per_class if official_split == "train" else min(per_class, 256)
                for label in ["bonafide", "spoof"]:
                    selected_ids.update(r[1] for r in sorted([r for r in protocol_rows if r[-1] == label],
                                                            key=lambda r: stable_seed(688, r[1]))[:cap])
        keep = [i for i in zf.infolist() if not i.is_dir() and
                not i.filename.startswith("__MACOSX/") and
                ("ASVspoof2019_LA_train/" in i.filename or "ASVspoof2019_LA_dev/" in i.filename
                 or "ASVspoof2019_LA_cm_protocols/" in i.filename)]
        for index, item in enumerate(keep, 1):
            if official_index is not None and official_index.get(Path(item.filename).name) != [item.CRC, item.file_size]:
                continue
            if selected_ids is not None and item.filename.endswith(".flac") and Path(item.filename).stem not in selected_ids:
                continue
            destination = (base / item.filename).resolve()
            if not destination.is_relative_to(base.resolve()):
                raise ValueError("Unsafe archive path")
            if destination.exists():
                if destination.stat().st_size != item.file_size:
                    destination.rename(destination.with_suffix(destination.suffix + ".interrupted"))
                else:
                    continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item) as src, destination.open("xb") as dst:
                while chunk := src.read(1024**2):
                    dst.write(chunk)
            if index % 5000 == 0:
                print(json.dumps({"extracted": index, "total": len(keep)}), flush=True)
    rows = []
    audio_map = {path.stem: path for path in base.rglob("*.flac")}
    for official_split, split in [("train", "train"), ("dev", "dev")]:
        protocol = next(base.rglob(f"ASVspoof2019.LA.cm.{official_split}.*.txt"))
        for line in protocol.read_text().splitlines():
            speaker, file_id, _, attack, label = line.split()
            if selected_ids is not None and file_id not in selected_ids:
                continue
            if file_id not in audio_map:
                raise FileNotFoundError(file_id)
            rows.append({"id": file_id, "path": str(audio_map[file_id]), "source_id": speaker, "split": split,
                         "label_fake": int(label == "spoof"), "domain": "speech", "generator": attack if label == "spoof" else "human",
                         "origin": "public_external", "label_status": "confirmed", "label_scope": "speech_authenticity",
                         "source_url": "https://zenodo.org/records/6906306", "license": "Open Data Commons Attribution; see bundled LICENSE_text.txt"})
    write(rows, "speech.jsonl", ["Official LA train/dev speaker partitions; no PA replay-as-fake mapping.",
                                 "AntiDeepfake post-training may include this corpus; these metrics are adaptation diagnostics.",
                                 "TTS/VC coverage does NOT imply all requested modern cloning/neural-codec generators are covered."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", choices=["music", "speech"])
    parser.add_argument("--per-class", type=int, default=0)
    args = parser.parse_args()
    if args.branch == "speech":
        speech(args.per_class)
    else:
        music()
