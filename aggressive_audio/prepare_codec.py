"""Add documented CoSG generations, retaining original labels and corpus separation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from contracts import load_manifest
from prepare import write

ROOT = Path(__file__).resolve().parent


def main():
    base = ROOT / "assets/codecfake_cosg"
    extraction = base / "audio"
    extraction.mkdir(exist_ok=True)
    with tarfile.open(base / "CodecFake_plus_CoSG.tar.xz", "r:xz") as archive:
        for item in archive:
            if not item.isfile() or Path(item.name).suffix.lower() not in {".wav", ".flac"}:
                continue
            path = (extraction / item.name).resolve()
            if not path.is_relative_to(extraction.resolve()):
                raise ValueError("Unsafe archive path")
            if path.exists():
                if path.stat().st_size != item.size:
                    raise ValueError("Incomplete existing extraction")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as src, path.open("xb") as dst:
                while chunk := src.read(1024**2):
                    dst.write(chunk)
    audio = {}
    for path in extraction.rglob("*"):
        if path.suffix.lower() in {".wav", ".flac"}:
            if path.stem in audio:
                raise ValueError(f"Duplicate CoSG audio id: {path.stem}")
            audio[path.stem] = path
    rows = load_manifest(ROOT / "manifests/speech.jsonl")
    added = []
    for line in (base / "CoSG_labels.txt").read_text().splitlines():
        generator, clip_id, quantizer, auxiliary, decoder, label = line.split()
        label = label.lower()
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"Unrecognized CoSG label: {label}")
        path = audio.get(clip_id)
        if not path:
            raise FileNotFoundError(clip_id)
        added.append({"id": "CoSG_" + clip_id, "path": str(path),
                      # Original prompt/speaker linkage isn't supplied in this protocol.
                      # Keep the entire small corpus in train instead of pretending a random split is independent.
                      "source_id": "CoSG_source_linkage_unresolved_training_only", "split": "train",
                      "label_fake": int(label == "spoof"), "domain": "speech",
                      "generator": generator if label == "spoof" else "human_CoSG",
                      "origin": "public_external", "label_status": "confirmed",
                      "label_scope": "CoSG original generated speech label; no CoRS relabelling",
                      "source_url": "https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset",
                      "license": "MIT on official dataset card; individual upstream demonstration provenance retained",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "codec_quantizer": quantizer, "auxiliary": auxiliary, "decoder": decoder})
    write(rows + added, "speech_extended.jsonl", [
        "ASVspoof LA official train/dev + CoSG labelled real and generated speech in train only.",
        "CoSG includes 17 source-model groups; final files selected per run are recorded in config.",
        "CoSG source prompt/speaker mapping is incomplete; no random CoSG dev split or unseen-generator claim.",
        "ASV-only dev cannot measure modern codec/cloning generalization; separate external evaluation remains necessary."])
    print(json.dumps({"added": len(added), "counts": dict(Counter(f"{r['label_fake']}/{r['generator']}" for r in added))}, indent=2))


if __name__ == "__main__":
    main()
