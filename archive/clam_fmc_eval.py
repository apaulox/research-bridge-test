#!/usr/bin/env python3
"""Evaluate the public MoM-CLAM checkpoint on a balanced FMC subset."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from clam_official_probe import CLAM, extract_embedding, load_wave


def equal_error_rate(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def read_items(manifest, real_dir, fake_dir, per_class):
    buckets = {0: [], 1: []}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name, label_name = line.split()
        label = int(label_name == "fake")
        path = (fake_dir if label else real_dir) / name
        if path.is_file() and len(buckets[label]) < per_class:
            buckets[label].append((path, label))
    if any(len(values) < per_class for values in buckets.values()):
        raise RuntimeError({key: len(values) for key, values in buckets.items()})
    items = buckets[0] + buckets[1]
    return sorted(items, key=lambda item: item[0].name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--wav2vec-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--durations", default="4,20,60,90")
    args = parser.parse_args()

    durations = [int(value) for value in args.durations.split(",")]
    items = read_items(args.manifest, args.real_dir, args.fake_dir, args.per_class)
    device = torch.device("cuda")
    started = time.perf_counter()

    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.mert_dir, local_files_only=True, trust_remote_code=True
    )
    mert = AutoModel.from_pretrained(
        args.mert_dir, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    mert_embeddings = {}
    for index, (path, _) in enumerate(items, 1):
        wave = load_wave(path)
        for duration in durations:
            mert_embeddings[(path.name, duration)] = extract_embedding(
                mert, mert_processor, wave, device, duration
            )
        if index % 25 == 0:
            print(f"MERT {index}/{len(items)}", flush=True)
    del mert
    torch.cuda.empty_cache()

    wav_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.wav2vec_dir, local_files_only=True, trust_remote_code=True
    )
    wav2vec = AutoModel.from_pretrained(
        args.wav2vec_dir, local_files_only=True, trust_remote_code=True
    ).to(device).eval()
    clam = CLAM().to(device).eval()
    clam.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )

    rows = []
    for index, (path, label) in enumerate(items, 1):
        wave = load_wave(path)
        row = {"FILE": path.name, "LABEL_FAKE": label}
        for duration in durations:
            wav_embedding = extract_embedding(
                wav2vec, wav_processor, wave, device, duration
            )
            with torch.inference_mode():
                logit = clam(
                    mert_embeddings[(path.name, duration)].unsqueeze(0).to(device),
                    wav_embedding.unsqueeze(0).to(device),
                ).view(-1)[0]
            row[f"LOGIT_{duration}"] = float(logit)
            row[f"PROB_{duration}"] = float(torch.sigmoid(logit))
        rows.append(row)
        if index % 25 == 0:
            print(f"W2V+CLAM {index}/{len(items)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = np.asarray([row["LABEL_FAKE"] for row in rows])
    for duration in durations:
        logits = np.asarray([row[f"LOGIT_{duration}"] for row in rows])
        print(
            f"duration={duration} AUC={roc_auc_score(labels, logits):.6f} "
            f"EER={equal_error_rate(labels, logits):.6f} "
            f"real_mean={logits[labels == 0].mean():.6f} "
            f"fake_mean={logits[labels == 1].mean():.6f} "
            f"min={logits.min():.6f} max={logits.max():.6f}",
            flush=True,
        )
    print(
        f"SAVED {args.output} files={len(rows)} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
