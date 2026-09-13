"""Manifest and per-file prediction contracts; independent of neural model libraries."""
import hashlib
import json
from pathlib import Path

import numpy as np


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def load_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    ids = set()
    groups = {}
    paths = {}
    for row in rows:
        for field in ["id", "path", "source_id", "split", "label_fake", "domain", "generator", "source_url", "license"]:
            if field not in row or row[field] is None or row[field] == "":
                raise ValueError(f"Missing required field {field}: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample id {row['id']}")
        ids.add(row["id"])
        if row.get("origin") != "public_external":
            raise ValueError("Only documented public external data may enter training")
        if row["label_fake"] not in [0, 1] or row["split"] not in ["train", "dev", "holdout"]:
            raise ValueError("Invalid label or split")
        path = Path(row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Deny known competition evaluation/sample inputs as an additional guard.
        if any(token in path.as_posix().lower() for token in ["/dacon/baseline/data/", "/data/test/", "/open/test/"]):
            raise ValueError(f"Competition input cannot enter a training manifest: {path}")
        for key, mapping in [(row["domain"] + ":" + row["source_id"], groups), (str(path), paths)]:
            previous = mapping.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split leakage: {key}")
        if row.get("label_status", "confirmed") != "confirmed":
            raise ValueError("Ambiguous generation/reconstruction labels must be quarantined")
    return rows


def fuse(voice_fake, music_fake, voice_present, music_present, method="mean"):
    values = np.asarray([voice_fake, music_fake, voice_present, music_present], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("All four probabilities must be finite and in [0,1]")
    a, b = voice_fake * voice_present, music_fake * music_present
    if method == "mean":
        return float((a + b) / 2)
    if method == "max":
        return float(max(a, b))
    raise ValueError(method)


def eer(labels, scores):
    from sklearn.metrics import roc_curve
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels.tolist()) != {0, 1}:
        return None
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite score")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)
