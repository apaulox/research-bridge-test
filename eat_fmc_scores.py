#!/usr/bin/env python3
"""Score the same balanced FMC subset with the bundled EAT-FMC detector."""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import torch

from clam_fmc_eval import read_items
from clam_official_probe import load_wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, required=True)
    parser.add_argument("--eat-dir", type=Path, required=True)
    parser.add_argument("--ensemble-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("eat_bundle", args.ensemble_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EAT_DIR = args.eat_dir

    items = read_items(args.manifest, args.real_dir, args.fake_dir, args.per_class)
    device = torch.device("cuda")
    model = module.load_eat_model(device)
    rows = []
    started = time.perf_counter()
    for index, (path, label) in enumerate(items, 1):
        probability = module.predict_eat_fake(model, load_wave(path), device)
        rows.append(
            {"FILE": path.name, "LABEL_FAKE": label, "EAT_PROB": probability}
        )
        if index % 25 == 0:
            print(f"EAT {index}/{len(items)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"SAVED {args.output} files={len(rows)} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
