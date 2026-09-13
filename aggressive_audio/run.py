"""Train controlled augmentation experiments. All outputs are local experimental artifacts."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from augment import AugmentConfig, augment, overlay_voice, transform
from contracts import eer, load_manifest, stable_seed

ROOT = Path(__file__).resolve().parent


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        divisor = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return np.asarray(audio, dtype=np.float32)


def crop(audio, rng=None, samples=64000):
    if len(audio) < samples:
        return np.tile(audio, (samples + len(audio) - 1) // len(audio))[:samples].copy()
    start = int(rng.integers(len(audio) - samples + 1)) if rng is not None else 0
    return audio[start:start + samples].copy()


def subset(rows, split, per_class, seed, balance_generators=False):
    selected = []
    for label in [0, 1]:
        pool = sorted([r for r in rows if r["split"] == split and r["label_fake"] == label],
                      key=lambda r: stable_seed(seed, r["id"]))
        if balance_generators:
            from collections import defaultdict, deque
            groups = defaultdict(deque)
            for row in pool:
                groups[row["generator"]].append(row)
            pool = []
            names = sorted(groups)
            while any(groups.values()):
                for name in names:
                    if groups[name]:
                        pool.append(groups[name].popleft())
        selected.extend(pool[:per_class] if per_class else pool)
    if {r["label_fake"] for r in selected} != {0, 1}:
        raise ValueError(f"Both classes required in {split}")
    return sorted(selected, key=lambda r: r["id"])


def validate(model, rows, branch, device, batch_size, stress=False, samples=64000):
    from models import logits
    records = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            waveforms = []
            for row in chunk:
                x = crop(read_audio(row["path"]), samples=samples)
                if stress:
                    rng = np.random.default_rng(stable_seed("fixed-validation-stress-v1", row["id"]))
                    x, _ = augment(x, rng, AugmentConfig(clean_probability=0))
                waveforms.append(x)
            x = torch.from_numpy(np.stack(waveforms)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                score = logits(model, x, branch).float().softmax(-1)[:, 0].cpu().tolist()
            records.extend({"id": row["id"], "label": row["label_fake"], "prob_fake": s,
                            "generator": row["generator"], "stress": stress} for row, s in zip(chunk, score))
    return eer([r["label"] for r in records], [r["prob_fake"] for r in records]), records


def train(args):
    from models import load_eat, load_speech, configure_training, train_mode, logits, adaptation_state
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    device = torch.device("cuda")
    manifest = Path(args.manifest) if args.manifest else ROOT / "manifests" / f"{args.branch}.jsonl"
    rows = load_manifest(manifest)
    training = subset(rows, "train", args.train_per_class, args.seed, args.balance_generators)
    development = subset(rows, "dev", args.dev_per_class, args.seed)
    speech_pool = []
    if args.voice_overlay:
        if args.branch != "music":
            raise ValueError("voice_overlay is a music-specific nuisance experiment")
        speech_manifest = Path(args.overlay_manifest) if args.overlay_manifest else ROOT / "manifests/speech.jsonl"
        speech_pool = [r for r in load_manifest(speech_manifest) if r["split"] == "train"]
    output = ROOT / "runs" / args.name
    augmentation_config = (AugmentConfig(clean_probability=0.5, max_transforms=1,
                           transforms=("noise", "gain", "resample"), light=True)
                           if args.aug == "light" else AugmentConfig())
    if args.resume:
        if not (output / "checkpoint_latest.pt").is_file():
            raise FileNotFoundError("No full training checkpoint. Legacy adaptation.pt files contain weights only.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    cfg = vars(args).copy()
    cfg.update({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selected_training_ids": [r["id"] for r in training], "selected_dev_ids": [r["id"] for r in development],
                "augmentation": asdict(augmentation_config), "class_order": ["fake", "real"],
                "selection": "mean of fixed clean/stress dev EER", "test_data_used": False,
                "metric_scope": "external adaptation diagnostic; upstream checkpoint overlap possible",
                "sample_rate": 16000, "samples": args.samples, "inference_crop": "first",
                "fusion": "not used in branch training; compare max vs mean with identical cached scores"})
    if not args.resume:
        (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    model = (load_eat() if args.branch == "music" else load_speech()).to(device)
    modules = configure_training(model, args.branch, args.last_layers)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    print(json.dumps({"event": "loaded", "branch": args.branch, "trainable_parameters": sum(p.numel() for p in parameters),
                      "train_rows": len(training), "dev_rows": len(development)}), flush=True)
    history, best = [], float("inf")
    trace_counts = Counter()
    started = time.monotonic()
    first_epoch, resume_step, resume_losses, prior_seconds = 0, 0, [], 0.0
    if args.resume:
        saved = torch.load(output / "checkpoint_latest.pt", map_location="cpu", weights_only=True)
        for key in ["branch", "aug", "seed", "lr", "batch_size", "accumulate", "last_layers", "voice_overlay",
                    "manifest_sha256", "selected_training_ids", "selected_dev_ids", "balance_generators", "samples"]:
            if saved["run_config"].get(key) != cfg.get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        current = model.state_dict()
        current.update(saved["state"])
        model.load_state_dict(current, strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best = saved["history"], saved["best"]
        trace_counts.update(saved["augmentation_counts"])
        first_epoch, resume_step = saved["next_epoch"], saved["next_step"]
        resume_losses, prior_seconds = saved["running_losses"], saved["elapsed_seconds"]
        restore_rng(saved["rng"])
        print(json.dumps({"resumed_epoch": first_epoch, "next_step": resume_step}), flush=True)

    def save_training(next_epoch, next_step, running_losses):
        # Called only after optimizer.step/zero_grad, so no partial accumulated gradients are omitted.
        payload = {"format_version": 1, "state": adaptation_state(model, modules),
                   "optimizer": optimizer.state_dict(), "rng": rng_state(), "run_config": cfg,
                   "next_epoch": next_epoch, "next_step": next_step, "running_losses": running_losses,
                   "history": history, "best": best, "augmentation_counts": dict(trace_counts),
                   "elapsed_seconds": prior_seconds + time.monotonic() - started}
        atomic_torch_save(payload, output / "checkpoint_latest.pt")
        (output / "resume_status.json").write_text(json.dumps({"next_epoch": next_epoch, "next_step": next_step,
              "checkpoint": "checkpoint_latest.pt", "optimizer_and_rng_saved": True}, indent=2))

    for epoch in range(first_epoch, args.epochs + 1):
        losses = resume_losses if epoch == first_epoch else []
        if epoch:
            train_mode(model, modules)
            # Same file order and crop RNG for augmented and unaugmented arms.
            order = np.random.default_rng(stable_seed(args.seed, "order", epoch)).permutation(len(training))
            optimizer.zero_grad(set_to_none=True)
            batch_count = (len(order) + args.batch_size - 1) // args.batch_size
            for step, start in enumerate(range(0, len(order), args.batch_size)):
                if epoch == first_epoch and step < resume_step:
                    continue
                chunk = [training[int(i)] for i in order[start:start + args.batch_size]]
                xs, ys = [], []
                for row in chunk:
                    crop_rng = np.random.default_rng(stable_seed(args.seed, "crop", epoch, row["id"]))
                    aug_rng = np.random.default_rng(stable_seed(args.seed, "augment", epoch, row["id"]))
                    x = crop(read_audio(row["path"]), crop_rng, samples=args.samples)
                    if args.voice_overlay:
                        mix_rng = np.random.default_rng(stable_seed(args.seed, "voice-overlay", epoch, row["id"]))
                        if mix_rng.random() < 0.5:
                            voice = speech_pool[int(mix_rng.integers(len(speech_pool)))]
                            x, info = overlay_voice(x, crop(read_audio(voice["path"]), mix_rng), mix_rng)
                            trace_counts[info["name"]] += 1
                    if args.aug in ("channel", "light"):
                        x, trace = augment(x, aug_rng, augmentation_config)
                        trace_counts.update(t["name"] for t in trace)
                    # Author checkpoint uses fake index 0, real index 1.
                    xs.append(x)
                    ys.append(1 - row["label_fake"])
                x = torch.from_numpy(np.stack(xs)).to(device)
                y = torch.tensor(ys, device=device)
                window_start = (step // args.accumulate) * args.accumulate
                window_size = min(args.accumulate, batch_count - window_start)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(logits(model, x, args.branch).float(), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss / window_size).backward()
                if (step + 1) % args.accumulate == 0 or step + 1 == batch_count:
                    torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                boundary = (step + 1) % args.accumulate == 0 or step + 1 == batch_count
                if boundary and ((step + 1) % args.checkpoint_steps == 0 or (output / "STOP").exists()):
                    save_training(epoch, step + 1, losses)
                    if (output / "STOP").exists():
                        (output / "paused.json").write_text(json.dumps({"epoch": epoch, "next_step": step + 1}))
                        print("Paused at optimizer boundary; full checkpoint saved.", flush=True)
                        return
                if step % 100 == 0:
                    print(json.dumps({"epoch": epoch, "step": step, "batches": batch_count, "loss": losses[-1]}), flush=True)
        clean, clean_rows = validate(model, development, args.branch, device, args.batch_size, samples=args.samples)
        stress, stress_rows = validate(model, development, args.branch, device, args.batch_size, stress=True, samples=args.samples)
        selection = (clean + stress) / 2
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                  "clean_eer": clean, "stress_eer": stress, "selection_metric": selection,
                  "seconds": prior_seconds + time.monotonic() - started,
                  "max_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}
        history.append(record)
        print(json.dumps(record), flush=True)
        state = {"state": adaptation_state(model, modules), "branch": args.branch,
                 "epoch": epoch, "run_config": cfg, "metrics": record}
        atomic_torch_save(state, output / "last_adaptation.pt")
        if selection < best:
            best = selection
            atomic_torch_save(state, output / "best_adaptation.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output / f"dev_epoch{epoch:02}.json").write_text(json.dumps(clean_rows + stress_rows, indent=2), encoding="utf-8")
        (output / "augmentation_counts.json").write_text(json.dumps(trace_counts, indent=2), encoding="utf-8")
        save_training(epoch + 1, 0, [])
    (output / "complete.json").write_text(json.dumps({"completed_epochs": args.epochs, "best_metric": best,
        "scope": "local external-data experiment, not a DACON submission or independent SOTA evaluation"}, indent=2))


def check_models(args):
    from models import load_eat, load_speech, configure_training, logits, train_mode
    torch.set_num_threads(4)
    torch.manual_seed(688)
    model = (load_eat() if args.branch == "music" else load_speech()).cuda().eval()
    modules = configure_training(model, args.branch)
    x = torch.randn(2, 64000, device="cuda") * 0.03
    with torch.inference_mode():
        alone = logits(model, x[:1], args.branch)
        batch = logits(model, x, args.branch)[:1]
    error = float((alone - batch).abs().max())
    if not torch.allclose(alone, batch, atol=5e-4, rtol=5e-4):
        raise AssertionError(f"Cross-file dependence or excessive batching error: {error}")
    train_mode(model, modules)
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(logits(model, x, args.branch).float(), torch.tensor([0, 1], device="cuda"))
    loss.backward()
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise AssertionError("Frozen encoder received gradients")
    optimizer.step()
    changed = any(not torch.equal(p, old) for p, old in zip(params, before))
    if not changed:
        raise AssertionError("No trainable weight changed")
    record = {"branch": args.branch, "batch_independence_max_abs_error": error,
              "probe_optimizer_step_changed_weights": changed, "loss": float(loss),
              "input": "synthetic waveform, execution check only", "performance_evidence": False}
    (ROOT / "records" / f"model-check-{args.branch}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "check"])
    parser.add_argument("--branch", choices=["music", "speech"], required=True)
    parser.add_argument("--name")
    parser.add_argument("--aug", choices=["none", "channel", "light"], default="channel")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=2048)
    parser.add_argument("--dev-per-class", type=int, default=256)
    parser.add_argument("--last-layers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=688)
    parser.add_argument("--voice-overlay", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--overlay-manifest")
    parser.add_argument("--balance-generators", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-steps", type=int, default=128)
    parser.add_argument("--samples", type=int, default=64000)
    args = parser.parse_args()
    if args.action == "train":
        if not args.name or Path(args.name).name != args.name:
            parser.error("--name must be a new run directory name")
        if args.batch_size < 2 or args.accumulate < 1 or args.epochs < 1 or args.checkpoint_steps < 1 or args.samples < 16000:
            parser.error("batch-size >=2, accumulate >=1, epochs >=1, checkpoint-steps >=1 required")
        train(args)
    else:
        check_models(args)


if __name__ == "__main__":
    main()
