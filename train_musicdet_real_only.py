"""Audited real-only MusicDET trainer; saves every epoch and selects fixed-direction dev EER."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def eer(real_scores, fake_scores):
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    order = np.argsort(scores, kind='mergesort')
    labels = labels[order]
    misses = np.concatenate([[0.0], np.cumsum(labels) / len(real_scores)])
    false_alarms = np.concatenate([[1.0],
        (len(fake_scores) - (np.arange(1, len(scores)+1) - np.cumsum(labels))) / len(fake_scores)])
    point = int(np.argmin(np.abs(misses - false_alarms)))
    return float((misses[point] + false_alarms[point]) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-protocol', type=Path)
    parser.add_argument('--dev-protocol', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=688)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.repo))
    from dataset import FakeMusicCapsDataset
    from model import SpecNF
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    train_protocol = args.train_protocol or args.repo / 'label/fakemusiccaps/openset/TTM01_train.txt'
    dev_protocol = args.dev_protocol or args.repo / 'label/fakemusiccaps/openset/TTM01_dev.txt'
    train_set = FakeMusicCapsDataset(str(args.audio_dir), str(train_protocol), only_real=True, random_sampling=True)
    dev_set = FakeMusicCapsDataset(str(args.audio_dir), str(dev_protocol), only_real=False, random_sampling=False)
    expected = {Path(row[0]).name for row in train_set.all_files} | {Path(row[0]).name for row in dev_set.all_files}
    missing = sorted(name for name in expected if not (args.audio_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f'{len(missing)} training/dev audio files missing; examples={missing[:10]}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    device = torch.device('cuda')
    model = SpecNF(K=2, L=1, R=5.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=5e-4)
    metadata = dict(model='spec-nf', task='fakemusiccaps', K=2, L=1, R=5.0,
                    audio_len=64600, only_real=True, seed=args.seed, batch_size=args.batch_size,
                    checkpoint_selection='minimum fixed-direction dev EER; real score=-NLL',
                    train_protocol=str(train_protocol), dev_protocol=str(dev_protocol),
                    source='audited wrapper around official MusicDET architecture/dataset')
    (args.output_dir / 'args.json').write_text(json.dumps(metadata, indent=2))
    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for waveform, _, labels in train_loader:
            assert torch.all(labels == 0)
            waveform = waveform.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            _, nll = model(waveform, labels)
            loss = nll.mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); scores=[]; labels=[]
        with torch.inference_mode():
            for waveform, _, batch_labels in dev_loader:
                waveform = waveform.to(device, non_blocking=True)
                real_prior = torch.zeros(len(waveform), dtype=torch.long, device=device)
                _, nll = model(waveform, real_prior)
                scores.extend((-nll).cpu().tolist()); labels.extend(batch_labels.tolist())
        scores=np.asarray(scores); labels=np.asarray(labels)
        dev_eer=eer(scores[labels == 0], scores[labels == 1])
        record=dict(epoch=epoch, train_nll=float(np.mean(losses)), dev_eer=dev_eer)
        history.append(record); print(json.dumps(record), flush=True)
        state={key:value.detach().cpu() for key,value in model.state_dict().items()}
        torch.save(state, args.output_dir / f'epoch_{epoch:02d}.pt')
        if dev_eer < best:
            best=dev_eer; torch.save(state, args.output_dir / 'anti-spoofing_feat_model.pt')
        (args.output_dir / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
