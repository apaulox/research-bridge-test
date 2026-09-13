"""Fit a fixed probability calibration on FMC dev and audit it on FMC eval."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader


GENERATORS = {
    'TTM01': 'MusicGen',
    'TTM02': 'MusicLDM',
    'TTM03': 'AudioLDM2',
    'TTM04': 'Stable Audio Open',
    'TTM05': 'Mustango',
}


def load_model(repo, checkpoint_dir, device):
    sys.path.insert(0, str(repo))
    from model import SpecNF

    config = json.loads((checkpoint_dir / 'args.json').read_text())
    model = SpecNF(K=config['K'], L=config['L'], R=config['R'])
    state = torch.load(
        checkpoint_dir / 'anti-spoofing_feat_model.pt',
        map_location='cpu',
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


def score_protocol(repo, audio_dir, protocol, model, device, batch_size, workers):
    from dataset import FakeMusicCapsDataset

    dataset = FakeMusicCapsDataset(
        str(audio_dir), str(protocol), only_real=False, random_sampling=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=workers > 0,
    )
    rows = []
    with torch.inference_mode():
        for waveform, filenames, labels in loader:
            waveform = waveform.to(device, non_blocking=True)
            real_prior = torch.zeros(len(waveform), dtype=torch.long, device=device)
            _, nll = model(waveform, real_prior)
            for filename, label, score in zip(filenames, labels, nll.cpu().tolist()):
                rows.append((Path(filename).stem, int(label), float(score)))
    return rows


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def metrics(labels, probabilities):
    return {
        'samples': int(len(labels)),
        'real': int((labels == 0).sum()),
        'fake': int((labels == 1).sum()),
        'roc_auc_fake': float(roc_auc_score(labels, probabilities)),
        'log_loss': float(log_loss(labels, probabilities, labels=[0, 1])),
        'brier': float(brier_score_loss(labels, probabilities)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model, training_config = load_model(args.repo, args.checkpoint_dir, device)

    dev_rows = []
    for prefix, generator in GENERATORS.items():
        rows = score_protocol(
            args.repo,
            args.audio_dir,
            args.protocol_dir / f'{prefix}_dev.txt',
            model,
            device,
            args.batch_size,
            args.workers,
        )
        dev_rows.extend((prefix, generator, *row) for row in rows)
        print(f'scored {prefix} dev: {len(rows)}', flush=True)

    dev_scores = np.asarray([row[4] for row in dev_rows], dtype=np.float64)
    dev_labels = np.asarray([row[3] for row in dev_rows], dtype=np.int64)
    calibrator = LogisticRegression(
        class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=688
    )
    calibrator.fit(dev_scores[:, None], dev_labels)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    if not math.isfinite(coefficient) or not math.isfinite(intercept):
        raise ValueError('Nonfinite calibration parameters')
    if coefficient <= 0:
        raise ValueError(f'FMC dev learned a reversed fake direction: coefficient={coefficient}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dev_csv = args.output.with_name('calibration_dev_scores.csv')
    with dev_csv.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['SPLIT', 'GENERATOR', 'ID', 'LABEL_FAKE', 'MUSIC_ANOMALY_SCORE'])
        writer.writerows(dev_rows)

    dev_probabilities = sigmoid(coefficient * dev_scores + intercept)
    report = {
        'schema_version': 1,
        'method': 'logistic_regression',
        'input_score': 'NLL under the MusicDET real prior; larger means fake',
        'output_probability': 'P(fake)',
        'coefficient': coefficient,
        'intercept': intercept,
        'class_weight': 'balanced',
        'calibration_data': 'available FMC TTM01-TTM05 dev protocols; shared real rows repeated per generator',
        'training_checkpoint': str(args.checkpoint_dir),
        'training_config': training_config,
        'dev_metrics': metrics(dev_labels, dev_probabilities),
        'eval_metrics': {},
    }

    pooled_eval_labels = []
    pooled_eval_probabilities = []
    for prefix, generator in GENERATORS.items():
        score_path = args.evaluation_dir / f'{prefix}_scores.csv'
        with score_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        labels = np.asarray([int(row['LABEL_FAKE']) for row in rows], dtype=np.int64)
        scores = np.asarray(
            [float(row['MUSIC_ANOMALY_SCORE']) for row in rows], dtype=np.float64
        )
        probabilities = sigmoid(coefficient * scores + intercept)
        report['eval_metrics'][prefix] = {
            'generator': generator,
            **metrics(labels, probabilities),
        }
        pooled_eval_labels.extend(labels.tolist())
        pooled_eval_probabilities.extend(probabilities.tolist())
    report['eval_metrics']['pooled'] = metrics(
        np.asarray(pooled_eval_labels), np.asarray(pooled_eval_probabilities)
    )
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
