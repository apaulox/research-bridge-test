#!/usr/bin/env python3
"""Score audio with an existing official MusicDET checkpoint (NOT a submission)."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.wma'}


def load_configuration(checkpoint_dir):
    config_path = checkpoint_dir / 'args.json'
    weight_path = checkpoint_dir / 'anti-spoofing_feat_model.pt'
    for path in (config_path, weight_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required trained-model file missing: {path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    required = {'model', 'task', 'K', 'L', 'R', 'audio_len', 'only_real'}
    if required - config.keys():
        raise ValueError(f'Incomplete training metadata: {sorted(required - config.keys())}')
    if config['model'] not in {'spec-nf', 'Spec-nf'}:
        raise ValueError('This probe supports MusicDET SpecNF only.')
    if config['task'] != 'fakemusiccaps':
        raise ValueError('Expected a FakeMusicCaps checkpoint. Check its training provenance.')
    if config['audio_len'] != 64600:
        raise ValueError('This audited architecture requires 64600 samples. Do not guess a new size.')
    if type(config['only_real']) is not bool:
        raise ValueError('only_real must be a JSON boolean from the training configuration.')
    if type(config['K']) is not int or config['K'] < 1 or config['L'] != 1:
        raise ValueError('Unsupported flow configuration; verify the original training code.')
    if not isinstance(config['R'], (int, float)) or not math.isfinite(config['R']):
        raise ValueError('Invalid R in training configuration.')
    return config, weight_path


def window_starts(length, size, mode):
    if length < 1 or size < 1:
        raise ValueError('Audio and window lengths must be positive.')
    if length <= size:
        return [0]
    if mode == 'center':
        return [(length - size) // 2]
    last = length - size
    starts = list(range(0, last + 1, size))
    if starts[-1] != last:
        starts.append(last)
    return starts


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--mode', choices=['center', 'sliding-mean', 'sliding-max'], default='center')
    parser.add_argument('--limit', type=int, default=None, help='Smoke test only; omit for all files.')
    args = parser.parse_args()
    config, weight_path = load_configuration(args.checkpoint_dir)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not args.audio_dir.is_dir():
        parser.error(f'Audio directory does not exist: {args.audio_dir}')
    files = sorted((p for p in args.audio_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.stem)
    if not files:
        parser.error('No audio files found')
    if len({p.stem for p in files}) != len(files):
        parser.error('Duplicate audio IDs (filename stems)')
    if args.limit is not None:
        files = files[:args.limit]
    metadata_path = args.output.with_suffix(args.output.suffix + '.json')
    partial_path = args.output.with_suffix(args.output.suffix + '.partial')
    for path in (args.output, metadata_path, partial_path):
        if path.exists():
            parser.error(f'Output already exists; choose a new experiment name: {path}')

    import librosa
    import numpy as np
    import torch
    import torch.nn.functional as functional
    # Namespace import avoids adding a generic `model` directory to sys.path.
    from vendor.MusicDET.model import SpecNF

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
    torch.manual_seed(688)
    device = torch.device(args.device)
    model = SpecNF(K=config['K'], L=config['L'], R=config['R'])
    state = torch.load(weight_path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected the official nonempty state_dict checkpoint.')
    if all(key.startswith('module.') for key in state):
        state = {key[len('module.'):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    uninitialized = [name for name, value in model.named_buffers()
                     if name.endswith('.inited') and value.item() != 1]
    if uninitialized:
        raise ValueError(f'Checkpoint has uninitialized ActNorm buffers: {uninitialized}')
    model.to(device).eval()
    size = config['audio_len']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    elapsed_start = time.perf_counter()
    fields = ['ID', 'MUSIC_ANOMALY_SCORE', 'SECONDS', 'WINDOWS', 'RMS', 'LOW_ENERGY', 'INFERENCE_SECONDS']
    with partial_path.open('x', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(files, 1):
            started = time.perf_counter()
            audio, _ = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError(f'Empty or nonfinite audio: {path}')
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            scores = []
            for start in window_starts(len(audio), size, args.mode):
                segment = torch.from_numpy(audio[start:start + size].copy())
                missing = size - segment.numel()
                if missing:
                    segment = functional.pad(segment, (missing // 2, missing - missing // 2))
                # Official dataset.py: center crop/pad first, then RMS normalization.
                segment = segment / torch.sqrt(torch.mean(segment ** 2) + 1e-8)
                with torch.inference_mode():
                    real_label = torch.zeros(1, dtype=torch.long, device=device)
                    _, nll = model(segment.unsqueeze(0).to(device), real_label)
                    score = float(nll.item())
                if not math.isfinite(score):
                    raise ValueError(f'Nonfinite MusicDET score: {path}')
                scores.append(score)
            # Official test.py returns -NLL (REAL direction). DACON fake direction is +NLL.
            score = max(scores) if args.mode == 'sliding-max' else sum(scores) / len(scores)
            writer.writerow(dict(ID=path.stem, MUSIC_ANOMALY_SCORE=score,
                                 SECONDS=audio.size / 16000, WINDOWS=len(scores), RMS=rms,
                                 LOW_ENERGY=int(rms < 1e-5),
                                 INFERENCE_SECONDS=time.perf_counter() - started))
            output.flush()
            print(f'[{index}/{len(files)}] {path.name}: anomaly={score:.6f}', flush=True)
    elapsed = time.perf_counter() - elapsed_start
    metadata = dict(training_config=config, mode=args.mode,
                    upstream_commit=(ROOT / 'vendor/MusicDET/UPSTREAM_COMMIT.txt').read_text().strip(),
                    checkpoint_sha256=file_hash(weight_path), files=len(files),
                    audio_dir=str(args.audio_dir.resolve()), device=args.device,
                    torch_version=torch.__version__, librosa_version=librosa.__version__,
                    scoring_seconds=elapsed, includes_model_loading_time=False,
                    score_definition='positive NLL under real prior; larger means more anomalous; NOT a probability',
                    status='raw probe only; no DACON submission generated')
    with metadata_path.open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    partial_path.rename(args.output)
    print(f'Saved {len(files)} raw scores to {args.output}; not a submission CSV.')


if __name__ == '__main__':
    main()
