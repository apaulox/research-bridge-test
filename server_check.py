#!/usr/bin/env python3
"""Read-only server inventory; does not install packages or load checkpoints."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=Path.cwd())
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args()
    versions = {}
    for name in ['torch', 'torchaudio', 'librosa', 'numpy', 'transformers', 'demucs', 'panns-inference']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=versions, ffmpeg=shutil.which('ffmpeg'),
                  baseline_dir=str(args.baseline_dir.resolve()),
                  baseline_files={name: (args.baseline_dir / name).exists()
                                  for name in ['script.py', 'requirements.txt', 'model/df_arena_1b',
                                               'model/htdemucs', 'model/panns',
                                               'data/test', 'data/sample_submission.csv']})
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        report['gpu'] = result.stdout.strip() or result.stderr.strip()
    else:
        report['gpu'] = 'nvidia-smi not found'
    if args.checkpoint_dir:
        report['musicdet_files'] = {name: (args.checkpoint_dir / name).is_file()
                                   for name in ['args.json', 'anti-spoofing_feat_model.pt']}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
