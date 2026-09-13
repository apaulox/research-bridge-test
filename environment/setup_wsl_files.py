"""Extract user-provided DACON bundle into a new WSL project directory."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import zipfile


def safe_extract(archive, destination):
    root = destination.resolve()
    for item in archive.infolist():
        target = (root / item.filename).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'Archive path escapes destination: {item.filename}')
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError(f'Archive symlink rejected: {item.filename}')
        if target.exists() and not item.is_dir():
            raise FileExistsError(target)
    archive.extractall(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    downloads = root / 'downloads'
    downloads.mkdir()
    baseline = root / 'baseline'
    baseline.mkdir()
    with zipfile.ZipFile(args.source / 'open.zip') as archive:
        safe_extract(archive, downloads)
    print('Outer archive extracted', flush=True)
    with zipfile.ZipFile(downloads / 'baseline_submit.zip') as archive:
        safe_extract(archive, baseline)
    shutil.copytree(downloads / 'data', baseline / 'data')
    notebooks = root / 'notebooks'
    notebooks.mkdir()
    candidates = list(args.source.glob('[[]Baseline_Inference[]]*.ipynb'))
    for path in candidates:
        shutil.copy2(path, notebooks / path.name)
    musicdet = root / 'musicdet'
    musicdet.mkdir()
    sources = ['archive/musicdet_probe.py', 'environment/server_check.py',
               'results/README_MusicDET.md', 'archive/test_probe_contract.py']
    for source in sources:
        shutil.copy2(args.workspace / source, musicdet / Path(source).name)
    shutil.copytree(args.workspace / 'archive/vendor', musicdet / 'vendor')
    (root / 'logs').mkdir()
    (root / 'setup_origin.json').write_text(json.dumps(dict(source=str(args.source),
        baseline=str(baseline), notebook_count=len(candidates)), indent=2))
    print(f'Project extracted to {root}', flush=True)


if __name__ == '__main__':
    main()
