"""Extract, convert, and audit FMC/MusicCaps audio for official MusicDET protocols."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import zipfile


AUDIO_EXTENSIONS = {'.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.wma'}

# FakeMusicCaps stores each generator in a directory and leaves the original
# MusicCaps YouTube ID as the filename.  MusicDET protocols encode the same
# generator in a TTM prefix instead.
GENERATOR_TO_TTM = {
    'MusicGen_medium': 'TTM01',
    'musicldm': 'TTM02',
    'audioldm2': 'TTM03',
    'stable_audio_open': 'TTM04',
    'mustango': 'TTM05',
}


def md5(path):
    value = hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def convert(task):
    source, destination = map(Path, task)
    import librosa
    import numpy as np
    import soundfile as sf
    if destination.exists():
        try:
            info = sf.info(destination)
            if info.samplerate == 16000 and info.channels == 1 and info.frames > 0:
                return destination.name, 'existing', info.frames
        except Exception:
            pass
        raise ValueError(f'Invalid existing output requires manual review: {destination}')
    audio, original_sr = librosa.load(source, sr=None, mono=True, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f'Invalid source audio: {source}')
    if original_sr != 48000:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=48000, res_type='soxr_hq')
    audio = librosa.resample(audio, orig_sr=48000, target_sr=16000, res_type='soxr_hq')
    temporary = destination.with_suffix('.tmp.wav')
    sf.write(temporary, audio.astype(np.float32), 16000, subtype='FLOAT')
    os.replace(temporary, destination)
    return destination.name, 'converted', len(audio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--real-dir', type=Path, required=True)
    parser.add_argument('--raw-fake-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--available-protocol-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if md5(args.archive) != 'db418dc95ab7dc378a55f29d6021fd66':
        raise ValueError('FMC archive MD5 mismatch')
    protocols = sorted((args.repo / 'label/fakemusiccaps/openset').glob('TTM*.txt'))
    rows = {}
    expected_stems = set()
    for protocol in protocols:
        content = [line.split()[:2] for line in protocol.read_text().splitlines() if line.strip()]
        rows[protocol.name] = content
        expected_stems.update(Path(name).stem for name, _ in content)
    fake_stems = {stem for stem in expected_stems if stem.startswith(('TTM01_', 'TTM02_', 'TTM03_', 'TTM04_', 'TTM05_'))}
    args.raw_fake_dir.mkdir(parents=True, exist_ok=True)
    extraction_marker = args.raw_fake_dir / '.extraction-complete.json'
    if not extraction_marker.exists():
        with zipfile.ZipFile(args.archive) as archive:
            mapping = {}
            for item in archive.infolist():
                member_path = Path(item.filename)
                suffix = member_path.suffix.lower()
                generator = member_path.parts[0] if member_path.parts else ''
                ttm = GENERATOR_TO_TTM.get(generator)
                if item.is_dir() or suffix not in AUDIO_EXTENSIONS or ttm is None:
                    continue
                stem = f'{ttm}_{member_path.stem}'
                if stem not in fake_stems:
                    continue
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f'Archive symlink rejected: {item.filename}')
                if stem in mapping:
                    raise ValueError(f'Duplicate fake audio stem: {stem}')
                mapping[stem] = item
            missing_members = sorted(fake_stems - mapping.keys())
            if missing_members:
                raise FileNotFoundError(f'{len(missing_members)} expected fake members absent; {missing_members[:10]}')
            for index, (stem, item) in enumerate(mapping.items(), 1):
                destination = args.raw_fake_dir / f'{stem}{Path(item.filename).suffix.lower()}'
                if not destination.exists():
                    temporary = destination.with_suffix(destination.suffix + '.part')
                    with archive.open(item) as source, temporary.open('xb') as output:
                        while chunk := source.read(8 * 1024**2):
                            output.write(chunk)
                    temporary.rename(destination)
                if index % 1000 == 0:
                    print(f'extracted {index}/{len(mapping)}', flush=True)
        extraction_marker.write_text(json.dumps(dict(files=len(mapping), archive_md5=md5(args.archive)), indent=2))
    real_map = {path.stem: path for path in args.real_dir.glob('*.wav')}
    fake_map = {path.stem: path for path in args.raw_fake_dir.iterdir()
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS}
    source_map = {**real_map, **fake_map}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    missing = []
    for stem in sorted(expected_stems):
        source = source_map.get(stem)
        if source is None:
            missing.append(stem)
        else:
            tasks.append((str(source), str(args.output_dir / f'{stem}.wav')))
    print(f'conversion candidates={len(tasks)} missing={len(missing)}', flush=True)
    converted = existing = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            _, status, _ = future.result()
            converted += status == 'converted'; existing += status == 'existing'
            if index % 500 == 0:
                print(f'processed {index}/{len(tasks)}', flush=True)
    available = {path.stem for path in args.output_dir.glob('*.wav')}
    args.available_protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol_report = {}
    for name, content in rows.items():
        kept = [(filename, label) for filename, label in content if Path(filename).stem in available]
        (args.available_protocol_dir / name).write_text(
            ''.join(f'{filename} {label}\n' for filename, label in kept))
        counts = {label: sum(row_label == label for _, row_label in kept) for label in ['real', 'fake']}
        protocol_report[name] = dict(original=len(content), available=len(kept), **counts)
    # Pair TTM01 dev real/fake by underlying MusicCaps ID for balanced model selection.
    ttm01_dev = rows['TTM01_dev.txt']
    real_ids = {Path(name).stem for name, label in ttm01_dev if label == 'real' and Path(name).stem in available}
    paired = [(name, label) for name, label in ttm01_dev
              if (Path(name).stem.removeprefix('TTM01_') in real_ids and Path(name).stem in available)]
    (args.available_protocol_dir / 'TTM01_dev_paired.txt').write_text(
        ''.join(f'{filename} {label}\n' for filename, label in paired))
    report = dict(expected=len(expected_stems), available=len(available & expected_stems), missing=missing,
                  converted=converted, existing=existing, protocols=protocol_report,
                  paired_dev_rows=len(paired))
    (args.available_protocol_dir / 'dataset_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
