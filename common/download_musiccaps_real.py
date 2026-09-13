"""Download exact public MusicCaps segments referenced by official MusicDET splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import soundfile as sf


def duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sleep', type=float, default=0.75)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    if args.workers < 1 or args.workers > 8:
        parser.error('--workers must be between 1 and 8')
    log_path = args.output_dir.parent / 'musiccaps_download.jsonl'

    def download(row):
        audio_id = row['ytid']
        start = float(row['start_s'])
        end = float(row['end_s'])
        expected = end - start
        final = args.output_dir / f'{audio_id}.wav'
        if final.exists():
            existing_seconds = duration(final)
            if abs(existing_seconds - expected) <= 0.1:
                return dict(id=audio_id, split=row['split'], status='existing',
                            duration=existing_seconds, elapsed=0, detail='')
            quarantine = args.output_dir.parent / 'quarantine_bad_duration'
            quarantine.mkdir(exist_ok=True)
            candidate = quarantine / final.name
            serial = 1
            while candidate.exists():
                candidate = quarantine / f'{audio_id}.{serial}.wav'
                serial += 1
            final.rename(candidate)
        yt_dlp = str(Path(sys.executable).with_name('yt-dlp'))
        command = [
            yt_dlp, '--no-playlist', '--force-keyframes-at-cuts',
            '--download-sections', f'*{start}-{end}', '-x', '--audio-format', 'wav',
            '--audio-quality', '0', '--retries', '5', '--fragment-retries', '5',
            '--no-overwrites', '-o', str(args.output_dir / f'{audio_id}.%(ext)s'),
            f'https://www.youtube.com/watch?v={audio_id}',
        ]
        started = time.time()
        result = subprocess.run(command, text=True, capture_output=True)
        status = 'ok'
        detail = ''
        seconds = None
        if result.returncode or not final.exists():
            status = 'failed'
            detail = (result.stderr or result.stdout)[-2000:]
        else:
            seconds = duration(final)
            if abs(seconds - expected) > 0.1:
                status = 'bad_duration'
                detail = f'{seconds:.4f} != {expected:.4f}'
        time.sleep(args.sleep + random.random() * args.sleep)
        return dict(id=audio_id, split=row['split'], status=status, duration=seconds,
                    elapsed=time.time()-started, detail=detail)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, row) for row in rows]
        for future in as_completed(futures):
            record = future.result()
            completed += 1
            if record['status'] != 'existing':
                with log_path.open('a') as log:
                    log.write(json.dumps(record) + '\n')
            print(f'[{completed}/{len(rows)}] {record["id"]}: {record["status"]}', flush=True)


if __name__ == '__main__':
    main()
