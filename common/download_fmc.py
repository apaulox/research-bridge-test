"""Resumable download and checksum verification for the official FMC archive."""
import argparse
import hashlib
from pathlib import Path
import time

import requests

URL = 'https://zenodo.org/api/records/15063698/files/FakeMusicCaps.zip/content'
SIZE = 12_889_873_014
MD5 = 'db418dc95ab7dc378a55f29d6021fd66'


def checksum(path):
    value = hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + '.part')
    if args.output.exists():
        if args.output.stat().st_size == SIZE and checksum(args.output) == MD5:
            print('Already complete and verified:', args.output)
            return
        raise FileExistsError(f'Existing final file failed verification: {args.output}')
    for attempt in range(1, 11):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            raise ValueError('Partial file is larger than expected')
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError('Server did not honor resume range')
                mode = 'ab' if offset else 'xb'
                with partial.open(mode) as handle:
                    last_report = offset
                    for chunk in response.iter_content(8 * 1024**2):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset - last_report >= 256 * 1024**2:
                            print(f'{offset / SIZE:6.2%}  {offset / 1024**3:.2f}/{SIZE / 1024**3:.2f} GiB', flush=True)
                            last_report = offset
            if partial.stat().st_size == SIZE:
                break
        except Exception as exc:
            print(f'attempt {attempt} interrupted at {offset}: {exc}', flush=True)
            if attempt == 10:
                raise
            time.sleep(min(30, attempt * 3))
    if partial.stat().st_size != SIZE:
        raise ValueError(f'Wrong size: {partial.stat().st_size} != {SIZE}')
    actual = checksum(partial)
    if actual != MD5:
        raise ValueError(f'MD5 mismatch: {actual} != {MD5}')
    partial.rename(args.output)
    print('Verified:', args.output, flush=True)


if __name__ == '__main__':
    main()
