"""Build and verify a DACON code-submission ZIP without external tools."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile


MAX_ZIP_BYTES = 10 * 1024**3
MAX_UNPACKED_BYTES = 32 * 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    required = [args.source / 'model', args.source / 'script.py', args.source / 'requirements.txt']
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing submission entries: {missing}')
    if args.output.exists():
        raise FileExistsError(args.output)

    files = []
    for root_name in ('model',):
        for path in sorted((args.source / root_name).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink is not allowed in model package: {path}')
            files.append(path)
    files.extend([args.source / 'script.py', args.source / 'requirements.txt'])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for index, path in enumerate(files, 1):
            member = path.relative_to(args.source).as_posix()
            archive.write(path, member)
            print(f'[{index}/{len(files)}] {member}', flush=True)

    with zipfile.ZipFile(args.output) as archive:
        names = archive.namelist()
        if 'script.py' not in names or 'requirements.txt' not in names:
            raise ValueError('Required root files missing from archive')
        if not any(name.startswith('model/') for name in names):
            raise ValueError('model/ is missing from archive')
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts:
                raise ValueError(f'Unsafe archive member: {name}')
        unpacked = sum(info.file_size for info in archive.infolist())
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f'CRC failure: {bad_member}')

    packed = args.output.stat().st_size
    if packed > MAX_ZIP_BYTES:
        raise ValueError(f'ZIP exceeds 10 GiB: {packed}')
    if unpacked > MAX_UNPACKED_BYTES:
        raise ValueError(f'Unpacked submission exceeds 32 GiB: {unpacked}')
    print(
        f'VERIFIED files={len(files)} zip_gib={packed / 1024**3:.3f} '
        f'unpacked_gib={unpacked / 1024**3:.3f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
