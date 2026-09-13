#!/usr/bin/env python3
"""Create and verify a ZIP64 submission archive without external tools."""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"No files under {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for index, path in enumerate(files, 1):
            archive.write(path, path.relative_to(source).as_posix())
            if index % 25 == 0 or index == len(files):
                print(f"WRITE {index}/{len(files)}", flush=True)

    with zipfile.ZipFile(args.output, "r", allowZip64=True) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"CRC failure: {bad_file}")
        if len(archive.infolist()) != len(files):
            raise RuntimeError("Archive file count mismatch")

    digest = sha256(args.output)
    print(
        f"VERIFIED files={len(files)} bytes={args.output.stat().st_size} "
        f"sha256={digest}",
        flush=True,
    )
    if args.copy_to is not None:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.copy_to)
        copied_digest = sha256(args.copy_to)
        if copied_digest != digest:
            raise RuntimeError("Copied archive SHA-256 mismatch")
        print(
            f"COPIED path={args.copy_to} bytes={args.copy_to.stat().st_size} "
            f"sha256={copied_digest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
