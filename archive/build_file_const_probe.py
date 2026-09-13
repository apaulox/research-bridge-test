"""Create a DACON diagnostic package with FILE_FAKE_PROB fixed to 0.5."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


SOURCE = Path("submit_musicdet_fmc_v1.zip")
OUTPUT = Path("submit_musicdet_fmc_file_const_probe.zip")
TARGET_LINE = '        row["FILE_FAKE_PROB"] = round(file_fake, 10)'
REPLACEMENT_LINE = (
    '        # Diagnostic probe: force File EER to 0.5 while preserving every '
    "other prediction."
    '\n        row["FILE_FAKE_PROB"] = 0.5'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    with zipfile.ZipFile(SOURCE, "r") as source:
        script = source.read("script.py").decode("utf-8")
        if script.count(TARGET_LINE) != 1:
            raise ValueError("Expected exactly one FILE_FAKE_PROB assignment")
        probe_script = script.replace(TARGET_LINE, REPLACEMENT_LINE)

        with zipfile.ZipFile(OUTPUT, "w", allowZip64=True) as output:
            for info in source.infolist():
                if info.filename == "script.py":
                    output.writestr(info, probe_script.encode("utf-8"))
                    continue
                with source.open(info, "r") as src, output.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)

    with zipfile.ZipFile(OUTPUT, "r") as probe:
        bad = probe.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC validation failed: {bad}")
        saved_script = probe.read("script.py").decode("utf-8")
        if 'row["FILE_FAKE_PROB"] = 0.5' not in saved_script:
            raise RuntimeError("Probe assignment is absent from the output archive")
        if TARGET_LINE in saved_script:
            raise RuntimeError("Original FILE prediction assignment remains")
        if probe.namelist() != source.namelist():
            raise RuntimeError("Archive membership changed")

    print(f"Created: {OUTPUT.resolve()}")
    print(f"Bytes: {OUTPUT.stat().st_size}")
    print(f"SHA256: {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
