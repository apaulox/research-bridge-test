"""Fetch a deterministic ASVspoof LA pilot subset from the official ZIP using HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib
import zipfile

import requests

from contracts import stable_seed
from prepare import write

ROOT = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/6906306/files/LA.zip?download=1"


def fetch_range(start, end):
    with requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"HTTP range unsupported ({response.status_code}); refusing whole-file fallback")
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Incorrect range response: {content_range}")
        data = response.content
        if len(data) != end - start + 1:
            raise RuntimeError("Incomplete range")
        return data, int(content_range.split("/")[1])


class RemoteFile(io.RawIOBase):
    def __init__(self):
        self.position = 0
        _, self.size = fetch_range(0, 0)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        end = self.size - 1 if size < 0 else min(self.size - 1, self.position + size - 1)
        if end < self.position:
            return b""
        data, _ = fetch_range(self.position, end)
        self.position = end + 1
        return data


def member_data(info):
    header, _ = fetch_range(info.header_offset, info.header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid local ZIP header")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + 30 + name_len + extra_len
    compressed, _ = fetch_range(start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        result = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        result = compressed
    else:
        raise ValueError(f"Unsupported compression {info.compress_type}")
    if len(result) != info.file_size or zlib.crc32(result) & 0xffffffff != info.CRC:
        raise ValueError("Official ZIP member CRC mismatch")
    return result


def main():
    base = ROOT / "assets/asv2019_subset"
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteFile()) as archive:
        items = {i.filename: i for i in archive.infolist()}
    rows = []
    tasks = []
    for official_split, cap in [("train", 512), ("dev", 128)]:
        protocol_name = next(n for n in items if f"ASVspoof2019.LA.cm.{official_split}." in n and n.endswith(".txt"))
        protocol = member_data(items[protocol_name]).decode()
        (base / Path(protocol_name).name).write_text(protocol, encoding="utf-8")
        candidates = [line.split() for line in protocol.splitlines() if line.strip()]
        for label in ["bonafide", "spoof"]:
            selected = sorted([r for r in candidates if r[-1] == label], key=lambda r: stable_seed(688, r[1]))[:cap]
            for speaker, file_id, _, attack, _ in selected:
                name = next(n for n in items if n.endswith(f"/ASVspoof2019_LA_{official_split}/flac/{file_id}.flac")
                            or n == f"ASVspoof2019_LA_{official_split}/flac/{file_id}.flac")
                destination = base / official_split / f"{file_id}.flac"
                tasks.append((items[name], destination))
                rows.append({"id": file_id, "path": str(destination), "source_id": speaker,
                             "split": official_split, "label_fake": int(label == "spoof"), "domain": "speech",
                             "generator": attack if label == "spoof" else "human", "origin": "public_external",
                             "label_status": "confirmed", "label_scope": "speech_authenticity",
                             "source_url": "https://zenodo.org/records/6906306", "license": "ODC-By, ASVspoof official README"})
    def fetch(task):
        info, path = task
        if path.exists():
            data = path.read_bytes()
            if len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
                raise ValueError("Existing audio CRC mismatch")
        else:
            data = member_data(info)
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_bytes(data)
            part.replace(path)
        return {"path": str(path), "zip_member": info.filename, "crc32": info.CRC,
                "sha256": hashlib.sha256(data).hexdigest()}
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, record in enumerate(pool.map(fetch, tasks), 1):
            records.append(record)
            if index % 32 == 0:
                print(json.dumps({"subset_downloaded": index, "total": len(tasks)}), flush=True)
    (ROOT / "records/speech-subset-files.json").write_text(json.dumps(records, indent=2))
    write(rows, "speech.jsonl", ["Official ZIP members verified by CRC and recorded SHA-256.",
                                 "Pilot: 512 per class train, 128 per class dev; source speaker partitions unchanged.",
                                 "Not unseen by the upstream AntiDeepfake checkpoint; not a DACON performance estimate."])


if __name__ == "__main__":
    main()
