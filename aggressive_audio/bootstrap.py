"""Read local inventory and fetch pinned public resources; never reads DACON test audio."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RECORDS = ROOT / "records"


def download(url, path, expected_md5=None):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                raise RuntimeError("Server did not honor resume; partial file retained")
            if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Incorrect resumed byte range")
            with part.open("ab" if offset else "xb") as handle:
                last = time.monotonic()
                for chunk in response.iter_content(4 * 1024**2):
                    handle.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - last > 15:
                        print(json.dumps({"download": path.name, "bytes": offset}), flush=True)
                        last = time.monotonic()
        part.rename(path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    if expected_md5 and digest.hexdigest() != expected_md5:
        raise ValueError(f"Checksum mismatch: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "md5": digest.hexdigest(), "url": url}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inventory", "speech-model", "speech-data", "speech-data-mirror", "audit-mirror", "source-metadata", "codec-generation"])
    args = parser.parse_args()
    RECORDS.mkdir(exist_ok=True)
    if args.action == "inventory":
        import importlib.util
        import torch
        base = Path("/home/huskypaul/dacon")
        record = {
            "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "dependencies": {k: bool(importlib.util.find_spec(k)) for k in
                             ["fairseq", "safetensors", "transformers", "scipy", "hydra", "bitarray", "sacrebleu"]},
            "fmc_audio_dirs": [str(p) for p in (base / "musicdet/vendor/MusicDET/datasets").rglob("*") if p.is_dir()],
            "eat_files": [str(p) for p in (base / "submissions/eat75_musicdet25_dependencyfix/model/eat_fmc").iterdir()],
            "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        }
    elif args.action == "audit-mirror":
        import zipfile
        from remote_subset import RemoteFile
        with zipfile.ZipFile(RemoteFile()) as official:
            expected = {Path(i.filename).name: (i.CRC, i.file_size) for i in official.infolist() if not i.is_dir()}
        with zipfile.ZipFile(ASSETS / "asv2019_mirror/LA.zip") as mirror:
            actual = {Path(i.filename).name: (i.CRC, i.file_size) for i in mirror.infolist() if not i.is_dir()}
            examples = mirror.namelist()[:8]
        common = expected.keys() & actual.keys()
        mismatches = [name for name in common if expected[name] != actual[name]]
        record = {"official_members": len(expected), "mirror_members": len(actual),
                  "common_members": len(common), "mismatches": len(mismatches),
                  "mismatch_examples": mismatches[:10], "mirror_examples": examples,
                  "official_index": expected}
        print(json.dumps({k:v for k,v in record.items() if k != "official_index"}, indent=2), flush=True)
    elif args.action == "codec-generation":
        from huggingface_hub import snapshot_download
        repo = "CodecFake/CodecFake_Plus_Dataset"
        revision = "5543f7efb44ba818e36093838202cb50ba126a99"
        path = snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=ASSETS / "codecfake_cosg",
                                 allow_patterns=["CodecFake_plus_CoSG.tar.xz", "CoSG_labels.txt", "README.md"], token=False)
        record = {"repo": repo, "revision": revision, "path": path, "subset": "CoSG actual generated speech",
                  "excluded": "CoRS codec round-trip resynthesis", "license_card": "MIT; preserve upstream demo sources"}
    elif args.action == "source-metadata":
        import requests
        from huggingface_hub import HfApi, hf_hub_download
        fmc = requests.get("https://zenodo.org/api/records/15063698", timeout=30)
        fmc.raise_for_status()
        info = HfApi().dataset_info("CodecFake/CodecFake_Plus_Dataset", files_metadata=True, token=False)
        path = hf_hub_download("CodecFake/CodecFake_Plus_Dataset", "CoSG_labels.txt", repo_type="dataset",
                               revision=info.sha, local_dir=ASSETS / "codecfake_metadata", token=False)
        record = {"fmc_license": fmc.json()["metadata"].get("license"), "fmc_metadata": fmc.json()["metadata"],
                  "codec_revision": info.sha, "codec_files": [{"path": s.rfilename, "size": s.size} for s in info.siblings],
                  "cosg_label_examples": Path(path).read_text().splitlines()[:12]}
    elif args.action == "speech-model":
        from huggingface_hub import HfApi, snapshot_download
        repo = "nii-yamagishilab/xls-r-1b-anti-deepfake"
        info = HfApi().model_info(repo)
        path = snapshot_download(repo, revision=info.sha, local_dir=ASSETS / "speech_xlsr1b",
                                 allow_patterns=["*.json", "*.safetensors", "README.md", "LICENSE*"])
        record = {"repo": repo, "revision": info.sha, "path": path,
                  "license": "CC-BY-NC-SA-4.0", "official_author_checkpoint": True,
                  "backbone": "wav2vec2 XLS-R 1B", "head": "temporal mean + Linear(1280,2)",
                  "class_order": ["fake", "real"], "prior_augmentation": "RawBoost (1)+(2)"}
    elif args.action == "speech-data-mirror":
        from huggingface_hub import HfApi, hf_hub_download
        repo = "JesseHuang922/VoxSentinel_Synthetic_Detection_Dataset"
        info = HfApi().dataset_info(repo)
        names = [s.rfilename for s in info.siblings if Path(s.rfilename).name == "LA.zip"]
        if len(names) != 1:
            raise ValueError(f"Expected one LA.zip mirror, found {names}")
        path = Path(hf_hub_download(repo, names[0], repo_type="dataset", revision=info.sha,
                                   local_dir=ASSETS / "asv2019_mirror"))
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != "30c98f11d8b2bc21f2c257bfd78bb5c5":
            raise ValueError("Mirror differs from the official ASVspoof ZIP; do not use")
        # Use the official ZIP only after byte-equivalence verification.
        target = ASSETS / "asv2019/LA.zip"
        if not target.exists():
            path.rename(target)
        record = {"mirror": repo, "revision": info.sha, "path": str(target),
                  "md5": digest.hexdigest(), "verified_against": "https://zenodo.org/records/6906306",
                  "authority": "Official ASVspoof archive checksum; mirror is transport only"}
    else:
        base_url = "https://zenodo.org/records/6906306/files/"
        record = {"dataset": "ASVspoof2019 LA", "files": []}
        for name in ["README.txt", "LICENSE_text.txt"]:
            record["files"].append(download(base_url + name + "?download=1", ASSETS / "asv2019" / name))
        record["files"].append(download(base_url + "LA.zip?download=1", ASSETS / "asv2019/LA.zip",
                                         "30c98f11d8b2bc21f2c257bfd78bb5c5"))
        record["note"] = "Use official train/dev only for adaptation. No claim of unseen data relative to upstream AntiDeepfake."
    (RECORDS / f"{args.action}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in {"official_index", "fmc_metadata"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
