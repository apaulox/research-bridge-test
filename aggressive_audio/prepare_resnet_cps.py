"""Fetch official PANNs ResNet38 and extract its unchanged inference definitions."""
import ast
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/panns_resnet38"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"
    path = ASSETS / "ResNet38_mAP=0.434.pth"
    if not path.exists():
        part = path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            with part.open("wb") as f:
                count = 0
                last = 0
                for data in response.iter_content(1024*1024):
                    f.write(data)
                    count += len(data)
                    if count-last >= 16*1024*1024:
                        print(json.dumps({"downloaded_mb":round(count/1e6,1)}), flush=True)
                        last = count
        part.replace(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    assert digest == "bf12f36aaabac4e0855e22d3c3239c1b", digest
    repo = "https://api.github.com/repos/qiuqiangkong/audioset_tagging_cnn/commits/master"
    r = requests.get(repo, timeout=30)
    r.raise_for_status()
    revision = r.json()["sha"]
    prefix = f"https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/{revision}"
    r = requests.get(prefix + "/pytorch/models.py", timeout=30)
    r.raise_for_status()
    source = r.text
    (ASSETS / "official_models.py").write_text(source)
    tree = ast.parse(source)
    definitions = {n.name:n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    needed = {"ResNet38"}
    while True:
        found = {n.id for name in needed for n in ast.walk(definitions[name]) if isinstance(n, ast.Name)} & definitions.keys()
        if found <= needed:
            break
        needed.update(found)
    utilities = requests.get(prefix + "/pytorch/pytorch_utils.py", timeout=30)
    utilities.raise_for_status()
    mixup = next(n for n in ast.parse(utilities.text).body if isinstance(n, ast.FunctionDef) and n.name == "do_mixup")
    header = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torchlibrosa.stft import Spectrogram, LogmelFilterBank\nfrom torchlibrosa.augmentation import SpecAugmentation\n\n"
    header += ast.get_source_segment(utilities.text, mixup) + "\n\n"
    extracted = header + "\n\n".join(ast.get_source_segment(source, n) for n in tree.body if getattr(n, "name", None) in needed)
    (ASSETS / "resnet38_architecture.py").write_text(extracted)
    license_response = requests.get(prefix + "/LICENSE.MIT", timeout=30)
    license_response.raise_for_status()
    (ASSETS / "LICENSE_PANNS.txt").write_text(license_response.text)
    record = {"url":url, "md5":digest, "bytes":path.stat().st_size, "source_revision":revision,
              "source_url":prefix + "/pytorch/models.py", "definitions":sorted(needed),
              "official_checkpoint":True, "additional_training":False}
    (ASSETS / "provenance.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
