"""Extract the actual ZIP and test offline inference against pre-package predictions."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "submit_eat_xlsr_aug_resnet38_mean_v1.zip"


def main():
    folder = Path("/home/huskypaul/dacon/submissions/verify_eat_xlsr_resnet38_mean_v1")
    folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for entry in z.infolist():
            target = (folder / entry.filename).resolve()
            assert target.is_relative_to(folder.resolve())
        z.extractall(folder)
    # Block network connections and forbidden training-time model libraries.
    wrapper = folder / "offline_check.py"
    wrapper.write_text('''import importlib.abc
import importlib.util
import os
import runpy
import socket
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fairseq', 'timm'}:
            raise ModuleNotFoundError('Excluded submission dependency: ' + fullname)
sys.meta_path.insert(0, Block())
old_find = importlib.util.find_spec
def find(name, package=None):
    if name.split('.')[0] in {'fairseq', 'timm'}:
        return None
    return old_find(name, package)
importlib.util.find_spec = find
def denied(*a, **k):
    raise RuntimeError('Network disabled during ZIP verification')
socket.socket.connect = denied
socket.create_connection = denied
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'fresh_hf_home')
os.environ['HF_MODULES_CACHE'] = os.path.join(os.path.dirname(__file__), 'fresh_hf_modules')
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
''')
    reference = ROOT / "records/resnet38_submission_reference.csv"
    output = folder / "verified_submission.csv"
    command = [sys.executable, "-B", str(wrapper), str(folder / "script.py"),
               "--test-dir", str(ROOT / "fusion_diagnostic/audio"),
               "--sample-submission", str(reference), "--output", str(output)]
    # Empty template probabilities to demonstrate predictions do not consume reference scores.
    with reference.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        refs = list(reader)
    template = folder / "sample_submission.csv"
    with template.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:row[k] if k == "ID" else 0 for k in fields} for row in reversed(refs))
    command[command.index("--sample-submission") + 1] = str(template)
    subprocess.run(command, cwd=folder, check=True)
    with output.open() as f:
        predictions = list(csv.DictReader(f))
    assert [r["ID"] for r in predictions] == [r["ID"] for r in reversed(refs)]
    refmap = {r["ID"]:r for r in refs}
    errors = {c:max(abs(float(r[c])-float(refmap[r["ID"]][c])) for r in predictions)
              for c in fields if c != "ID"}
    assert max(errors.values()) < 1e-4, errors
    digest = hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest() if sys.version_info >= (3,11) else None
    if digest is None:
        h = hashlib.sha256()
        with ARCHIVE.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):
                h.update(chunk)
        digest = h.hexdigest()
    result = {"archive":str(ARCHIVE), "bytes":ARCHIVE.stat().st_size, "sha256":digest,
              "zip_crc_passed":True, "offline_extracted_zip_inference_passed":True,
              "forbidden_dependencies":["fairseq", "timm"], "files":len(predictions),
              "reversed_template_order_preserved":True, "max_probability_errors":errors,
              "python":sys.version, "server_execution_verified":False}
    (ROOT / "records/resnet38_submission_zip_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
