"""Audit the delivered ZIP and execute its extracted runtime with timm unavailable."""
import ast
import csv
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / "submit_eat_xlsr_aug_resnet38_mean_v1.zip"
STAGE = Path("/home/huskypaul/dacon/submissions/verify_eat_xlsr_resnet38_mean_v1")


def main():
    direct, references, checked = [], [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        requirements = z.read("requirements.txt").decode()
        assert "timm" not in requirements.lower()
        bundled = [n for n in z.namelist() if any(p.lower() in {"timm", "timm.py"} or p.lower().startswith("timm-") for p in Path(n).parts)]
        assert not bundled, bundled
        for name in z.namelist():
            if not name.endswith((".py", ".json", ".txt")):
                continue
            content = z.read(name)
            assert content == (STAGE / name).read_bytes(), name
            if not name.endswith(".py"):
                continue
            checked.append(name)
            source = content.decode("utf-8-sig")
            references.extend({"file":name,"line":i,"text":line.strip()} for i,line in enumerate(source.splitlines(),1) if "timm" in line.lower())
            for node in ast.walk(ast.parse(source)):
                modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ""] if isinstance(node,ast.ImportFrom) else []
                direct.extend({"file":name,"module":m} for m in modules if m.split(".")[0] == "timm")
    assert not direct, direct
    attempts, probes = [], []
    assert not any(n.split(".")[0] == "timm" for n in sys.modules)
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "timm":
                attempts.append(fullname)
                raise ModuleNotFoundError("timm blocked by final ZIP audit")
    sys.meta_path.insert(0, Block())
    real_find = importlib.util.find_spec
    def find(name, package=None):
        if name.split(".")[0] == "timm":
            probes.append(name)
            return None
        return real_find(name, package)
    importlib.util.find_spec = find
    assert importlib.util.find_spec("timm") is None
    try:
        __import__("timm")
    except ModuleNotFoundError as e:
        assert "final ZIP audit" in str(e)
    else:
        raise AssertionError("Import guard failed")
    attempts.clear()
    probes.clear()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_MODULES_CACHE"] = str(STAGE / "fresh_timm_audit_modules")
    output = STAGE / "timm_audit_submission.csv"
    sys.argv = [str(STAGE / "script.py"), "--test-dir", str(ROOT / "fusion_diagnostic/audio"),
                "--sample-submission", str(STAGE / "sample_submission.csv"), "--output", str(output)]
    print(json.dumps({"python_files_checked":len(checked),"direct_timm_imports":direct,"guard_self_test_passed":True}), flush=True)
    runpy.run_path(str(STAGE / "script.py"), run_name="__main__")
    loaded = [n for n in sys.modules if n.split(".")[0] == "timm"]
    assert not attempts and not loaded, (attempts, loaded)
    with output.open() as f:
        current = list(csv.DictReader(f))
    with (STAGE / "verified_submission.csv").open() as f:
        previous = list(csv.DictReader(f))
    assert len(current) == len(previous) == 64
    delta = 0.0
    for a,b in zip(current,previous):
        assert a["ID"] == b["ID"]
        delta = max(delta, *(abs(float(a[k])-float(b[k])) for k in a if k != "ID"))
    assert delta == 0.0, delta
    report = {"archive":ARCHIVE.name,"python_files_checked":len(checked),"requirements":requirements.strip(),
              "runtime_sources_match_zip":True,"bundled_timm_packages":bundled,"direct_timm_imports":direct,
              "textual_references":references,"guard_self_test_passed":True,"runtime_import_attempts":attempts,
              "optional_availability_probes":probes,"loaded_timm_modules":loaded,"files_tested":64,
              "max_probability_difference":delta,"passed":True,"python":sys.version,
              "limit":"Local Python 3.10 runtime; not an execution on the DACON server."}
    (ROOT / "records/final_zip_timm_audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
