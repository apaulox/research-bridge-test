"""Verify actual optimizer/RNG resume against an uninterrupted tiny EAT run."""
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent


def main():
    token = str(int(time.time()))
    common = [sys.executable, "-B", str(ROOT / "run.py"), "train", "--branch", "music", "--aug", "channel",
              "--epochs", "1", "--train-per-class", "16", "--dev-per-class", "4", "--checkpoint-steps", "8"]
    reference = "resume_test_reference_" + token
    paused = "resume_test_paused_" + token
    log_dir = ROOT / "records"
    with (log_dir / "resume_test_reference.log").open("w") as log:
        subprocess.run(common + ["--name", reference], stdout=log, stderr=subprocess.STDOUT, check=True)
    with (log_dir / "resume_test_paused.log").open("w") as log:
        process = subprocess.Popen(common + ["--name", paused], stdout=log, stderr=subprocess.STDOUT)
        folder = ROOT / "runs" / paused
        while not (folder / "config.json").exists():
            if process.poll() is not None:
                raise RuntimeError("Resume test failed before creating run configuration")
            time.sleep(0.2)
        (folder / "STOP").write_text("Resume functionality test")
        if process.wait(timeout=120):
            raise RuntimeError("Pause test run failed")
    if not (folder / "paused.json").exists():
        raise AssertionError("Training did not pause at optimizer boundary")
    (folder / "STOP").rename(folder / "STOP.handled")
    with (log_dir / "resume_test_resumed.log").open("w") as log:
        subprocess.run(common + ["--name", paused, "--resume"], stdout=log, stderr=subprocess.STDOUT, check=True)
    a = torch.load(ROOT / "runs" / reference / "last_adaptation.pt", map_location="cpu", weights_only=True)
    b = torch.load(folder / "last_adaptation.pt", map_location="cpu", weights_only=True)
    mismatch = [name for name in a["state"] if not torch.equal(a["state"][name], b["state"][name])]
    report = {"reference": reference, "resumed": paused, "bitwise_equal": not mismatch,
              "mismatch_parameters": mismatch, "performance_evidence": False,
              "purpose": "Checkpoint correctness only; do not report tiny-run EER as model performance"}
    (log_dir / "resume_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if mismatch:
        raise AssertionError("Resumed weights differ from uninterrupted run")


if __name__ == "__main__":
    main()
