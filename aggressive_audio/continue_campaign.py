"""Resume the bounded campaign without overwriting completed or interrupted runs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    common = [sys.executable, "-B", str(ROOT / "run.py"), "train", "--epochs", "2",
              "--train-per-class", "512", "--dev-per-class", "128", "--balance-generators"]
    tasks = [
        ("music_last2_channel_v2", ["--branch", "music", "--aug", "channel", "--last-layers", "2", "--lr", "1e-6"]),
        ("speech_probe_none_v1", ["--branch", "speech", "--aug", "none", "--manifest", str(ROOT / "manifests/speech_extended.jsonl")]),
        ("speech_probe_channel_v1", ["--branch", "speech", "--aug", "channel", "--manifest", str(ROOT / "manifests/speech_extended.jsonl")]),
        ("music_head_channel_1024_v1", ["--branch", "music", "--aug", "channel"]),
        ("music_head_channel_overlay_1024_v1", ["--branch", "music", "--aug", "channel", "--voice-overlay",
                                              "--overlay-manifest", str(ROOT / "manifests/speech_extended.jsonl")]),
    ]
    for name, options in tasks:
        folder = ROOT / "runs" / name
        if (folder / "complete.json").exists():
            print(json.dumps({"already_complete": name}), flush=True)
            continue
        if (ROOT / "STOP_CAMPAIGN").exists() or (folder / "STOP").exists():
            print(json.dumps({"paused_before": name}), flush=True)
            return
        command = common + ["--name", name] + options
        if folder.exists():
            if not (folder / "checkpoint_latest.pt").is_file():
                raise RuntimeError(f"Incomplete run has no full checkpoint: {name}")
            command.append("--resume")
        print(json.dumps({"starting": name}), flush=True)
        logfile = ROOT / "records" / (name + ".log")
        with logfile.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(logfile.read_text()[-3000:], flush=True)
            raise RuntimeError(f"Run failed: {name}")
        if not (folder / "complete.json").exists():
            print(json.dumps({"paused": name, "resume": str(folder / "checkpoint_latest.pt")}), flush=True)
            return
        history = json.loads((folder / "history.json").read_text())
        print(json.dumps({"completed": name, "last_metrics": history[-1]}), flush=True)
    print("Controlled campaign complete.", flush=True)


if __name__ == "__main__":
    main()
