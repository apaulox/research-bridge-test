"""A bounded sequence of controlled trials. Every child must succeed before the next starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["music-last2", "speech", "music-overlay"])
    args = p.parse_args()
    common = [sys.executable, "-B", str(ROOT / "run.py"), "train", "--epochs", "2",
              "--train-per-class", "512", "--dev-per-class", "128", "--balance-generators"]
    if args.stage == "music-last2":
        commands = [common + ["--branch", "music", "--name", f"music_last2_{aug}_v1", "--aug", aug,
                             "--last-layers", "2", "--lr", "1e-6"] for aug in ["none", "channel"]]
    elif args.stage == "speech":
        commands = [common + ["--branch", "speech", "--name", f"speech_probe_{aug}_v1", "--aug", aug,
                             "--manifest", str(ROOT / "manifests/speech_extended.jsonl")]
                    for aug in ["none", "channel"]]
    else:
        # A matched control with the SAME head-only learning rate and data size.
        commands = [common + ["--branch", "music", "--name", "music_head_channel_1024_v1", "--aug", "channel"],
                    common + ["--branch", "music", "--name", "music_head_channel_overlay_1024_v1", "--aug", "channel",
                              "--voice-overlay", "--overlay-manifest", str(ROOT / "manifests/speech_extended.jsonl")]]
    for command in commands:
        print(json.dumps({"launch": command}), flush=True)
        subprocess.run(command, check=True)
        name = command[command.index("--name") + 1]
        if (ROOT / "runs" / name / "STOP").exists():
            print(json.dumps({"paused_stage": args.stage, "run": name}), flush=True)
            return
    print(json.dumps({"completed_stage": args.stage}), flush=True)


if __name__ == "__main__":
    main()
