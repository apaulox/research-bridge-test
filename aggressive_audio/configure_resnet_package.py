import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "submission_eat_xlsr_aug_mean_v1"
NEW = ROOT / "submission_eat_xlsr_aug_resnet38_mean_v1"


def main():
    OLD.rename(NEW)
    assets = ROOT / "assets/panns_resnet38"
    panns = NEW / "model/panns"
    (panns / "Cnn14_mAP=0.431.pth").rename(ROOT / "assets/cancelled_cps_Cnn14.pth")
    for name in ["ResNet38_mAP=0.434.pth", "LICENSE_PANNS.txt", "provenance.json"]:
        shutil.copy2(assets / name, panns / name)
    groups = json.loads((panns / "component_labels.json").read_text())
    assert len(groups["voice"]) == 18 and len(groups["music"]) == 117
    groups["voice"].append("A capella")
    (panns / "component_labels.json").write_text(json.dumps(groups, indent=2))
    runtime = NEW / "model/runtime"
    shutil.copy2(assets / "resnet38_architecture.py", runtime / "resnet38_architecture.py")
    shutil.copy2(ROOT / "resnet_presence.py", runtime / "resnet_presence.py")
    script = (ROOT / "submission_script.py").read_text()
    script = script.replace("from eat_loader import load_eat", "from eat_loader import load_eat\nfrom resnet_presence import load_presence")
    script = script.replace('base.HTDEMUCS_DIR = ROOT / "model/htdemucs"',
                            'base.HTDEMUCS_DIR = ROOT / "model/htdemucs"\n    base.load_panns_model = lambda target_device: load_presence(base.PANNS_DIR, target_device)')
    (NEW / "script.py").write_text(script)
    info = NEW / "model/MODEL_INFO.md"
    text = info.read_text().replace("# EAT + XLS-R augmented mean v1", "# EAT + XLS-R augmented + ResNet38 CPS mean v1")
    text = text.replace("PANNs presence.", "PANNs ResNet38 official mAP=0.434 presence; Voice 19 (A capella added), Music 117.")
    text = text.replace("- PANNs/HTDemucs: unchanged competition baseline assets.",
                        "- PANNs ResNet38: official https://zenodo.org/records/3987831 ; unchanged pretrained weights, no additional training. Code MIT notice included.\n- HTDemucs: unchanged competition baseline assets.")
    info.write_text(text)
    print(json.dumps({"stage":str(NEW), "voice_labels":19, "music_labels":117}), flush=True)


if __name__ == "__main__":
    main()
