"""Offline PANNs ResNet38 presence head with an explicit AudioSet label contract."""
import csv
import json
from pathlib import Path
import torch
from resnet38_architecture import ResNet38


class PresenceModel:
    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    def inference(self, waveforms):
        with torch.inference_mode():
            result = self.model(torch.as_tensor(waveforms, dtype=torch.float32, device=self.device))
        return result["clipwise_output"].cpu().numpy(), None


def load_presence(folder, device):
    folder = Path(folder)
    with (folder / "class_labels_indices.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {r["display_name"]:int(r["index"]) for r in rows}
    groups = json.loads((folder / "component_labels.json").read_text())
    assert len(groups["voice"]) == len(set(groups["voice"])) == 19
    assert len(groups["music"]) == len(set(groups["music"])) == 117
    assert "A capella" in groups["voice"]
    assert "A capella" not in groups["music"]
    model = ResNet38(sample_rate=32000, window_size=1024, hop_size=320,
                     mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    state = torch.load(folder / "ResNet38_mAP=0.434.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return PresenceModel(model, device), [lookup[n] for n in groups["voice"]], [lookup[n] for n in groups["music"]]
