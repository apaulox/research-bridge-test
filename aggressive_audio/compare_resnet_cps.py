"""Record effects of the CPS replacement without claiming a DACON CPS score."""
import csv
import json
from pathlib import Path
from contracts import eer

ROOT = Path(__file__).resolve().parent
with (ROOT / "fusion_diagnostic/predictions_v1/submission_mean.csv").open() as f:
    old = {r["ID"]:r for r in csv.DictReader(f)}
with (ROOT / "records/resnet38_submission_reference.csv").open() as f:
    new = {r["ID"]:r for r in csv.DictReader(f)}
assert old.keys() == new.keys()
labels = json.loads((ROOT / "fusion_diagnostic/labels.json").read_text())
columns = [c for c in next(iter(old.values())) if c != "ID"]
changes = {c:max(abs(float(old[k][c])-float(new[k][c])) for k in old) for c in columns}
result = {"max_absolute_probability_changes":changes,
          "original_cnn14_file_eer":eer([r["label_file_fake"] for r in labels], [float(old[r["ID"]]["FILE_FAKE_PROB"]) for r in labels]),
          "resnet38_file_eer":eer([r["label_file_fake"] for r in labels], [float(new[r["ID"]]["FILE_FAKE_PROB"]) for r in labels]),
          "voice_labels":19, "music_labels":117, "added_voice_label":"A capella",
          "ads_weights_changed":False,
          "runtime_note":"Speech moved from fairseq to torchaudio with math SDPA. Same-backend original/conversion probability difference <1e-6 on conversion probes. Relative to the earlier fused-backend diagnostic, speech probabilities differ by up to the measured value; not bitwise-identical inference.",
          "scope":"64 reused-source external FILE diagnostic; no DACON CPS labels or score available.",
          "user_reported_score":{"total":0.6899,"ads":0.656,"cps":0.995,"new_zip_result":False}}
(ROOT / "records/resnet38_cps_comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
