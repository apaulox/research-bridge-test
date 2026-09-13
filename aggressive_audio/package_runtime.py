"""Assemble inference-only code and provenance, excluding training audio/state."""
import ast
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "submission_eat_xlsr_aug_mean_v1"


def main():
    runtime = STAGE / "model/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    helpers = (ROOT.parent / "script_eat75_musicdet25_fmc.py").read_text()
    helpers = helpers[:helpers.index("# 5. DF-Arena")]
    (runtime / "audio_helpers.py").write_text(helpers)
    source = (ROOT / "models.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "load_eat")
    loader = ast.get_source_segment(source, node).replace("path=DEFAULT_EAT", "path")
    (runtime / "eat_loader.py").write_text("from pathlib import Path\nimport sys\nimport torch\n\n" + loader + "\n")
    shutil.copy2(ROOT / "submission_script.py", STAGE / "script.py")
    # These packages are provided by the DACON baseline image. Do not replace its CUDA torch stack.
    shutil.copy2(ROOT.parent / "submission_audit/wsl/baseline_submit/requirements.txt", STAGE / "requirements.txt")
    provenance = STAGE / "model/provenance"
    provenance.mkdir(exist_ok=True)
    for name in ["music_head_channel_1024_v1", "speech_probe_channel_v1"]:
        dst = provenance / name
        dst.mkdir(exist_ok=True)
        for file in ["config.json", "history.json", "augmentation_counts.json"]:
            shutil.copy2(ROOT / "runs" / name / file, dst / file)
    for file in ["README.md", "RESULTS_2026-09-12.md"]:
        shutil.copy2(ROOT / file, provenance / file)
    info = """# EAT + XLS-R augmented mean v1

Inference: EAT+AASIST music only (first 4 seconds of original mix); AntiDeepfake XLS-R1B
and temporal-mean linear probe on HTDemucs vocals (64600-sample segments, segment max).
PANNs presence. FILE=(VOICE_PRESENT*VOICE_FAKE + MUSIC_PRESENT*MUSIC_FAKE)/2.
Class 0 in both model logits is fake. File probabilities do not use other input files.

Music: official DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2 weights, then our
music_head_channel_1024_v1 head adaptation merged into model_state.pt.
Speech: official nii-yamagishilab/xls-r-1b-anti-deepfake revision
60c96543210117bdb256636e7961e16463f99291, then speech_probe_channel_v1 probe adaptation.
Encoder weights mapped to torchaudio; conversion check records numerical agreement.
This is an adapted derivative, not an unchanged official checkpoint.

Training provenance: FMC/MusicCaps real and five music generators; ASVspoof2019 LA
and CodecFake+ CoSG, balanced selected training rows. Exact configurations are included.
No DACON test training, pseudo-labeling, or cross-file calibration.
No training audio, optimizer states, local evaluation WAVs, or evaluation labels in ZIP.

Attribution and terms:
- DeepFense code: bundled Apache 2.0 notice; model card https://huggingface.co/DeepFense/FakeMusicCaps_EAT_AASIST_NoAug_Seed2
- FMC: https://zenodo.org/records/15063698 CC-BY-NC-4.0
- MusicCaps metadata: https://huggingface.co/datasets/google/MusicCaps CC-BY-SA-4.0; individual audio rights separate.
- Speech original and adapted derivative: CC-BY-NC-SA-4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/
  https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake ; original model card included.
- ASVspoof2019: https://zenodo.org/records/6906306 ODC-By
- CoSG: https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset MIT dataset card; source demo terms separate.
- PANNs/HTDemucs: unchanged competition baseline assets.

Use for noncommercial competition evaluation under the above component terms.
Local execution verification does not constitute DACON server acceptance or a licensing ruling.
Run: python script.py --test-dir data/test --sample-submission data/sample_submission.csv --output output/submission.csv
"""
    (STAGE / "model/MODEL_INFO.md").write_text(info)
    print(json.dumps({"runtime_ready":str(STAGE)}))


if __name__ == "__main__":
    main()
