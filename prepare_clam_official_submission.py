"""Stage, smoke-test, and package the official CLAM-only DACON submission."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parent
DACON = Path('/home/huskypaul/dacon')
SOURCE = DACON / 'submissions/eat75_clam25_fmc_v2'
STAGE = DACON / 'submissions/clam_official_v1'
VALIDATION = DACON / 'clam/validation/official_v1'
OUTPUT = WORKSPACE / 'submit_clam_official_v1.zip'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    code = (WORKSPACE / 'script_clam_official.py').read_text(encoding='utf-8')
    compile(code, 'script.py', 'exec')
    previous = ast.parse((SOURCE / 'script.py').read_text())
    current = ast.parse(code)
    old_functions = {node.name: ast.dump(node) for node in previous.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node) for node in current.body if isinstance(node, ast.FunctionDef)}
    preserved = [
        'get_segment_starts', 'extract_segment', 'load_audio',
        'load_panns_model', 'make_panns_segments', 'predict_presence',
        'predict_presence_for_all_files', 'load_htdemucs_model',
        'separate_voice_and_music', 'load_df_arena_model', 'predict_fake',
        'combine_file_fake_score', 'predict_fake_scores_for_all_files',
    ]
    for name in preserved:
        assert old_functions[name] == new_functions[name], name
    assert 'EAT_ENSEMBLE' not in code and 'CALIBRATION_COEFFICIENT' not in code
    resume = '--resume' in sys.argv
    if (STAGE.exists() and not resume) or OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite an existing submission')
    if resume:
        assert (STAGE / 'script.py').read_text() == code, 'Staged code changed'
    STAGE.mkdir(parents=True, exist_ok=resume)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (STAGE / 'model').mkdir(exist_ok=resume)
    # Shared immutable model bytes, independent script/metadata files.
    for name in ('df_arena_1b', 'htdemucs', 'panns', 'clam'):
        if not resume:
            shutil.copytree(SOURCE / 'model' / name, STAGE / 'model' / name,
                            copy_function=os.link,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.cache'))
    shutil.copy2(WORKSPACE / 'script_clam_official.py', STAGE / 'script.py')
    shutil.copy2(SOURCE / 'requirements.txt', STAGE / 'requirements.txt')
    shutil.copy2(SOURCE / 'model/MODEL_INFO.txt', STAGE / 'model/MODEL_INFO.txt')
    official_head = DACON / 'clam/MoM-CLAM/best_model_triplet_loss_margin_0.2.pth'
    staged_head = STAGE / 'model/clam/best_model_triplet_loss_margin_0.2.pth'
    # Locate the official repository checkpoint without guessing its subfolder.
    matches = list((DACON / 'clam/MoM-CLAM').rglob(official_head.name))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one official checkpoint, found {matches}')
    assert sha256(matches[0]) == sha256(staged_head), 'Official checkpoint mismatch'
    info = {
        'music_model': 'Official pretrained MoM-CLAM only',
        'repository': 'https://github.com/StarkVision-AI/MoM-CLAM',
        'repository_commit': 'db7fff06e25415c119e58b8e4c4f48c840a845c8',
        'checkpoint': staged_head.name,
        'checkpoint_sha256': sha256(staged_head),
        'mert': 'm-a-p/MERT-v1-95M',
        'mert_revision': '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5',
        'wav2vec2': 'm3hrdadfi/wav2vec2-base-100k-gtzan-music-genres',
        'wav2vec2_revision': 'caf978c8328a2cec4229b3eb0b41b162e379caa1',
        'training': 'Published MoM checkpoint; no local fine-tuning',
        'dataset_scope': 'MoM real/synthetic songs; exact released checkpoint training subset unverified',
        'music_probability': 'sigmoid(raw_logit), official real=0 and fake=1',
        'fmc_calibration': False,
        'eat_ensemble': False,
        'input': 'Original mix, native-rate torchaudio decode, direct per-backbone resampling',
        'preprocessing': 'MERT 24kHz, Wav2Vec2 16kHz; mono; first 90s; trailing zero padding',
        'preprocessing_change_from_v2': 'Remove intermediate librosa 16kHz conversion for official extractor parity',
        'file_probability': 'max(VOICE_PRESENT_PROB * VOICE_FAKE_PROB, MUSIC_PRESENT_PROB * MUSIC_FAKE_PROB)',
        'preserved_baseline_functions': preserved,
        'requirements': 'Identical to working baseline/v2; no torch or torchvision installation',
    }
    (STAGE / 'model/CLAM_OFFICIAL_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    print('STAGED official checkpoint, baseline branches preserved', flush=True)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with (VALIDATION / 'smoke.log').open('w') as log:
        try:
            subprocess.run([sys.executable, str(WORKSPACE / 'verify_clam_official.py'),
                            '--stage', str(STAGE), '--validation', str(VALIDATION)],
                           check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            log.flush()
            print((VALIDATION / 'smoke.log').read_text()[-10000:], flush=True)
            raise
    print((VALIDATION / 'report.json').read_text(), flush=True)
    with (VALIDATION / 'package.log').open('w') as log:
        subprocess.run([sys.executable, str(WORKSPACE / 'build_dacon_submission_zip.py'),
                        '--source', str(STAGE), '--output', str(OUTPUT)],
                       check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
    summary = dict(info)
    summary.update(zip_bytes=OUTPUT.stat().st_size, zip_sha256=sha256(OUTPUT),
                   validation=json.loads((VALIDATION / 'report.json').read_text()))
    (WORKSPACE / 'clam_official_v1_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copy2(VALIDATION / 'submission.csv', WORKSPACE / 'clam_official_v1_smoke.csv')
    print(f'COMPLETE {OUTPUT} bytes={OUTPUT.stat().st_size} sha256={summary["zip_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
