"""Build the CLAM batch16 reproduction with baseline checks and no timm."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

WORKSPACE = Path(__file__).resolve().parents[1]
DACON = Path('/home/huskypaul/dacon')
SOURCE = DACON / 'submissions/clam_official_v1'
STAGE = DACON / 'submissions/clam_official_v2'
VALIDATION = DACON / 'clam/validation/official_v2'
LOCAL_ZIP = DACON / 'submissions/submit_clam_official_v2.zip'
OUTPUT = WORKSPACE / 'submit_clam_official_v2.zip'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    resume = '--resume' in sys.argv
    if OUTPUT.exists() or LOCAL_ZIP.exists() or (STAGE.exists() and not resume):
        raise FileExistsError('Do not overwrite an existing v2 package')
    STAGE.mkdir(parents=True, exist_ok=resume)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    (STAGE / 'model').mkdir(exist_ok=resume)
    code = (WORKSPACE / 'archive/script_clam_official_v2.py').read_text()
    compile(code, 'script.py', 'exec')
    baseline = ast.parse((DACON / 'baseline/script.py').read_text())
    new = ast.parse(code)
    baseline_functions = {n.name: ast.dump(n) for n in baseline.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef)}
    differences = [name for name in baseline_functions if baseline_functions[name] != new_functions.get(name)]
    assert set(differences) == {'parse_arguments', 'predict_fake_scores_for_all_files', 'main'}, differences
    for name in ('df_arena_1b', 'htdemucs', 'panns', 'clam'):
        if not (STAGE / 'model' / name).exists():
            shutil.copytree(SOURCE / 'model' / name, STAGE / 'model' / name,
                            copy_function=os.link, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.cache'))
    shutil.copy2(WORKSPACE / 'archive/script_clam_official_v2.py', STAGE / 'script.py')
    shutil.copy2(SOURCE / 'requirements.txt', STAGE / 'requirements.txt')
    shutil.copy2(SOURCE / 'model/MODEL_INFO.txt', STAGE / 'model/MODEL_INFO.txt')
    assert (STAGE / 'requirements.txt').read_bytes() == (DACON / 'baseline/requirements.txt').read_bytes()
    info = json.loads((SOURCE / 'model/CLAM_OFFICIAL_INFO.json').read_text())
    info.update(version='v2', music_model='Official CLAM head batch16 reproduction',
                music_probability='torch.sigmoid(float32 raw_logit), real=0 fake=1',
                batch_size=16, batch_tail='Keep remainder without padding',
                batch_order='Sort IDs then local torch randperm with seed=42; restore baseline CSV order',
                caveat='Preserves released checkpoint cross-file attention; does not fix its learned batch dependence',
                official_differences=['Explicit seed 42 for reproducible grouping; official test loader shuffle is not explicitly seeded',
                                      'Local offline loading; encoder eval mode explicit; baseline failure handling and CSV interface'],
                changes_from_v1=['Head batch1 -> batch16', 'Stable float64 sigmoid -> official float32 sigmoid'],
                baseline_unchanged_functions=[name for name in baseline_functions if name not in differences])
    (STAGE / 'model/CLAM_OFFICIAL_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    for path in STAGE.rglob('*.py'):
        assert 'timm' not in path.read_text().lower(), path
    print('STAGED v2, 19 original baseline functions match; no timm source', flush=True)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with (VALIDATION / 'verification.log').open('w') as log:
        try:
            subprocess.run([sys.executable, str(WORKSPACE / 'archive/verify_clam_official_v2.py'),
                            '--stage', str(STAGE), '--validation', str(VALIDATION)],
                           check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            log.flush()
            print((VALIDATION / 'verification.log').read_text()[-16000:], flush=True)
            raise
    report = json.loads((VALIDATION / 'report.json').read_text())
    print(json.dumps(report, indent=2), flush=True)
    with (VALIDATION / 'package.log').open('w') as log:
        subprocess.run([sys.executable, str(WORKSPACE / 'common/build_dacon_submission_zip.py'),
                        '--source', str(STAGE), '--output', str(LOCAL_ZIP)],
                       check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    expected_hash = digest(LOCAL_ZIP)
    with LOCAL_ZIP.open('rb') as source, OUTPUT.open('xb') as target:
        shutil.copyfileobj(source, target, length=8*1024*1024)
    assert digest(OUTPUT) == expected_hash
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.read('script.py') == (STAGE / 'script.py').read_bytes()
        assert archive.read('requirements.txt') == (STAGE / 'requirements.txt').read_bytes()
    manifest = dict(info, zip_sha256=expected_hash, zip_bytes=OUTPUT.stat().st_size, validation=report)
    (WORKSPACE / 'results/clam_official_v2_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(VALIDATION / 'submission.csv', WORKSPACE / 'results/clam_official_v2_smoke.csv')
    print(f'COMPLETE {OUTPUT} bytes={OUTPUT.stat().st_size} SHA256={expected_hash}', flush=True)


if __name__ == '__main__':
    main()
