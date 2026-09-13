"""Add launchers/notebooks and record verification for the prepared WSL project."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import nbformat

root = Path('/home/huskypaul/dacon')
workspace = Path('/mnt/c/Users/husky_gp7j99y/Documents/ChatGPT/DACON')
baseline = root / 'baseline'
environment = '/home/huskypaul/miniforge3/envs/dacon-cu128'
activate = '''# Run: source ~/dacon/activate.sh
source /home/huskypaul/miniforge3/etc/profile.d/conda.sh
conda activate dacon-cu128
cd /home/huskypaul/dacon/baseline
'''
launcher = f'''#!/usr/bin/env bash
set -euo pipefail
export PATH="{environment}/bin:$PATH"
cd /home/huskypaul/dacon
exec python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.root_dir=/home/huskypaul/dacon
'''
(root / 'activate.sh').write_text(activate)
(root / 'start_jupyter.sh').write_text(launcher)
(root / 'start_jupyter.sh').chmod(0o755)

notebook = nbformat.v4.new_notebook()
notebook.metadata.kernelspec = dict(display_name='Python (DACON CUDA 12.8)', language='python', name='dacon-cu128')
notebook.cells = [
    nbformat.v4.new_markdown_cell('# DACON CUDA 환경 시작\n\n위에서 아래로 Shift+Enter로 실행하세요. 제공된 3개 파일은 동작 확인용입니다. 이 결과로 대회 성능을 판단하지 않습니다. MusicDET 학습 완료 가중치는 아직 없습니다.'),
    nbformat.v4.new_code_cell("from pathlib import Path\nimport os\nimport torch\nos.chdir(Path.home() / 'dacon/baseline')\nprint('작업 폴더:', Path.cwd())\nprint('PyTorch:', torch.__version__)\nprint('CUDA:', torch.version.cuda)\nassert torch.cuda.is_available()\nprint('GPU:', torch.cuda.get_device_name(0))"),
    nbformat.v4.new_markdown_cell('## 베이스라인 실행\n\nPANNs → HTDemucs → DF-Arena를 실행합니다. 기존 output/submission.csv가 있으면 새 결과로 갱신됩니다.'),
    nbformat.v4.new_code_cell('%run script.py'),
    nbformat.v4.new_code_cell("import pandas as pd\npd.read_csv('output/submission.csv')"),
]
nbformat.validate(notebook)
nbformat.write(notebook, baseline / '00_start_here.ipynb')
originals = list((root / 'notebooks').glob('*.ipynb'))
if len(originals) == 1:
    original = nbformat.read(originals[0], as_version=4)
    original.metadata.kernelspec = notebook.metadata.kernelspec
    for cell in original.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    # Running copy sits next to model/ and data/, preserving the original code.
    nbformat.validate(original)
    nbformat.write(original, baseline / '01_original_baseline.ipynb')

with (baseline / 'data/sample_submission.csv').open(encoding='utf-8-sig') as handle:
    expected = list(csv.DictReader(handle))
with (baseline / 'output/submission.csv').open(encoding='utf-8-sig') as handle:
    actual = list(csv.DictReader(handle))
assert [r['ID'] for r in expected] == [r['ID'] for r in actual]
columns = ['FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB']
for row in actual:
    for column in columns:
        value = float(row[column])
        assert math.isfinite(value) and 0 <= value <= 1, (row['ID'], column)
    combined = max(float(row['VOICE_FAKE_PROB']) * float(row['VOICE_PRESENT_PROB']),
                   float(row['MUSIC_FAKE_PROB']) * float(row['MUSIC_PRESENT_PROB']))
    assert abs(combined - float(row['FILE_FAKE_PROB'])) < 1e-8

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()

hash_report = []
for line in (baseline / 'model/SHA256SUMS.txt').read_text().splitlines():
    if not line.strip():
        continue
    expected_hash, relative = line.split(maxsplit=1)
    relative = relative.lstrip('*')
    candidate = (baseline / relative).resolve()
    if not candidate.is_file():
        candidate = (baseline / 'model' / relative).resolve()
    assert candidate.is_relative_to(baseline), candidate
    assert digest(candidate) == expected_hash, relative
    hash_report.append(relative)

report = dict(baseline_sample_count=len(actual), prediction_ranges_valid=True,
              ids_match=True, max_fusion_verified=True, model_checksums_verified=hash_report,
              baseline_script_sha256=digest(baseline / 'script.py'),
              notebook_kernel='dacon-cu128', musicdet_checkpoint_available=False)
(root / 'logs/final_verification.json').write_text(json.dumps(report, indent=2))
output_folder = workspace / 'environment/wsl_environment'
output_folder.mkdir(exist_ok=True)
for name in ['cuda_verification.json', 'environment.json', 'final_verification.json', 'baseline-smoke.log']:
    shutil.copy2(root / 'logs' / name, output_folder / name)
shutil.copy2(root / 'setup/requirements-lock.txt', output_folder / 'requirements-lock.txt')
shutil.copy2(baseline / 'output/submission.csv', output_folder / 'baseline_smoke_submission.csv')
shutil.copy2(baseline / '00_start_here.ipynb', output_folder / '00_start_here.ipynb')
print(json.dumps(report, indent=2))
