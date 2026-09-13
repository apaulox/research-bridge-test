"""Compare immutable baseline assets and repeat its unmodified voice inference."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path('/home/huskypaul/dacon')
validation = root / 'clam/validation/official_v2'
baseline = root / 'baseline'
stage = root / 'submissions/clam_official_v2'
different_assets = []
checked = 0
for folder in ('df_arena_1b', 'panns', 'htdemucs'):
    for path in (baseline / 'model' / folder).rglob('*'):
        if not path.is_file() or '.cache' in path.parts or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        relative = path.relative_to(baseline)
        other = stage / relative
        if not other.is_file():
            different_assets.append(str(relative))
            continue
        if path.stat().st_ino != other.stat().st_ino:
            def digest(item):
                result = hashlib.sha256()
                with item.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                        result.update(chunk)
                return result.digest()
            if digest(path) != digest(other):
                different_assets.append(str(relative))
        checked += 1
print(json.dumps({'checked_baseline_assets': checked, 'different_assets': different_assets}), flush=True)
assert not different_assets
spec = importlib.util.spec_from_file_location('baseline_repeat', baseline / 'script.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result_file = validation / 'baseline_repeat.csv'
sys.argv = ['script.py', '--test-dir', str(validation / 'test'),
            '--sample-submission', str(validation / 'sample_submission.csv'), '--output', str(result_file)]
module.main()
def read(path):
    with path.open() as handle:
        return {row['ID']: row for row in csv.DictReader(handle)}
original = read(validation / 'baseline_submission.csv')
repeat = read(result_file)
v2 = read(validation / 'submission.csv')
rows = []
for audio_id in original:
    values = [float(data[audio_id]['VOICE_FAKE_PROB']) for data in (original, repeat, v2)]
    rows.append({'ID': audio_id, 'baseline': values[0], 'baseline_repeat': values[1], 'v2': values[2],
                 'baseline_self_delta': abs(values[0]-values[1]), 'v2_delta': abs(values[1]-values[2])})
report = dict(checked_assets=checked, different_assets=different_assets,
              baseline_self_max_delta=max(row['baseline_self_delta'] for row in rows),
              v2_vs_repeat_max_delta=max(row['v2_delta'] for row in rows), rows=rows)
print(json.dumps(report, indent=2), flush=True)
(Path(__file__).resolve().parent / 'clam_v2_voice_diagnostic.json').write_text(json.dumps(report, indent=2))
