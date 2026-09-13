"""Validate relocation and the current pipeline's shared source references."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
moves = json.loads((ROOT / 'results/file_moves.json').read_text())
for old, new in moves.items():
    assert (ROOT / new).is_file(), new
    assert not (ROOT / old).is_file(), old

sources = [ROOT / new for new in moves.values() if new.endswith('.py')]
sources += list((ROOT / 'aggressive_audio').glob('*.py'))
sources += list((ROOT / 'environment').glob('*.py'))
for path in set(sources):
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

# Moving the shared helper must not change audio processing or its function bodies.
before = subprocess.check_output(['git', 'show', '19617e8:script_eat75_musicdet25_fmc.py'], cwd=ROOT).decode()
after = (ROOT / 'common/script_eat75_musicdet25_fmc.py').read_text(encoding='utf-8')
assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
for filename in ['infer.py', 'package_runtime.py']:
    text = (ROOT / 'aggressive_audio' / filename).read_text()
    assert 'ROOT.parent / "common/script_eat75_musicdet25_fmc.py"' in text
assert (ROOT / 'results/submission_audit/wsl/baseline_submit/requirements.txt').is_file()
for filename in ['README.md', 'common/README.md', 'environment/README.md', 'archive/README.md', 'results/README.md']:
    path = ROOT / filename
    for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
        if '://' not in link:
            assert (path.parent / link).exists(), (filename, link)
print(json.dumps({'relocated_files': len(moves), 'python_syntax_checked': len(set(sources)),
                  'shared_audio_code_unchanged': True, 'current_pipeline_paths_exist': True,
                  'navigation_links_valid': True}))
