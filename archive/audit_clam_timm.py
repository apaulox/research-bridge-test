"""Audit shipped Python sources, then smoke-test with every timm import blocked."""

import ast
import csv
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zipfile


WORKSPACE = Path(__file__).resolve().parents[1]
STAGE = Path('/home/huskypaul/dacon/submissions/clam_official_v1')
VALIDATION = Path('/home/huskypaul/dacon/clam/validation/official_v1/timm_audit')


class BlockTimm(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'timm' or fullname.startswith('timm.'):
            self.attempts.append(fullname)
            raise ModuleNotFoundError('timm intentionally blocked for submission audit')
        return None


def main():
    started = time.perf_counter()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    code_files = []
    imports = []
    textual_references = []
    with zipfile.ZipFile(WORKSPACE / 'submit_clam_official_v1.zip') as archive:
        names = archive.namelist()
        assert not any(name.startswith('model/eat_fmc/') for name in names)
        requirements = archive.read('requirements.txt').decode()
        assert 'timm' not in requirements.lower()
        for name in names:
            if not name.endswith(('.py', '.json', '.txt')):
                continue
            data = archive.read(name)
            # Execute only staged runtime files byte-identical to the archive.
            assert data == (STAGE / name).read_bytes(), name
            if not name.endswith('.py'):
                continue
            code_files.append(name)
            source = data.decode('utf-8-sig')
            for number, line in enumerate(source.splitlines(), 1):
                if 'timm' in line.lower():
                    textual_references.append({'file': name, 'line': number, 'text': line.strip()})
            for node in ast.walk(ast.parse(source)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                imports.extend({'file': name, 'module': module} for module in modules
                               if module == 'timm' or module.startswith('timm.'))
    assert not imports, imports
    assert not any(name == 'timm' or name.startswith('timm.') for name in sys.modules)
    # Avoid reusing previously cached Hugging Face custom model modules.
    os.environ['HF_MODULES_CACHE'] = str(VALIDATION / 'fresh_hf_modules')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    blocker = BlockTimm()
    sys.meta_path.insert(0, blocker)
    # A genuinely absent optional package returns None from find_spec.
    # Raising there would incorrectly break Transformers' availability probe.
    original_find_spec = importlib.util.find_spec
    availability_probes = []

    def no_timm_spec(name, package=None):
        if name == 'timm' or name.startswith('timm.'):
            availability_probes.append(name)
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = no_timm_spec
    # Prove the guard works before testing the pipeline.
    assert importlib.util.find_spec('timm') is None
    try:
        __import__('timm')
    except ModuleNotFoundError as error:
        assert 'intentionally blocked' in str(error)
    else:
        raise AssertionError('timm guard did not block import')
    blocker.attempts.clear()
    availability_probes.clear()
    output = VALIDATION / 'submission.csv'
    sys.argv = [str(STAGE / 'script.py'),
                '--test-dir', '/home/huskypaul/dacon/baseline/data/test',
                '--sample-submission', '/home/huskypaul/dacon/baseline/data/sample_submission.csv',
                '--output', str(output)]
    print(f'AUDITED {len(code_files)} shipped Python files; running with timm blocked', flush=True)
    runpy.run_path(str(STAGE / 'script.py'), run_name='__main__')
    loaded = [name for name in sys.modules if name == 'timm' or name.startswith('timm.')]
    assert not loaded and not blocker.attempts, (loaded, blocker.attempts)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    with (WORKSPACE / 'results/clam_official_v1_smoke.csv').open() as handle:
        previous = list(csv.DictReader(handle))
    assert len(rows) == len(previous) == 3
    max_delta = 0.0
    for row, old in zip(rows, previous):
        assert row['ID'] == old['ID']
        for key in row:
            if key != 'ID':
                max_delta = max(max_delta, abs(float(row[key]) - float(old[key])))
    assert max_delta < 1e-5, max_delta
    result = dict(zip_file='submit_clam_official_v1.zip',
                  shipped_python_files_checked=len(code_files),
                  staged_runtime_matches_zip=True,
                  direct_timm_imports=imports, textual_timm_references=textual_references,
                  requirements=requirements.strip(),
                  timm_guard_self_test_passed=True,
                  runtime_timm_import_attempts=blocker.attempts,
                  optional_timm_availability_probes=availability_probes,
                  loaded_timm_modules=loaded,
                  fresh_hf_module_cache=True, full_pipeline_samples=len(rows),
                  maximum_probability_delta=max_delta,
                  runtime_versions={name: importlib.metadata.version(name)
                                    for name in ('torch', 'torchaudio', 'transformers')},
                  seconds=time.perf_counter() - started,
                  verdict='PASS: submitted CLAM pipeline runs without timm',
                  scope='WSL dacon-cu128, provided smoke audio; not a replica of the DACON server')
    (WORKSPACE / 'results/clam_official_v1_timm_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
