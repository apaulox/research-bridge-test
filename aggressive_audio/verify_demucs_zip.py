"""Verify the actual stem submission offline on eight different mixture cases."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parent / 'submit_eat_xlsr_demucs_resnet38_mean_v1.zip'


def main():
    folder = Path('/home/huskypaul/dacon/submissions/verify_eat_xlsr_demucs_resnet38_mean_v1')
    folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        assert {'script.py', 'requirements.txt'} <= set(archive.namelist())
        for entry in archive.infolist():
            assert (folder / entry.filename).resolve().is_relative_to(folder.resolve())
        archive.extractall(folder)
    # Reuse the exact previously audited network/import blocker, extracted as a literal.
    tree = ast.parse((ROOT / 'verify_submission_zip.py').read_text())
    wrapper_text = next(n.args[0].value for n in ast.walk(tree)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == 'write_text' and n.args
                        and isinstance(n.args[0], ast.Constant)
                        and isinstance(n.args[0].value, str)
                        and n.args[0].value.startswith('import importlib.abc'))
    wrapper = folder / 'offline_check.py'
    wrapper.write_text(wrapper_text)
    labels = json.loads((ROOT / 'fusion_diagnostic/labels.json').read_text())
    selected = labels[::8]
    audio = folder / 'test_audio'
    audio.mkdir()
    for row in selected:
        shutil.copy2(row['path'], audio / (row['ID'] + '.wav'))
    with (ROOT / 'records/resnet38_submission_reference.csv').open() as handle:
        fields = csv.DictReader(handle).fieldnames
    template = folder / 'sample_submission.csv'
    with template.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row['ID'] if k == 'ID' else 0 for k in fields} for row in reversed(selected))
    output = folder / 'verified_submission.csv'
    subprocess.run([sys.executable, '-B', str(wrapper), str(folder / 'script.py'),
                    '--test-dir', str(audio), '--sample-submission', str(template),
                    '--output', str(output)], cwd=folder, check=True)
    with output.open() as handle:
        predictions = list(csv.DictReader(handle))
    assert [r['ID'] for r in predictions] == [r['ID'] for r in reversed(selected)]
    for row in predictions:
        values = {k: float(row[k]) for k in fields if k != 'ID'}
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values.values())
        expected = (values['VOICE_PRESENT_PROB'] * values['VOICE_FAKE_PROB']
                    + values['MUSIC_PRESENT_PROB'] * values['MUSIC_FAKE_PROB']) / 2
        assert abs(values['FILE_FAKE_PROB'] - expected) < 1e-12
    # Confirm packaged heads exactly match the completed runs.
    import torch
    for branch in ['music', 'speech']:
        delta = torch.load(ROOT / f'runs/{branch}_demucs_medium_v1/best_adaptation.pt', map_location='cpu', weights_only=True)['state']
        path = folder / ('model/eat_fmc/model_state.pt' if branch == 'music' else 'model/speech_xlsr/model.pt')
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key, value in delta.items():
            actual = state[key] if branch == 'music' else state['probe'][key.removeprefix('proj_fc.')]
            assert torch.equal(actual, value), key
        del state
    digest = hashlib.sha256()
    with ARCHIVE.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    result = {'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
              'sha256': digest.hexdigest(), 'zip_crc_passed': True,
              'offline_extracted_zip_inference_passed': True, 'blocked': ['network', 'timm', 'fairseq'],
              'files': len(predictions), 'scenarios': [r['scenario'] for r in selected],
              'template_order_preserved': True, 'mean_fusion_verified': True,
              'adapted_heads_exact_match': True, 'python': sys.version,
              'server_execution_verified': False}
    (ROOT / 'records/demucs_submission_zip_verification.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(output, ROOT / 'records/demucs_submission_smoke.csv')
    (ROOT.parent / 'results' / (ARCHIVE.name + '.sha256')).write_text(digest.hexdigest() + '  ' + ARCHIVE.name + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
