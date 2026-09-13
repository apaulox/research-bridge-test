"""Compare CLAM v2 with official batching/extractors and the DACON baseline."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    dacon = Path('/home/huskypaul/dacon')
    repo = dacon / 'clam/MoM-CLAM'
    args.validation.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_MODULES_CACHE=str(args.validation / 'fresh_hf_modules'),
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    attempts = []

    class NoTimm(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'timm' or fullname.startswith('timm.'):
                attempts.append(fullname)
                raise ModuleNotFoundError('timm intentionally blocked')
            return None

    sys.meta_path.insert(0, NoTimm())
    original_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, package=None: (
        None if name == 'timm' or name.startswith('timm.')
        else original_find_spec(name, package))
    try:
        __import__('timm')
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('timm blocker failed')
    attempts.clear()
    import torch
    import torchaudio
    # Validation-only controls: the unmodified baseline varies by ~0.003 on
    # repeat GPU runs. Apply the same deterministic settings to both programs.
    # These settings are not added to the submitted script.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    script = load('submission_v2', args.stage / 'script.py')
    voice_inputs = {'v2': [], 'baseline': []}
    def recorded_predict(function, name):
        def predict(*arguments, **keywords):
            # Baseline model output itself varies on repeated identical calls.
            # Verify exact waveform delivery rather than requiring its stochastic
            # output to be bit-identical across separate calls.
            audio = arguments[2]
            voice_inputs[name].append(hashlib.sha256(audio.tobytes()).hexdigest())
            return function(*arguments, **keywords)
        return predict
    script.predict_fake = recorded_predict(script.predict_fake, 'v2')
    official_module = load('official_clam', repo / 'models/clam.py')
    official = official_module.CLAM(13, 13, 768, 768).eval()
    model = script.load_clam_head(torch.device('cpu'))
    official.load_state_dict(model.state_dict(), strict=True)
    torch.manual_seed(2718)
    batch_checks = []
    for count in (1, 3, 16, 17, 33):
        files = [Path(f'CHECK_{i:04d}.wav') for i in range(count)]
        mert = {path.stem: torch.randn(13, 768) for path in files}
        wav = {path.stem: torch.randn(13, 768) for path in files}
        before = torch.get_rng_state().clone()
        actual = script.predict_clam_embedding_batches(model, files, mert, wav, torch.device('cpu'))
        assert torch.equal(before, torch.get_rng_state()), 'Music shuffle changed baseline RNG'
        # Reference uses the official training forward method and a real DataLoader.
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(42)).tolist()
        dataset = [(index, mert[files[index].stem], wav[files[index].stem]) for index in permutation]
        expected = {}
        with torch.inference_mode():
            for indices, x, y in torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False):
                values = torch.sigmoid(official.forward_training(x, y)[0].view(-1)).tolist()
                expected.update({files[index].stem: value for index, value in zip(indices.tolist(), values)})
        assert actual == expected, count
        reordered = script.predict_clam_embedding_batches(model, list(reversed(files)), mert, wav, torch.device('cpu'))
        assert actual == reordered, 'CSV ordering changed music grouping'
        batch_checks.append({'files': count, 'official_probability_max_delta': 0.0,
                             'tail_batch': count % 16, 'id_order_invariant': True})
    del model, official
    print('PASS official batch16, partial batches, ID mapping and baseline RNG', flush=True)

    # Use 17 distinct existing files solely to exercise a full and partial batch.
    # No labels are used by the submitted model or for fitting any parameter.
    candidates = {0: [], 1: []}
    for line in (dacon / 'musicdet/label_available/TTM01_eval.txt').read_text().splitlines():
        name, label = line.split()
        category = int(label == 'fake')
        root = dacon / 'musicdet/data_raw' / ('fmc' if category else 'musiccaps')
        path = root / name
        if path.is_file() and len(candidates[category]) < (9 if category else 8):
            candidates[category].append(path)
        if len(candidates[0]) == 8 and len(candidates[1]) == 9:
            break
    selected = candidates[0] + candidates[1]
    assert len(selected) == 17
    audio_dir = args.validation / 'test'
    audio_dir.mkdir(exist_ok=True)
    for index, path in enumerate(selected):
        target = audio_dir / f'AUDIT_{index:04d}{path.suffix}'
        if not target.exists():
            os.link(path, target)
    sample_path = args.validation / 'sample_submission.csv'
    ids = [path.stem for path in sorted(audio_dir.iterdir())][::-1]
    with sample_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ID'] + script.PREDICTION_COLUMNS)
        writer.writeheader()
        for audio_id in ids:
            writer.writerow({'ID': audio_id, **{column: 0 for column in script.PREDICTION_COLUMNS}})

    # Execute function bodies from the actual official extractor files, avoiding
    # their top-level online model-loading code. Compare live model embeddings.
    original_extract = script.extract_clam_embedding
    extractor_checks = {}
    cached_mert, cached_wav = {}, {}

    def checked_extract(model, processor, path, device):
        rate = int(processor.sampling_rate)
        actual = original_extract(model, processor, path, device)
        (cached_mert if rate == 24000 else cached_wav)[path.stem] = actual
        if rate not in extractor_checks:
            is_mert = rate == 24000
            filename = 'mertextraction.py' if is_mert else 'wave2vec2extract.py'
            name = 'mert_embedding_fn' if is_mert else 'wav2vec2_embedding_fn'
            tree = ast.parse((repo / 'extractors' / filename).read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = {'torch': torch, 'torchaudio': torchaudio, 'T': torchaudio.transforms,
                         'model_mert' if is_mert else 'model': model,
                         'processor_mert' if is_mert else 'processor': processor}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
            expected = namespace[name](str(path), duration=90, device=device).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            extractor_checks[rate] = float((actual - expected).abs().max())
        return actual

    script.extract_clam_embedding = checked_extract
    output = args.validation / 'submission.csv'
    sys.argv = ['script.py', '--test-dir', str(audio_dir), '--sample-submission', str(sample_path), '--output', str(output)]
    script.main()
    with output.open() as handle:
        current = list(csv.DictReader(handle))
    assert [row['ID'] for row in current] == ids
    # Quantify the actual batch-vs-singleton change on these audio embeddings.
    head = script.load_clam_head(torch.device('cpu'))
    with torch.inference_mode():
        single_scores = {
            audio_id: torch.sigmoid(head(cached_mert[audio_id][None], cached_wav[audio_id][None])).item()
            for audio_id in ids}
    music_change = max(abs(float(row['MUSIC_FAKE_PROB']) - single_scores[row['ID']]) for row in current)
    del head, cached_mert, cached_wav
    baseline = load('dacon_original_baseline', dacon / 'baseline/script.py')
    baseline.predict_fake = recorded_predict(baseline.predict_fake, 'baseline')
    baseline_output = args.validation / 'baseline_submission.csv'
    sys.argv = ['script.py', '--test-dir', str(audio_dir), '--sample-submission', str(sample_path), '--output', str(baseline_output)]
    baseline.main()
    with baseline_output.open() as handle:
        old = {row['ID']: row for row in csv.DictReader(handle)}
    max_delta = {column: 0.0 for column in ('VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB')}
    for row in current:
        numbers = {key: float(row[key]) for key in script.PREDICTION_COLUMNS}
        assert all(0 <= value <= 1 for value in numbers.values())
        for column in max_delta:
            max_delta[column] = max(max_delta[column], abs(numbers[column]-float(old[row['ID']][column])))
        expected_file = max(numbers['VOICE_PRESENT_PROB']*numbers['VOICE_FAKE_PROB'],
                            numbers['MUSIC_PRESENT_PROB']*numbers['MUSIC_FAKE_PROB'])
        assert abs(numbers['FILE_FAKE_PROB']-expected_file) < 2e-10
    # Original baseline calls predict_fake on voice then music for each file.
    assert voice_inputs['v2'] == voice_inputs['baseline'][::2], 'Voice waveform changed'
    assert max_delta['VOICE_PRESENT_PROB'] == 0.0
    assert max_delta['MUSIC_PRESENT_PROB'] == 0.0
    assert not attempts
    assert not any(name == 'timm' or name.startswith('timm.') for name in sys.modules)
    result = dict(batch_checks=batch_checks, official_extractor_embedding_max_deltas=extractor_checks,
                  distinct_audio_files=17, max_unchanged_branch_deltas=max_delta,
                  comparison_uses_validation_only_deterministic_settings=True,
                  voice_waveforms_identical_to_original_baseline=True,
                  voice_output_note='Code, assets and input preserved; repeated baseline inference itself varies. Numerical output differences are reported, not hidden.',
                  max_music_change_vs_singleton=music_change,
                  file_formula_verified=True, csv_order_verified=True,
                  timm_import_attempts=attempts, seconds=time.perf_counter()-started,
                  scope='Functional parity tests; external audio used only for validation, not training/calibration')
    (args.validation / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
