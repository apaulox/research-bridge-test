"""Verify official head/preprocessing parity and run the complete sample pipeline."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch
import torchaudio


def load_module(name, path):
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
    script = load_module('clam_submission', args.stage / 'script.py')
    reference = load_module('official_clam', dacon / 'clam/MoM-CLAM/models/clam.py')
    official = reference.CLAM(13, 13, 768, 768).eval()
    head = script.load_clam_head(torch.device('cpu'))
    official.load_state_dict(head.state_dict(), strict=True)
    torch.manual_seed(314159)
    embeddings = [torch.randn(1, 13, 768), torch.randn(1, 13, 768)]
    with torch.inference_mode():
        torch.testing.assert_close(head(*embeddings), official(*embeddings), rtol=0, atol=0)
    del head, official
    # Stereo 44.1 kHz exercises native-rate resampling instead of 16->24 kHz.
    wave = torch.stack([torch.sin(torch.arange(44100) * 0.1),
                        torch.cos(torch.arange(44100) * 0.15)]) * 0.1
    test_audio = args.validation / 'native_rate_stereo.wav'
    torchaudio.save(str(test_audio), wave, 44100)
    for sample_rate in (16000, 24000):
        audio, original_rate = torchaudio.load(str(test_audio))
        audio = torchaudio.transforms.Resample(original_rate, sample_rate)(audio)
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Literal official extractor padding convention.
        audio = torch.cat([audio, torch.zeros(1, sample_rate * 90 - audio.shape[1])], dim=1)
        torch.testing.assert_close(script.make_clam_waveform(test_audio, sample_rate),
                                   audio.squeeze(0), rtol=0, atol=0)
    raw_logits = []
    original_loader = script.load_clam_head

    def capture_loader(device):
        model = original_loader(device)
        model.register_forward_hook(lambda _model, _inputs, output: raw_logits.append(float(output.view(-1)[0])))
        return model

    script.load_clam_head = capture_loader
    result_path = args.validation / 'submission.csv'
    sys.argv = ['script.py', '--test-dir', str(dacon / 'baseline/data/test'),
                '--sample-submission', str(dacon / 'baseline/data/sample_submission.csv'),
                '--output', str(result_path), '--device', 'cuda']
    script.main()
    with result_path.open() as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    with (dacon / 'baseline/data/sample_submission.csv').open(encoding='utf-8-sig') as handle:
        sample = list(csv.DictReader(handle))
    assert columns == ['ID'] + script.PREDICTION_COLUMNS
    assert [row['ID'] for row in rows] == [row['ID'] for row in sample]
    assert len(raw_logits) == len(rows) == 3
    maximum_baseline_delta = 0.0
    old_path = dacon / 'clam/validation/clam_eat_v2_no_timm_smoke.csv'
    with old_path.open() as handle:
        old = {row['ID']: row for row in csv.DictReader(handle)}
    for row, raw in zip(rows, raw_logits):
        values = {key: float(row[key]) for key in script.PREDICTION_COLUMNS}
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in values.values())
        expected = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()
        assert abs(values['MUSIC_FAKE_PROB'] - expected) < 6e-11
        expected_file = max(values['VOICE_PRESENT_PROB'] * values['VOICE_FAKE_PROB'],
                            values['MUSIC_PRESENT_PROB'] * values['MUSIC_FAKE_PROB'])
        assert abs(values['FILE_FAKE_PROB'] - expected_file) < 2e-10
        for column in ('VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB'):
            delta = abs(values[column] - float(old[row['ID']][column]))
            maximum_baseline_delta = max(maximum_baseline_delta, delta)
            assert delta < 1e-5, (row['ID'], column, delta)
    report = dict(samples=len(rows), official_head_exact_parity=True,
                  official_preprocessing_exact_parity=True,
                  music_matches_uncalibrated_official_sigmoid=True,
                  file_max_fusion_verified=True, column_and_id_order_verified=True,
                  max_voice_presence_delta_vs_v2=maximum_baseline_delta,
                  raw_music_logits=raw_logits,
                  music_probabilities=[float(row['MUSIC_FAKE_PROB']) for row in rows],
                  seconds=time.perf_counter() - started,
                  scope='Three provided smoke samples; no DACON leaderboard evaluation')
    (args.validation / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
