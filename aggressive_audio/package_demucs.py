"""Assemble the completed stem experiment without altering previous submissions."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / 'submission_eat_xlsr_demucs_resnet38_mean_v1'


def assemble():
    shutil.copytree(ROOT / 'submission_eat_xlsr_aug_resnet38_mean_v1', STAGE,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'output'))
    path = STAGE / 'script.py'
    source = path.read_text(encoding='utf-8')
    start = source.index('    for path in files:\n        audio = base.load_audio(path)')
    end = source.index('    folder = ROOT / "model/speech_xlsr"', start)
    source = source[:start] + source[end:]
    old = '        voice, _ = base.separate_voice_and_music(path, demucs, device)'
    assert source.count(old) == 1
    source = source.replace(old, '''        voice, accompaniment = base.separate_voice_and_music(path, demucs, device)
        if len(accompaniment) == 0:
            raise ValueError("Empty separated music: " + str(path))
        if len(accompaniment) < 64000:
            accompaniment = np.tile(accompaniment, (64000 + len(accompaniment) - 1) // len(accompaniment))
        scores[path.stem] = {"MUSIC_FAKE_PROB": probability(music, accompaniment[:64000], device)}''')
    path.write_text(source, encoding='utf-8')
    provenance = STAGE / 'model/provenance'
    for branch in ['music', 'speech']:
        name = f'{branch}_demucs_medium_v1'
        dst = provenance / name
        dst.mkdir()
        for filename in ['config.json', 'history.json', 'augmentation_counts.json', 'complete.json']:
            shutil.copy2(ROOT / 'runs' / name / filename, dst / filename)
    shutil.copy2(ROOT / 'stem_experiment_medium_v1/RESULTS.md', provenance / 'DEMUCS_RESULTS.md')
    info = STAGE / 'model/MODEL_INFO.md'
    text = info.read_text(encoding='utf-8')
    text = text.replace('EAT + XLS-R augmented + ResNet38 CPS mean v1', 'EAT + XLS-R Demucs branches + ResNet38 CPS mean v1')
    text = text.replace('first 4 seconds of original mix', 'first 4 seconds of HTDemucs drums+bass+other sum')
    text = text.replace('music_head_channel_1024_v1 head adaptation', 'music_demucs_medium_v1 AASIST+Linear adaptation')
    text = text.replace('speech_probe_channel_v1 probe adaptation', 'speech_demucs_medium_v1 probe adaptation')
    text += '\nBoth encoders remain frozen. Training uses the same previous selected IDs, augmentation, seed and 2 epochs; only inputs change to separated stems. Previous run provenance is retained for comparison. Inference keeps previous speech 64600-sample segment/max policy and presence-gated mean fusion.\n'
    info.write_text(text, encoding='utf-8')
    print(STAGE, flush=True)


def merge():
    import torch
    for branch in ['music', 'speech']:
        delta = torch.load(ROOT / f'runs/{branch}_demucs_medium_v1/best_adaptation.pt', map_location='cpu', weights_only=True)
        assert delta['branch'] == branch
        if branch == 'music':
            path = STAGE / 'model/eat_fmc/model_state.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert all(not k.startswith('frontend.') for k in delta['state'])
            assert all(k in state and state[k].shape == v.shape for k, v in delta['state'].items())
            state.update(delta['state'])
        else:
            path = STAGE / 'model/speech_xlsr/model.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert set(delta['state']) == {'proj_fc.weight', 'proj_fc.bias'}
            state['probe'] = {k.removeprefix('proj_fc.'): v for k, v in delta['state'].items()}
        torch.save(state, path)
        del state
        print(f'Merged {branch} best adaptation', flush=True)
    (STAGE / 'model/provenance/stem_export.json').write_text(json.dumps({
        'music_run': 'music_demucs_medium_v1', 'speech_run': 'speech_demucs_medium_v1',
        'encoder_conversion': 'Reused previously verified frozen torchaudio XLS-R encoder; replaced only probe',
        'routing': 'HTDemucs vocals -> speech; drums+bass+other -> music',
    }, indent=2))


if __name__ == '__main__':
    {'assemble': assemble, 'merge': merge}[sys.argv[1]]()
