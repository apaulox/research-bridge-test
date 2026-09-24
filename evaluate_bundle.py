"""Run from a downloaded W&B best artifact; FAIR-Play data stays local."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--output-dir', default='evaluation_outputs')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    bundle = source.parent
    metadata = json.loads((bundle / 'metadata.json').read_text())
    backbone = metadata['config']['visual_backbone']
    if backbone != 'dinov2_vitb14_reg':
        raise ValueError('Portable runner currently supports DINOv2+register only')
    audio = str(Path(args.audio_dir).resolve())
    video = str(Path(args.video_dir).resolve())
    output = str(Path(args.output_dir).resolve())
    if not Path(audio).is_dir() or not Path(video).is_dir():
        raise FileNotFoundError('FAIR-Play audios and extracted frames are required')
    if not torch.cuda.is_available():
        raise RuntimeError('This inference implementation requires a CUDA GPU')
    original_load = torch.hub.load
    def local_load(repo, model, *values, **kwargs):
        if repo == 'facebookresearch/dinov2':
            # Full frozen backbone weights are already in visual_best.pth.
            kwargs.update(source='local', pretrained=False)
            repo = str(bundle / 'dinov2_source')
        return original_load(repo, model, *values, **kwargs)
    torch.hub.load = local_load
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = ['demo_batch.py', '--visual_backbone', backbone,
                '--head_layout', metadata['config'].get('head_layout', 'multi'),
                '--split_file', str(bundle / 'split.json'),
                '--weights_visual', str(bundle / 'weights/visual_best.pth'),
                '--weights_audio', str(bundle / 'weights/audio_best.pth'),
                '--audio_dir', audio, '--video_dir', video,
                '--output_dir_root', output, '--hop_size', '0.05',
                '--checkpoints_dir', str(bundle / 'inference_options')]
    try:
        runpy.run_path(str(source / 'demo_batch.py'), run_name='__main__')
    finally:
        torch.hub.load = original_load
    sys.argv = ['evaluate.py', '--results_root', output, '--normalization', 'True']
    runpy.run_path(str(source / 'evaluate.py'), run_name='__main__')


if __name__ == '__main__':
    main()
