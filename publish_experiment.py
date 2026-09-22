"""Publish completed local training, scalar history and a portable best model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('--entity', default='stringwoo22-hanyang-university')
    parser.add_argument('--project', default='DenseSSL')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if Path(args.experiment).name != args.experiment:
        raise ValueError('Experiment must be a directory name')
    checkpoint = root / 'checkpoints' / args.experiment / 'mono2binaural'
    state = torch.load(checkpoint / 'training_latest.pth', map_location='cpu', weights_only=False)
    config = state['ablation_config']
    has_attention = 'attention_weight' in state['contrastive_criterion']
    if has_attention != (config['semantic_pool'] == 'visual_attention'):
        raise ValueError('Checkpoint pooling metadata and parameters disagree')
    best = torch.load(checkpoint / 'criterion_best.pth', map_location='cpu', weights_only=False)
    if best['ablation_config'] != config:
        raise ValueError('Best and latest checkpoint configurations disagree')
    history = defaultdict(dict)
    acc = EventAccumulator(str(checkpoint / 'tensorboard'), size_guidance={'scalars': 0}).Reload()
    for tag in acc.Tags()['scalars']:
        for event in acc.Scalars(tag):
            history[event.step][tag] = event.value
    val = [(step, data['data/val_loss']) for step, data in history.items() if 'data/val_loss' in data]
    if not val:
        raise ValueError('No validation history')
    best_step, best_value = min(val, key=lambda item: item[1])
    if abs(best_value - state['best_err']) > 1e-5:
        raise ValueError('TensorBoard minimum does not match saved best error')
    history[best_step]['data/best_checkpoint_val_loss'] = best_value
    summary = {'validation/best': float(state['best_err']), 'validation/best_step': best_step,
               'validation/final': max(val)[1], 'training/total_steps': state['total_steps'],
               'training/completed_epoch': state['next_epoch'] - 1}
    log = root / ('train_' + args.experiment + '.log')
    if log.exists():
        matches = re.findall(r'saving the best model \(epoch (\d+), total_steps (\d+)\)', log.read_text())
        epochs = [int(epoch) for epoch, step in matches if int(step) == best_step]
        if epochs:
            summary['validation/best_epoch'] = epochs[-1]
    if args.evaluation:
        evaluation_text = args.evaluation.read_text()
        for label, metric in [('STFT L2 Distance:', 'stft'), ('Average Envelope Distance:', 'envelope')]:
            match = re.search(re.escape(label) + r'\s+(\S+)\s+(\S+)\s+(\S+)', evaluation_text)
            if not match:
                raise ValueError('Missing evaluation metric: ' + label)
            for suffix, value in zip(('mean', 'std', 'se'), match.groups()):
                summary['test/' + metric + '_' + suffix] = float(value)
    for name in ('visual_best.pth', 'audio_best.pth', 'criterion_best.pth', 'opt.txt'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    print(json.dumps({'config': config, 'summary': summary, 'history_steps': len(history)}, indent=2))
    if args.check_only:
        return
    # Stable identity and a history cursor make retrying an upload safe.
    run_id = 'pool-' + args.experiment
    run = wandb.init(entity=args.entity, project=args.project, id=run_id, name=args.experiment,
                     resume='allow', mode='online', config=config, job_type='completed-training')
    try:
        run.define_metric('global_step')
        run.define_metric('data/*', step_metric='global_step')
        cursor = int(run.summary.get('uploaded_through_step', -1))
        for step in sorted(history):
            if step > cursor:
                run.log({'global_step': step, **history[step]})
        run.summary.update(summary)
        run.summary['uploaded_through_step'] = max(history)
        with tempfile.TemporaryDirectory(prefix='densessl-publish-') as temp:
            temp = Path(temp)
            metadata = {'experiment': args.experiment, 'config': config, 'summary': summary,
                        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                        'source_snapshot': 'Current working files, including uncommitted ablation changes',
                        'dataset_included': False, 'checkpoint_kind': 'best validation; inference only, not optimizer resume'}
            (temp / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            # Package versions, without URLs or credentials from pip configuration.
            packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=freeze'], text=True)
            (temp / 'environment.txt').write_text(packages)
            artifact = wandb.Artifact(args.experiment + '-best', type='model', metadata=metadata)
            artifact.add_file(str(temp / 'metadata.json'), name='metadata.json')
            artifact.add_file(str(temp / 'environment.txt'), name='environment.txt')
            for name in ('visual_best.pth', 'audio_best.pth', 'criterion_best.pth', 'opt.txt'):
                artifact.add_file(str(checkpoint / name), name='weights/' + name)
            artifact.add_file(config['split_file'], name='split.json')
            # Explicit source allowlist excludes credentials, data, outputs and logs.
            sources = list(root.glob('*.py'))
            for directory in ('models', 'options', 'data', 'util'):
                sources.extend((root / directory).rglob('*.py'))
            sources.extend(root.glob('LICENSE*'))
            for path in sorted(set(sources)):
                if path.is_file() and not path.is_symlink():
                    artifact.add_file(str(path), name='source/' + path.relative_to(root).as_posix())
            if config['visual_backbone'] == 'dinov2_vitb14_reg':
                hub = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
                if not (hub / 'hubconf.py').exists():
                    raise FileNotFoundError('DINOv2 cached source missing: ' + str(hub))
                for path in sorted(hub.rglob('*')):
                    if path.is_file() and not path.is_symlink() and (path.suffix in ('.py', '.yaml') or path.name.startswith('LICENSE')):
                        artifact.add_file(str(path), name='dinov2_source/' + path.relative_to(hub).as_posix())
            if args.evaluation:
                artifact.add_file(str(args.evaluation), name='evaluation.txt')
            logged = run.log_artifact(artifact, aliases=['best', 'latest'])
            logged.wait()
            run.summary['best_artifact'] = logged.qualified_name
            print('W&B run:', run.url)
            print('Best artifact:', logged.qualified_name)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == '__main__':
    main()
