"""Evaluate a trained MusicDET checkpoint on available FMC generator splits."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

GENERATORS = {
    'TTM01': 'MusicGen', 'TTM02': 'MusicLDM', 'TTM03': 'AudioLDM2',
    'TTM04': 'Stable Audio Open', 'TTM05': 'Mustango',
}


def eer(real_scores, fake_scores):
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
    order = np.argsort(scores, kind='mergesort'); labels = labels[order]
    misses = np.concatenate([[0.0], np.cumsum(labels) / len(real_scores)])
    false_alarms = np.concatenate([[1.0],
        (len(fake_scores) - (np.arange(1, len(scores)+1) - np.cumsum(labels))) / len(fake_scores)])
    index = int(np.argmin(np.abs(misses - false_alarms)))
    return float((misses[index] + false_alarms[index]) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    from dataset import FakeMusicCapsDataset
    from model import SpecNF
    config=json.loads((args.checkpoint_dir/'args.json').read_text())
    model=SpecNF(K=config['K'],L=config['L'],R=config['R'])
    state=torch.load(args.checkpoint_dir/'anti-spoofing_feat_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); model.cuda().eval()
    reports=[]
    for prefix,generator in GENERATORS.items():
        protocol=args.protocol_dir/f'{prefix}_eval.txt'
        dataset=FakeMusicCapsDataset(str(args.audio_dir),str(protocol),random_sampling=False)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                          pin_memory=True,persistent_workers=args.workers>0)
        values=[]
        with torch.inference_mode():
            for waveform,filenames,labels in loader:
                waveform=waveform.cuda(non_blocking=True)
                real_prior=torch.zeros(len(waveform),dtype=torch.long,device='cuda')
                _,nll=model(waveform,real_prior)
                for filename,label,score in zip(filenames,labels,nll.cpu().tolist()):
                    values.append((Path(filename).stem,int(label),float(score)))
        labels=np.asarray([v[1] for v in values]); fake_scores=np.asarray([v[2] for v in values])
        real_scores=-fake_scores[labels==0]; generated_real_direction=-fake_scores[labels==1]
        report=dict(split=prefix,generator=generator,samples=len(values),real=int((labels==0).sum()),
                    fake=int((labels==1).sum()),eer=eer(real_scores,generated_real_direction),
                    roc_auc_fake=float(roc_auc_score(labels,fake_scores)))
        reports.append(report); print(json.dumps(report),flush=True)
        with (args.output_dir/f'{prefix}_scores.csv').open('w',newline='') as handle:
            writer=csv.writer(handle); writer.writerow(['ID','LABEL_FAKE','MUSIC_ANOMALY_SCORE'])
            writer.writerows(values)
    summary=dict(generators=reports,mean_eer=float(np.mean([r['eer'] for r in reports])),
                 score_definition='NLL under real prior; higher means fake',checkpoint=str(args.checkpoint_dir))
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
