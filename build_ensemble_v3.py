"""Stage, dev-calibrate, validate and package the batch16 ensemble."""
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

W = Path(__file__).resolve().parent
D = Path('/home/huskypaul/dacon')
S = D / 'submissions/eat75_clam25_fmc_v3'
V = D / 'clam/validation/ensemble_v3'
OLD = D / 'submissions/eat75_clam25_fmc_v2'

def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def functions(path):
    return {n.name: ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def main():
    V.mkdir(parents=True, exist_ok=True)
    if not S.exists():
        shutil.copytree(OLD / 'model', S / 'model', copy_function=os.link,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.cache'))
    shutil.copy2(W / 'script_eat75_clam25_fmc_v3.py', S / 'script.py')
    shutil.copy2(OLD / 'requirements.txt', S / 'requirements.txt')
    old, new = functions(OLD / 'script.py'), functions(S / 'script.py')
    changed = [k for k in old if old[k] != new.get(k)]
    assert set(changed) == {'calibrate_clam_fake', 'predict_music_scores_for_all_files'}, changed
    base = functions(D / 'baseline/script.py')
    base_changed = [k for k in base if base[k] != new.get(k)]
    assert set(base_changed) == {'predict_fake_scores_for_all_files', 'main'}, base_changed
    assert (S/'requirements.txt').read_bytes() == (D/'baseline/requirements.txt').read_bytes()
    from audit_clam_timm import BlockTimm
    blocker = BlockTimm()
    sys.meta_path.insert(0, blocker)
    original_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, package=None: None if name == 'timm' or name.startswith('timm.') else original_spec(name, package)
    os.environ['HF_MODULES_CACHE'] = str(V/'fresh_hf_modules')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    spec = importlib.util.spec_from_file_location('ensemble_v3', S/'script.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, roc_auc_score
    device = torch.device('cuda')
    groups = {}
    paths = {}
    for name in ['TTM01_dev_100x2', 'TTM01_eval_100x2'] + [f'TTM0{i}_eval_50x2' for i in range(2,6)]:
        with (D/'clam/validation'/f'{name}_eat.csv').open() as f:
            rows = list(csv.DictReader(f))
        groups[name] = rows
        for row in rows:
            folder = 'fmc' if int(row['LABEL_FAKE']) else 'musiccaps'
            path = D/'musicdet/data_raw'/folder/row['FILE']
            assert path.is_file(), path
            assert path.stem not in paths or paths[path.stem] == path
            paths[path.stem] = path
    dev_ids = {r['FILE'] for r in groups['TTM01_dev_100x2']}
    assert all(not dev_ids.intersection(r['FILE'] for r in rows) for name,rows in groups.items() if 'eval' in name)
    streams = []
    for backbone in ['MERT-v1-95M', 'wav2vec2-base-100k-gtzan-music-genres']:
        cache = V / f'{backbone}_embeddings.pt'
        embeddings = torch.load(cache, weights_only=True) if cache.exists() else {}
        missing = [p for key,p in sorted(paths.items()) if key not in embeddings]
        if missing:
            model, processor = m.load_clam_backbone(m.CLAM_DIR/backbone, device)
            for i,path in enumerate(missing, 1):
                embeddings[path.stem] = m.extract_clam_embedding(model, processor, m.load_audio(path), device)
                if i % 25 == 0 or i == len(missing):
                    torch.save(embeddings, cache)
                    print(f'{backbone}: {i}/{len(missing)}', flush=True)
            del model, processor
            m.clear_cuda()
        streams.append(embeddings)
    head = m.load_clam_head(device)
    logits = {}
    for name,rows in groups.items():
        files = [paths[Path(r['FILE']).stem] for r in rows]
        logits[name] = m.predict_clam_batch_logits(head, files, *streams, device)
    dev = groups['TTM01_dev_100x2']
    x = np.array([logits['TTM01_dev_100x2'][Path(r['FILE']).stem] for r in dev]).reshape(-1,1)
    y = np.array([int(r['LABEL_FAKE']) for r in dev])
    fit = LogisticRegression(C=1.0, solver='lbfgs').fit(x,y)
    calibration = dict(coefficient=float(fit.coef_[0,0]), intercept=float(fit.intercept_[0]), batch_size=16,
                       shuffle_seed=42, fit_split='TTM01_dev_100x2', real=100, fake=100,
                       input='v2 original mix decoded at 16k; 90s zero pad; MERT resample 24k',
                       purpose='Dev-only logistic score calibration; no checkpoint training')
    dump(S/'model/clam/fmc_batch16_calibration.json', calibration)
    def metrics(labels, scores):
        fpr,tpr,_ = roc_curve(labels,scores)
        fnr = 1-tpr
        idx = np.argmin(np.abs(fpr-fnr))
        return dict(eer=float((fpr[idx]+fnr[idx])/2), auc=float(roc_auc_score(labels,scores)))
    report = dict(calibration=calibration, changed_functions=changed, baseline_changed_functions=base_changed,
                  evaluation={}, unique_audio=len(paths), caveat='Batch peer dependence remains; seed and 75:25 fixed before eval.')
    for name,rows in groups.items():
        output = []
        for r in rows:
            raw = logits[name][Path(r['FILE']).stem]
            calibrated = m.calibrate_clam_fake(raw)
            eat = float(r['EAT_PROB'])
            output.append(dict(FILE=r['FILE'], LABEL_FAKE=int(r['LABEL_FAKE']), raw=raw,
                               calibrated=calibrated, eat=eat, ensemble=m.combine_music_fake_score(eat,calibrated)))
        with (V/f'{name}_v3.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
        report['evaluation'][name] = {key:metrics([r['LABEL_FAKE'] for r in output],[r[key] for r in output]) for key in ['raw','calibrated','eat','ensemble']}
    del head, streams
    m.clear_cuda()
    print(json.dumps(report,indent=2), flush=True)
    # Exercise full submission on 17 files: includes a final partial head batch.
    smoke = D/'clam/validation/official_v2'
    sys.argv = ['script.py','--test-dir',str(smoke/'test'),'--sample-submission',str(smoke/'sample_submission.csv'), '--output',str(V/'submission.csv')]
    m.main()
    with (V/'submission.csv').open(encoding='utf-8-sig') as f:
        submitted = list(csv.DictReader(f))
    with (smoke/'sample_submission.csv').open(encoding='utf-8-sig') as f:
        expected = list(csv.DictReader(f))
    assert len(submitted) == 17
    assert [r['ID'] for r in submitted] == [r['ID'] for r in expected]
    for r in submitted:
        assert all(0 <= float(r[k]) <= 1 for k in m.PREDICTION_COLUMNS)
        expected_file = max(float(r['VOICE_FAKE_PROB'])*float(r['VOICE_PRESENT_PROB']), float(r['MUSIC_FAKE_PROB'])*float(r['MUSIC_PRESENT_PROB']))
        assert abs(float(r['FILE_FAKE_PROB'])-expected_file) < 1e-8
    assert not blocker.attempts, blocker.attempts
    report.update(smoke_rows=17, timm_import_attempts=blocker.attempts, csv_order_and_file_max='passed',
                  contract_audit=json.loads((V/'contract_audit.json').read_text()))
    dump(V/'report.json',report)
    dump(S/'model/ENSEMBLE_V3_INFO.json',report)
    local = D/'submissions/submit_eat75_clam25_fmc_v3.zip'
    subprocess.run([sys.executable,str(W/'build_dacon_submission_zip.py'),'--source',str(S),'--output',str(local)],check=True)
    output = W/local.name
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    sha=digest(local)
    with local.open('rb') as src,output.open('xb') as dst: shutil.copyfileobj(src,dst,8*1024*1024)
    assert digest(output)==sha
    report.update(zip_sha256=sha,zip_bytes=output.stat().st_size)
    dump(W/'eat75_clam25_fmc_v3_manifest.json',report)
    shutil.copy2(V/'submission.csv', W/'eat75_clam25_fmc_v3_smoke.csv')
    print('DONE '+str(output),flush=True)

if __name__ == '__main__':
    main()
