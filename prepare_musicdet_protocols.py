"""Extract FMC protocols and audit their split/metadata mapping before training."""
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

root = Path.home() / 'dacon/musicdet'
prep = root / 'data_preparation'
repo = root / 'vendor/MusicDET'
with zipfile.ZipFile(prep / 'official_protocols.zip') as archive:
    for entry in archive.infolist():
        if entry.is_dir() or not entry.filename.startswith('label/fakemusiccaps/openset/'):
            continue
        target = (repo / entry.filename).resolve()
        if not target.is_relative_to(repo.resolve()):
            raise ValueError(entry.filename)
        data = archive.read(entry)
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
with (prep / 'musiccaps-public.csv').open() as handle:
    metadata = {r['ytid']: r for r in csv.DictReader(handle)}
all_protocols = {}
report = {}
for path in sorted((repo / 'label/fakemusiccaps/openset').glob('*.txt')):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    assert all(len(r) >= 2 and r[1] in {'real', 'fake'} for r in rows), (path.name, rows[:3])
    # Official dataset.py consumes the first two fields; protocols also contain a numeric label.
    rows = [row[:2] for row in rows]
    assert len({r[0] for r in rows}) == len(rows), f'Duplicates: {path}'
    report[path.name] = dict(Counter(r[1] for r in rows))
    report[path.name]['examples'] = rows[:2]
    all_protocols[path.stem] = rows
splits = {}
for split in ['train', 'dev', 'eval']:
    rows = all_protocols['TTM01_' + split]
    splits[split] = {Path(name).stem for name, label in rows if label == 'real'}
    report['same_real_' + split] = all(
        {Path(name).stem for name, label in all_protocols[f'TTM0{i}_{split}'] if label == 'real'} == splits[split]
        for i in range(2, 6))
report['real_split_overlaps'] = {
    f'{a}/{b}': len(splits[a] & splits[b])
    for a, b in [('train', 'dev'), ('train', 'eval'), ('dev', 'eval')]}
assert not any(report['real_split_overlaps'].values()), 'Real train/dev/eval overlap'
report['unmatched_real_ids'] = sorted(set.union(*splits.values()) - metadata.keys())
if not report['unmatched_real_ids']:
    with (prep / 'required_real_audio.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ytid', 'start_s', 'end_s', 'split'])
        writer.writeheader()
        for split, ids in splits.items():
            for audio_id in sorted(ids):
                item = metadata[audio_id]
                writer.writerow(dict(ytid=audio_id, start_s=item['start_s'], end_s=item['end_s'], split=split))
report['training_started'] = False
report['real_audio_downloaded'] = False
(prep / 'protocol_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
