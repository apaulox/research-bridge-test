"""Read public MoM metadata and saved notebook outputs without training."""
import csv
import collections
import json
from pathlib import Path

repo = Path('/home/huskypaul/dacon/clam/MoM-CLAM')
held_out = {'riffusion', 'suno_3', 'AI_COVERS', 'suno_4', 'Yue'}
report = {}
for name in ('ai_generated_music_metadata.csv', 'real_songs_2.csv', 'real_yt_covers.csv'):
    with (repo / name).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    entry = {'rows': len(rows), 'columns': columns}
    for field in ('model_name', 'no_vocal', 'is_instrumental', 'split', 'type', 'label'):
        if field in columns:
            entry[field] = dict(collections.Counter(row[field] for row in rows))
    if name.startswith('ai_'):
        entry['train_validation_candidates'] = dict(collections.Counter(
            row['model_name'] for row in rows if row['model_name'] not in held_out))
        entry['test_candidates'] = dict(collections.Counter(
            row['model_name'] for row in rows if row['model_name'] in held_out))
    report[name] = entry
notebook = json.loads((repo / 'train_triplet_loss.ipynb').read_text())
interesting = []
for index, cell in enumerate(notebook['cells']):
    source = ''.join(cell.get('source', []))
    if any(term in source for term in ('value_counts', 'len(train', 'len(test', 'remove_filename',
                                       'train_real_dataset', 'fake_df_train', 'label =', 'label=')):
        output = ''.join(''.join(item.get('text', [])) + ''.join(item.get('data', {}).get('text/plain', []))
                         for item in cell.get('outputs', []))
        interesting.append({'cell': index, 'source': source[:1600], 'output': output[-1800:]})
report['notebook_evidence'] = interesting
print(json.dumps(report, indent=2, ensure_ascii=False))
