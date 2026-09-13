import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = []
for platform in ('windows', 'wsl'):
    records.extend(json.loads((root / 'submission_audit' / platform / 'archive_inventory.json').read_text(encoding='utf-8')))
by_name = {Path(r['path']).name: r for r in records}
pairs = [
    ('submit_eat75_musicdet.zip', 'submit_eat75_musicdet25_fmc_v1.zip'),
    ('submit_eat75_musicdet25.zip', 'submit_eat75_musicdet25_dependencyfix.zip'),
    ('submit_eat75_musicdet25_fmc_v1.zip', 'submit_eat75_musicdet25_fmc_v2.zip'),
    ('submit_eat75_musicdet25_fmc_v2.zip', 'submit_eat75_musicdet25_dependencyfix.zip'),
    ('submit_eat75_clam25_fmc_v1.zip', 'submit_eat75_clam25_fmc_v2.zip'),
    ('submit_eat75_clam25_fmc_v2.zip', 'submit_eat75_clam25_fmc_v3.zip'),
    ('submit_musicdet_fmc_v1.zip', 'musicdet_fmc_file_probe.zip'),
    ('submit_clam_official_v1.zip', 'submit_clam_official_v2.zip'),
]
result=[]
for left,right in pairs:
    a,b=by_name[left],by_name[right]
    ea={x['name']:(x['bytes'],x['crc']) for x in a['entries']}
    eb={x['name']:(x['bytes'],x['crc']) for x in b['entries']}
    changes=[n for n in sorted(ea.keys()|eb.keys()) if ea.get(n)!=eb.get(n)]
    item=dict(left=left,right=right,changed_entries=changes,script_identical=a['texts'].get('script.py')==b['texts'].get('script.py'),
              model_weights_same_crc_and_size=not any(n.endswith(('.pt','.pth','.bin','.safetensors','.th')) for n in changes))
    result.append(item)
    print(json.dumps(item))
(root / 'submission_audit' / 'archive_comparisons.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
