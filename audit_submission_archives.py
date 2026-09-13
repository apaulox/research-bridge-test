"""Read archive metadata and small provenance files without loading model pickles."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--wsl', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
archives = list(a.root.glob('*.zip'))
if a.wsl:
    archives = list((a.root / 'submissions').glob('*.zip')) + list((a.root / 'downloads').glob('*.zip'))
reports = []
for archive in sorted(archives):
    with zipfile.ZipFile(archive) as z:
        record = dict(path=str(archive), bytes=archive.stat().st_size, entries=[], texts={})
        for item in z.infolist():
            record['entries'].append(dict(name=item.filename, bytes=item.file_size, crc=item.CRC))
            name = item.filename
            wanted = name in ('script.py', 'requirements.txt') or ('INFO' in name and name.endswith('.txt')) or (name.endswith('.json') and any(x in name.lower() for x in ('calibration','args','config','manifest'))) or name.endswith('README.md')
            if wanted and item.file_size < 100000:
                content = z.read(item).decode('utf-8', errors='replace')
                record['texts'][name] = content
                target = a.output / archive.stem / name
                if not target.resolve().is_relative_to(a.output.resolve()):
                    raise ValueError(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8')
        record['entry_fingerprint'] = hashlib.sha256(json.dumps(record['entries'],sort_keys=True).encode()).hexdigest()
        reports.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('entries','texts')}))
(a.output / 'archive_inventory.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
if a.wsl:
    gathered = {}
    for pattern in ('musicdet/runs/**/*.json','musicdet/data_preparation/**/*.json','musicdet/evaluation/**/*.json','clam/validation/**/*.json','eat_musicdet/**/*.json'):
        for source in a.root.glob(pattern):
            if source.stat().st_size < 100000:
                gathered[str(source)] = source.read_text(errors='replace')
    (a.output / 'wsl_records.json').write_text(json.dumps(gathered,ensure_ascii=False,indent=2),encoding='utf-8')
    print('WSL_RECORDS', list(gathered))
