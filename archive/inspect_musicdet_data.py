"""Fetch small public metadata files before downloading training audio."""
import csv
import io
import json
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlsplit
import zipfile

import requests

root = Path.home() / 'dacon/musicdet/data_preparation'
root.mkdir(parents=True, exist_ok=True)
sources = {
    'musiccaps-public.csv': 'https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv',
    'official_protocols.zip': 'https://drive.google.com/uc?export=download&id=1CnC8G6Kp6WfF3XcX0tJI6F6RrfAAY7uI',
}
report = {}

class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.active = True
            self.action = attrs.get('action')
        if self.active and tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False

for name, url in sources.items():
    try:
        if name.endswith('.zip'):
            first = requests.get(url, timeout=(10, 30), stream=True)
            first.raise_for_status()
            if 'text/html' in first.headers.get('Content-Type', ''):
                form = DownloadForm()
                form.feed(first.text)
                if form.action and urlsplit(form.action).hostname == 'drive.usercontent.google.com':
                    # Normal public-file "download anyway" form, not authentication.
                    url = requests.Request('GET', form.action, params=form.fields).prepare().url
            first.close()
        with requests.get(url, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024**2:
                    raise ValueError('Metadata download exceeded 100 MB limit')
                chunks.append(chunk)
            content = b''.join(chunks)
        if name.endswith('.csv'):
            rows = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
            if not rows or not {'ytid', 'start_s', 'end_s'}.issubset(rows[0]):
                raise ValueError('Not the expected MusicCaps metadata')
            report[name] = dict(rows=len(rows), columns=list(rows[0]))
        else:
            if not zipfile.is_zipfile(io.BytesIO(content)):
                title = re.search(r'<title>(.*?)</title>', content.decode('utf-8', errors='replace'), re.S)
                raise ValueError(f'Expected ZIP, received {response.headers.get("Content-Type")}; '
                                 f'bytes={len(content)}; signature={content[:8].hex()}; '
                                 f'page_title={title.group(1) if title else "none"}')
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                report[name] = dict(files=[dict(name=i.filename, bytes=i.file_size)
                                          for i in archive.infolist()])
        target = root / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f'Different metadata already exists: {target}')
        target.write_bytes(content)
        report[name]['saved'] = str(target)
        report[name]['url'] = url
    except Exception as exc:
        report[name] = dict(error=f'{type(exc).__name__}: {exc}', url=url)
print(json.dumps(report, indent=2))
(root / 'source_check.json').write_text(json.dumps(report, indent=2))
