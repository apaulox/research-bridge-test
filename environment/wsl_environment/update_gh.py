"""Install an official checksum-verified GitHub CLI binary for the WSL user."""
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'dacon-gh-setup'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


release = json.loads(fetch('https://api.github.com/repos/cli/cli/releases/latest'))
version = release['tag_name'].removeprefix('v')
arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}[platform.machine()]
name = f'gh_{version}_linux_{arch}.tar.gz'
assets = {asset['name']: asset['browser_download_url'] for asset in release['assets']}
checksums = fetch(assets[f'gh_{version}_checksums.txt']).decode()
expected = next(line.split()[0] for line in checksums.splitlines()
                if line.split()[-1].lstrip('*') == name)
payload = fetch(assets[name])
assert hashlib.sha256(payload).hexdigest() == expected, 'Checksum mismatch'
target = Path.home() / '.local/bin/gh'
target.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    binary = archive.extractfile(f'gh_{version}_linux_{arch}/bin/gh').read()
temporary = target.with_suffix('.new')
temporary.write_bytes(binary)
temporary.chmod(0o755)
temporary.replace(target)
subprocess.run([str(target), '--version'], check=True)
print(f'Installed: {target}; archive SHA256: {expected}', flush=True)
