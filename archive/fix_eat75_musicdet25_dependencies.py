"""Repair only packaging dependencies of the exact reported submission ZIP."""
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import zipfile
import zlib

W=Path(__file__).resolve().parents[1]
D=Path('/home/huskypaul/dacon')
S=D/'submissions/eat75_musicdet25_dependencyfix'
V=D/'musicdet/validation/dependencyfix'
SOURCE=W/'submit_eat75_musicdet25.zip'
GOOD=D/'submissions/eat75_clam25_fmc_v2'
EXISTING=D/'submissions/eat75_musicdet25_fmc_v1'
OUT=W/'submit_eat75_musicdet25_dependencyfix.zip'
LOCAL=D/'submissions'/OUT.name

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
    return h.hexdigest()

def main():
    assert not OUT.exists() and not LOCAL.exists()
    S.mkdir(parents=True,exist_ok=True)
    V.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(SOURCE) as z:
        for entry in z.infolist():
            if entry.is_dir(): continue
            path=S/entry.filename
            assert path.resolve().is_relative_to(S.resolve())
            if path.exists(): continue
            path.parent.mkdir(parents=True,exist_ok=True)
            candidate=EXISTING/entry.filename
            crc=0
            if candidate.is_file() and candidate.stat().st_size==entry.file_size:
                with candidate.open('rb') as f:
                    for data in iter(lambda:f.read(8*1024*1024),b''): crc=zlib.crc32(data,crc)
                if crc==entry.CRC:
                    os.link(candidate,path)
                    continue
            with z.open(entry) as src,path.open('xb') as dst: shutil.copyfileobj(src,dst,8*1024*1024)
        changes={}
        for name in ['eat_model.py','model_core.py']:
            relative='model/eat_fmc/eat_architecture/'+name
            original=z.read(relative).decode()
            replacement=(GOOD/relative).read_text()
            changes[relative]=''.join(difflib.unified_diff(original.splitlines(True),replacement.splitlines(True)))
            temp=S/(relative+'.tmp')
            temp.write_text(replacement)
            temp.replace(S/relative)
        temp=S/'requirements.tmp'
        temp.write_bytes((D/'baseline/requirements.txt').read_bytes())
        temp.replace(S/'requirements.txt')
        assert (S/'script.py').read_bytes()==z.read('script.py')
        (V/'dependency_patch.diff').write_text('\n'.join(changes.values()))
    for path in S.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            assert not any(n=='timm' or n.startswith('timm.') for n in modules),path
    from audit_clam_timm import BlockTimm
    blocker=BlockTimm()
    sys.meta_path.insert(0,blocker)
    old_spec=importlib.util.find_spec
    importlib.util.find_spec=lambda name,package=None: None if name=='timm' or name.startswith('timm.') else old_spec(name,package)
    os.environ['HF_MODULES_CACHE']=str(V/'fresh_hf_modules')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    spec=importlib.util.spec_from_file_location('submission',S/'script.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.argv=['script.py','--test-dir',str(D/'baseline/data/test'),'--sample-submission',str(D/'baseline/data/sample_submission.csv'),'--output',str(V/'submission.csv')]
    if not Path(sys.argv[4]).exists(): sys.argv[4]=str(D/'baseline/sample_submission.csv')
    m.main()
    assert not blocker.attempts
    subprocess.run([sys.executable,str(W/'common/build_dacon_submission_zip.py'),'--source',str(S),'--output',str(LOCAL)],check=True)
    with zipfile.ZipFile(SOURCE) as original,zipfile.ZipFile(LOCAL) as fixed:
        assert set(original.namelist())==set(fixed.namelist())
        changed=[name for name in original.namelist() if (original.getinfo(name).CRC,original.getinfo(name).file_size)!=(fixed.getinfo(name).CRC,fixed.getinfo(name).file_size)]
        assert set(changed)=={'requirements.txt','model/eat_fmc/eat_architecture/eat_model.py','model/eat_fmc/eat_architecture/model_core.py'},changed
    expected=sha(LOCAL)
    with LOCAL.open('rb') as src,OUT.open('xb') as dst: shutil.copyfileobj(src,dst,8*1024*1024)
    assert sha(OUT)==expected
    report=dict(source=SOURCE.name,output=OUT.name,sha256=expected,bytes=OUT.stat().st_size,changed_files=changed,
                script_identical=True,other_zip_entries_identical_crc_and_size=True,timm_import_attempts=blocker.attempts,
                smoke='Execution only, dummy inputs; no performance interpretation',
                limitation='DACON installer logs unavailable; server-side torch/torchaudio conflict not directly reproduced')
    (W/'results/eat75_musicdet25_dependencyfix_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
