"""Create a portable package from tracked files plus the explicitly identified data inputs."""
from pathlib import Path
import argparse, hashlib, json, subprocess, zipfile
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args()
destination=args.destination.resolve();destination.parent.mkdir(parents=True,exist_ok=True)
paths=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().strip('\x00').split('\x00')
paths+=['data/study_raw/macro_avg-00000-of-00001.parquet','data/study_raw/README.md','data/study_raw/manifest.json']
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for name in sorted(set(paths)): z.write(ROOT/name,name)
    z.writestr('study_revision.json',json.dumps({'source_commit':revision,'baseline':'61d5728eca5167516f62daf67032ece8f3893af7','branch':'course/empirical-study'},indent=2))
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))
print(f'{destination}: {destination.stat().st_size:,} bytes, SHA256 {hashlib.sha256(destination.read_bytes()).hexdigest()}')
