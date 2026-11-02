"""Retrieve only the pinned evaluation table and its dataset card."""
from pathlib import Path
import hashlib, json
import requests
ROOT = Path(__file__).resolve().parents[1]
REV = '9919b5a0e61e57a85021263918fa82d6ceaee038'
out = ROOT / 'data/study_raw'
out.mkdir(parents=True, exist_ok=True)
records = []
for name in ['data/macro_avg-00000-of-00001.parquet', 'README.md']:
    url = f'https://huggingface.co/datasets/allenai/DataDecide-eval-results/resolve/{REV}/{name}'
    p = out / Path(name).name
    if not p.exists():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with p.with_suffix(p.suffix+'.partial').open('wb') as f:
                for b in r.iter_content(1024*1024):
                    f.write(b)
        p.with_suffix(p.suffix+'.partial').rename(p)
    records.append({'file':p.name, 'source':url, 'revision':REV, 'bytes':p.stat().st_size,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(out/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print(json.dumps(records,indent=2))
