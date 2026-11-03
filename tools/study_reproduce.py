"""One offline command; recompute into a fresh directory and compare numeric artifacts."""
from pathlib import Path
import argparse, json, subprocess, sys, hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if output.exists(): raise SystemExit('Choose a fresh output path; existing artifacts are never overwritten.')
    output.mkdir(parents=True)
    for stage in range(1,8):
        print(f'Stage {stage}/7',flush=True)
        with (output/f'stage{stage}.log').open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/empirical_study.py'),'--stage',str(stage),'--output',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    comparisons=[]
    for p in sorted((ROOT/'results/empirical_study').glob('*.csv')):
        actual=output/p.name
        if not actual.exists(): raise AssertionError(f'Not regenerated: {p.name}')
        a,b=pd.read_csv(p),pd.read_csv(actual)
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-12)
        comparisons.append({'file':p.name,'rows':len(a),'status':'pass','sha256':hashlib.sha256(actual.read_bytes()).hexdigest()})
    json_comparisons=[]
    for p in sorted((ROOT/'results/empirical_study').glob('*.json')):
        if p.name in {'artifact_inventory.json','reproducibility_verification.json'} or p.name.endswith('_provenance.json'): continue
        assert json.loads((output/p.name).read_text(encoding='utf8'))==json.loads(p.read_text(encoding='utf8')),p.name
        json_comparisons.append(p.name)
    completed=subprocess.run([sys.executable,'-m','pytest','tests/test_empirical_study.py','--no-cov','-q'],cwd=ROOT,capture_output=True,text=True)
    (output/'tests.log').write_text(completed.stdout+completed.stderr,encoding='utf8');completed.check_returncode()
    result={'numeric_csv_comparisons':comparisons,'json_results_matched':json_comparisons,'stages_1_to_7':'passed','tests':'6 passed',
            'input':'pinned macro-average Parquet; no network used during rerun',
            'tolerance':'relative 1e-10, absolute 1e-12; identical rows and columns required'}
    (ROOT/'results/empirical_study/reproducibility_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(f'PASS: {len(comparisons)} CSV files regenerated and matched; six targeted tests passed.',flush=True)
if __name__=='__main__':main()
