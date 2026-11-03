"""Write executable inspection notebooks and capture actual Python cell output."""
from pathlib import Path
import contextlib, io, json, os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks';OUT.mkdir(exist_ok=True)
items=[
 ('01_cross_scale_predictability',2,'cross_scale_predictability.csv','spearman_by_scale.png','How well do smaller-size rankings predict the 1B reference?','cross_scale_predictability.md'),
 ('02_task_heterogeneity',3,'task_heterogeneity.csv','task_ranking_heatmap.png','Does predictability differ by evaluation task?','task_heterogeneity.md'),
 ('03_seed_uncertainty',4,'seed_subsampling_summary.csv','seed_subsampling_predictability.png','How stable is selection across all available seed subsets?','seed_uncertainty.md'),
 ('04_extrapolation_comparison',5,'extrapolation_summary.csv','extrapolation_error_comparison.png','Which source-only fitting methods best predict target scores and winners?','extrapolation_comparison.md'),
 ('05_robustness_analysis',6,'robustness_matrix.csv','robustness_comparison.png','Which conclusions persist under alternative analytical specifications?','robustness_analysis.md')]
for name,stage,table,figure,question,doc in items:
    setup="from pathlib import Path\nimport pandas as pd\nROOT = Path.cwd() if (Path.cwd()/'tools').exists() else Path.cwd().parent\n"
    code=setup+f"result = pd.read_csv(ROOT/'results/empirical_study/{table}')\nprint(result.to_string(index=False))\n"
    namespace={};capture=io.StringIO()
    previous=Path.cwd();os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(capture):exec(code,namespace)
    finally:os.chdir(previous)
    cells=[{'cell_type':'markdown','metadata':{},'source':f'# {question}\n\nThe numerical output below was executed from saved study results. To recompute the underlying analysis, run `python tools/empirical_study.py --stage {stage}` from the repository root, after prerequisite stages. `tools/study_reproduce.py` runs the complete offline sequence into a fresh directory.\n\n'+(ROOT/'docs'/doc).read_text(encoding='utf8')},
           {'cell_type':'code','metadata':{},'execution_count':1,'source':code,'outputs':[{'output_type':'stream','name':'stdout','text':capture.getvalue()}]},
           {'cell_type':'markdown','metadata':{},'source':f'![Study result](../figures/empirical_study/{figure})\n\n## Interpretation\n\n'+(ROOT/'docs/integrated_findings.md').read_text(encoding='utf8')}]
    # Regeneration cell is intentionally unexecuted here; full regeneration is verified by the standalone runner.
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'source':f"# Optional regeneration (may take several minutes):\n# import subprocess, sys\n# subprocess.run([sys.executable, str(ROOT/'tools/empirical_study.py'), '--stage', '{stage}'], cwd=ROOT, check=True)\n",'outputs':[]})
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':4}
    (OUT/f'{name}.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
print('Five executable notebooks written; saved-result display cells executed successfully.')
