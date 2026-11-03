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
    checks={
      2:"""from scipy.stats import spearmanr
obs = pd.read_csv(ROOT/'results/empirical_study/analysis_observations.csv', float_precision='round_trip')
means = obs.groupby(['task','params_num','recipe']).score.mean().unstack('recipe')
recomputed = []
for task, table in means.groupby(level='task'):
    table = table.droplevel('task'); target = table.loc[table.index.max()]
    for size, source in table.iloc[:-1].iterrows():
        recomputed.append((task,size,spearmanr(source,target).statistic))
check = pd.DataFrame(recomputed,columns=['task','params_num','new_rho']).merge(result,on=['task','params_num'],validate='one_to_one')
assert len(check)==len(result) and (check.new_rho-check.spearman).abs().max()<1e-12
print('Independently recomputed all',len(check),'task/size rank correlations.')""",
      3:"""by_size = pd.read_csv(ROOT/'results/empirical_study/cross_scale_predictability.csv')
by_task = by_size.groupby('task').agg(rho=('spearman','mean'), top1=('top1','mean'), regret=('regret','mean'))
saved = result.set_index('task').reindex(by_task.index)
assert (by_task.rho-saved.mean_spearman).abs().max()<1e-12
assert (by_task.top1-saved.top1_fraction).abs().max()<1e-12
print(by_task.sort_values('rho',ascending=False).round(4))""",
      4:"""from itertools import combinations
obs = pd.read_csv(ROOT/'results/empirical_study/analysis_observations.csv',float_precision='round_trip')
obs = obs[obs.task!='olmes_10_macro_avg']; outcomes=[]
for (task,size),g in obs.groupby(['task','params_num']):
    scores=g.pivot(index='recipe',columns='seed',values='score').sort_index()
    reference=scores.mean(axis=1)
    for k in (1,2,3):
        for subset in combinations(scores.columns,k):
            winner=scores[list(subset)].mean(axis=1).idxmax()
            outcomes.append((k,float(reference[winner]==reference.max())))
observed=pd.DataFrame(outcomes,columns=['n_seeds','winner_agreement']).groupby('n_seeds').mean()
assert (observed.winner_agreement-result.set_index('n_seeds').top1).abs().max()<1e-12
print(observed)""",
      5:"""import numpy as np
import runpy
predict = runpy.run_path(str(ROOT/'tools/empirical_study.py'))['predict_curve']
obs = pd.read_csv(ROOT/'results/empirical_study/analysis_observations.csv',float_precision='round_trip')
g=obs[(obs.task=='arc_easy')&(obs.recipe=='C4')].groupby('params_num').score.mean()
training=g[g.index<=750_000_000]; reference=g.loc[1_000_000_000]
saved=pd.read_csv(ROOT/'results/empirical_study/extrapolation_predictions.csv')
for method in ['largest_observed','linear','log_linear','power_law']:
    pred,_,_=predict(training.index,training,1_000_000_000,method)
    row=saved[(saved.task=='arc_easy')&(saved.recipe=='C4')&(saved.cutoff==750_000_000)&(saved.method==method)].iloc[0]
    assert np.isclose(row.prediction,pred,atol=1e-10)
    print(method, 'prediction',round(pred,5),'held-out observed',round(reference,5))""",
      6:"""import numpy as np
pairs = pd.read_csv(ROOT/'results/empirical_study/pairwise_effect_sizes.csv',float_precision='round_trip')
for alpha in [.01,.025,.05,.10]:
    p=pairs.assign(sign=np.where(pairs.p_value<alpha/14,np.sign(pairs.mean_difference),0))
    signs=p.groupby(['task','recipe_a','recipe_b']).sign.agg(['min','max'])
    reversing=((signs['min']<0)&(signs['max']>0)).sum()
    within=((signs['min']==0)&(signs['max']==0)).sum()
    stable=len(signs)-reversing-within
    print(alpha, 'reversing',reversing,'stable',stable,'within noise',within)
    if alpha==.05: assert (reversing,stable,within)==(32,1424,1844)"""}
    capture=io.StringIO()
    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(capture):exec(checks[stage],namespace)
    finally:os.chdir(previous)
    cells.insert(2,{'cell_type':'code','metadata':{},'execution_count':2,'source':checks[stage],
                    'outputs':[{'output_type':'stream','name':'stdout','text':capture.getvalue()}]})
    # Regeneration cell is intentionally unexecuted here; full regeneration is verified by the standalone runner.
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'source':f"# Optional regeneration (may take several minutes):\n# import subprocess, sys\n# subprocess.run([sys.executable, str(ROOT/'tools/empirical_study.py'), '--stage', '{stage}'], cwd=ROOT, check=True)\n",'outputs':[]})
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':4}
    (OUT/f'{name}.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf8')
print('Five executable notebooks written; saved-result display cells executed successfully.')
