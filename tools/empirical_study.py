"""Sequential, offline empirical study. Run --stage 1 through --stage 8."""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, subprocess, sys
import numpy as np
import pandas as pd
from scipy import stats, optimize
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AGG = 'olmes_10_macro_avg'
KEY = ['task','params_num','recipe']
SEED = 20261103
OUT = ROOT/'results/empirical_study'
FIG = ROOT/'figures/empirical_study'

def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v

def save_json(name, value):
    (OUT/name).write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf8')

def csv(name, rows):
    d = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(OUT/name,index=False,float_format='%.12g')
    return d

def note(name, text):
    (ROOT/'docs'/name).write_text(text,encoding='utf8')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def data(): return pd.read_csv(OUT/'analysis_observations.csv')

def means(d): return d.groupby(KEY,sort=True).score.mean().unstack('recipe')

def metrics(pred, target, recipes, tie='lexical'):
    p,y=np.asarray(pred),np.asarray(target)
    assert len(p)==len(y)==len(recipes) and np.isfinite(p).all() and np.isfinite(y).all()
    # Exact floating-score ties; lexical rule fixed without seeing target scores.
    order=sorted(range(len(p)),key=lambda i:(-p[i],str(recipes[i])))
    winners=np.flatnonzero(p==p.max())
    target_best=y.max()
    chosen=[order[0]] if tie=='lexical' else winners
    regret=float(np.mean(target_best-y[chosen]))
    return {'spearman':float(stats.spearmanr(p,y).statistic),
            'kendall_tau_b':float(stats.kendalltau(p,y,variant='b').statistic),
            'top1':float(np.mean(y[chosen]==target_best)),
            'top5_hit':float(any(y[i]==target_best for i in order[:5])),
            'regret':regret,'normalized_regret':regret/np.ptp(y) if np.ptp(y)>0 else np.nan,
            'selected_recipe':str(recipes[order[0]]),'n_predicted_top_ties':len(winners),
            'n_target_top_ties':int(sum(y==target_best)), 'n_recipes':len(p)}

def lineplot(d,x,y,name,xlabel='Model size (millions of parameters)', ylabel=None):
    fig,ax=plt.subplots(figsize=(8,4.8))
    for task,g in d.groupby('task',sort=True):
        ax.plot(g[x]/1e6,g[y],marker='o',markersize=3,linewidth=2 if task==AGG else 1,
                alpha=1 if task==AGG else .75,label=task)
    ax.set_xscale('log'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel or y)
    ax.grid(alpha=.2); ax.legend(fontsize=6.5,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout(); fig.savefig(FIG/name,dpi=180); plt.close(fig)

def stage1():
    p=ROOT/'data/study_raw/macro_avg-00000-of-00001.parquet'
    manifest=json.loads((p.parent/'manifest.json').read_text())
    assert sha(p)==manifest[0]['sha256'], 'Input checksum mismatch'
    raw=pd.read_parquet(p)
    required=['params','data','task','seed','step','tokens','compute','metrics','chinchilla']
    assert set(raw.columns)==set(required), raw.columns
    assert not raw[required].isna().any().any()
    ids=['params','data','task','seed','step']
    dup=int(raw.duplicated(ids).sum())
    assert dup==0, 'Duplicate observations require resolution'
    d=raw[raw.step>0].copy()
    final=d[d.step==d.groupby(['params','data','seed']).step.transform('max')].copy()
    final['score']=final.metrics.map(lambda x:json.loads(x)['primary_metric'])
    assert np.isfinite(final.score).all() and final.score.between(0,1).all()
    final=final.rename(columns={'data':'recipe','params':'params_str'})
    final['params_num']=final.params_str.map(lambda x:int(float(x[:-1])*(1e9 if x[-1]=='B' else 1e6)))
    assert not final.duplicated(KEY+['seed']).any()
    counts=final.groupby(KEY).agg(n_seeds=('seed','nunique'),n_scores=('score','size')).reset_index()
    expected=final.task.nunique()*final.params_num.nunique()*final.recipe.nunique()
    assert len(counts)==expected and (counts.n_seeds==3).all() and (counts.n_scores==3).all()
    # Same budget within a size, and task rows aligned to the same final checkpoint.
    budgets=final.groupby('params_num')[['step','tokens','compute']].nunique()
    assert (budgets==1).all().all(), 'Incompatible within-scale training budgets'
    assert list(final.chinchilla.unique())==['5xC']
    pivot=final.pivot(index=['recipe','params_num','seed'],columns='task',values='score')
    agg_error=float(np.max(np.abs(pivot[AGG]-pivot.drop(columns=AGG).mean(axis=1))))
    assert agg_error<1e-12
    cols=['recipe','params_str','params_num','task','seed','step','tokens','compute','score']
    final=final[cols].sort_values(KEY+['seed'])
    csv('analysis_observations.csv',final)
    csv('data_quality_audit.csv',counts)
    csv('training_budgets.csv',final.groupby(['params_num','params_str'])[['step','tokens','compute']].first().reset_index())
    summary={'raw_rows':len(raw),'untrained_rows_excluded':int((raw.step==0).sum()),
      'positive_step_rows':len(d),'final_rows':len(final),'intermediate_rows_excluded':len(d)-len(final),
      'recipes':final.recipe.nunique(),'scales':final.params_num.nunique(),'tasks':sorted(final.task.unique()),
      'aggregate':AGG,'seeds_per_cell':3,'missing_cells':expected-len(counts),'duplicates':dup,
      'aggregate_max_error':agg_error,'seed_labels_by_scale':final.groupby('params_str').seed.unique().map(list).to_dict(),
      'source_manifest':manifest,'metric':'primary_metric','direction':'higher is better',
      'pairing':'not established; no cross-scale paired inference',
      'final_checkpoint':'maximum positive step per recipe, size and seed, without using scores'}
    save_json('data_quality_audit.json',summary)
    # Independent recount from individual historical classifications, not summary fields.
    historical=json.loads((ROOT/'results/p1_09_rank_reversals.json').read_text())
    def find_pairs(obj,path=''):
        result=[]
        if isinstance(obj,dict):
            if 'pairs' in obj:
                records=obj['pairs']; keys=[(r['recipe_a'],r['recipe_b']) for r in records]
                assert len(set(keys))==len(keys)
                result.append({'path':path,'counts':pd.Series([r['classification'] for r in records]).value_counts().to_dict(),'n':len(records)})
            for k,v in obj.items(): result+=find_pairs(v,path+'/'+k)
        return result
    save_json('historical_census_recount.json',{'source_sha256':sha(ROOT/'results/p1_09_rank_reversals.json'),
              'provenance':historical['provenance'],'recount':find_pairs(historical)})
    note('data_quality_audit.md',f'''# Stage 1 Data quality audit

The pinned source contains {len(raw):,} rows. Excluded {(raw.step==0).sum():,} untrained rows and {len(d)-len(final):,} intermediate rows by the maximum-step rule, leaving {len(final):,} final observations. No duplicate keys, missing final cells, nonfinite scores or out-of-range scores were found. There are {final.recipe.nunique()} recipes, {final.params_num.nunique()} sizes, 10 individual task families and one aggregate; every cell has three seeds. Budgets are consistent across recipes and seeds within each size. Tokens change with size: this is a joint size/training-budget ladder, not an isolated causal effect of parameter count.

The aggregate is the arithmetic mean of the ten component scores (maximum discrepancy {agg_error:.3g}). It is excluded from task-level summaries. The primary metric is a task-specific accuracy score supplied by DataDecide; metric definitions can differ between tasks, so all rankings are within task, and raw regret is summarized explicitly as accuracy percentage points.

Small auxiliary labels occur below 1B and large auxiliary labels at 1B. Label matching does not prove paired runs. Welch comparisons use an independent-run assumption, three seeds per recipe, unbiased sample variance and a per-trajectory Bonferroni correction over sizes. With three observations, normality is not testable and the inference remains assumption-dependent. This does not control errors across every recipe pair and task.

Historical pair classifications were independently recounted and checked for duplicates in historical_census_recount.json. Recomputing their statistics from the newly loaded data occurs in Stage 6; recounting alone is not statistical reproduction. The inventory preserves older artifact provenance.

All input hashes, filtering counts, seed labels, budgets and coverage assertions are saved in results/empirical_study. All four proposed empirical analyses can proceed using this complete panel. No missing values were imputed.
''')
    print(json.dumps(clean(summary),indent=2))

def stage2():
    m=means(data()); rows=[]
    for task,g in m.groupby(level='task',sort=True):
        g=g.droplevel('task'); target=g.index.max(); y=g.loc[target]
        for n,p in g.iterrows():
            if n==target: continue
            rows.append({'task':task,'params_num':n,'target':target,**metrics(p,y,g.columns)})
    d=csv('cross_scale_predictability.csv',rows); save_json('cross_scale_predictability.json',rows)
    for y,f in [('spearman','spearman_by_scale.png'),('kendall_tau_b','kendall_by_scale.png'),
                ('top1','top1_accuracy_by_scale.png'),('regret','selection_regret_by_scale.png')]:
        lineplot(d,'params_num',y,f)
    note('cross_scale_predictability.md','''# Stage 2 Cross-scale predictability

The largest size (1B) is held out as the reference. Means of three runs are compared for the same 25 recipes at every smaller size, separately for each task. Spearman uses average ranks and Kendall uses tau-b, both tie-adjusted. Exact predicted top ties are broken by lexical recipe ID, without target access. Any recipe tied for the maximum target mean counts as correct. Top-five hit means at least one target maximizer is in the selected five, not top-five set overlap. Regret is the target maximum minus the selected recipe's target mean; normalized regret divides by the target range, undefined at zero range.

Top-1 is a binary outcome for a single task and size, not an estimated probability of success. Task-averaged values later describe this fixed set of tasks. Target means are noisy reference measurements, not known true expected performances. No p-values based on independent recipes or tasks are claimed. The 1B self-comparison is excluded from predictive results.
''')
    print(d.groupby('params_num')[['spearman','top1','regret']].mean().to_string())

def stage3():
    d=pd.read_csv(OUT/'cross_scale_predictability.csv'); individual=d[d.task!=AGG]
    summary=d.groupby('task').agg(mean_spearman=('spearman','mean'),min_spearman=('spearman','min'),
           max_spearman=('spearman','max'),top1_fraction=('top1','mean'),top5_fraction=('top5_hit','mean'),
           mean_regret=('regret','mean'),max_regret=('regret','max')).reset_index()
    csv('task_heterogeneity.csv',summary)
    csv('scale_summary.csv',individual.groupby('params_num')[['spearman','kendall_tau_b','top1','top5_hit','regret','normalized_regret']].mean().reset_index())
    m=means(data()); rows=[]
    tasks=[t for t in sorted(data().task.unique()) if t!=AGG]
    for n in sorted(data().params_num.unique()):
        for a,b in itertools.combinations(tasks,2):
            x,y=m.loc[(a,n)],m.loc[(b,n)]
            rows.append({'params_num':n,'task_a':a,'task_b':b,
                'spearman':stats.spearmanr(x,y).statistic,
                'winner_agreement':float(metrics(x,y,x.index)['top1'])})
    csv('task_recipe_agreement.csv',rows)
    heat=d.pivot(index='task',columns='params_num',values='spearman')
    fig,ax=plt.subplots(figsize=(9,5)); im=ax.imshow(heat,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_yticks(range(len(heat)),heat.index); ax.set_xticks(range(len(heat.columns)),[f'{n/1e6:g}' for n in heat.columns],rotation=45)
    ax.set_xlabel('Source model size (M); target 1000M'); fig.colorbar(im,ax=ax,label='Spearman correlation'); fig.tight_layout()
    fig.savefig(FIG/'task_ranking_heatmap.png',dpi=180); plt.close(fig)
    note('task_heterogeneity.md','''# Stage 3 Task heterogeneity

Task summaries average equally over the 13 source sizes. These are descriptive summaries over dependent, unevenly spaced sizes, not independent trials. The aggregate is shown separately and excluded from scale_summary.csv. Task pair rankings and winner agreement are recorded at each size. Differences between tasks are not interpreted causally, and no one task is validated as a universal proxy. The results use exactly the same available recipes throughout.
''')
    print(summary.to_string(index=False))

STAGES={1:stage1,2:stage2,3:stage3}

def main():
    global OUT,FIG
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',type=int,required=True)
    parser.add_argument('--output',type=Path); args=parser.parse_args()
    if args.output: OUT=args.output.resolve(); FIG=OUT/'figures'
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    STAGES[args.stage]()
    save_json(f'stage_{args.stage}_provenance.json',{'stage':args.stage,'seed':SEED,
      'command':f'python tools/empirical_study.py --stage {args.stage}',
      'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'script_sha256':sha(Path(__file__)),'python':platform.python_version(),
      'packages':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}})

if __name__=='__main__': main()
