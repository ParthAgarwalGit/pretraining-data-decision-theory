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
    d.to_csv(OUT/name,index=False,float_format='%.17g')
    return d

def note(name, text):
    (ROOT/'docs'/name).write_text(text,encoding='utf8')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def data(): return pd.read_csv(OUT/'analysis_observations.csv',float_precision='round_trip')

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

def welch(mu_a,mu_b,var_a,var_b,na=3,nb=3):
    a,b=var_a/na,var_b/nb
    se=np.sqrt(a+b)
    if na<2 or nb<2 or se==0: return np.nan,np.nan,np.nan
    t=(mu_a-mu_b)/se; df=(a+b)**2/(a*a/(na-1)+b*b/(nb-1))
    return t,df,float(2*stats.t.sf(abs(t),df))

def stage4():
    d=data(); rows=[]; subs=[]; pairs=[]
    for (task,n),g in d.groupby(['task','params_num'],sort=True):
        a=g.pivot(index='recipe',columns='seed',values='score').sort_index()
        mu=a.mean(axis=1); sd=a.std(axis=1,ddof=1); var=sd**2
        target=d[(d.task==task)&(d.params_num==d.params_num.max())].groupby('recipe').score.mean().reindex(a.index)
        for recipe in a.index:
            se=sd[recipe]/np.sqrt(3); half=stats.t.ppf(.975,2)*se
            rows.append({'task':task,'params_num':n,'recipe':recipe,'mean':mu[recipe],'sd':sd[recipe],
              'se':se,'ci_low':mu[recipe]-half,'ci_high':mu[recipe]+half,
              'cv':sd[recipe]/mu[recipe] if mu[recipe]!=0 else np.nan})
        for ra,rb in itertools.combinations(a.index,2):
            t,df,p=welch(mu[ra],mu[rb],var[ra],var[rb])
            pairs.append({'task':task,'params_num':n,'recipe_a':ra,'recipe_b':rb,'mean_difference':mu[ra]-mu[rb],
              'se_difference':np.sqrt((var[ra]+var[rb])/3),'welch_t':t,'welch_df':df,'p_value':p,
              'resolved_unadjusted_05':bool(p<.05),'resolved_bonferroni_14':bool(p<.05/14),
              'target_order_agrees':float(np.sign(mu[ra]-mu[rb])==np.sign(target[ra]-target[rb]))})
        for k in [1,2,3]:
            for subset in itertools.combinations(a.columns,k):
                ss=a.loc[:,list(subset)]; pred=ss.mean(axis=1)
                resolved=[]; agreements=[]
                for ra,rb in itertools.combinations(a.index,2):
                    agreements.append(float(np.sign(pred[ra]-pred[rb])==np.sign(mu[ra]-mu[rb])))
                    if k>=2:
                        t,df,p=welch(pred[ra],pred[rb],ss.loc[ra].var(ddof=1),ss.loc[rb].var(ddof=1),k,k)
                        resolved.append(float(p<.05/14))
                subs.append({'task':task,'params_num':n,'n_seeds':k,'subset':'|'.join(subset),
                    **metrics(pred,mu,a.index),'pairwise_order_agreement':np.mean(agreements),
                    'resolved_fraction':np.mean(resolved) if resolved else np.nan,
                    'target_regret':metrics(pred,target,a.index)['regret']})
    u=csv('seed_uncertainty.csv',rows); s=csv('seed_subsampling.csv',subs)
    p=csv('pairwise_effect_sizes.csv',pairs)
    csv('seed_subsampling_summary.csv',s[s.task!=AGG].groupby('n_seeds')[['spearman','top1','pairwise_order_agreement','resolved_fraction','regret','target_regret']].mean().reset_index())
    # Stratify by an explicitly descriptive t-statistic, not a causal noise decomposition.
    q=p[(p.task!=AGG)&(p.params_num<d.params_num.max())].copy()
    q['signal_bin']=pd.cut(q.welch_t.abs(),[-np.inf,1,2,4,np.inf],labels=['<1','1-2','2-4','>=4'])
    sig=q.groupby('signal_bin',observed=True).agg(n=('target_order_agrees','size'),agreement=('target_order_agrees','mean')).reset_index()
    csv('signal_to_noise_summary.csv',sig)
    lineplot(u.groupby(['task','params_num']).sd.mean().reset_index(),'params_num','sd','seed_variation_by_scale.png',ylabel='Mean across recipes of seed SD')
    fig,ax=plt.subplots(figsize=(6,4)); v=s[s.task!=AGG].groupby('n_seeds')[['spearman','top1']].mean()
    ax.plot(v.index,v.spearman,'o-',label='Ranking correlation'); ax.plot(v.index,v.top1,'s-',label='Winner agreement')
    ax.set_xticks([1,2,3]); ax.set_xlabel('Seeds in subset'); ax.set_ylabel('Agreement with full-seed reference'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'seed_subsampling_predictability.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(sig.signal_bin.astype(str),sig.agreement)
    ax.set_xlabel('Absolute Welch t statistic at source scale');ax.set_ylabel('Pair ordering agreement with 1B');ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(FIG/'effect_size_vs_ranking_agreement.png',dpi=180);plt.close(fig)
    note('seed_uncertainty.md','''# Stage 4 Seed uncertainty

All one-, two- and three-seed subsets within each size are enumerated. Each subset label is applied across recipes as a descriptive diagnostic, not assumed to pair independent runs or scales. Three-seed agreement with the three-seed reference is mechanically perfect. Nested subsets share observations with the reference; their agreement is optimistic relative to an independent replication.

Individual score intervals use t(2) times the standard error, with independent approximately normal run scores assumed. They are pointwise, unadjusted, unclipped and based on only three runs; no reliable population coverage claim is made. The CV is descriptive on a positive accuracy scale and can be misleading near zero. One-seed uncertainty and resolution are undefined, recorded as null rather than zero. Two-sided Welch comparisons use each recipe's own variance and replicate count. Zero pooled variance is unresolved rather than treated as infinite confidence. The Bonferroni denominator 14 controls comparisons across sizes for a single pair under the assumptions; it does not control all pairwise tests together.

Signal-to-noise bins use the absolute Welch statistic (a standardized mean difference by its SE, not Cohen's d). Associations with target ordering are descriptive and dependent. They cannot separate noise from systematic scale effects causally. The target is a three-seed reference, not truth. Seed labels are not paired across scales. No bootstrap replications are presented as extra training runs.
''')
    print(pd.read_csv(OUT/'seed_subsampling_summary.csv').to_string(index=False))

def predict_curve(x,y,target,method,alpha_bounds=(.01,3.0)):
    """All fitting depends only on x,y. Target supplies a covariate, never a score."""
    x=np.asarray(x,dtype=float)/1e6; z=float(target)/1e6; y=np.asarray(y)
    if method=='largest_observed': return float(y[-1]),False,[]
    if method in ('linear','log_linear'):
        xx=x if method=='linear' else np.log(x); zz=z if method=='linear' else np.log(z)
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),xx]),y,rcond=None)[0]
        return float(coef@[1,zz]),False,coef.tolist()
    # Variable projection: solve E and A exactly at each alpha, then refine all local grid minima.
    def fit_alpha(alpha):
        basis=np.column_stack([np.ones(len(x)),x**(-alpha)])
        coef=np.linalg.lstsq(basis,y,rcond=None)[0]
        return float(np.sum((basis@coef-y)**2)),coef
    grid=np.geomspace(*alpha_bounds,60); costs=[fit_alpha(a)[0] for a in grid]
    candidates=[(costs[0],grid[0]),(costs[-1],grid[-1])]
    for i in range(1,len(grid)-1):
        if costs[i]<=costs[i-1] and costs[i]<=costs[i+1]:
            fit=optimize.minimize_scalar(lambda a:fit_alpha(a)[0],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-10})
            if fit.success: candidates.append((fit.fun,fit.x))
    _,alpha=min(candidates); _,coef=fit_alpha(alpha)
    boundary=bool(np.isclose(alpha,alpha_bounds[0]) or np.isclose(alpha,alpha_bounds[1]))
    return float(coef@[1,z**(-alpha)]),boundary,[*coef.tolist(),float(alpha)]

def stage5():
    m=means(data()); predictions=[]; evaluations=[]; failures=[]
    for task,g in m.groupby(level='task',sort=True):
        g=g.droplevel('task'); target=g.index.max(); y=g.loc[target]
        for cutoff in g.index[3:-1]:
            train=g[g.index<=cutoff]; assert train.index.max()<target
            for method in ['largest_observed','linear','log_linear','power_law']:
                pred=[]; bad=False
                for recipe in g.columns:
                    try:
                        value,bound,coef=predict_curve(train.index,train[recipe],target,method)
                        if not np.isfinite(value): raise ValueError('Nonfinite prediction')
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        failures.append({'task':task,'cutoff':cutoff,'recipe':recipe,'method':method,'reason':str(exc)})
                        value=np.nan;bound=False;coef=[];bad=True
                    pred.append(value)
                    predictions.append({'task':task,'cutoff':cutoff,'target':target,'recipe':recipe,'method':method,
                        'prediction':value,'observed_target':y[recipe],'alpha_boundary':bound,
                        'outside_score_range':bool(value<0 or value>1),'coefficients':json.dumps(coef),
                        'n_training_scales':len(train),'scale_ratio':target/cutoff})
                if not bad:
                    evaluations.append({'task':task,'cutoff':cutoff,'target':target,'method':method,
                       'rmse':float(np.sqrt(np.mean((np.array(pred)-y)**2))),
                       'mae':float(np.mean(np.abs(np.array(pred)-y))),
                       'scale_ratio':target/cutoff,**metrics(pred,y,g.columns)})
    p=csv('extrapolation_predictions.csv',predictions); e=csv('extrapolation_metrics.csv',evaluations)
    csv('extrapolation_failures.csv',pd.DataFrame(failures,columns=['task','cutoff','recipe','method','reason']))
    csv('extrapolation_summary.csv',e[e.task!=AGG].groupby('method')[['rmse','mae','spearman','top1','top5_hit','regret']].mean().reset_index())
    csv('extrapolation_diagnostics.csv',p.groupby('method')[['alpha_boundary','outside_score_range']].agg(['sum','mean']).reset_index())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method,g in e[e.task!=AGG].groupby('method'):
        v=g.groupby('cutoff')[['rmse','regret']].mean()
        for ax,y in zip(axes,['rmse','regret']): ax.plot(v.index/1e6,v[y],marker='o',label=method);ax.set_xscale('log');ax.set_xlabel('Largest fitted size (M)');ax.set_ylabel(y);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'extrapolation_error_comparison.png',dpi=180);plt.close(fig)
    note('extrapolation_comparison.md','''# Stage 5 Extrapolation specification

Predictions target 1B, using every source cutoff from the fourth available size (10M) to 750M. All 25 recipes, three-seed means, equal scale weights and identical permitted source ranges are used for every method. The largest-observed baseline uses only the final observation within that permitted range. It is intentionally a ranking baseline; using its unchanged score as a numeric prediction may have substantial level bias.

Let x=N/1,000,000. Linear is a+b*x. Log-linear is a+b*ln(x), not a logarithm of the score. Power-law is E+A*x**(-alpha), alpha in [0.01,3], E and A unconstrained. At each alpha the two linear coefficients are solved by least squares. A fixed 60-point log grid plus bounded refinements of local minima and endpoints determines the minimum. This normalized, profiled implementation was selected before examining target results. It shares the existing PowerLawN functional form but intentionally does not inherit its unit-dependent amplitude bounds. Scores are not clipped. Boundary fits and out-of-range predictions are retained and reported. Alpha bounds are a modeling assumption, examined in Stage 6, not tuned against target scores.

Fitting receives source scores and target size only; target scores enter solely after prediction. Full failures are logged and are not silently dropped. Headline summaries average 10 tasks and 10 cutoffs equally; they are descriptive on a fixed, dependent panel. Method performance at each cutoff is also saved. RMSE and correct winner selection answer different questions. A smaller RMSE does not imply smaller selection regret.
''')
    print(pd.read_csv(OUT/'extrapolation_summary.csv').to_string(index=False)); print('Failed fits:',len(failures))

def stage6():
    d=data(); m=means(d); p=pd.read_csv(OUT/'pairwise_effect_sizes.csv',float_precision='round_trip')
    census=[]; pair_classes=[]
    for alpha in [.01,.025,.05,.10]:
        for (task,ra,rb),g in p.groupby(['task','recipe_a','recipe_b'],sort=True):
            signs=np.sign(g.loc[g.p_value<alpha/len(g),'mean_difference'])
            label='within_noise' if len(signs)==0 else ('reversing' if signs.min()<0<signs.max() else 'stable')
            pair_classes.append({'alpha':alpha,'task':task,'recipe_a':ra,'recipe_b':rb,'classification':label,'resolved_sizes':len(signs)})
    pc=csv('rank_reversal_pairs.csv',pair_classes)
    for (alpha,task),g in pc.groupby(['alpha','task']):
        counts=g.classification.value_counts()
        census.append({'alpha':alpha,'task':task,**{k:int(counts.get(k,0)) for k in ['reversing','stable','within_noise']},'total':len(g)})
    c=csv('rank_reversal_sensitivity.csv',census)
    old=json.loads((ROOT/'results/p1_09_rank_reversals.json').read_text())['data']
    # Locate the calibrated branch by its field name; never conflate uncorrected counts.
    def get_old(obj,path=''):
        out=[]
        if isinstance(obj,dict):
            if 'pairs' in obj and 'bonferroni' in path:
                out=[{'task':obj['task'],'recipe_a':r['recipe_a'],'recipe_b':r['recipe_b'],'old':r['classification']} for r in obj['pairs']]
            for k,v in obj.items(): out+=get_old(v,path+'/'+k)
        return out
    oldpairs=pd.DataFrame(get_old(old)); assert len(oldpairs)==3300
    joined=pc[pc.alpha==.05].merge(oldpairs,on=['task','recipe_a','recipe_b'],validate='one_to_one')
    assert len(joined)==3300
    discrepancies=joined[joined.classification!=joined.old]
    csv('historical_census_discrepancies.csv',discrepancies)
    save_json('rank_reversal_reproduction.json',{'pairs_compared':len(joined),'classification_discrepancies':len(discrepancies),
        'fresh_counts':joined.classification.value_counts().to_dict(),'historical_counts':joined.old.value_counts().to_dict(),
        'correction':'two-sided Welch t; alpha/14 per pair trajectory; no global pair/task correction'})
    variants=[]
    for task,g in m.groupby(level='task',sort=True):
        g=g.droplevel('task')
        for target in [530_000_000,750_000_000,1_000_000_000]:
            for n,x in g[g.index<target].iterrows():
                for rule in ['lexical','average_winners']:
                    variants.append({'task':task,'target':target,'params_num':n,'tie_rule':rule,**metrics(x,g.loc[target],g.columns,rule)})
    v=csv('robustness_targets_ties.csv',variants)
    # Leave-one-recipe-out affects the candidate set; do not call this independent replication.
    recipe_rows=[]
    for task,g in m.groupby(level='task',sort=True):
        g=g.droplevel('task');target=g.index.max()
        for omitted in g.columns:
            subset=g.drop(columns=omitted)
            for n in [4_000_000,530_000_000,750_000_000]:
                recipe_rows.append({'task':task,'omitted_recipe':omitted,'params_num':n,**metrics(subset.loc[n],subset.loc[target],subset.columns)})
    csv('robustness_recipe_exclusion.csv',recipe_rows)
    cross=pd.read_csv(OUT/'cross_scale_predictability.csv');ind=cross[cross.task!=AGG]
    loo=[]
    for task in sorted(ind.task.unique()):
        sub=ind[ind.task!=task]
        for n,g in sub.groupby('params_num'):
            loo.append({'omitted_task':task,'params_num':n,**g[['spearman','top1','regret']].mean().to_dict()})
    csv('robustness_leave_one_task_out.csv',loo)
    # Fixed alternatives, reported as sensitivity rather than selecting the best bound after seeing targets.
    alt=[]
    for task,g in m.groupby(level='task',sort=True):
        g=g.droplevel('task');target=g.index.max()
        for cutoff in [20_000_000,150_000_000,750_000_000]:
            for bounds in [(.001,1.0),(.01,5.0)]:
                train=g[g.index<=cutoff]
                pred=[predict_curve(train.index,train[r],target,'power_law',bounds)[0] for r in g.columns]
                alt.append({'task':task,'cutoff':cutoff,'bounds':str(bounds),**metrics(pred,g.loc[target],g.columns),
                            'rmse':float(np.sqrt(np.mean((np.array(pred)-g.loc[target])**2)))})
    csv('robustness_power_bounds.csv',alt)
    matrix=[
      {'finding':'Larger source sizes generally improve ranking','status':'evaluate target, recipe and task sensitivity','evidence':'robustness_targets_ties.csv; robustness_recipe_exclusion.csv; robustness_leave_one_task_out.csv'},
      {'finding':'Reversal prevalence','status':'threshold-sensitive; report all counts','evidence':'rank_reversal_sensitivity.csv'},
      {'finding':'Power-law advantage','status':'assumption- and metric-dependent','evidence':'extrapolation_metrics.csv; robustness_power_bounds.csv'},
      {'finding':'General seed requirement','status':'inconclusive with three seeds','evidence':'seed_subsampling.csv'}]
    csv('robustness_matrix.csv',matrix)
    save_json('robustness_summary.json',{'census_by_alpha_all_entries':c.groupby('alpha')[['reversing','stable','within_noise']].sum().reset_index().to_dict('records'),
       'historical_discrepancies':len(discrepancies),'variants':matrix})
    note('robustness_analysis.md','''# Stage 6 Robustness

The fresh census independently calculates Welch statistics from the seed scores and compares individual classifications against the historical calibrated branch. Thresholds 0.01, 0.025, 0.05 and 0.10 are all reported, with 0.05 retained as primary. Correction is across 14 scales per recipe pair. Stable means no conflicting resolved signs, possibly with only one resolved size; within-noise means no resolved size, not equivalence. Aggregate and individual task counts are separable in the saved CSV.

Ranking sensitivity uses targets 530M, 750M and 1B, lexical versus uniform-over-tied-winners selection, leave-one-recipe-out candidate sets, and leave-one-individual-task-out summaries. The full panel has no missing observations, so stricter complete-case filtering is identical and is not presented as new evidence. The primary power-law bounds are preserved; two predetermined alternatives at three source cutoffs are saved separately. All are descriptive sensitivity analyses on the same dependent observations. No hypothesis was selected solely because it produced a favorable result.
''')
    print(c.groupby('alpha')[['reversing','stable','within_noise']].sum().to_string());print('Historical discrepancies:',len(discrepancies))

STAGES={1:stage1,2:stage2,3:stage3,4:stage4,5:stage5,6:stage6}

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
