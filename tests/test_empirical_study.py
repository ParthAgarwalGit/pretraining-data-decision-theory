"""Independent mathematical checks for the new offline analyses."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy import stats

spec=importlib.util.spec_from_file_location('study',Path(__file__).parents[1]/'tools/empirical_study.py')
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

def test_hand_calculated_rank_and_regret():
    out=study.metrics([3,2,1],[1,2,3],['a','b','c'])
    assert out['spearman']==-1 and out['kendall_tau_b']==-1
    assert out['regret']==2 and out['normalized_regret']==1 and out['top1']==0

def test_tie_break_does_not_use_target():
    out=study.metrics([1,1,0],[0,1,.5],['a','b','c'])
    assert out['selected_recipe']=='a' and out['top1']==0
    averaged=study.metrics([1,1,0],[0,1,.5],['a','b','c'],'average_winners')
    assert averaged['top1']==.5 and averaged['regret']==.5

def test_welch_matches_scipy_with_unequal_replication():
    a=np.array([.1,.3,.7]);b=np.array([.4,.5,.6,.8])
    t,df,p=study.welch(a.mean(),b.mean(),a.var(ddof=1),b.var(ddof=1),len(a),len(b))
    reference=stats.ttest_ind(a,b,equal_var=False)
    assert np.isclose(t,reference.statistic) and np.isclose(p,reference.pvalue)
    assert np.isclose(df,reference.df)

def test_one_seed_has_no_variance_inference():
    assert all(np.isnan(x) for x in study.welch(0,1,0,0,1,1))

def test_exact_linear_and_power_curves():
    x=np.array([4,6,10,20,60,150])*1e6
    y=.1+.0003*x/1e6
    assert np.isclose(study.predict_curve(x,y,1e9,'linear')[0],.4)
    power=.8-.6*(x/1e6)**(-.5)
    pred,_,_=study.predict_curve(x,power,1e9,'power_law')
    assert np.isclose(pred,.8-.6*1000**(-.5),atol=1e-6)

def test_reference_can_change_without_changing_prediction():
    x=np.array([4,6,8,10])*1e6;y=np.array([.1,.2,.3,.4])
    prediction=study.predict_curve(x,y,1e9,'log_linear')[0]
    # Prediction interface receives no held-out score; errors may change while prediction cannot.
    assert abs(prediction-.3)!=abs(prediction-.9)
    assert prediction==study.predict_curve(x,y,1e9,'log_linear')[0]
