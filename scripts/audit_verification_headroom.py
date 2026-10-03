"""Independent cache, cohort and solved-oracle checks; no refitting or retuning."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from financepaper.experiments.verification_headroom import load, npz
from financepaper.experiments.decisive_validation import hashes,validate_hashes,write_json,polish_adapter
from financepaper.data.polish import load_polish
from financepaper.reliability.validation import POLISH_NAMES,POLISH_GROUPS,aggregate_groups
from financepaper.reliability.reason_sets import verified


def audit(root):
    frozen,cfg,cohort=load(root)
    for path in (root/"complete.json",root/"analysis/report_receipt.json"):
        validate_hashes(json.loads(path.read_text())["artifacts"])
    X,_,clusters=load_polish(cfg["polish_path"])
    bundle=joblib.load(frozen["predictor"])
    parts=json.loads((Path(cfg["historical_root"])/"polish/partition_ids.json").read_text())
    assert not set(cohort['ids'])&set(bundle['preprocessor'].train_ids)
    assert set(bundle['donors'].reference.index)<=set(parts['train'])
    g=pd.Series(clusters,index=X.index)
    assert not set(cohort['clusters'])&set(g.loc[parts['outer']])
    assert len(np.unique(cohort['clusters']))==cfg['customers']
    assert np.all(~cohort['mcar10']|cohort['mcar30'])
    X=X.loc[cohort['ids']];adapter=polish_adapter(bundle);policies=joblib.load(frozen['policy'])['policies']
    checked=dict(customers=len(X),states=0,query_actions=0,oracle_solutions=0,
                 natural_restorations=0,reference_mismatches=0,spot_attributed_states=0)
    for condition in cfg['conditions']:
        ev=npz(root/condition/'states.npz');truth=npz(root/condition/'verification_only.npz')
        owner,field=ev['owner'],ev['field'];natural=X.isna().to_numpy();artificial=cohort[condition]
        hidden=natural|artificial;n=len(X)
        assert not (natural&artificial).any()
        assert len(owner)==n+artificial.sum()
        np.testing.assert_array_equal(owner[:n],np.arange(n));assert (field[:n]==-1).all()
        assert set(zip(owner[n:],field[n:]))==set(zip(*np.where(artificial)))
        assert np.all(~ev['candidate']|ev['candidate'][:n][owner])
        for target in cfg['targets']:
            np.testing.assert_array_equal(truth[target],verified(truth['restored_phi'],ev['hidden'][:n],target))
        # Rebuild selected actual reveals independently of the generator helper.
        indices=np.linspace(0,len(owner)-1,12,dtype=int)
        rows=X.to_numpy()[owner[indices]].copy()
        rows[artificial[owner[indices]]]=np.nan
        for pos,state in enumerate(indices):
            j=field[state]
            if j>=0:rows[pos,j]=X.to_numpy()[owner[state],j]
        assert np.isnan(rows[natural[owner[indices]]]).all()
        attr=adapter.attribute(pd.DataFrame(rows,columns=POLISH_NAMES))
        assert attr['valid'].all()
        np.testing.assert_allclose(attr['phi'],ev['feature_phi'][indices],atol=1e-8)
        grouped,_=aggregate_groups(attr['phi'],hidden[owner[indices]],POLISH_NAMES,POLISH_GROUPS)
        np.testing.assert_allclose(grouped,ev['phi'][indices],atol=1e-8)
        for target in cfg['targets']:
            for alpha in cfg['alphas']:
                for method in cfg['terminal_methods']:
                    sets=policies[(target,method,alpha,False)].release(ev)
                    for objective in ('reasons','customers'):
                        achieved=[]
                        for budget in (0,1):
                            result=npz(root/'analysis'/f'{condition}_{target}_{alpha}_oracle_{method}_{objective}_b{budget}.npz')
                            actions,selected=result['actions'],result['selected'];emitted=actions>=0
                            np.testing.assert_array_equal(owner[actions[emitted]],np.flatnonzero(emitted))
                            np.testing.assert_array_equal(selected[emitted],sets[actions[emitted]])
                            assert not selected[~emitted].any()
                            assert (selected&~truth[target]).sum()<=alpha*selected.sum()+1e-8
                            if budget==0:assert (field[actions[emitted]]==-1).all()
                            achieved.append(selected.sum() if objective=='reasons' else selected.any(1).sum())
                            checked['oracle_solutions']+=1
                        assert achieved[1]>=achieved[0]
        checked['states']+=len(owner);checked['query_actions']+=int(artificial.sum())
        checked['spot_attributed_states']+=len(indices)
    bounds=pd.read_csv(root/'analysis/elementary_bounds.csv')
    metrics=pd.read_csv(root/'analysis/metrics.csv')
    for _,row in bounds.iterrows():
        f=metrics[(metrics.condition==row.condition)&(metrics.target==row.target)&(metrics.alpha==row.alpha)]
        feasible=f[f.method.str.startswith('zero_')&(f.risk<=row.alpha)]
        assert abs(row.best_feasible_zero_coverage-feasible.customer_ge1.max())<1e-12
    checked['status']='PASS'
    checked['freeze']=hashes([root/'protocol_freeze.json'])
    checked['audit_source']=hashes([Path(__file__)])
    write_json(root/'audit_receipt.json',checked)
    print(json.dumps(checked,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/verification_headroom')
    audit(Path(parser.parse_args().output))
