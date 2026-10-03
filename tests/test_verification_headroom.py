"""Meaningful oracle, entity/mask and fixed-claim contracts; synthetic fixtures."""
from itertools import product

import numpy as np
import pytest

from financepaper.evaluation.verification_headroom import (
    actual_reveal_states, optimize_oracle, selected_outputs, paired_intervals,
)
from financepaper.reliability.validation import aggregate_groups


def test_reveal_exhaustive_stop_natural_unknown_and_no_other_truth():
    full=np.array([[1.,2.,np.nan,4.],[5.,6.,7.,8.]])
    artificial=np.array([[True,True,False,False],[False,False,False,True]])
    owner,field,values,remaining=actual_reveal_states(full,artificial)
    assert len(values)==len(full)+artificial.sum()
    assert np.array_equal(field[:2],[-1,-1])
    for i,(r,j) in enumerate(zip(owner,field)):
        original=full[r].copy();original[artificial[r]]=np.nan
        if j>=0:original[j]=full[r,j]
        np.testing.assert_equal(values[i],original)
        assert remaining[i].sum()==artificial[r].sum()-(j>=0)
    assert np.isnan(values[owner==0,2]).all()
    bad=artificial.copy();bad[0,2]=True
    with pytest.raises(ValueError,match="naturally"):
        actual_reveal_states(full,bad)


def test_acquired_field_cannot_enter_fixed_reason_claim():
    phi=np.array([[1.,100.,3.]])
    original_hidden=np.array([[False,True,False]])
    grouped,hidden=aggregate_groups(phi,original_hidden,('a','b','c'),{'g':['a','b'],'h':['c']})
    np.testing.assert_array_equal(grouped,[[1.,3.]])
    assert not hidden.any()


@pytest.mark.parametrize("budget",[0,1])
@pytest.mark.parametrize("objective",["reasons","customers"])
def test_oracle_matches_exhaustive_empirical_risk_and_stop(budget,objective):
    # owners 0,1 each have STOP and reveal. Both a budget tradeoff and a cost tie.
    sets=np.array([[1,1,0],[1,0,0],[1,0,0],[1,1,1]],bool)
    truth=np.array([[1,0,1],[1,1,0]],bool)
    owner=np.array([0,1,0,1]);field=np.array([-1,-1,0,2]);alpha=.25
    action,receipt=optimize_oracle(sets,truth,owner,field,alpha=alpha,budget=budget,objective=objective)
    selected=selected_outputs(sets,action)
    def key(actions):
        emitted=selected_outputs(sets,np.array(actions));count=emitted.sum()
        customers=emitted.any(1).sum();cost=sum(a>=0 and field[a]>=0 for a in actions)
        return (count,customers,-cost) if objective=='reasons' else (customers,count,-cost)
    options=[[-1,*np.flatnonzero((owner==i)&((field<0)|(budget==1))) ] for i in range(2)]
    feasible=[]
    for actions in product(*options):
        emitted=selected_outputs(sets,np.array(actions))
        if (emitted&~truth).sum()<=alpha*emitted.sum():feasible.append(key(actions))
    assert key(action)==max(feasible)
    assert receipt['optimal']
    assert (selected&~truth).sum()<=alpha*selected.sum()
    if budget==0:assert np.all(field[action[action>=0]]<0)


def test_oracle_cannot_edit_one_failed_reason_from_a_set():
    sets=np.array([[1,1]],bool);truth=np.array([[1,0]],bool)
    action,_=optimize_oracle(sets,truth,np.array([0]),np.array([-1]),alpha=.1,budget=1,objective='customers')
    assert action.tolist()==[-1]


def test_equivalent_reveal_prefers_zero_cost_stop():
    sets=np.ones((2,2),bool);truth=np.ones((1,2),bool)
    action,_=optimize_oracle(sets,truth,np.array([0,0]),np.array([2,-1]),alpha=.1,budget=1,objective='customers')
    assert action.tolist()==[1]


def test_empty_risk_is_not_zero_and_bootstrap_uses_customer_counts():
    selected=np.zeros((2,3),bool);truth=np.ones_like(selected)
    weights=np.array([[1,1],[2,0],[0,2]])
    row,samples=paired_intervals(selected,truth,truth,np.zeros(2),weights)
    assert np.isnan(row['risk']) and np.isnan(samples['risk']).all()
    assert row['customer_ge1']==0
    selected[0,0]=True;truth[0,0]=False
    row,samples=paired_intervals(selected,truth,np.ones_like(truth),np.array([1,0]),weights)
    assert row['risk']==1 and row['customer_ge1']==.5
    np.testing.assert_equal(samples['customer_ge1'],[.5,1,0])


def test_invalid_oracle_inputs_rejected():
    with pytest.raises(ValueError):
        optimize_oracle(np.ones((1,2)),np.ones((1,2),bool),np.array([0]),np.array([-1]),
                        alpha=.1,budget=1,objective='customers')
