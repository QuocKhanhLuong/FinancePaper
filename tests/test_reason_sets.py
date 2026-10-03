import inspect
import numpy as np
import pytest
from financepaper.reliability.reason_sets import (
    evidence, verified, selected_set, fit_policy, ReasonSetPolicy, strongest,
    matched_random, make_grids, inference_output,
)


def sample(n=1):
    phi=np.tile([.5,.3,.2,.1],(n,1))
    draws=np.repeat(phi[:,None,:],8,axis=1)
    return evidence(phi,draws,draws,np.zeros_like(phi,bool),np.full(n,.7))


def test_variable_cardinality_more_than_two_and_target_separation():
    ev=sample()
    release=selected_set(ev,"stable_both",(1,0,.01),"meaningful")
    assert release.sum()==4
    assert verified(ev["phi"],ev["hidden"],"meaningful").sum()==4
    assert verified(ev["phi"],ev["hidden"],"ranked").sum()==2
    assert "restored" not in inspect.signature(evidence).parameters
    out=inference_output(ev,ReasonSetPolicy("stable_both","meaningful",.1,False,(1,0,.01)),
                         list("abcd"),[.6],[.7])[0]
    assert out["status"]=="ALL_CANDIDATES" and len(out["stable_reasons"])==4
    assert "verification_only" not in out


def test_intersection_catches_single_family_false_stability_and_observed_groups_only():
    ev=sample()
    ev["conditional_quantiles"][:,:,2]=-1
    ev["conditional_sign"][:,2]=0
    a=selected_set(ev,"stable_donor",(1,0,.01),"meaningful")
    both=selected_set(ev,"stable_both",(1,0,.01),"meaningful")
    assert a.sum()==4 and both.sum()==3
    phi=np.array([[.8,.7,.1]])
    hidden=np.array([[True,False,False]])
    draws=np.repeat(phi[:,None,:],8,axis=1)
    x=evidence(phi,draws,draws,hidden,[.8])
    assert not x["candidate"][0,0]


def test_matched_controls_preserve_per_customer_size_and_are_truth_free():
    ev=sample(4);counts=np.array([0,1,2,4]);ids=np.arange(4)
    strength=strongest(ev,counts)
    random=matched_random(ev,counts,ids)
    np.testing.assert_equal(strength.sum(1),counts)
    np.testing.assert_equal(random.sum(1),counts)
    np.testing.assert_equal(random,matched_random(ev,counts,ids))
    assert not np.any(random&~ev["candidate"])


def test_policy_matches_reason_risk_not_any_reason_risk():
    ev=sample(100)
    survive=np.ones((100,4),bool);survive[:20,0]=False
    policy=fit_policy(ev,survive,np.zeros(100),np.arange(100),method="frequency",
                      grid=[(0.,)],target="meaningful",alpha=.05)
    chosen=policy.release(ev)
    assert chosen.sum()==400
    assert (chosen&~survive).sum()/chosen.sum()==.05
    assert (chosen&~survive).any(1).mean()==.2


def test_robust_environment_and_cluster_bound_fail_closed():
    ev=sample(100)
    survive=np.ones((100,4),bool);survive[90:]=False
    for conservative in [False,True]:
        policy=fit_policy(ev,survive,np.r_[np.zeros(90),np.ones(10)],np.arange(100),
                          method="frequency",grid=[(0.,),(1.,)],target="meaningful",alpha=.05,
                          conservative=conservative)
        assert not policy.release(ev).any()
    # Duplicating one customer cannot manufacture independent calibration evidence.
    all_survive=np.ones((100,4),bool)
    p=fit_policy(ev,all_survive,np.zeros(100),np.zeros(100),method="frequency",grid=[(0.,)],
                 target="meaningful",alpha=.15,conservative=True)
    assert p.parameters is None


def test_continuous_grids_and_invalid_evidence():
    assert len(make_grids(sample())["stable_donor"])==108
    with pytest.raises(ValueError,match="Nonfinite"):
        evidence([[np.nan]],[[[1.]]],[[[1.]]],np.array([[False]]),[.5])
