"""Scientific boundary tests for the new validation, not metric-only snapshots."""
import inspect
import numpy as np
import pandas as pd
import pytest
from financepaper.data.schema import FEATURE_NAMES, CATEGORICAL_FEATURES
from financepaper.data.polish import verification_masks, restore_artificial, polish_partitions
from financepaper.evaluation.revision import revision_event
from financepaper.explanations.reasons import extract_reasons
from financepaper.reliability.donors import ConditionalDonors
from financepaper.reliability.validation import (ValidationDonors, TAIWAN_GROUPS, POLISH_GROUPS,
    POLISH_NAMES, aggregate_groups, revision_arrays, fit_validation_policy, ValidationPolicy)


def test_vector_event_equals_frozen_reference():
    rng=np.random.default_rng(521)
    before,after=rng.normal(size=(2,200,23));hidden=rng.random((200,23))<.3
    result=revision_arrays(before,after,hidden)
    for i in range(200):
        reasons=extract_reasons(before[i],hidden[i],FEATURE_NAMES)
        expected=revision_event(reasons,after[i],hidden[i],FEATURE_NAMES)
        assert result["eligible"][i]==(expected is not None)
        if expected is not None:
            assert result["event"][i]==expected
    draws=np.stack([after,after+.03],1)
    result2=revision_arrays(before,draws,hidden)
    assert np.array_equal(result2["event"][:,0],result["event"])


def test_signed_groups_preserve_observed_estimand_and_whole_sum():
    rng=np.random.default_rng(92)
    for names,groups in ((FEATURE_NAMES,TAIWAN_GROUPS),(POLISH_NAMES,POLISH_GROUPS)):
        phi=rng.normal(size=(20,len(names)));hidden=rng.random(phi.shape)<.3
        a,h=aggregate_groups(phi,hidden,names,groups)
        b,_=aggregate_groups(phi,hidden,names,groups,observed_only=False)
        np.testing.assert_allclose(a.sum(1),(phi*~hidden).sum(1))
        np.testing.assert_allclose(b.sum(1),phi.sum(1))
        changed=phi.copy();changed[hidden]+=1000
        np.testing.assert_allclose(aggregate_groups(changed,hidden,names,groups)[0],a)
        assert h.shape==(20,len(groups))


def test_k8_donors_exactly_preserved_and_prefixes_nested():
    rng=np.random.default_rng(41)
    train=pd.DataFrame(rng.normal(size=(80,23))*100,columns=FEATURE_NAMES,index=np.arange(1,81))
    for f in CATEGORICAL_FEATURES:
        train[f]=rng.integers(0,4,len(train))
    query=train.iloc[:12].copy();hidden=rng.random(query.shape)<.3;query=query.mask(hidden)
    old=ConditionalDonors(train,seed=33).complete(query,draws=8,seed=184)
    sampler=ValidationDonors(train,CATEGORICAL_FEATURES,seed=33)
    result,ids=sampler.sample(query,hidden,k=16,seed=184)
    np.testing.assert_equal(result[:,:8],np.stack([x.to_numpy() for x in old],axis=1))
    for k in (1,2,4,8):
        x,i=sampler.sample(query,hidden,k=k,seed=184)
        np.testing.assert_equal(x,result[:,:k]);np.testing.assert_array_equal(i,ids[:,:k])
    with pytest.raises(ValueError,match="erased"):
        sampler.sample(train.iloc[:12],hidden)


def test_natural_unknowns_never_become_verification_truth():
    rng=np.random.default_rng(9)
    X=pd.DataFrame(rng.normal(size=(64,4)),columns=list("abcd"))
    X.iloc[0,0]=np.nan
    masks=verification_masks(X,51)
    assert not masks["mcar30"][0,0]
    h=masks["mcar30"];partial=X.mask(h)
    restored=restore_artificial(partial,X,h)
    pd.testing.assert_frame_equal(restored,X)
    donor=ValidationDonors(X.dropna())
    draws,_=donor.sample(partial,h)
    assert np.isnan(draws[0,:,0]).all()
    for i in range(len(X)):
        np.testing.assert_equal(draws[i,:,~h[i]],np.broadcast_to(partial.iloc[i].to_numpy()[~h[i]][:,None],draws[i,:,~h[i]].shape))
    bad=h.copy();bad[0,0]=True
    with pytest.raises(ValueError,match="natural"):
        restore_artificial(partial,X,bad)


def test_duplicate_records_stay_together():
    rng=np.random.default_rng(18)
    X=pd.DataFrame(rng.normal(size=(400,3)))
    groups=np.repeat(np.arange(200),2); y=np.repeat(np.arange(200)%2,2)
    parts=polish_partitions(X,y,groups)
    sets=[set(groups[idx]) for idx in parts.values()]
    assert sum(map(len,parts.values()))==400
    for i,a in enumerate(sets):
        for b in sets[i+1:]:
            assert not a&b


def test_robust_policy_and_inference_boundary():
    frame=pd.DataFrame(dict(record_id=np.tile(np.arange(200),2),condition=np.repeat(["a","b"],200),
        probability=np.tile(np.r_[np.zeros(100),np.ones(100)],2),eligible=True,
        event=np.r_[np.zeros(200),np.zeros(100),np.ones(100)],stratum=1,assigned=True))
    pooled=fit_validation_policy(frame,family="pooled",risk_target=.3,conservative=False,environments=["a","b"])
    robust=fit_validation_policy(frame,family="robust",risk_target=.3,conservative=False,environments=["a","b"])
    assert pooled.thresholds==(1.,)
    assert robust.thresholds==(0.,)
    assert set(inspect.signature(ValidationPolicy.release).parameters)=={"self","probability","eligible","strata"}
    selected=robust.release(frame.probability,frame.eligible,frame.stratum)
    assert selected.sum()==200
    # An absent observable stratum cannot be optimistically released.
    stratified=fit_validation_policy(frame,family="stratified",risk_target=.3,conservative=False,environments=["a","b"])
    assert stratified.thresholds[0] is None and stratified.thresholds[2] is None


def test_empty_release_risk_not_reported_as_safe_threshold():
    f=pd.DataFrame(dict(probability=[.5]*80,eligible=False,event=0,stratum=0,assigned=True,condition="a"))
    p=fit_validation_policy(f,family="pooled",risk_target=.1,conservative=True,environments=["a"])
    assert p.thresholds==(None,)
    assert not p.release(f.probability,f.eligible,f.stratum).any()
