import numpy as np
import pytest
from financepaper.evaluation.revision import revision_event
from financepaper.evaluation.revision_audit import event_components, normalized_shift, observed_groups
from financepaper.data.schema import FEATURE_NAMES
from financepaper.data.temporal import TemporalPreprocessor
from financepaper.explain.permutation import permutation_attributions


def test_rank_tie_tolerance_does_not_move_sign_boundary():
    names = ("a","b","c","d")
    phi = np.array([.004,.2,.3,.005])
    hidden = np.zeros(4,bool)
    event = event_components(("a","b","c"),phi,hidden,names,rank_epsilon=.01)
    assert not event["event"]
    assert event_components(("a","b","c"),[-.004,.2,.3,.005],hidden,names,rank_epsilon=.01)["sign_change"]


def test_primary_exactly_matches_original_target():
    rng = np.random.default_rng(932)
    for _ in range(100):
        after = rng.normal(size=23)
        hidden = rng.random(23)<.3
        idx = np.flatnonzero(~hidden)[:3]
        reasons = tuple(FEATURE_NAMES[j] for j in idx)
        assert event_components(reasons,after,hidden)["event"] == revision_event(reasons,after,hidden,FEATURE_NAMES)


def test_semantic_groups_exclude_restored_hidden_truth():
    before = np.ones((2,23))
    mask = np.zeros((2,23),bool)
    mask[:,FEATURE_NAMES.index("PAY_0")] = True
    after = before.copy()
    after[:,FEATURE_NAMES.index("PAY_0")] = 1e9
    np.testing.assert_array_equal(observed_groups(before,mask)[0],observed_groups(after,mask)[0])
    np.testing.assert_array_equal(normalized_shift(before,after,mask),[0,0])


def test_shared_game_is_exact_for_additive_and_uses_no_target(credit_frame):
    X,_=credit_frame
    prep=TemporalPreprocessor().fit(X.iloc[:100])
    refs=prep.transform(X.iloc[100:104])
    batch=prep.transform(X.iloc[104:108])
    def f(b):
        return b.temporal.sum((1,2)).astype(float)+2*b.static.sum(1)+b.static_observed.sum(1)
    attr=permutation_attributions(batch,refs,f,permutations=8)
    assert attr["valid"].all()
    np.testing.assert_allclose(attr["original"].sum(1), f(batch)-attr["baseline_logit"],atol=1e-6)
    other=permutation_attributions(batch,refs,f,permutations=8,seed=2)
    np.testing.assert_allclose(attr["original"],other["original"],atol=1e-5)


def test_donors_preserve_observed_and_cannot_access_hidden_truth(credit_frame):
    from financepaper.reliability.donors import ConditionalDonors
    X,_=credit_frame
    donor=ConditionalDonors(X.iloc[:100],max_reference=80)
    partial=X.iloc[100:110].astype(float).copy()
    partial.iloc[:,[0,8]]=np.nan
    out=donor.complete(partial,draws=3)
    again=donor.complete(partial,draws=3)
    for a,b in zip(out,again):
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a.to_numpy()[partial.notna()],partial.to_numpy()[partial.notna()])
        assert not a.isna().any().any()
        assert set(a.iloc[:,0]).issubset(set(donor.reference.iloc[:,0]))


def test_inference_boundary_rejects_restoration_fields():
    from financepaper.reliability.current import CurrentEvidence,current_features,RECIPES
    args=dict(probability=np.array([.3]),hidden=np.zeros((1,23),bool),
        attribution=np.ones((1,23)),completion_probabilities=np.array([[.2,.4]]))
    a=current_features(CurrentEvidence(**args))
    assert set(sum(RECIPES.values(),[])).issubset(a.columns)
    for forbidden in ("prob_full","restored_prediction","revision_event","y","full_attribution"):
        with pytest.raises(TypeError): CurrentEvidence(**args,**{forbidden:np.array([999])})


def test_policy_ties_unknowns_and_zero_release():
    from financepaper.reliability.policy import fit_policy
    n=1000
    good=fit_policy(np.full(n,.2),np.ones(n,bool),np.zeros(n),conservative=True)
    assert good.calibration_released==n
    assert good.release([.2,.2],[True,False]).tolist()==[True,False]
    bad=fit_policy(np.full(n,.2),np.ones(n,bool),np.ones(n),conservative=True)
    assert bad.threshold is None and bad.upper_bound is None
    assert not bad.release([0.],[True])[0]
    unknown=fit_policy(np.zeros(10),np.ones(10,bool),np.full(10,np.nan))
    assert unknown.threshold is None
    small=fit_policy(np.zeros(5),np.ones(5,bool),np.zeros(5),conservative=True)
    assert small.threshold is None  # zero observed errors is not zero population risk


def test_condition_assignment_is_customer_level_and_outcome_blind():
    import pandas as pd
    from financepaper.reliability.policy import one_condition_per_customer
    frame=pd.DataFrame([(i,c) for i in range(100) for c in ("complete","mcar10","mcar20","mcar30","mar30")],columns=["record_id","condition"])
    a=one_condition_per_customer(frame,11)
    assert a.sum()==100 and frame[a].record_id.nunique()==100
    frame["revision_event"]=np.arange(len(frame))%2
    np.testing.assert_array_equal(a,one_condition_per_customer(frame,11))


def test_current_stats_ignore_hidden_attribution_truth():
    from financepaper.reliability.current import CurrentEvidence,current_features,RECIPES
    hidden=np.zeros((2,23),bool); hidden[:,7]=True
    phi=np.tile(np.linspace(-1,1,23),(2,1))
    def features(a):
        return current_features(CurrentEvidence(np.array([.3,.5]),hidden,a,np.full((2,8),.4)))
    before=features(phi)
    phi[:,7]=1e12
    np.testing.assert_array_equal(before[RECIPES["mask_explanation"]],features(phi)[RECIPES["mask_explanation"]])


def test_partitions_exclude_historical_test_and_reserved_customers():
    import pandas as pd
    from types import SimpleNamespace
    from financepaper.experiments.revision_study import make_partitions,load_config
    cfg,_=load_config("configs/revision_study.yaml")
    data=SimpleNamespace(y=pd.Series(np.random.default_rng(88).binomial(1,.22,30000)))
    folds,excluded=make_partitions(data,cfg)
    outer=[]
    assert len(excluded)==9000
    for parts in folds:
        all_ids=np.concatenate(list(parts.values()))
        assert len(all_ids)==len(set(all_ids))==21000
        assert not set(all_ids).intersection(excluded)
        outer.extend(parts["outer"])
    assert len(outer)==len(set(outer))==21000


def test_serving_schema_has_no_verification_and_never_releases_missing_reason():
    from financepaper.reliability.current import CurrentEvidence,RECIPES
    from financepaper.reliability.inference import diagnostic_output
    from financepaper.reliability.policy import ReleasePolicy
    from financepaper.training.calibration import PositiveSlopePlattCalibrator
    class ConstantSelector:
        def predict_proba(self,features): return np.tile([.9,.1],(len(features),1))
    hidden=np.zeros((1,23),bool);hidden[:,0]=True
    evidence=CurrentEvidence(np.array([.3]),hidden,np.ones((1,23)),np.full((1,8),.3))
    selector=dict(model=ConstantSelector(),columns=RECIPES["mask_explanation"],calibrator=PositiveSlopePlattCalibrator(1,0))
    result=diagnostic_output(evidence,np.array([-.8]),selector,ReleasePolicy(.2,100,50,0,.08,False))[0]
    assert "verification_only" not in result and "debug" not in result
    assert FEATURE_NAMES[0] not in result["explanation"]["top_features"]
    assert result["reliability"]["release_decision"]
