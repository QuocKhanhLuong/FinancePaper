"""Leakage, reduction and artifact guards for the exploratory baseline audit."""
import inspect

import numpy as np
import pytest

from financepaper.data.schema import FEATURE_NAMES
from financepaper.reliability.distribution_audit import (
    current_features, sha256, split_roles, validate_current, verification_labels, verify_hashes,
)


def cache():
    n, d = 3, len(FEATURE_NAMES)
    hidden = np.zeros((n,d), bool)
    hidden[:, FEATURE_NAMES.index("AGE")] = True
    partial = np.ones((n,d))
    partial[hidden] = np.nan
    phi = np.zeros((n,d))
    phi[:,FEATURE_NAMES.index("PAY_0")] = .6
    phi[:,FEATURE_NAMES.index("BILL_AMT1")] = .4
    phi[:,FEATURE_NAMES.index("PAY_AMT1")] = .1
    return {"record_ids": np.arange(n), "hidden": hidden, "natural": np.zeros_like(hidden),
        "artificial": hidden.copy(), "partial_values": partial, "phi": phi,
        "probability": np.full(n,.3), "valid": np.ones(n,bool),
        "completion_phi": np.repeat(phi[:,None,:],16,axis=1),
        "completion_probability": np.full((n,16),.3),
        "completion_valid": np.ones((n,16),bool), "completions": np.ones((n,16,d)),
        "donor_ids": np.full((n,16),100)}


def features(c):
    return current_features(*(c[k] for k in ("probability", "completion_probability", "hidden", "phi",
                                            "completion_phi", "valid", "completion_valid")))


def test_roles_deterministic_disjoint_and_order_invariant():
    a = split_roles(np.arange(8), [10,11], fit_count=5)
    b = split_roles(np.arange(8)[::-1], [11,10], fit_count=5)
    for role in a:
        np.testing.assert_array_equal(a[role],b[role])
    assert len(set(np.concatenate(list(a.values())))) == 10


@pytest.mark.parametrize("fit,evaluation", [([1,1,2],[4]), ([1,2,3],[3]), ([1,2,3],[4,4])])
def test_roles_reject_duplicates_and_overlap(fit,evaluation):
    with pytest.raises(ValueError):
        split_roles(fit,evaluation,fit_count=1)


@pytest.mark.parametrize("broken", ["truth", "donor", "observed", "natural", "identity", "dtype"])
def test_contract_rejects_leakage_or_misalignment(broken):
    c = cache()
    validate_current(c,[0,1,2],[100])
    if broken == "truth": c["partial_values"][c["hidden"]] = 2
    elif broken == "donor": c["donor_ids"][0,0] = 0
    elif broken == "observed": c["completions"][0,0,0] = 9
    elif broken == "natural": c["natural"][0,0] = True
    elif broken == "identity": c["record_ids"][0] = 1
    else: c["hidden"] = c["hidden"].astype(int)
    with pytest.raises(ValueError):
        validate_current(c,[0,1,2],[100])


def test_feature_boundary_reduction_and_exact_rank_event():
    c = cache()
    p,e,scores,eligible = features(c)
    assert p.shape == (3,44) and e.shape == (3,48)
    np.testing.assert_array_equal(e[:,:p.shape[1]],p)
    assert eligible.all() and not scores["mc8"].any()
    assert not scores["rank_instability"].any()
    assert set(inspect.signature(current_features).parameters).isdisjoint(
        {"restored_phi", "truth", "target", "verification", "y"})
    changed = c["completion_phi"].copy()
    changed[:,0,FEATURE_NAMES.index("PAY_AMT1")] = 1
    c["completion_phi"] = changed
    pp,ee,ss,_ = features(c)
    np.testing.assert_array_equal(pp,p)  # Explanation-only changes cannot affect P.
    assert not np.array_equal(ee,e)
    np.testing.assert_allclose(ss["mc8"],1/8)
    np.testing.assert_allclose(ss["rank_instability"],.5/8)


def test_labels_are_offline_and_last_eight_only_affect_validity():
    c = cache()
    p,e,s,_ = features(c)
    restored = c["phi"].copy()
    restored[:,FEATURE_NAMES.index("PAY_AMT1")] = 1
    labels,valid = verification_labels(c["phi"],c["hidden"],restored,np.ones(3,bool))
    assert labels.all() and valid.all()
    c["truth"] = np.full((3,23),999)
    c["verification"] = restored
    c["completion_probability"][:,8:] = 1
    c["completion_phi"][:,8:] = 99
    pp,ee,ss,eligible = features(c)
    np.testing.assert_array_equal(pp,p)
    np.testing.assert_array_equal(ee,e)
    assert eligible.all()
    c["completion_valid"][0,15] = False
    assert not features(c)[3][0]


def test_source_drift_is_rejected(tmp_path):
    p = tmp_path / "receipt"
    p.write_text("before")
    hashes = {p.name: sha256(p)}
    verify_hashes(hashes,tmp_path)
    p.write_text("after")
    with pytest.raises(ValueError,match="drift"):
        verify_hashes(hashes,tmp_path)
