import inspect

import numpy as np
import pytest

from financepaper.reliability.stable_core import (
    StableCorePolicy, completion_evidence, fit_stable_policy, stable_core_metrics, verified_survival,
)


def evidence(n=1):
    current = np.tile([.6, .3, -.2], (n, 1))
    completions = np.tile([[[.7, .4, -.1], [.6, -.1, .3]]], (n, 1, 1))
    return completion_evidence(current, completions, np.zeros_like(current, bool))


def test_partial_release_and_no_verification_inputs():
    ev = evidence()
    np.testing.assert_equal(ev.candidates, [[True, True, False]])
    np.testing.assert_allclose(ev.joint_frequency, [[1., .5, .5]])
    selected = StableCorePolicy(1., 1., .1, "robust").release(ev)
    np.testing.assert_equal(selected, [[True, False, False]])
    assert "restored" not in inspect.signature(completion_evidence).parameters
    # Changing later truth affects only the evaluator, never the serving decision.
    a = verified_survival([[.6, .3, -.2]], np.zeros((1, 3), bool))
    b = verified_survival([[-.6, .3, .8]], np.zeros((1, 3), bool))
    assert stable_core_metrics(selected, a, ev.candidates)["reason_precision"] == 1.
    assert stable_core_metrics(selected, b, ev.candidates)["reason_precision"] == 0.
    np.testing.assert_equal(selected, StableCorePolicy(1., 1., .1, "robust").release(ev))


def test_near_ties_hidden_groups_and_up_to_one_candidate():
    cur = [[.1, -.1, 100.]]
    comp = [[[.1, .10000001, 100.]]]
    ev = completion_evidence(cur, comp, np.array([[False, False, True]]), k=1)
    assert ev.candidates.sum() == 1 and ev.rank_frequency[0, 0] == 1.
    assert not ev.candidates[0, 2]


def test_robust_calibration_does_not_pool_away_failing_environment():
    ev = evidence(10000)
    survived = np.ones((10000, 3), bool)
    survived[9000:, 0] = False  # Every first reason fails in a small environment.
    ids = np.arange(10000)
    env = np.array(["easy"] * 9000 + ["hard"] * 1000)
    robust = fit_stable_policy(ev, survived, ids, env, alpha=.1)
    assert not robust.release(ev).any()
    pooled = fit_stable_policy(ev, survived, ids, env, alpha=.1, family="pooled", conservative=False)
    assert pooled.release(ev).any()


def test_conservative_calibration_can_release_and_handles_empty_sets():
    ev = evidence(10000)
    survived = np.ones((10000, 3), bool)
    policy = fit_stable_policy(ev, survived, np.arange(10000), np.zeros(10000), alpha=.1)
    assert policy.release(ev).sum() == 20000
    metrics = stable_core_metrics(np.zeros((10000, 3), bool), survived, ev.candidates)
    assert metrics["false_stable_rate"] is None
    assert metrics["customer_coverage"] == 0
    with pytest.raises(ValueError, match="Repeated"):
        fit_stable_policy(ev, survived, np.zeros(10000), np.zeros(10000), alpha=.1)
