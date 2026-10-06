"""Independent exact oracles and rejection tests for the round-five controls."""
import numpy as np
import pytest

from financepaper.explanations.conditional_moments import CompiledAttributions, FiniteJointLaw, Leaf
from financepaper.explanations.rank_event_audit import (
    FiniteContrastAudit, coalition_shap, toy_compiler, toy_margin,
)
from financepaper.reliability.validation import revision_arrays


def setup_toy(additive=False):
    support = np.array([-1., -.5, 0., 1., 2.])
    points = np.column_stack([np.ones((5, 3)), support])
    compiler = toy_compiler(additive=additive)
    current = compiler.values(points[[0]])[0, :3]
    query = FiniteContrastAudit(compiler, np.eye(4)[:, :3], current, np.ones(3, bool))
    return compiler, points, query


@pytest.mark.parametrize("additive", [False, True])
def test_compiler_against_independent_coalition_oracle(additive):
    compiler, points, _ = setup_toy(additive)
    # Include all split boundaries, not just positive-support atoms.
    points = np.vstack([points, [[1, 1, 1, h] for h in [-.75, -.25, .5, 1.5]],
                        [[0, 1, 0, 2], [1, 0, 1, -1]]])
    predict = lambda x: toy_margin(x, additive=additive)
    oracle = coalition_shap(predict, points, np.zeros(4))
    np.testing.assert_allclose(compiler.values(points), oracle, atol=1e-12, rtol=0)
    np.testing.assert_allclose(oracle.sum(1), predict(points), atol=1e-12, rtol=0)


def test_matched_moments_do_not_identify_rank_risk():
    compiler, points, query = setup_toy()
    partial, observed = np.array([1, 1, 1, np.nan]), np.array([1, 1, 1, 0], bool)
    a = FiniteJointLaw(points, [.5, 0, 0, .5, 0])
    b = FiniteJointLaw(points, [0, .8, 0, 0, .2])
    ma, mb = [compiler.moments(partial, observed, law) for law in (a, b)]
    np.testing.assert_allclose(ma.mean, mb.mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ma.covariance, mb.covariance, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ma.mean, [2, 2, 2.125, 0], atol=1e-12)
    np.testing.assert_allclose(ma.covariance[:2, :2], [[.25, -.25], [-.25, .25]], atol=1e-12)
    assert query.probability(partial, observed, a) == .5
    assert query.probability(partial, observed, b) == .2
    np.testing.assert_allclose(toy_margin(points), 6.125)


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_direct_contrasts_match_frozen_event_and_ignore_hidden_placeholders(seed):
    compiler, points, query = setup_toy()
    weights = np.random.default_rng(seed).uniform(.1, 1, len(points))
    weights /= weights.sum()
    law = FiniteJointLaw(points, weights)
    partial, observed = np.array([1, 1, 1, np.nan]), np.array([1, 1, 1, 0], bool)
    direct = query.events(partial, observed, law)
    frozen = revision_arrays(compiler.values(points[[0]])[:, :3],
                             compiler.values(points)[None, :, :3], np.zeros((1, 3), bool), k=2)
    assert frozen["eligible"].item()
    np.testing.assert_array_equal(direct, frozen["event"][0])
    np.testing.assert_array_equal(direct, [False, False, False, True, True])
    # Group 0 was not in current top-2, but can displace group 1.
    assert 0 not in query.candidates
    for placeholder in [-1000., 1000., np.inf]:
        altered = partial.copy()
        altered[3] = placeholder
        np.testing.assert_array_equal(direct, query.events(altered, observed, law))
    order = np.random.default_rng(seed).permutation(len(points))
    permuted = FiniteJointLaw(points[order], weights[order])
    assert query.probability(partial, observed, permuted) == pytest.approx(weights @ direct)


def test_additive_and_degenerate_controls():
    _, points, query = setup_toy(additive=True)
    observed, partial = np.array([1, 1, 1, 0], bool), [1, 1, 1, np.nan]
    assert query.probability(partial, observed, FiniteJointLaw(points, np.ones(5)/5)) == 0
    _, _, query = setup_toy()
    assert query.probability(partial, observed, FiniteJointLaw(points[[0]], [1])) == 0
    assert query.probability(partial, observed, FiniteJointLaw(points[[-1]], [1])) == 1


def test_tie_tolerance_and_sign_not_postverification_magnitude():
    # One active leaf per coordinate permits independent positive/negative scores.
    compiler = CompiledAttributions(
        [Leaf(v, ((j, .5, np.inf),)) for j, v in enumerate([.005, .005, 1.])], np.zeros((1, 3)))
    points = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]])
    query = FiniteContrastAudit(compiler, np.eye(3), [.03, .02, 0], np.ones(3, bool), k=1)
    # .005 remains positive despite being below the *selection* threshold .01.
    np.testing.assert_array_equal(query.events([np.nan]*3, np.zeros(3, bool),
                                  FiniteJointLaw(points, [1/3]*3)), [False, True, True])
    loose = FiniteContrastAudit(compiler, np.eye(3), [.03, .02, 0], np.ones(3, bool),
                               k=1, rank_epsilon=1.)
    assert not loose.events([np.nan]*3, np.zeros(3, bool), FiniteJointLaw(points[[2]], [1])).item()


def test_any_outsider_overtake_is_not_the_frozen_event_at_ties():
    compiler = CompiledAttributions(
        [Leaf(v, ((j, .5, np.inf),)) for j, v in enumerate([1., 1., 2.])], np.zeros((1, 3)))
    before, after = np.array([[2., 1.5, 0.]]), np.array([[1., 1., 2.]])
    query = FiniteContrastAudit(compiler, np.eye(3), before[0], np.ones(3, bool), k=2)
    # The outsider overtakes both displayed reasons, but only one reason is
    # strictly above each. Frozen tolerant rank membership retains both.
    assert after[0, 2] > after[0, query.candidates].min()
    reference = revision_arrays(before, after, np.zeros((1, 3), bool), k=2)
    assert not reference["event"].item()
    assert not query.events([np.nan]*3, np.zeros(3, bool), FiniteJointLaw([[1., 1., 1.]], [1.])).item()


@pytest.mark.parametrize("weights", [[.2]*4, [.2, .2, .2, .2, -.2], [np.nan]*5, [0]*5])
def test_corrupted_weights_fail(weights):
    _, points, _ = setup_toy()
    with pytest.raises(ValueError):
        FiniteJointLaw(points, weights)


@pytest.mark.parametrize("current,available,k", [([0, 0, 0], [True]*3, 2),
    ([1, 1, 1], [False]*3, 2), ([np.nan, 1, 1], [True]*3, 2),
    ([1, 1, 1], [True]*3, 4), ([1, 1, 1], [1]*3, 2)])
def test_invalid_or_ineligible_candidate_state_fails(current, available, k):
    compiler, _, _ = setup_toy()
    with pytest.raises(ValueError):
        FiniteContrastAudit(compiler, np.eye(4)[:, :3], current, available, k=k)


def test_invalid_observed_values_fail():
    _, points, query = setup_toy()
    law = FiniteJointLaw(points, np.ones(5)/5)
    with pytest.raises(ValueError):
        query.events([np.nan]*4, np.array([1, 1, 1, 0], bool), law)
    with pytest.raises(ValueError):
        query.events([1, 1, 1, 0], [1, 1, 1, 0], law)


def test_constant_compiler_with_no_terms_is_handled():
    compiler = CompiledAttributions([Leaf(3., ())], np.zeros((1, 1)))
    query = FiniteContrastAudit(compiler, np.eye(1), [1.], [True], k=1)
    assert query.probability([np.nan], np.array([False]), FiniteJointLaw([[0]], [1])) == 1
