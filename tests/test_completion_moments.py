"""Definition-level checks of research estimands, not model performance tests."""
import numpy as np
import pytest

from financepaper.evaluation.completion_moments import (
    FiniteMarginal, Leaf, NormalMarginal, attribution_moments,
    cell_enumeration_moments, coefficient_moments, covariance_witness,
    enumeration_moments, evaluate, finite_states, point_shapley, threshold_states,
    tree_leaves, joint_law_witness,
)


def random_tree(rng, dimensions, depth):
    if depth == 0:
        return float(rng.normal())
    return dict(feature=int(rng.integers(dimensions)), threshold=float(rng.choice([-.5, .5])),
                left=random_tree(rng, dimensions, depth-1), right=random_tree(rng, dimensions, depth-1))


@pytest.mark.parametrize("seed", range(8))
def test_against_coalition_definition_and_strong_baselines(seed):
    rng = np.random.default_rng(seed)
    leaves = tree_leaves(random_tree(rng, 3, 3))+tree_leaves(random_tree(rng, 3, 2))
    p = [FiniteMarginal((-1., 0., 1.), tuple(rng.dirichlet(np.ones(3)))) for _ in range(3)]
    q = [FiniteMarginal((-1., 0., 1.), tuple(rng.dirichlet(np.ones(3)))) for _ in range(3)]
    oracle = enumeration_moments(leaves, p, q, [0, 1, 2], definition_oracle=True)
    for result in (attribution_moments(leaves, p, q, [0, 1, 2]),
                   coefficient_moments(leaves, p, q, [0, 1, 2]),
                   cell_enumeration_moments(leaves, p, q, [0, 1, 2], pairwise=True)):
        for name in ("mean", "second", "covariance"):
            np.testing.assert_allclose(result[name], oracle[name], rtol=1e-11, atol=2e-13)


def test_shap_is_not_constant_on_original_leaf():
    leaves = [Leaf(1., ((0, .5, np.inf), (1, .5, np.inf)))]
    p = [FiniteMarginal((1.,), (1.,)), FiniteMarginal((0.,), (1.,))]
    x = [[0., 0.], [0., 1.]]
    np.testing.assert_array_equal(evaluate(leaves, x), [0, 0])
    np.testing.assert_allclose(point_shapley(leaves, p, x, [0])[:, 0], [0., -.5])


def test_shared_feature_cross_tree_cancellation():
    leaves = [Leaf(1., ((0, .5, np.inf), (2, .5, np.inf))),
              Leaf(-1., ((1, .5, np.inf), (2, .5, np.inf)))]
    p = [FiniteMarginal((0.,), (1.,))]*3
    q = [FiniteMarginal((1.,), (1.,))]*2+[FiniteMarginal((0., 1.), (.5, .5))]
    result = attribution_moments(leaves, p, q, [0, 1, 2])
    np.testing.assert_allclose(result["covariance"], [[1/16, -1/16, 0], [-1/16, 1/16, 0], [0, 0, 0]], atol=1e-15)
    independent_trees = sum(attribution_moments([leaf], p, q, [0, 1, 2])["covariance"] for leaf in leaves)
    assert independent_trees[2, 2] > 0  # Wrong ablation loses the exact cancellation.


def test_continuous_completion_threshold_cell_oracle():
    leaves = tree_leaves(dict(feature=0, threshold=.2, left=-1., right=dict(
        feature=0, threshold=.9, left=2., right=dict(feature=1, threshold=-.3, left=4., right=-2.))))
    p, q = [NormalMarginal(0, 1)]*2, [NormalMarginal(.3, .7), NormalMarginal(-.2, 1.4)]
    truth = cell_enumeration_moments(leaves, p, q, [0, 1])
    got = attribution_moments(leaves, p, q, [0, 1])
    np.testing.assert_allclose(got["covariance"], truth["covariance"], atol=2e-14)
    with pytest.raises(OverflowError):
        threshold_states(leaves, q, max_states=1)


@pytest.mark.parametrize("scale", [0., 1., 100.])
@pytest.mark.parametrize("reference_mode", ["point", "full_support"])
def test_arbitrary_zero_sum_covariance_realization(scale, reference_mode):
    b = scale*np.array([[1., 2.], [-3., 1.], [2., -3.]])
    mu = np.array([.1, -.3, .7])
    leaves, p, q = covariance_witness(b, mu, reference_mode=reference_mode)
    x, _ = finite_states(q)
    np.testing.assert_allclose(evaluate(leaves, x), mu.sum()*(1 if reference_mode == "point" else 2), atol=3e-13)
    phi = point_shapley(leaves, p, x, list(range(5)))
    np.testing.assert_allclose(phi[:, :3], mu+x[:, 3:] @ b.T, atol=1e-13)
    np.testing.assert_allclose(phi[:, 3:], 0, atol=1e-13)
    result = attribution_moments(leaves, p, q, list(range(5)))
    np.testing.assert_allclose(result["covariance"][:3, :3], b @ b.T, atol=1e-8)
    np.testing.assert_allclose(result["covariance"][3:, :], 0, atol=1e-8)


def test_moments_do_not_identify_sign_probability():
    # Same mean zero and second moment one; different P(X>0).
    x = np.array([-1., 1.]); w = np.array([.5, .5])
    z = np.array([-2., 0., 1.]); v = np.array([1/6, .5, 1/3])
    np.testing.assert_allclose([w@x, w@(x*x)], [v@z, v@(z*z)])
    assert w[x > 0].sum() != v[z > 0].sum()


def test_arbitrary_joint_law_one_hidden_field_full_support_reference():
    desired = np.array([[1., 4., -2.], [-3., 2., 5.], [0., 0., 1.], [3., -1., 2.]])
    weights = np.array([.1, .2, .3, .4])
    leaves, p, q = joint_law_witness(desired, weights)
    x, w = finite_states(q)
    mu = weights @ desired
    np.testing.assert_allclose(evaluate(leaves, x), 2*mu.sum(), atol=1e-13)
    phi = point_shapley(leaves, p, x, [0, 1, 2, 3])
    np.testing.assert_allclose(phi[:, :3], desired, atol=1e-13)
    np.testing.assert_allclose(phi[:, 3], -(desired-mu).sum(axis=1), atol=1e-13)
    oracle = enumeration_moments(leaves, p, q, [0, 1, 2, 3], definition_oracle=True)
    truth = (desired.T*weights) @ desired-np.outer(mu, mu)
    np.testing.assert_allclose(oracle["covariance"][:3, :3], truth, atol=1e-12)
    np.testing.assert_allclose(attribution_moments(leaves, p, q, [0, 1, 2, 3])["covariance"], oracle["covariance"], atol=1e-12)
