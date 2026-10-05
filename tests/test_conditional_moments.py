"""Independent coalition oracles for the numerical feasibility prototype."""
from itertools import product
import math

import numpy as np
import pytest

from financepaper.explanations.conditional_moments import (
    CompiledAttributions, FiniteJointLaw, Leaf, MomentBudgetExceeded, ProductLaw,
    leaves_from_xgboost,
)


def score(leaves, points):
    out = np.zeros(len(points))
    for leaf in leaves:
        inside = np.ones(len(points), bool)
        for j, lo, hi in leaf.bounds:
            inside &= (points[:, j] >= lo) & (points[:, j] < hi)
        out += leaf.value * inside
    return out


def coalition_oracle(leaves, background, point):
    """Enumerate feature subsets and actual background splices, not path formulas."""
    d = len(point)
    games = {}
    for mask in product([False, True], repeat=d):
        x = background.copy()
        x[:, mask] = point[np.array(mask)]
        games[mask] = score(leaves, x).mean()
    result = np.zeros(d)
    for mask, value in games.items():
        s = sum(mask)
        for j in range(d):
            if not mask[j]:
                added = list(mask)
                added[j] = True
                weight = math.factorial(s)*math.factorial(d-s-1)/math.factorial(d)
                result[j] += weight*(games[tuple(added)]-value)
    return result


BACKGROUND = np.array([[-1., -1., .5], [-1., -1., -.5], [1., 1., 1.], [1., -1., -1.]])
WORLDS = [
    # Additive leaves from independent stumps, including an irrelevant coordinate.
    [Leaf(-1., ((0, -np.inf, 0.),)), Leaf(2., ((0, 0., np.inf),)),
     Leaf(.6, ((1, -np.inf, 0.),)), Leaf(-.4, ((1, 0., np.inf),))],
    # Prediction-constant left leaf, but SHAP depends on feature 1 there.
    [Leaf(0., ((0, -np.inf, 0.),)),
     Leaf(1., ((0, 0., np.inf), (1, -np.inf, 0.))),
     Leaf(3., ((0, 0., np.inf), (1, 0., np.inf)))],
    # Already intersected bounds from a repeated-feature split and a second tree.
    [Leaf(-2., ((0, -np.inf, -.5),)), Leaf(1., ((0, -.5, .5),)),
     Leaf(4., ((0, .5, np.inf),)), Leaf(2., ((1, 0., np.inf),))],
]


@pytest.mark.parametrize('leaves', WORLDS)
def test_exhaustive_all_observation_subsets_and_correlated_law(leaves):
    compiler = CompiledAttributions(leaves, BACKGROUND)
    support = np.array(list(product([-1., 0., 1.], repeat=3)))
    weights = np.arange(1, len(support)+1, dtype=float)
    weights /= weights.sum()  # Joint, nonfactorizing completion law.
    x = np.array([.25, -.25, .75])
    for pattern in product([False, True], repeat=3):
        observed = np.array(pattern)
        law = FiniteJointLaw(support, weights)
        full = support.copy()
        full[:, observed] = x[observed]
        phis = np.array([coalition_oracle(leaves, BACKGROUND, point) for point in full])
        mean = weights @ phis
        covariance = (phis-mean).T @ ((phis-mean)*weights[:, None])
        partial = x.copy()
        partial[~observed] = np.nan
        result = compiler.moments(partial, observed, law)
        generic = compiler.moments(partial, observed, law, specialize=False)
        np.testing.assert_allclose(compiler.values(full), phis, atol=1e-12)
        np.testing.assert_allclose(result.mean, mean, atol=1e-12)
        np.testing.assert_allclose(result.covariance, covariance, atol=1e-12)
        np.testing.assert_allclose(result.covariance, generic.covariance, atol=1e-12)
        assert np.linalg.eigvalsh(result.covariance).min() > -1e-11
        # Efficiency includes all cross-tree covariance terms.
        raw = score(leaves, full)
        np.testing.assert_allclose(result.covariance.sum(), weights @ (raw-weights@raw)**2, atol=1e-11)


def test_product_joint_agreement_hidden_truth_exclusion_and_grouping():
    compiler = CompiledAttributions(WORLDS[1], BACKGROUND)
    atoms = [np.array([-1., 0., 1.])]*3
    weights = [np.array([.2, .3, .5])]*3
    law = ProductLaw(atoms, weights)
    points = np.array(list(product(*atoms)))
    joint_weights = np.array([np.prod(w) for w in product(*weights)])
    observed = np.array([True, False, False])
    r1 = compiler.moments([-1., np.nan, np.nan], observed, law)
    r2 = compiler.moments([-1., 999., -100.], observed, FiniteJointLaw(points, joint_weights))
    np.testing.assert_allclose(r1.mean, r2.mean, atol=1e-12)
    np.testing.assert_allclose(r1.covariance, r2.covariance, atol=1e-12)
    grouping = np.array([[1., 0.], [1., 0.], [0., 1.]])
    grouped = compiler.moments([-1., np.nan, np.nan], observed, law, grouping=grouping)
    np.testing.assert_allclose(grouped.mean, r1.mean @ grouping, atol=1e-12)
    np.testing.assert_allclose(grouped.covariance, grouping.T@r1.covariance@grouping, atol=1e-12)


def test_point_mass_reduction_and_background_immutability():
    background = BACKGROUND.copy()
    compiler = CompiledAttributions(WORLDS[1], background)
    background[:] = 1000
    point = np.array([[1., 1., 1.]])
    result = compiler.moments([np.nan]*3, np.zeros(3, bool), FiniteJointLaw(point, [1.]))
    np.testing.assert_allclose(result.mean, coalition_oracle(WORLDS[1], BACKGROUND, point[0]), atol=1e-12)
    np.testing.assert_allclose(result.covariance, 0, atol=1e-12)
    assert not compiler.background.flags.writeable


def test_off_path_prediction_constant_does_not_imply_attribution_constant():
    background = np.array(list(product([0., 1.], repeat=2)))
    leaves = [Leaf(0., ((0, -np.inf, .5),)),
              Leaf(1., ((0, .5, np.inf), (1, -np.inf, .5))),
              Leaf(3., ((0, .5, np.inf), (1, .5, np.inf)))]
    compiler = CompiledAttributions(leaves, background)
    points = np.array([[0., 0.], [0., 1.]])
    np.testing.assert_array_equal(score(leaves, points), [0., 0.])
    np.testing.assert_allclose(compiler.values(points), [[-.75, -.25], [-1.25, .25]])
    moments = compiler.moments([0., np.nan], np.array([True, False]), FiniteJointLaw(points, [.5, .5]))
    assert np.trace(moments.covariance) > 0
    assert abs(moments.covariance.sum()) < 1e-12


def test_equal_moments_do_not_determine_positive_reason_probability():
    a, wa = np.array([-1., 1.]), np.array([.5, .5])
    b, wb = np.array([-2., .5]), np.array([.2, .8])
    assert np.isclose(wa@a, wb@b)
    assert np.isclose(wa@(a*a), wb@(b*b))
    assert wa@(a > 0) != wb@(b > 0)


def test_budget_and_input_guards():
    compiler = CompiledAttributions(WORLDS[1], BACKGROUND)
    law = FiniteJointLaw(BACKGROUND, [.25]*4)
    with pytest.raises(MomentBudgetExceeded):
        compiler.moments([np.nan]*3, np.zeros(3, bool), law, max_terms=0)
    with pytest.raises(MomentBudgetExceeded):
        compiler.moments([np.nan]*3, np.zeros(3, bool), law, timeout_seconds=-1)
    with pytest.raises(ValueError):
        compiler.moments([0]*3, [1, 0, 0], law)
    with pytest.raises(ValueError):
        compiler.moments([np.nan]*3, np.ones(3, bool), law)
    with pytest.raises(ValueError):
        ProductLaw([[0, 1]], [[.2, .2]])
    with pytest.raises(ValueError):
        FiniteJointLaw([[np.nan]], [1.])


def test_constant_model_zero_attribution():
    compiler = CompiledAttributions([Leaf(10., ())], BACKGROUND)
    result = compiler.moments([np.nan]*3, np.zeros(3, bool), FiniteJointLaw(BACKGROUND, [.25]*4))
    np.testing.assert_array_equal(result.mean, np.zeros(3))
    np.testing.assert_array_equal(result.covariance, np.zeros((3, 3)))


def test_xgboost_parser_repeated_splits_threshold_atoms_and_shap():
    import xgboost as xgb
    import shap
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 3)).astype(np.float32)
    y = (x[:, 0]**2 + x[:, 1] > 1).astype(int)
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, n_jobs=1, random_state=11)
    model.fit(x, y)
    leaves = leaves_from_xgboost(model)
    compiler = CompiledAttributions.from_xgboost(model, x[:16])
    atoms = x[:8].copy()
    for i, leaf in enumerate(leaves[:8]):
        for j, lo, hi in leaf.bounds:
            if np.isfinite(lo):
                atoms[i, j] = lo
            elif np.isfinite(hi):
                atoms[i, j] = hi
    explainer = shap.TreeExplainer(model, shap.maskers.Independent(x[:16], max_samples=16),
                                   feature_perturbation='interventional', model_output='raw')
    np.testing.assert_allclose(compiler.values(atoms), explainer.shap_values(atoms), atol=2e-5)
    compiled2 = CompiledAttributions.from_xgboost(model, x[:16])
    np.testing.assert_array_equal(compiler.coefficients, compiled2.coefficients)

    # A float64 immediately below a float32 threshold rounds *to* that threshold.
    neighbors = []
    for leaf in leaves:
        for j, lo, hi in leaf.bounds:
            for endpoint in [lo, hi]:
                if np.isfinite(endpoint):
                    for value in [np.nextafter(endpoint, -np.inf), endpoint,
                                  np.nextafter(endpoint, np.inf)]:
                        p = x[0].astype(float)
                        p[j] = value
                        neighbors.append(p)
    neighbors = np.array(neighbors)
    expected = explainer.shap_values(neighbors.astype(np.float32))
    np.testing.assert_allclose(compiler.values(neighbors), expected, atol=2e-5)
    law = FiniteJointLaw(neighbors, np.full(len(neighbors), 1/len(neighbors)))
    moments = compiler.moments([np.nan]*3, np.zeros(3, bool), law)
    np.testing.assert_allclose(moments.mean, expected.mean(axis=0), atol=2e-5)
    np.testing.assert_allclose(moments.covariance, np.cov(expected.T, bias=True), atol=2e-5)


def test_parser_explicit_repeated_feature_intersection():
    import json
    tree = {'nodeid': 0, 'split': 'f0', 'split_condition': .5, 'yes': 1, 'no': 2,
            'children': [{'nodeid': 1, 'leaf': 0},
                         {'nodeid': 2, 'split': 'f0', 'split_condition': 1.5, 'yes': 3, 'no': 4,
                          'children': [{'nodeid': 3, 'leaf': 1}, {'nodeid': 4, 'leaf': 3}]}]}
    class Booster:
        feature_names = None
        def save_config(self):
            return json.dumps({'learner': {'gradient_booster': {'name': 'gbtree'},
                                           'learner_model_param': {'num_class': '0'}}})
        def get_dump(self, **_):
            return [json.dumps(tree)]
    class Model:
        def get_booster(self):
            return Booster()
    leaves = leaves_from_xgboost(Model())
    assert leaves[1] == Leaf(1., ((0, .5, 1.5),))
    np.testing.assert_array_equal(score(leaves, np.array([[.49], [.5], [1.49], [1.5]])), [0, 1, 1, 3])


def test_same_law_hidden_placeholder_invariance():
    compiler = CompiledAttributions(WORLDS[1], BACKGROUND, input_dtype=np.float32)
    law = FiniteJointLaw(BACKGROUND, [.25]*4)
    observed = np.array([True, False, False])
    a = compiler.moments([-.25, np.nan, np.nan], observed, law)
    b = compiler.moments([-.25, 1e200, -1e200], observed, law)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.covariance, b.covariance)


def test_equal_moment_different_sign_actual_attribution_laws():
    leaves = [Leaf(-2., ((0, -np.inf, -1.5),)), Leaf(-1., ((0, -1.5, -.5),)),
              Leaf(.5, ((0, -.5, .75),)), Leaf(1., ((0, .75, np.inf),))]
    compiler = CompiledAttributions(leaves, [[-1.], [1.]])
    laws = [FiniteJointLaw([[-1.], [1.]], [.5, .5]),
            FiniteJointLaw([[-2.], [.5]], [.2, .8])]
    results = [compiler.moments([np.nan], np.array([False]), q) for q in laws]
    np.testing.assert_allclose(results[0].mean, results[1].mean, atol=1e-12)
    np.testing.assert_allclose(results[0].covariance, results[1].covariance, atol=1e-12)
    sign_support = [float(q.weights@(compiler.values(q.points)[:, 0] > 0)) for q in laws]
    np.testing.assert_allclose(sign_support, [.5, .8])


@pytest.mark.parametrize('leaves', WORLDS)
def test_known_fourier_basis_control_against_independent_coalition_oracle(leaves):
    from financepaper.explanations.moment_baselines import uniform_four_atom_moments
    compiler = CompiledAttributions(leaves, BACKGROUND)
    atoms = np.array([-1.5, -.5, .5, 1.5])
    support = np.array(list(product(atoms, repeat=3)))
    for pattern in product([False, True], repeat=3):
        observed = np.array(pattern)
        x = np.array([.3, -.2, .1])
        complete = support.copy()
        complete[:, observed] = x[observed]
        truth = np.array([coalition_oracle(leaves, BACKGROUND, p) for p in complete])
        x[~observed] = np.nan
        result = uniform_four_atom_moments(compiler, x, observed, atoms)
        np.testing.assert_allclose(result['mean'], truth.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(result['covariance'], np.cov(truth.T, bias=True), atol=1e-12)
