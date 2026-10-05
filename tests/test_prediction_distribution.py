import inspect
from itertools import combinations
from math import factorial

import numpy as np
import pytest
from scipy.special import expit

from financepaper.reliability.prediction_distribution import prediction_distribution_features


def exhaustive_shap(function, x, background):
    """Independent point-reference coalition oracle for the fixed sanity case."""
    d = len(x)
    phi = np.zeros(d)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for size in range(d):
            weight = factorial(size)*factorial(d-size-1)/factorial(d)
            for subset in combinations(others, size):
                z = background.copy()
                z[list(subset)] = x[list(subset)]
                before = function(z)
                z[j] = x[j]
                phi[j] += weight*(function(z)-before)
    return phi


def test_known_confidences_and_ties():
    p = np.array([[0.1, 0.2, 0.3, 0.9], [0.1, 0.9, 0.5, 0.2]])
    out = prediction_distribution_features(np.array([0.8, 0.5]), p)
    np.testing.assert_allclose(out["hard_confidence"], [0.75, 0.5])
    np.testing.assert_allclose(out["soft_confidence"], [0.625, 0.575])
    np.testing.assert_allclose(out["current_action_agreement"], [0.25, 0.5])
    np.testing.assert_equal(out["hard_class"], [0, 1])
    assert out["completion_variance"][0] == pytest.approx(0.096875)


def test_single_completion_extremes_and_order_invariance():
    out = prediction_distribution_features(np.array([0, 1]), np.array([[1], [0]]))
    np.testing.assert_equal(out["completion_variance"], [0, 0])
    np.testing.assert_equal(out["current_entropy"], [0, 0])
    np.testing.assert_equal(out["hard_confidence"], [1, 1])
    np.testing.assert_equal(out["current_action_agreement"], [0, 0])
    a = np.array([[0.1, 0.2, 0.9], [0.51, 0.8, 0.99]])
    first = prediction_distribution_features(np.array([0.5, 0.7]), a)
    second = prediction_distribution_features(np.array([0.5, 0.7]), a[:, ::-1])
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=1e-15)


@pytest.mark.parametrize("current,draws", [
    ([], np.empty((0, 2))), ([0.5], np.empty((1, 0))),
    ([0.5], [[0.1], [0.9]]), ([[0.5]], [[0.1]]), ([0.5], [0.1]),
    ([np.nan], [[0.1]]), ([0.5], [[np.inf]]), ([0.5], [[-0.1]]),
    ([1.1], [[0.2]]), ([0.5], [[1.01]]),
])
def test_invalid_input_rejected(current, draws):
    with pytest.raises(ValueError):
        prediction_distribution_features(current, draws)


def test_current_information_only_signature_and_no_mutation():
    assert tuple(inspect.signature(prediction_distribution_features).parameters) == (
        "current_probability", "completion_probability"
    )
    current, draws = np.array([0.3]), np.array([[0.2, 0.4]])
    c, d = current.copy(), draws.copy()
    prediction_distribution_features(current, draws)
    np.testing.assert_equal(current, c)
    np.testing.assert_equal(draws, d)


def test_constant_prediction_can_have_changed_observed_reasons():
    f = lambda x: x[0]*x[2] + x[1]*(1-x[2])
    background = np.array([0., 0., 0.25])
    completions = np.array([[1., 1., 0.], [1., 1., 1.]])
    phi = np.stack([exhaustive_shap(f, x, background) for x in completions])
    np.testing.assert_allclose(phi, [[0.125, 0.875, 0], [0.625, 0.375, 0]], atol=1e-15)
    np.testing.assert_allclose(phi.sum(1), [1, 1])
    assert np.argmax(phi[0, :2]) != np.argmax(phi[1, :2])
    assert np.all(phi[:, :2] > 0.01)  # neither a sign tolerance nor rank tie artifact
    p = expit([f(x) for x in completions])
    out = prediction_distribution_features(p[:1], p[None, :])
    assert out["completion_variance"][0] == 0
    assert out["hard_confidence"][0] == 1
    assert out["current_action_agreement"][0] == 1


def test_additive_and_probability_shift_controls():
    f = lambda x: x[0] + x[1]
    b = np.array([0., 0., 0.25])
    for h in [0., 1.]:
        np.testing.assert_allclose(exhaustive_shap(f, np.array([1., 1., h]), b), [1, 1, 0])
    out = prediction_distribution_features(np.array([0.51]), np.array([[0.51, 0.99]]))
    assert out["hard_confidence"][0] == 1
    assert out["completion_variance"][0] > 0
