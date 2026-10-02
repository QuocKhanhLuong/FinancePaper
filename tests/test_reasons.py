import numpy as np
import pytest

from financepaper.explanations.reasons import extract_reasons
from financepaper.evaluation.revision import revision_event, reason_diagnostics

NAMES = ("a", "b", "c", "d", "hidden")
MASK = np.array([False, False, False, False, True])


def test_positive_observed_exactly_k_or_withhold():
    assert extract_reasons([4, 3, 2, -1, 100], MASK, NAMES) == ("a", "b", "c")
    assert extract_reasons([4, 3, 0, -1, 100], MASK, NAMES) == ()
    assert extract_reasons([1, 1, 1, 1, 100], MASK, NAMES) == ("a", "b", "c")
    assert extract_reasons([1, 0.01, -1, -1, 100], MASK, NAMES, k=2, min_attribution=0.01) == ()
    assert extract_reasons([4, 3, 2, 1, 100], np.ones(5, dtype=bool), NAMES) == ()


def test_restored_hidden_values_never_compete():
    assert revision_event(("a", "b", "c"), [4, 3, 2, 1, 10000], MASK, NAMES) is False
    assert revision_event((), [4, 3, 2, 1, 10000], MASK, NAMES) is None


@pytest.mark.parametrize("value", [-0.1, 0, 1e-7])
def test_sign_failure(value):
    assert revision_event(("a", "b", "c"), [4, 3, value, 0.1, 100], MASK, NAMES)


def test_rank_eviction_and_ties():
    reasons = ("a", "b", "c")
    assert revision_event(reasons, [4, 3, 1, 2, 100], MASK, NAMES)
    assert not revision_event(reasons, [4, 3, 1, 2, 100], MASK, NAMES, rank_tolerance=1)
    assert not revision_event(reasons, [4, 3, 1, 1 + 5e-7, 100], MASK, NAMES)
    assert not revision_event(reasons, [2, 4, 3, 1, 100], MASK, NAMES)


def test_reason_input_validation():
    with pytest.raises(ValueError):
        extract_reasons([1, 2], [False], ["a", "b"])
    with pytest.raises(ValueError):
        extract_reasons([np.nan] * 5, MASK, NAMES)
    with pytest.raises(ValueError):
        revision_event(("a", "hidden", "c"), [4, 3, 2, 1, 100], MASK, NAMES)
    with pytest.raises(ValueError):
        revision_event(("a", "b"), [4, 3, 2, 1, 100], MASK, NAMES)


def test_diagnostics_empty_and_unchanged():
    phi = [4, 3, 2, 1, 100]
    d = reason_diagnostics(phi, phi, MASK, NAMES)
    assert d == {"observed_attribution_mae": 0, "observed_sign_agreement": 1,
                 "top_k_overlap": 1, "rank_correlation": 1}
    assert all(v is None for v in reason_diagnostics(phi, phi, np.ones(5, dtype=bool), NAMES).values())
