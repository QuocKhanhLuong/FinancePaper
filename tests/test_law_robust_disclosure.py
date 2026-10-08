from fractions import Fraction as F
import pytest

from financepaper.evaluation.law_robust_disclosure import (
    law_frontier, law_risk_upper, law_witness, transfer_risk_upper,
)


@pytest.mark.parametrize("z,epsilon", [(F(1, 4), F(0)), (F(1, 4), F(1, 10)),
    (F(1, 4), F(1, 4)), (F(1, 4), F(1)), (F(0), F(1, 10)), (F(1), F(0))])
def test_exact_law_witness_and_inverse(z, epsilon):
    witness = law_witness(z, epsilon)
    r, q, gaps = witness["reference"], witness["completion"], witness["normalized_gaps"]
    assert sum(abs(a-b) for a,b in zip(r,q))/2 <= epsilon
    assert min(witness["table"]) >= 0 and max(witness["table"]) <= 1
    assert witness["table"][3] == witness["table"][7] == F(1, 2)
    mean = sum(a*b for a,b in zip(q,gaps))
    assert mean == law_frontier(z, epsilon)
    assert law_risk_upper(mean, epsilon) == z


@pytest.mark.parametrize("mean", [F(0), F(1, 4), F(1, 2), F(1)])
def test_matched_and_unrestricted_limits(mean):
    assert law_risk_upper(mean, 0) == (1-mean)/(1+mean)
    assert law_risk_upper(mean, 1) == 1-mean


def test_concrete_improvement_over_generic_same_information_transfer():
    assert law_risk_upper(F(1, 2), F(1, 10)) == F(11, 30)
    assert transfer_risk_upper(F(1, 2), F(1, 10)) == F(1, 2)


@pytest.mark.parametrize("call", [lambda: law_risk_upper(-1, 0),
    lambda: law_risk_upper(0, 2), lambda: law_frontier(2, 0),
    lambda: law_witness(F(1, 2), 0, -1)])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
