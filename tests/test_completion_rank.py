from fractions import Fraction as F

import pytest

from financepaper.evaluation.bounded_completion import finite_definition_matrix
from financepaper.evaluation.completion_rank import (
    rank_cells, rank_frontier, rank_pair_operator, rank_risk_upper,
    rank_witness, universal_risk_upper,
)


@pytest.mark.parametrize("m,p", [(2, F(1, 2)), (3, F(1, 10)), (4, F(9, 10))])
def test_pair_operator_against_independent_coalition_definition(m, p):
    q = (F(1, 5), F(3, 10), F(1, 2))
    for h in range(3):
        matrix = finite_definition_matrix(m, p, q, h)
        direct = tuple(a-b for a, b in zip(matrix[0], matrix[1], strict=True))
        assert rank_pair_operator(m, p, q, h) == direct


@pytest.mark.parametrize("m", [2, 3, 5])
def test_attainment_and_strict_reversal(m):
    p, z = F(1, 2), F(1, 4)
    operators = [rank_pair_operator(m, p, [z, 1-z], h) for h in (0, 1)]
    table = rank_witness(m, p, z)
    gaps = [sum(a*b for a, b in zip(op, table, strict=True)) for op in operators]
    assert gaps[0] == 0 < gaps[1]
    assert (z*gaps[0]+(1-z)*gaps[1])/(1-p) == rank_frontier(m, p, z)["normalized_mean"]
    perturbed = rank_witness(m, p, z, perturbation=F(1, 1000))
    assert sum(a*b for a, b in zip(operators[0], perturbed, strict=True)) < 0
    assert all(table[2**m*h+2**m-1] == F(1, 2) for h in (0, 1))


def test_closed_form_inverse_and_endpoints():
    for z in (F(1, 100), F(1, 4), F(9, 10)):
        mean = (1-z)/(1+z)
        assert rank_frontier(2, F(1, 2), z)["normalized_mean"] == mean
        upper = rank_risk_upper(2, F(1, 2), mean)
        assert z <= upper <= z+F(1, 2**64)
        assert float(upper) <= universal_risk_upper(mean) < float(1-mean)
    assert rank_risk_upper(2, F(1, 2), 0) == 1
    assert rank_risk_upper(2, F(1, 2), 1) == 0


def test_cardinality_probabilities_and_monotone_posterior():
    cells = rank_cells(12, F(1, 10))
    etas = [b/a for a, b in cells]
    assert sum(a for a, _ in cells) == 1
    assert sum(b for _, b in cells) == F(1, 2)
    assert all(a < b for a, b in zip(etas, etas[1:]))


@pytest.mark.parametrize("call", [
    lambda: rank_cells(1, F(1, 2)), lambda: rank_cells(3, 0),
    lambda: rank_frontier(3, F(1, 2), 2),
    lambda: rank_risk_upper(3, F(1, 2), -1),
    lambda: rank_pair_operator(3, F(1, 2), [F(1, 3)], 0),
    lambda: rank_witness(3, F(1, 2), 0),
])
def test_invalid_contracts(call):
    with pytest.raises(ValueError):
        call()
