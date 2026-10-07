from fractions import Fraction

import numpy as np
import pytest

from financepaper.evaluation.bounded_completion import (
    completion_kernel, definition_matrix, directional_width, scalar_width, sharp_tree, fixed_law_factor,
)
from financepaper.evaluation.completion_moments import (
    FiniteMarginal, attribution_moments, coalition_oracle, evaluate, tree_leaves,
)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_exact_kernel_from_independent_coalition_definition(m):
    p = Fraction(1, 3)
    kernel = completion_kernel(m, p)
    low, high = definition_matrix(m, p, 0), definition_matrix(m, p, 1)
    for i in range(m):
        assert [b-a for a, b in zip(low[i], high[i], strict=True)] == [-v for v in kernel[i]]+kernel[i]
        assert sum(abs(v) for v in kernel[i][:-1]) == scalar_width(m, p)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_sharp_bounded_tree_matches_definition_and_variance(m):
    p = Fraction(1, 2)
    leaves = tree_leaves(sharp_tree(m, constant=.3))
    assert len(leaves) == 2*m+2
    assert all(0 <= leaf.value <= 1 for leaf in leaves)
    reference = [FiniteMarginal((0., 1.), (.5, .5))]*(m+1)
    completion = [FiniteMarginal((1.,), (1.,))]*m+[reference[-1]]
    x = np.array([[1.]*m+[0.], [1.]*m+[1.]])
    phi = [coalition_oracle(lambda y: evaluate(leaves, y), reference, row, [0])[0] for row in x]
    np.testing.assert_allclose(abs(phi[1]-phi[0]), float(scalar_width(m, p)), atol=1e-14)
    moments = attribution_moments(leaves, reference, completion, [0])
    np.testing.assert_allclose(moments["covariance"][0, 0], float(scalar_width(m, p)**2/4), atol=1e-14)


def test_exact_known_widths_and_directional_bounds():
    assert scalar_width(1, Fraction(1, 2)) == Fraction(1, 4)
    assert scalar_width(2, Fraction(1, 2)) == Fraction(7, 24)
    kernel = completion_kernel(2, Fraction(1, 2))
    assert directional_width(kernel, [1, 1]) == Fraction(5, 12)
    assert directional_width(kernel, [1, -1]) == Fraction(1, 2)
    assert fixed_law_factor(["1/10", "9/10"])[0] == Fraction(9, 100)
    assert fixed_law_factor(["1/5", "3/10", "1/2"])[0] == Fraction(1, 4)


@pytest.mark.parametrize("m,p", [(0, .5), (1, 0), (1, 1)])
def test_reject_outside_full_support_scope(m, p):
    with pytest.raises(ValueError):
        scalar_width(m, p)
