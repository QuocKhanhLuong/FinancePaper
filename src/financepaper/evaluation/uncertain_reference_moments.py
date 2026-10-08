"""Reference-moment certificates with deterministic or simultaneous lower bounds."""
from fractions import Fraction as F
from math import ceil, log, sqrt

from financepaper.evaluation.completion_rank import rank_cells, rank_frontier
from financepaper.evaluation.reference_moments import reference_moment_bound


def moment_box(center, radius):
    center, radius = tuple(map(F, center)), F(radius)
    if not center or any(not -1 <= b <= 1 for b in center) or radius < 0:
        raise ValueError("Unit-interval signed moments and nonnegative radius required")
    return (tuple(max(F(-1), b-radius) for b in center),
            tuple(min(F(1), b+radius) for b in center))


def interval_risk_bound(m, p, lower, upper):
    lower, upper = tuple(map(F, lower)), tuple(map(F, upper))
    cells = rank_cells(m, F(p))
    if len(lower) != len(cells) or len(upper) != len(lower) or any(
        not -1 <= l <= u <= 1 for l, u in zip(lower, upper, strict=True)
    ):
        raise ValueError("Ordered moment box in [-1,1] required")
    assert all(a > b > 0 for a, b in cells)
    return reference_moment_bound(m, p, lower)


def hoeffding_radius(n, coordinates, delta):
    """One-sided union radius for iid vectors in [-1,1]^coordinates.

    Components may depend on each other. Radius uses floating log/sqrt,
    plus 1e-12 and upward rounding to a 1e-12 rational grid; not a formal
    interval-arithmetic implementation of transcendental functions.
    """
    delta = F(delta)
    if (not isinstance(n, int) or isinstance(n, bool) or n < 1 or
        not isinstance(coordinates, int) or isinstance(coordinates, bool) or
        coordinates < 1 or not 0 < delta < 1):
        raise ValueError("Positive sample/coordinate counts and 0<delta<1 required")
    value = sqrt(2*log(coordinates/float(delta))/n)
    return F(ceil((value+1e-12)*10**12), 10**12)


def c5_certifies(m, p, mean_lower, budget):
    """Exact forward comparison avoids inverse-frontier rounding at a budget."""
    budget, mean_lower = F(budget), F(mean_lower)
    if not 0 < budget < 1 or not -1 <= mean_lower <= 1:
        raise ValueError("Need interior risk budget and a signed unit mean")
    return mean_lower >= rank_frontier(m, F(p), budget)["normalized_mean"]
