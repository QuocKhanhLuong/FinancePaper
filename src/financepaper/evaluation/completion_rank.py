"""Exact rank-tail frontier for one missing SHAP player; restricted research model.

Signed pair gaps only. The reference and completion hidden law must agree.
All arithmetic in the finite frontier and its witness is rational.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb, factorial, sqrt


@lru_cache(maxsize=128)
def rank_cells(m, p):
    """Return (alpha, beta) for cardinality K; eta=beta/alpha.

    Positive beta-integral expansion avoids cancellation. O(m^2) rational
    summands; arbitrary-precision bit cost is not treated as constant.
    """
    p = Fraction(p)
    if not isinstance(m, int) or isinstance(m, bool) or m < 2 or not 0 < p < 1:
        raise ValueError("m>=2 and 0<p<1 required")
    r = m-2
    cells = []
    for k in range(r+1):
        alpha, beta = Fraction(0), Fraction(0)
        for j in range(k+1):
            coefficient = comb(r, k)*comb(k, j)*p**(k-j)*(1-p)**(r-k+j)
            alpha += coefficient*Fraction(factorial(j)*factorial(r-k), factorial(j+r-k+1))
            beta += coefficient*Fraction(factorial(j+1)*factorial(r-k), factorial(j+r-k+2))
        cells.append((alpha, beta))
    assert sum(a for a, _ in cells) == 1
    assert sum(b for _, b in cells) == Fraction(1, 2)
    return tuple(cells)


def rank_frontier(m, p, failure_mass):
    """Sharp maximum E(Delta)/(1-p) at a prescribed nonpositive-gap mass.

    z=0 returns the continuous endpoint 1; z=1 the all-tie endpoint 0.
    Returns the optimal losses for an attaining two-hidden-state model.
    """
    z = Fraction(failure_mass)
    if not 0 <= z <= 1:
        raise ValueError("Failure mass must be in [0,1]")
    cells = rank_cells(m, Fraction(p))
    loss = [Fraction(0)]*len(cells)
    need = Fraction(1, 2)
    for k in sorted(range(len(cells)), key=lambda k: cells[k][1]/cells[k][0], reverse=True):
        alpha, beta = cells[k]
        efficiency = z+(1-z)*beta/alpha
        take = min(Fraction(1), need/(alpha*efficiency))
        loss[k] = take
        need -= take*alpha*efficiency
        if need == 0:
            break
    assert need == 0
    cost = sum(a*l for (a, _), l in zip(cells, loss, strict=True))
    return dict(normalized_mean=1-2*z*cost, cost=cost, loss=tuple(loss))


def rank_risk_upper(m, p, normalized_mean, iterations=64):
    """Conservative rational upper endpoint of the inverse frontier.

    No estimated-mean uncertainty is handled here. Callers must validate the
    model/distribution assumptions and use an exact mean or valid lower bound.
    """
    mean = Fraction(normalized_mean)
    rank_cells(m, Fraction(p))
    if not 0 <= mean <= 1:
        raise ValueError("Normalized nonnegative mean must be in [0,1]")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("Positive integer iterations required")
    if mean == 0:
        return Fraction(1)
    if mean == 1:
        return Fraction(0)
    low, high = Fraction(0), Fraction(1)
    for _ in range(iterations):
        mid = (low+high)/2
        if rank_frontier(m, p, mid)["normalized_mean"] >= mean:
            low = mid
        else:
            high = mid
    return high


def universal_risk_upper(normalized_mean):
    """Floating evaluation of the analytic envelope; not interval arithmetic."""
    mean = float(normalized_mean)
    if not 0 <= mean <= 1:
        raise ValueError("Normalized nonnegative mean must be in [0,1]")
    a = 1-mean
    return ((a+sqrt(a*a+8*a))/4)**2


def rank_pair_operator(m, p, probabilities, hidden_value):
    """Pair (0,1) linear operator, derived from the positive-mixture formula.

    Table columns are a + 2**m*h, where low m bits encode A. This is compared
    to the independent coalition definition in the executable audit.
    """
    p = Fraction(p)
    cells = rank_cells(m, p)
    q = tuple(Fraction(v) for v in probabilities)
    if not q or min(q) <= 0 or sum(q) != 1 or hidden_value not in range(len(q)):
        raise ValueError("Positive normalized hidden probabilities required")
    row = [Fraction(0)]*(2**m*len(q))
    for h, mass in enumerate(q):
        for state in range(2**(m-2)):
            k = state.bit_count()
            alpha, beta = cells[k]
            coefficient = (1-p)*(mass*(alpha-beta)+(beta if h == hidden_value else 0))/comb(m-2, k)
            row[2**m*h+4*state+1] = coefficient
            row[2**m*h+4*state+2] = -coefficient
    return tuple(row)


def rank_witness(m, p, failure_mass, constant=Fraction(1, 2), perturbation=0):
    """Bounded exact table; bad-state ties, or strict reversals if perturbed.

    perturbation moves bad contrasts toward -1, leaving all predictions at
    query A=1 fixed. The table representation is exponential in m.
    """
    z, c, epsilon = map(Fraction, (failure_mass, constant, perturbation))
    if not 0 < z < 1 or not 0 <= c <= 1 or not 0 <= epsilon <= 1:
        raise ValueError("Need 0<z<1, 0<=c<=1 and 0<=perturbation<=1")
    losses = rank_frontier(m, p, z)["loss"]
    table = []
    for h in (0, 1):
        for a in range(2**m):
            if a % 4 in (0, 3):
                value = c
            else:
                k = (a//4).bit_count()
                d = (1-2*losses[k])*(1-epsilon)-epsilon if h == 0 else Fraction(1)
                value = (1+d)/2 if a % 4 == 1 else (1-d)/2
            table.append(value)
    assert min(table) >= 0 and max(table) <= 1
    return tuple(table)
