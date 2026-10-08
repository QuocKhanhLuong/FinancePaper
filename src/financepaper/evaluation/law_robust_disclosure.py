"""Sharp two-observed-player SHAP pair certificate under a TV law mismatch."""
from fractions import Fraction as F


def unit(value, label):
    value = F(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be in [0,1]")
    return value


def law_frontier(failure_mass, tv_budget):
    """Maximum normalized mean at given nonpositive pair-gap probability."""
    z, epsilon = unit(failure_mass, "Failure mass"), unit(tv_budget, "TV budget")
    return (1-z)/(1+max(z-epsilon, F(0)))


def law_risk_upper(normalized_mean, tv_budget):
    """Exact rational upper bound; mean and TV budget require external validity."""
    t, epsilon = unit(normalized_mean, "Normalized mean"), unit(tv_budget, "TV budget")
    return min(1-t, (1-t+t*epsilon)/(1+t))


def transfer_risk_upper(normalized_mean, tv_budget):
    """Generic matched-law certificate plus TV event/mean transfer baseline."""
    t, epsilon = unit(normalized_mean, "Normalized mean"), unit(tv_budget, "TV budget")
    lower_mean = max(t-epsilon, F(0))
    return min(1-t, epsilon+(1-lower_mean)/(1+lower_mean))


def law_witness(failure_mass, tv_budget, constant=F(1, 2)):
    """Return finite R,Q and exact bounded table; columns a0+2*a1+4*h."""
    z, epsilon, c = unit(failure_mass, "Failure mass"), unit(tv_budget, "TV budget"), unit(constant, "Prediction")
    r = max(z-epsilon, F(0))
    b = (1-r)/(1+r)
    contrasts = (-b, F(1))
    table = tuple(v for d in contrasts for v in (c, (1+d)/2, (1-d)/2, c))
    return dict(reference=(r, 1-r), completion=(z, 1-z), table=table,
                normalized_gaps=tuple((d+b)/2 for d in contrasts))
