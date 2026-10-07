"""Exact leaf-pair attribution moments for a specified product completion law.

Research prototype: no inferred completion distribution, calibration or human
validity. SHAP is interventional with a fixed product reference, on raw output.
"""
from dataclasses import dataclass
from itertools import product
from math import factorial
from math import prod

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr


@dataclass(frozen=True)
class FiniteMarginal:
    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self):
        if (not self.values or len(self.values) != len(self.probabilities)
                or not np.all(np.isfinite(self.values))
                or not np.all(np.isfinite(self.probabilities))
                or min(self.probabilities) < 0
                or not np.isclose(sum(self.probabilities), 1, rtol=0, atol=1e-12)):
            raise ValueError("Invalid finite marginal")

    def mass(self, lower, upper):
        return float(sum(p for x, p in zip(self.values, self.probabilities, strict=True) if lower < x <= upper))


@dataclass(frozen=True)
class NormalMarginal:
    mean: float
    sd: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("Normal marginal requires finite mean and positive SD")

    def mass(self, lower, upper):
        if lower >= upper:
            return 0.0
        a, b = (lower-self.mean)/self.sd, (upper-self.mean)/self.sd
        # Use the survival representation in the positive tail to avoid 1-1.
        return float(ndtr(-a)-ndtr(-b) if a > 0 else ndtr(b)-ndtr(a))


@dataclass(frozen=True)
class Leaf:
    value: float
    # Canonical (feature, lower-exclusive, upper-inclusive) intervals.
    bounds: tuple[tuple[int, float, float], ...]

    def __post_init__(self):
        ids = [x[0] for x in self.bounds]
        if not np.isfinite(self.value) or ids != sorted(set(ids)):
            raise ValueError("Finite leaf value and unique sorted feature IDs required")
        if any(j < 0 or not isinstance(j, int) or np.isnan(lo) or np.isnan(hi) or lo >= hi for j, lo, hi in self.bounds):
            raise ValueError("Invalid leaf interval")


def tree_leaves(tree, bounds=None):
    """Compile a numeric binary tree, intersecting repeated-feature constraints.

    Nodes are {'feature': int, 'threshold': float, 'left': node, 'right': node};
    terminals are finite numbers. Routing is x <= threshold to the left.
    """
    bounds = {} if bounds is None else bounds
    if not isinstance(tree, dict):
        return [Leaf(float(tree), tuple((j, *bounds[j]) for j in sorted(bounds)))]
    if set(tree) != {"feature", "threshold", "left", "right"}:
        raise ValueError("Unexpected tree fields")
    j, threshold = tree["feature"], tree["threshold"]
    if not isinstance(j, int) or j < 0 or not np.isfinite(threshold):
        raise ValueError("Invalid split")
    lower, upper = bounds.get(j, (-np.inf, np.inf))
    leaves = []
    for child, lo, hi in (("left", lower, min(upper, threshold)), ("right", max(lower, threshold), upper)):
        if lo < hi:
            leaves.extend(tree_leaves(tree[child], {**bounds, j: (lo, hi)}))
    return leaves


def evaluate(leaves, x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if not np.isfinite(x).all():
        raise ValueError("Completion samples must be finite")
    values = np.zeros(len(x))
    for leaf in leaves:
        active = np.ones(len(x), dtype=bool)
        for j, lo, hi in leaf.bounds:
            active &= (x[:, j] > lo) & (x[:, j] <= hi)
        values += leaf.value * active
    return values


def _validate(leaves, reference, completion, targets):
    if len(reference) != len(completion) or not reference:
        raise ValueError("Reference/completion dimensions differ or are empty")
    if len(set(targets)) != len(targets) or any(not isinstance(i, int) or i < 0 or i >= len(reference) for i in targets):
        raise ValueError("Invalid target indices")
    if any(j >= len(reference) for leaf in leaves for j, _, _ in leaf.bounds):
        raise ValueError("Leaf feature outside distribution")


def _rule(depth):
    x, w = leggauss(max(1, (depth+1)//2))
    return (x+1)/2, w/2


def point_shapley(leaves, reference, x, targets):
    """Exact polynomial-integration point baseline; quadrature SHAP is prior art."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    _validate(leaves, reference, reference, targets)
    if x.shape[1] != len(reference) or not np.isfinite(x).all():
        raise ValueError("Invalid complete points")
    result = np.zeros((len(x), len(targets)))
    for leaf in leaves:
        t, weights = _rule(len(leaf.bounds))
        p = {j: reference[j].mass(lo, hi) for j, lo, hi in leaf.bounds}
        indicators = {j: ((x[:, j] > lo) & (x[:, j] <= hi)).astype(float) for j, lo, hi in leaf.bounds}
        for pos, target in enumerate(targets):
            if target not in p:
                continue
            integrand = np.ones((len(x), len(t)))
            for j in p:
                if j != target:
                    integrand *= p[j] + t[None, :] * (indicators[j][:, None]-p[j])
            result[:, pos] += leaf.value * (indicators[target]-p[target]) * (integrand @ weights)
    return result


def attribution_moments(leaves, reference, completion, targets):
    """Exact (up to floating arithmetic) mean, raw second moment and covariance.

    Leaf pairs retain shared-feature and cross-tree dependence. The reference
    and completion laws each factor across FEATURES, never across leaf events.
    """
    _validate(leaves, reference, completion, targets)
    k = len(targets)
    mean, second = np.zeros(k), np.zeros((k, k))
    compiled = []
    for leaf in leaves:
        bounds = {j: (lo, hi) for j, lo, hi in leaf.bounds}
        p = {j: reference[j].mass(*interval) for j, interval in bounds.items()}
        q = {j: completion[j].mass(*interval) for j, interval in bounds.items()}
        t, weights = _rule(len(bounds))
        compiled.append((leaf.value, bounds, p, q, t, weights))
        for pos, target in enumerate(targets):
            if target in bounds:
                integrand = np.ones(len(t))
                for j in bounds:
                    if j != target:
                        integrand *= p[j]+t*(q[j]-p[j])
                mean[pos] += leaf.value*(q[target]-p[target])*(integrand @ weights)
    for va, ba, pa, qa, t, wt in compiled:
        for vb, bb, pb, qb, u, wu in compiled:
            common = ba.keys() & bb.keys()
            intersection = {j: completion[j].mass(max(ba[j][0], bb[j][0]), min(ba[j][1], bb[j][1])) for j in common}
            for i, first in enumerate(targets):
                if first not in ba:
                    continue
                for h, second_target in enumerate(targets):
                    if second_target not in bb:
                        continue
                    values = np.ones((len(t), len(u)))
                    for j in ba.keys() | bb.keys():
                        if j in ba:
                            a0, a1 = (-pa[j], 1.0) if j == first else (pa[j]*(1-t[:, None]), t[:, None])
                        if j in bb:
                            b0, b1 = (-pb[j], 1.0) if j == second_target else (pb[j]*(1-u[None, :]), u[None, :])
                        if j in common:
                            values *= a0*b0 + a1*b0*qa[j] + a0*b1*qb[j] + a1*b1*intersection[j]
                        elif j in ba:
                            values *= a0+a1*qa[j]
                        else:
                            values *= b0+b1*qb[j]
                    second[i, h] += va*vb*(wt @ values @ wu)
    # Symmetrization removes ordering roundoff; do not project/clamp eigenvalues.
    second = (second+second.T)/2
    covariance = second-np.outer(mean, mean)
    return dict(mean=mean, second=second, covariance=covariance)


def finite_states(marginals):
    if not all(isinstance(m, FiniteMarginal) for m in marginals):
        raise ValueError("Enumeration oracle needs finite marginals")
    states, weights = [], []
    for indices in product(*(range(len(m.values)) for m in marginals)):
        weight = np.prod([m.probabilities[i] for m, i in zip(marginals, indices, strict=True)])
        if weight > 0:
            states.append([m.values[i] for m, i in zip(marginals, indices, strict=True)])
            weights.append(weight)
    return np.asarray(states), np.asarray(weights)


def coalition_oracle(predict, reference, x, targets):
    """Independent definition-level oracle: enumerate coalitions AND backgrounds."""
    backgrounds, probabilities = finite_states(reference)
    d = len(reference)
    values = {}
    for code in range(2**d):
        hybrid = backgrounds.copy()
        included = [j for j in range(d) if code & (1 << j)]
        hybrid[:, included] = np.asarray(x)[included]
        values[code] = float(probabilities @ predict(hybrid))
    answer = []
    for target in targets:
        total = 0.0
        for code in range(2**d):
            if not code & (1 << target):
                size = code.bit_count()
                weight = factorial(size)*factorial(d-size-1)/factorial(d)
                total += weight*(values[code | (1 << target)]-values[code])
        answer.append(total)
    return np.asarray(answer)


def enumeration_moments(leaves, reference, completion, targets, *, definition_oracle=False):
    x, weights = finite_states(completion)
    if definition_oracle:
        shapley = np.asarray([coalition_oracle(lambda batch: evaluate(leaves, batch), reference, row, targets) for row in x])
    else:
        shapley = point_shapley(leaves, reference, x, targets)
    mean = weights @ shapley
    second = (shapley.T*weights) @ shapley
    return dict(mean=mean, second=second, covariance=second-np.outer(mean, mean), states=len(x))


def sample_product(marginals, draws, rng):
    columns = []
    for m in marginals:
        if isinstance(m, FiniteMarginal):
            columns.append(rng.choice(m.values, draws, p=m.probabilities))
        elif isinstance(m, NormalMarginal):
            columns.append(rng.normal(m.mean, m.sd, draws))
        else:
            raise ValueError("Unsupported sampling marginal")
    return np.asarray(columns).T


def covariance_witness(factor, mean, *, reference_mode="point"):
    """Construct depth-2 trees realizing B B^T while output stays constant.

    B's columns must sum to zero. Observed a=1; hidden h are fair +/-1.
    Reference can be point zero or a full-support product reference on the
    observed binary and hidden +/-1 domains. Zero leaves are implicit. No fit.
    """
    factor, mean = np.asarray(factor, dtype=float), np.asarray(mean, dtype=float)
    if (factor.ndim != 2 or mean.shape != (factor.shape[0],)
            or not np.isfinite(factor).all() or not np.isfinite(mean).all()
            or not np.allclose(factor.sum(axis=0), 0, rtol=0, atol=1e-12)):
        raise ValueError("Finite factor with zero column sums and compatible mean required")
    m, rank = factor.shape
    if reference_mode not in ("point", "full_support"):
        raise ValueError("Unknown reference mode")
    multiplier = 1 if reference_mode == "point" else 2
    leaves = []
    for i in range(m):
        leaves.append(Leaf(float(multiplier*mean[i]), ((i, .5, np.inf),)))
        for j in range(rank):
            leaves.append(Leaf(float(2*multiplier*factor[i, j]), ((i, .5, np.inf), (m+j, .5, np.inf))))
            leaves.append(Leaf(float(-2*multiplier*factor[i, j]), ((i, .5, np.inf), (m+j, -np.inf, -.5))))
    reference = [FiniteMarginal((0.,), (1.,))]*(m+rank)
    if reference_mode == "full_support":
        reference = ([FiniteMarginal((0., 1.), (.5, .5))]*m
                     +[FiniteMarginal((-1., 1.), (.5, .5))]*rank)
    completion = ([FiniteMarginal((1.,), (1.,))]*m
                  +[FiniteMarginal((-1., 1.), (.5, .5))]*rank)
    return leaves, reference, completion


def joint_law_witness(attributions, probabilities):
    """Realize any finite observed-SHAP law using one hidden ordered feature.

    Full-support product reference; same hidden law P_h=Q_h. The observed
    coordinates equal one under Q. Predictor is constant on every Q state.
    Returns explicit nonzero leaf terms of depth<=2 regression trees.
    """
    values, weights = np.asarray(attributions, float), np.asarray(probabilities, float)
    if (values.ndim != 2 or min(values.shape) < 1 or weights.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Finite attribution rows and strictly positive normalized masses required")
    states, m = values.shape
    mu = weights @ values
    centered = values-mu
    leaves = []
    for i in range(m):
        leaves.append(Leaf(float(2*mu[i]), ((i, .5, np.inf),)))
        # (a-1) is -1 on a=0, zero on a=1. Ordered step representation.
        leaves.append(Leaf(float(-4*centered[0, i]), ((i, -np.inf, .5),)))
        for s in range(1, states):
            jump = centered[s, i]-centered[s-1, i]
            leaves.append(Leaf(float(-4*jump), ((i, -np.inf, .5), (m, s-.5, np.inf))))
    hidden = FiniteMarginal(tuple(float(s) for s in range(states)), tuple(weights))
    reference = [FiniteMarginal((0., 1.), (.5, .5))]*m+[hidden]
    completion = [FiniteMarginal((1.,), (1.,))]*m+[hidden]
    return leaves, reference, completion


def threshold_states(leaves, completion, *, max_states=65536):
    """Exact cells induced by only the supplied leaf thresholds (strong oracle).

    Continuous distributions also work: one representative per nonempty cell.
    Dummy coordinates have one representative and do not multiply state count.
    """
    columns = []
    for j, marginal in enumerate(completion):
        cuts = sorted({v for leaf in leaves for h, lo, hi in leaf.bounds if h == j
                       for v in (lo, hi) if np.isfinite(v)})
        cells = []
        for lo, hi in zip([-np.inf]+cuts, cuts+[np.inf], strict=True):
            mass = marginal.mass(lo, hi)
            if mass > 0:
                # Upper endpoints belong to this cell; all leaf indicators agree.
                representative = hi if np.isfinite(hi) else (lo+max(1, abs(lo)) if np.isfinite(lo) else 0.)
                cells.append((representative, mass))
        columns.append(cells)
    count = prod(map(len, columns))
    if count > max_states:
        raise OverflowError(f"Threshold cells {count} exceed declared cap {max_states}")
    rows = list(product(*columns))
    return (np.asarray([[c[0] for c in row] for row in rows]),
            np.asarray([prod(c[1] for c in row) for row in rows]))


def cell_enumeration_moments(leaves, reference, completion, targets, *, pairwise=False, max_states=65536):
    """Enumerate global cells or, more strongly, only each leaf-pair's cells."""
    _validate(leaves, reference, completion, targets)
    if not pairwise:
        x, w = threshold_states(leaves, completion, max_states=max_states)
        phi = point_shapley(leaves, reference, x, targets)
        mean, second = w @ phi, (phi.T*w) @ phi
        return dict(mean=mean, second=second, covariance=second-np.outer(mean, mean), states=len(x))
    k, maximum = len(targets), 0
    mean, second = np.zeros(k), np.zeros((k, k))
    for a, leaf in enumerate(leaves):
        x, w = threshold_states([leaf], completion, max_states=max_states)
        mean += w @ point_shapley([leaf], reference, x, targets)
        for b in range(a, len(leaves)):
            other = leaves[b]
            x, w = threshold_states([leaf, other], completion, max_states=max_states)
            maximum = max(maximum, len(x))
            left = point_shapley([leaf], reference, x, targets)
            right = point_shapley([other], reference, x, targets)
            pair = (left.T*w) @ right
            second += pair if a == b else pair+pair.T
    return dict(mean=mean, second=second, covariance=second-np.outer(mean, mean), max_pair_states=maximum)


def coefficient_moments(leaves, reference, completion, targets):
    """Polynomial coefficient-product baseline; also polynomial, not enumeration.

    Integrates the same bivariate moment polynomial by monomial coefficients.
    Floating cancellation can be severe; no clipping/PSD projection is used.
    """
    _validate(leaves, reference, completion, targets)
    k = len(targets)
    mean, second = np.zeros(k), np.zeros((k, k))
    compiled = []
    for leaf in leaves:
        bounds = {j: (lo, hi) for j, lo, hi in leaf.bounds}
        p = {j: reference[j].mass(*b) for j, b in bounds.items()}
        q = {j: completion[j].mass(*b) for j, b in bounds.items()}
        compiled.append((leaf.value, bounds, p, q))
        for i, target in enumerate(targets):
            if target in bounds:
                coefficients = np.ones(1)
                for j in bounds:
                    if j != target:
                        coefficients = np.convolve(coefficients, [p[j], q[j]-p[j]])
                mean[i] += leaf.value*(q[target]-p[target])*np.sum(coefficients/np.arange(1, len(coefficients)+1))
    for va, ba, pa, qa in compiled:
        for vb, bb, pb, qb in compiled:
            for i, first in enumerate(targets):
                if first not in ba:
                    continue
                for h, last in enumerate(targets):
                    if last not in bb:
                        continue
                    polynomial = np.ones((1, 1))
                    for j in sorted(ba.keys() | bb.keys()):
                        a0, a1 = (([-pa[j]], [1.]) if j == first else ([pa[j], -pa[j]], [0., 1.])) if j in ba else ([1.], [0.])
                        b0, b1 = (([-pb[j]], [1.]) if j == last else ([pb[j], -pb[j]], [0., 1.])) if j in bb else ([1.], [0.])
                        overlap = completion[j].mass(max(ba[j][0], bb[j][0]), min(ba[j][1], bb[j][1])) if j in ba and j in bb else 0.
                        factor = (np.outer(a0, b0)+qa.get(j, 0)*np.outer(a1, b0)
                                  +qb.get(j, 0)*np.outer(a0, b1)+overlap*np.outer(a1, b1))
                        out = np.zeros((polynomial.shape[0]+factor.shape[0]-1, polynomial.shape[1]+factor.shape[1]-1))
                        for r in range(factor.shape[0]):
                            for s in range(factor.shape[1]):
                                out[r:r+polynomial.shape[0], s:s+polynomial.shape[1]] += factor[r, s]*polynomial
                        polynomial = out
                    integrals = 1/np.outer(np.arange(1, polynomial.shape[0]+1), np.arange(1, polynomial.shape[1]+1))
                    second[i, h] += va*vb*np.sum(polynomial*integrals)
    return dict(mean=mean, second=second, covariance=second-np.outer(mean, mean))
