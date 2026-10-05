"""Research-only exact moments of a fixed interventional attribution map.

The path expansion and moment identities are established algebra, not a claimed
new SHAP method. See docs/method_pivot/ROUND2_METHOD_SPEC.md. Numeric finite-input
trees only: native NaN routing and categorical tree nodes are explicitly outside
this prototype. Completion laws and the explainer background are distinct.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import time

import numpy as np


class MomentBudgetExceeded(RuntimeError):
    """No approximate fallback is performed when the exact budget is exceeded."""


def _frozen(a):
    out = np.array(a, dtype=float, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Leaf:
    value: float
    # Distinct feature, inclusive lower bound, exclusive upper bound.
    bounds: tuple[tuple[int, float, float], ...]

    def __post_init__(self):
        indices = [j for j, _, _ in self.bounds]
        if len(set(indices)) != len(indices) or any(j < 0 for j in indices):
            raise ValueError("Leaf bounds must use distinct nonnegative features")
        if not np.isfinite(self.value) or any(
            np.isnan(lo) or np.isnan(hi) or lo >= hi for _, lo, hi in self.bounds
        ):
            raise ValueError("Invalid leaf value or interval")


def leaves_from_xgboost(model) -> tuple[Leaf, ...]:
    """Extract additive raw leaf scores; the constant base margin cancels in SHAP."""
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    if config['learner']['gradient_booster']['name'] != 'gbtree':
        raise ValueError("Only numeric gbtree ensembles supported")
    if config['learner']['learner_model_param']['num_class'] not in ('0', '1'):
        raise ValueError("Multiclass outputs unsupported")
    feature_names = booster.feature_names
    result = []

    def visit(node, bounds):
        if 'leaf' in node:
            result.append(Leaf(float(node['leaf']), tuple(
                (j, *bounds[j]) for j in sorted(bounds)
            )))
            return
        if isinstance(node['split_condition'], list):
            raise ValueError("Categorical tree nodes unsupported")
        name = node['split']
        j = feature_names.index(name) if feature_names else int(name[1:])
        # XGBoost compares float32 inputs against float32 thresholds.
        threshold = float(np.float32(node['split_condition']))
        lo, hi = bounds.get(j, (-np.inf, np.inf))
        children = {child['nodeid']: child for child in node['children']}
        for child_id, interval in (
            (node['yes'], (lo, min(hi, threshold))),
            (node['no'], (max(lo, threshold), hi)),
        ):
            if interval[0] < interval[1]:
                updated = dict(bounds)
                updated[j] = interval
                visit(children[child_id], updated)

    for tree in booster.get_dump(dump_format='json'):
        visit(json.loads(tree), {})
    return tuple(result)


class ProductLaw:
    """Supplied conditional hidden law; clamping is not Bayesian conditioning."""
    def __init__(self, atoms, weights):
        if len(atoms) != len(weights) or not len(atoms):
            raise ValueError("One atom/weight vector required per coordinate")
        self.atoms = tuple(_frozen(a) for a in atoms)
        self.weights = tuple(_frozen(w) for w in weights)
        self.dimension = len(atoms)
        for a, w in zip(self.atoms, self.weights):
            _validate_law(a[:, None], w)

    def masses(self, lower, upper):
        result = np.ones(len(lower))
        for j, (atoms, weights) in enumerate(zip(self.atoms, self.weights)):
            result *= ((atoms[None, :] >= lower[:, j, None]) &
                       (atoms[None, :] < upper[:, j, None])) @ weights
        return result

    def fix_observed(self, x, observed):
        return ProductLaw(
            [np.array([x[j]]) if observed[j] else a for j, a in enumerate(self.atoms)],
            [np.array([1.]) if observed[j] else w for j, w in enumerate(self.weights)],
        )

    def astype(self, dtype):
        return ProductLaw([a.astype(dtype) for a in self.atoms], self.weights)


def _validate_law(points, weights):
    if (points.ndim != 2 or weights.ndim != 1 or len(points) != len(weights)
            or len(points) == 0 or not np.isfinite(points).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("Finite normalized nonnegative probability law required")


class FiniteJointLaw:
    """Supplied conditional donor/mixture weights, preserving hidden dependence.

    The caller must already provide q(X_hidden | observed); this class does not
    infer conditional weights by matching a query against an unconditional pool.
    """
    def __init__(self, points, weights):
        self.points, self.weights = _frozen(points), _frozen(weights)
        _validate_law(self.points, self.weights)
        self.dimension = self.points.shape[1]

    def masses(self, lower, upper):
        # Chunk rectangles to bound the R x support x dimension intermediate.
        result = np.zeros(len(lower))
        for start in range(0, len(lower), 64):
            lo, hi = lower[start:start+64], upper[start:start+64]
            inside = ((self.points[None] >= lo[:, None]) &
                      (self.points[None] < hi[:, None])).all(axis=2)
            result[start:start+64] = inside @ self.weights
        return result

    def fix_observed(self, x, observed):
        points = self.points.copy()
        points[:, observed] = x[observed]
        return FiniteJointLaw(points, self.weights)

    def astype(self, dtype):
        return FiniteJointLaw(self.points.astype(dtype), self.weights)


@dataclass(frozen=True)
class MomentResult:
    mean: np.ndarray
    covariance: np.ndarray
    compiled_terms: int
    residual_terms: int
    rectangle_queries: int
    joint_matrix_bytes: int
    elapsed_seconds: float


class CompiledAttributions:
    @classmethod
    def from_xgboost(cls, model, background):
        """Use XGBoost's numeric float32 execution semantics for all query laws."""
        return cls(leaves_from_xgboost(model), background, input_dtype=np.float32)

    def __init__(self, leaves, background, *, input_dtype=np.float64):
        self.input_dtype = np.dtype(input_dtype)
        if self.input_dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError("Only float32/float64 numeric input execution supported")
        background = _frozen(np.asarray(background, dtype=self.input_dtype))
        if background.ndim != 2 or not len(background) or not np.isfinite(background).all():
            raise ValueError("Nonempty finite background matrix required")
        self.background = background
        self.dimension = background.shape[1]
        coefficients = {}
        for leaf in leaves:
            if any(j >= self.dimension for j, _, _ in leaf.bounds):
                raise ValueError("Leaf feature exceeds background dimension")
            d = len(leaf.bounds)
            if not d:
                continue
            for size in range(d+1):
                for subset in combinations(range(d), size):
                    included = set(subset)
                    rectangle = tuple(leaf.bounds[j] for j in subset)
                    b_inside = np.ones(len(background), dtype=bool)
                    for j, (feature, lo, hi) in enumerate(leaf.bounds):
                        if j not in included:
                            b_inside &= (background[:, feature] >= lo) & (background[:, feature] < hi)
                    mass = b_inside.mean()
                    if not mass:
                        continue
                    row = coefficients.setdefault(rectangle, np.zeros(self.dimension))
                    for j, (feature, _, _) in enumerate(leaf.bounds):
                        weight = (math.factorial(size-1)*math.factorial(d-size)
                                  if j in included else
                                  -math.factorial(size)*math.factorial(d-size-1))
                        row[feature] += leaf.value * mass * weight / math.factorial(d)
        # Do not apply an attribution tolerance to remove small valid terms.
        self.rectangles = tuple(k for k, c in coefficients.items() if np.any(c != 0))
        self.coefficients = _frozen([coefficients[k] for k in self.rectangles]).reshape(-1, self.dimension)
        self.lower, self.upper = self._bounds(self.rectangles)

    def _bounds(self, rectangles):
        lower = np.full((len(rectangles), self.dimension), -np.inf)
        upper = np.full_like(lower, np.inf)
        for i, rectangle in enumerate(rectangles):
            for j, lo, hi in rectangle:
                lower[i, j], upper[i, j] = lo, hi
        return lower, upper

    def values(self, points):
        points = np.asarray(points, dtype=self.input_dtype).astype(float)
        if points.ndim != 2 or points.shape[1] != self.dimension or not np.isfinite(points).all():
            raise ValueError("Finite complete points of the compiled dimension required")
        out = np.zeros((len(points), self.dimension))
        for rectangle, coefficient in zip(self.rectangles, self.coefficients):
            inside = np.ones(len(points), dtype=bool)
            for j, lo, hi in rectangle:
                inside &= (points[:, j] >= lo) & (points[:, j] < hi)
            out[inside] += coefficient
        return out

    def moments(self, x, observed, law, *, grouping=None, specialize=True,
                max_terms=2000, max_matrix_bytes=32*1024**2, timeout_seconds=60.):
        start = time.perf_counter()
        x, observed = np.asarray(x, dtype=float), np.asarray(observed)
        if (x.shape != (self.dimension,) or observed.shape != x.shape
                or observed.dtype != bool or not np.isfinite(x[observed]).all()
                or law.dimension != self.dimension):
            raise ValueError("Explicit boolean observation mask and matching law required")
        # Only observed coordinates are read. Hidden placeholders may be NaN.
        partial = np.full(self.dimension, np.nan)
        partial[observed] = x[observed].astype(self.input_dtype)
        if not np.isfinite(partial[observed]).all():
            raise ValueError("Observed values overflow model input dtype")
        x = partial
        law = law.fix_observed(x, observed).astype(self.input_dtype)
        if specialize:
            merged = {}
            for rectangle, coefficient in zip(self.rectangles, self.coefficients):
                if any(observed[j] and not lo <= x[j] < hi for j, lo, hi in rectangle):
                    continue
                key = tuple((j, lo, hi) for j, lo, hi in rectangle if not observed[j])
                merged.setdefault(key, np.zeros(self.dimension))[:] += coefficient
            rectangles = tuple(k for k, c in merged.items() if np.any(c != 0))
            coefficients = np.array([merged[k] for k in rectangles]).reshape(-1, self.dimension)
            lower, upper = self._bounds(rectangles)
        else:
            rectangles = self.rectangles
            coefficients, lower, upper = self.coefficients, self.lower, self.upper
        if grouping is not None:
            grouping = np.asarray(grouping, dtype=float)
            if grouping.ndim != 2 or grouping.shape[0] != self.dimension or not np.isfinite(grouping).all():
                raise ValueError("Grouping must map input features to output groups")
            coefficients = coefficients @ grouping
        r = len(rectangles)
        if r > max_terms or r*r*8 > max_matrix_bytes:
            raise MomentBudgetExceeded(f"residual_terms={r}, matrix_bytes={r*r*8}")
        p = law.masses(lower, upper)
        joint = np.empty((r, r))
        for i in range(r):
            if time.perf_counter() - start > timeout_seconds:
                raise MomentBudgetExceeded("Moment query exceeded wall-clock budget")
            joint[i] = law.masses(np.maximum(lower, lower[i]), np.minimum(upper, upper[i]))
        mean = coefficients.T @ p
        covariance = coefficients.T @ joint @ coefficients - np.outer(mean, mean)
        # Symmetrization removes only floating multiplication order noise.
        covariance = (covariance + covariance.T) / 2
        return MomentResult(mean, covariance, len(self.rectangles), r, r+r*r,
                            joint.nbytes, time.perf_counter()-start)
