"""Finite-law reduction controls, not a proposed rank-inference algorithm.

Linear projection followed by finite weighted counting is standard. This module
exists to falsify moment-only identification and check the frozen revision event.
No learned completion law, financial outcome, or formal numeric certificate.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from financepaper.explanations.conditional_moments import (
    CompiledAttributions, FiniteJointLaw, Leaf,
)


def toy_margin(points, *, additive=False):
    """Declared piecewise margin, independently evaluated without tree leaves."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("Finite N x 4 complete points required")
    h = np.zeros(len(points)) if additive else np.array([-1., -.5, 0., 1., 2.])[
        np.searchsorted([-.75, -.25, .5, 1.5], points[:, 3], side="right")]
    b = points[:, :3] >= .5
    return b[:, 0]*(2+h) + b[:, 1]*(2-h) + b[:, 2]*2.125


def toy_compiler(*, additive=False):
    leaves = []
    if additive:
        leaves = [Leaf(v, ((j, .5, np.inf),)) for j, v in enumerate([2., 2., 2.125])]
    else:
        cuts = [-np.inf, -.75, -.25, .5, 1.5, np.inf]
        for h, lo, hi in zip([-1., -.5, 0., 1., 2.], cuts[:-1], cuts[1:]):
            leaves.extend([
                Leaf(2+h, ((0, .5, np.inf), (3, lo, hi))),
                Leaf(2-h, ((1, .5, np.inf), (3, lo, hi))),
            ])
        leaves.append(Leaf(2.125, ((2, .5, np.inf),)))
    return CompiledAttributions(leaves, np.zeros((1, 4)))


def coalition_shap(predict, points, background):
    """Small independent exhaustive point-reference game, exponential by design."""
    points, background = np.asarray(points, float), np.asarray(background, float)
    if (points.ndim != 2 or points.shape[1] > 8 or background.shape != (points.shape[1],)
            or not np.isfinite(points).all() or not np.isfinite(background).all()):
        raise ValueError("Finite points and a point background, at most 8 players")
    d = points.shape[1]
    out = np.zeros_like(points)
    for j in range(d):
        other = [i for i in range(d) if i != j]
        for size in range(d):
            weight = math.factorial(size)*math.factorial(d-size-1)/math.factorial(d)
            for subset in combinations(other, size):
                absent = np.broadcast_to(background, points.shape).copy()
                absent[:, list(subset)] = points[:, list(subset)]
                present = absent.copy()
                present[:, j] = points[:, j]
                out[:, j] += weight*(predict(present)-predict(absent))
    return out


class FiniteContrastAudit:
    """Project compiled terms into candidate signs and all observed contrasts.

    This baseline still visits every supplied atom. It makes no speedup claim.
    Grouping is supplied in the frozen observed-member convention. Inputs contain
    only current information and a supplied completion law; no restored outcome.
    """

    def __init__(self, compiler, grouping, current, available, *, k=2,
                 magnitude=.01, rank_epsilon=1e-6):
        grouping, current = np.asarray(grouping, float), np.asarray(current, float)
        available = np.asarray(available)
        if (current.ndim != 1 or grouping.shape != (compiler.dimension, len(current))
                or available.shape != current.shape or available.dtype != bool
                or not np.isfinite(grouping).all() or not np.isfinite(current).all()
                or not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(current)
                or not np.isfinite([magnitude, rank_epsilon]).all()
                or magnitude < 0 or rank_epsilon < 0):
            raise ValueError("Invalid current-state grouping, mask, k or tolerance")
        eligible = (current > magnitude) & available
        if eligible.sum() < k:
            raise ValueError("Ineligible explanation; cannot assign zero revision risk")
        self.candidates = np.argsort(-np.where(eligible, current, -np.inf), kind="stable")[:k]
        grouped = compiler.coefficients @ grouping
        self.sign_coefficients = grouped[:, self.candidates]
        self.competitors = [np.flatnonzero(available & (np.arange(len(current)) != g))
                            for g in self.candidates]
        self.contrasts = [grouped[:, ids] - grouped[:, [g]]
                          for g, ids in zip(self.candidates, self.competitors)]
        self.compiler, self.k, self.rank_epsilon = compiler, k, rank_epsilon

    def events(self, partial, observed, law):
        """Count the frozen event on finite support; hidden placeholders are ignored."""
        partial, observed = np.asarray(partial, float), np.asarray(observed)
        if (partial.shape != (self.compiler.dimension,) or observed.shape != partial.shape
                or observed.dtype != bool or not np.isfinite(partial[observed]).all()
                or not isinstance(law, FiniteJointLaw) or law.dimension != len(partial)):
            raise ValueError("Boolean observation mask and finite completion law required")
        # Read only observed coordinates. This is clamping an already conditional
        # law, not estimating q(hidden | observed) from an unconditional pool.
        points = law.points.copy()
        points[:, observed] = partial[observed]
        points = points.astype(self.compiler.input_dtype)
        if not np.isfinite(points).all():
            raise ValueError("Inputs overflow model dtype")
        inside = np.ones((len(points), len(self.compiler.rectangles)), dtype=bool)
        for r, rectangle in enumerate(self.compiler.rectangles):
            for j, lo, hi in rectangle:
                inside[:, r] &= (points[:, j] >= lo) & (points[:, j] < hi)
        signs = inside @ self.sign_coefficients
        event = (signs <= 1e-6).any(axis=1)  # historical sign threshold is fixed
        for contrast in self.contrasts:
            event |= ((inside @ contrast) > self.rank_epsilon).sum(axis=1) >= self.k
        return event

    def probability(self, partial, observed, law):
        return float(law.weights @ self.events(partial, observed, law))
