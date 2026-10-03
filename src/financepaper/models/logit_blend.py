"""Fixed convex combinations of fitted raw-logit predictors (not a new method)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import expit


@dataclass
class LogitBlend:
    additive: Any
    interaction: Any
    weight: float

    def __post_init__(self):
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("blend weight must be finite and in [0,1]")

    @staticmethod
    def _logits(model, matrix):
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(matrix), dtype=float)
        return np.asarray(model.predict(matrix, output_margin=True), dtype=float)

    def decision_function(self, matrix):
        return ((1 - self.weight) * self._logits(self.additive, matrix)
                + self.weight * self._logits(self.interaction, matrix))

    def predict_proba(self, matrix):
        p = expit(self.decision_function(matrix))
        return np.column_stack([1 - p, p])

    def combine_attributions(self, additive_phi, interaction_phi):
        a, b = np.asarray(additive_phi), np.asarray(interaction_phi)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("attributions must have equal shapes and finite values")
        return (1 - self.weight) * a + self.weight * b
