"""Held-out probability calibration for raw binary risk logits.

The temporal pilot uses a positive-slope Platt model so that calibration cannot
reverse the ordering learned by the predictor.  The calibrator has no access
to a model or to hidden/restored values; callers must fit it only on the
dedicated probability-calibration split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _as_1d_float(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return np.array(array, dtype=float, copy=True)


class PositiveSlopePlattCalibrator:
    """Monotone logistic calibration of a predictor's raw logits.

    The fitted probability is ``sigmoid(slope * logit + intercept)`` with
    ``slope > 0`` enforced by optimizing ``log(slope)``.  This is a
    post-processing model; it must be fit on an independent calibration split
    and never on the final test records.
    """

    VERSION = "positive-slope-platt-v1"

    def __init__(self, slope: float | None = None, intercept: float | None = None,
                 *, n_samples: int | None = None, optimizer_success: bool | None = None,
                 optimizer_message: str | None = None) -> None:
        if slope is not None and (not np.isfinite(slope) or slope <= 0):
            raise ValueError("slope must be finite and strictly positive")
        if intercept is not None and not np.isfinite(intercept):
            raise ValueError("intercept must be finite")
        self.slope = float(slope) if slope is not None else None
        self.intercept = float(intercept) if intercept is not None else None
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.optimizer_success = optimizer_success
        self.optimizer_message = optimizer_message

    @property
    def fitted(self) -> bool:
        return self.slope is not None and self.intercept is not None

    def fit(self, logits: Any, y: Any) -> "PositiveSlopePlattCalibrator":
        """Fit on the supplied arrays and return ``self``.

        No global state or samples beyond the two passed arrays are read.  A
        binary calibration split is required because a one-class split cannot
        identify both a slope and an intercept.
        """

        z = _as_1d_float(logits, "logits")
        target = np.asarray(y)
        if target.ndim == 2 and 1 in target.shape:
            target = target.reshape(-1)
        if target.ndim != 1 or target.size != z.size:
            raise ValueError("y must be a one-dimensional array aligned with logits")
        if not np.isfinite(target.astype(float)).all() or not np.all(np.isin(target, [0, 1])):
            raise ValueError("y must contain only finite binary labels")
        target = target.astype(float, copy=True)
        if np.unique(target).size != 2:
            raise ValueError("Positive-slope Platt calibration requires both classes")

        prevalence = float(np.clip(target.mean(), 1e-6, 1.0 - 1e-6))
        initial = np.array([0.0, np.log(prevalence / (1.0 - prevalence))], dtype=float)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            # The bound below keeps exp(log_slope) finite while retaining a
            # strictly positive slope for all finite optimizer states.
            log_slope, intercept = float(parameters[0]), float(parameters[1])
            slope = float(np.exp(log_slope))
            margin = slope * z + intercept
            loss = float(np.mean(np.logaddexp(0.0, margin) - target * margin))
            residual = expit(margin) - target
            gradient = np.array([
                float(np.mean(residual * slope * z)),
                float(np.mean(residual)),
            ])
            return loss, gradient

        result = minimize(
            lambda params: objective(params),
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-30.0, 30.0), (None, None)],
            options={"maxiter": 1000, "ftol": 1e-13, "gtol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(
                "Positive-slope Platt optimization failed: " + str(result.message)
            )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("Positive-slope Platt optimization produced non-finite parameters")
        slope = float(np.exp(np.clip(result.x[0], -30.0, 30.0)))
        intercept = float(result.x[1])
        if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
            raise RuntimeError("Positive-slope Platt optimization produced invalid parameters")
        self.slope = slope
        self.intercept = intercept
        self.n_samples = int(z.size)
        self.optimizer_success = bool(result.success)
        self.optimizer_message = str(result.message)
        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Calibrator must be fit before transforming logits")

    def transform(self, logits: Any) -> np.ndarray:
        """Transform raw logits into calibrated probabilities."""

        self._check_fitted()
        array = np.asarray(logits, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("logits must be finite")
        return expit(self.slope * array + self.intercept)

    predict_proba = transform
    predict = transform

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-serializable calibration metadata."""

        self._check_fitted()
        return {
            "version": self.VERSION,
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "log_slope": float(np.log(self.slope)),
            "n_samples": self.n_samples,
            "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PositiveSlopePlattCalibrator":
        if not isinstance(values, dict):
            raise TypeError("Calibration state must be a mapping")
        if "slope" not in values or "intercept" not in values:
            raise ValueError("Calibration state requires slope and intercept")
        return cls(
            float(values["slope"]), float(values["intercept"]),
            n_samples=values.get("n_samples"),
            optimizer_success=values.get("optimizer_success"),
            optimizer_message=values.get("optimizer_message"),
        )


def fit_platt(logits: Any, y: Any) -> PositiveSlopePlattCalibrator:
    """Fit and return a positive-slope Platt calibrator on supplied arrays."""

    return PositiveSlopePlattCalibrator().fit(logits, y)


# Concise aliases make the API convenient without introducing a second
# implementation or a sklearn dependency.
PlattCalibrator = PositiveSlopePlattCalibrator
fit_positive_slope_platt = fit_platt

__all__ = [
    "PositiveSlopePlattCalibrator",
    "PlattCalibrator",
    "fit_platt",
    "fit_positive_slope_platt",
]
