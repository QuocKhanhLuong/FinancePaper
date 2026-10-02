"""Target-independent missingness mechanisms for the temporal pilot.

The Taiwan workbook is complete, so MAR here is a declared stress simulator,
not an estimate of the data-collection process.  ``AnchorMAR`` uses only two
always-observed anchors (age and credit limit) to produce a field mask.  The
target and every non-anchor field are excluded from the mask probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from financepaper.data.schema import FEATURE_NAMES


ANCHOR_FIELDS = ("AGE", "LIMIT_BAL")
_NON_ANCHOR_FIELDS = tuple(field for field in FEATURE_NAMES if field not in ANCHOR_FIELDS)


def _validate_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if list(X.columns) != list(FEATURE_NAMES):
        raise ValueError(
            "X columns must be in the canonical Taiwan schema order; "
            f"expected {list(FEATURE_NAMES)}, got {list(X.columns)}"
        )
    if X.index.has_duplicates:
        raise ValueError("X index must identify records uniquely")
    try:
        values = X.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric feature values") from exc
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("missingness simulators require complete, finite records")


def _validate_rate(rate: float) -> float:
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("missingness rate must be numeric") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("missingness rate must be between zero and one")
    return value


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # Clipping keeps this helper independent of scipy while avoiding overflow
    # warnings for unusually large synthetic anchor values.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _resolve_rates(
    n_rows: int,
    rates: float | Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve a scalar or candidate rate schedule."""

    if np.isscalar(rates):
        return np.full(n_rows, _validate_rate(float(rates)), dtype=float)
    candidates = np.asarray(list(rates), dtype=float)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("rates must be a scalar or a non-empty one-dimensional sequence")
    if not np.isfinite(candidates).all() or not ((0 <= candidates).all() and (candidates <= 1).all()):
        raise ValueError("every augmentation rate must be between zero and one")
    # Every sequence is a candidate schedule.  This keeps a four-element
    # schedule unambiguous on a four-record smoke fixture: each record still
    # samples independently from all candidates.  Per-row schedules can be
    # added as a separate explicit API later without overloading this helper.
    return rng.choice(candidates, size=n_rows, replace=True).astype(float)


def augmentation_mask(
    X: pd.DataFrame,
    rates: float | Sequence[float],
    seed: int,
) -> pd.DataFrame:
    """Draw independent target-free training masks.

    ``rates`` may be a scalar or a candidate schedule such as
    ``(0.0, 0.1, 0.2, 0.3)``.  The input values are inspected only for
    shape and axes; changing feature values or a separately held target cannot
    change the generated mask for a fixed seed.
    """

    _validate_frame(X)
    rng = np.random.default_rng(seed)
    row_rates = _resolve_rates(len(X), rates, rng)
    mask = rng.random((len(X), X.shape[1])) < row_rates[:, None]
    return pd.DataFrame(mask, index=X.index, columns=X.columns, dtype=bool)


@dataclass
class AnchorMAR:
    """A fixed-anchor MAR stress mechanism fitted on training anchors only.

    The per-cell probability is

    ``sigmoid(intercept + .75*z_age - .75*z_limit)``.

    The intercept is chosen by bisection so that the expected missing fraction
    over all 23 columns is ``rate`` on the training anchors.  AGE and LIMIT_BAL
    themselves are always observed.  This implementation does not use a target
    or any non-anchor value when fitting or masking.
    """

    rate: float = 0.30
    intercept: float | None = None
    age_mean: float | None = None
    age_scale: float | None = None
    limit_mean: float | None = None
    limit_scale: float | None = None
    train_ids: tuple[Any, ...] = ()

    def fit(
        self,
        X_train: pd.DataFrame,
        rate: float | None = None,
        train_ids: Sequence[Any] | None = None,
    ) -> "AnchorMAR":
        _validate_frame(X_train)
        requested = self.rate if rate is None else _validate_rate(rate)
        # Since two anchors are protected, non-anchor probabilities must be
        # larger than the all-field rate to achieve it.  Reject impossible
        # requests instead of silently clipping the mechanism.
        non_anchor_rate = requested * len(FEATURE_NAMES) / len(_NON_ANCHOR_FIELDS)
        if not 0.0 <= non_anchor_rate <= 1.0:
            raise ValueError(
                "requested all-field MAR rate is incompatible with two observed anchors"
            )
        if train_ids is not None and len(train_ids) != len(X_train):
            raise ValueError("train_ids must align with X_train")
        self.rate = requested
        self.train_ids = tuple(
            X_train.index.tolist() if train_ids is None else list(train_ids)
        )

        age = X_train["AGE"].to_numpy(dtype=float)
        limit = X_train["LIMIT_BAL"].to_numpy(dtype=float)
        self.age_mean = float(np.mean(age))
        self.limit_mean = float(np.mean(limit))
        age_std = float(np.std(age, ddof=0))
        limit_std = float(np.std(limit, ddof=0))
        self.age_scale = age_std if age_std > 0 else 1.0
        self.limit_scale = limit_std if limit_std > 0 else 1.0
        z_age = (age - self.age_mean) / self.age_scale
        z_limit = (limit - self.limit_mean) / self.limit_scale
        linear = 0.75 * z_age - 0.75 * z_limit

        low, high = -40.0, 40.0
        for _ in range(100):
            middle = (low + high) / 2.0
            expected = float(np.mean(_sigmoid(middle + linear)))
            if expected < non_anchor_rate:
                low = middle
            else:
                high = middle
        self.intercept = float((low + high) / 2.0)
        return self

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.intercept,
                self.age_mean,
                self.age_scale,
                self.limit_mean,
                self.limit_scale,
            )
        ):
            raise RuntimeError("AnchorMAR must be fitted before mask generation")

    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Return non-anchor per-row probabilities for diagnostics/tests."""

        self._require_fitted()
        _validate_frame(X)
        age = (X["AGE"].to_numpy(dtype=float) - self.age_mean) / self.age_scale
        limit = (X["LIMIT_BAL"].to_numpy(dtype=float) - self.limit_mean) / self.limit_scale
        return _sigmoid(self.intercept + 0.75 * age - 0.75 * limit)

    def mask(self, X: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Generate an independent MAR mask while protecting the two anchors."""

        probabilities = self.probabilities(X)
        rng = np.random.default_rng(seed)
        generated = rng.random((len(X), len(_NON_ANCHOR_FIELDS))) < probabilities[:, None]
        mask = pd.DataFrame(False, index=X.index, columns=X.columns, dtype=bool)
        mask.loc[:, list(_NON_ANCHOR_FIELDS)] = generated
        # Explicit assignments keep the guarantee visible even if schema order
        # or the anchor tuple changes in a later extension.
        mask.loc[:, list(ANCHOR_FIELDS)] = False
        return mask

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "rate": float(self.rate),
            "anchor_fields": list(ANCHOR_FIELDS),
            "non_anchor_fields": list(_NON_ANCHOR_FIELDS),
            "intercept": float(self.intercept),
            "age_mean": float(self.age_mean),
            "age_scale": float(self.age_scale),
            "limit_mean": float(self.limit_mean),
            "limit_scale": float(self.limit_scale),
            "train_ids": [
                value.item() if isinstance(value, np.generic) else value for value in self.train_ids
            ],
            "coefficient_age": 0.75,
            "coefficient_limit": -0.75,
        }


__all__ = ["ANCHOR_FIELDS", "AnchorMAR", "augmentation_mask"]
