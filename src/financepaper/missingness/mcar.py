"""Independent Bernoulli cell masking and evaluation-only restoration."""

import numpy as np
import pandas as pd


def _check_mask(X: pd.DataFrame, mask: pd.DataFrame) -> None:
    if not X.index.equals(mask.index) or not X.columns.equals(mask.columns):
        raise ValueError("Mask axes must exactly match the original records")
    if not all(dtype == bool for dtype in mask.dtypes):
        raise ValueError("Mask must contain boolean values; True means hidden")
    if mask.isna().any().any():
        raise ValueError("Mask cannot contain missing values")


def mcar_mask(X: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("MCAR rate must be between zero and one")
    # Read shape/axes only: values and labels cannot influence missingness.
    return pd.DataFrame(
        np.random.default_rng(seed).random(X.shape) < rate,
        index=X.index, columns=X.columns,
    )


def apply_mask(X: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    _check_mask(X, mask)
    if X.isna().any().any():
        raise ValueError("The restoration benchmark requires complete source records")
    return X.astype(float).mask(mask)


def restore_hidden(observed: pd.DataFrame, truth: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Evaluation only. Never pass truth to an imputer or predictor beforehand."""
    _check_mask(observed, mask)
    _check_mask(truth, mask)
    if truth.isna().any().any() or not observed.isna().equals(mask):
        raise ValueError("Truth must be complete and observed NaNs must match the mask")
    if not np.array_equal(
        observed.to_numpy()[~mask.to_numpy()], truth.to_numpy()[~mask.to_numpy()]
    ):
        raise ValueError("Observed values differ from the reference record")
    return observed.where(~mask, truth)
