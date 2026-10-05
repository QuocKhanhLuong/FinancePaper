"""Finite-completion prediction controls, not explanation-risk probabilities.

Independent definitions inspired by the MVU hard/soft confidence distinction.
No official MVU implementation is copied; this is not learned DMV.
"""
from __future__ import annotations

import numpy as np


def prediction_distribution_features(
    current_probability: np.ndarray, completion_probability: np.ndarray
) -> dict[str, np.ndarray]:
    """Summarize [N,K] completion predictions using current information only.

    Class ties and vote ties select class 1. Variance uses denominator K.
    The interface deliberately has no verification/outcome inputs. All returned
    confidence values concern decisions, not correctness or reason stability.
    """
    current = np.asarray(current_probability, dtype=float)
    draws = np.asarray(completion_probability, dtype=float)
    if current.ndim != 1 or draws.ndim != 2:
        raise ValueError("Expected current [N] and completion probabilities [N,K]")
    if len(current) == 0 or draws.shape[0] != len(current) or draws.shape[1] == 0:
        raise ValueError("N and K must be positive and row counts must match")
    for name, values in (("current", current), ("completion", draws)):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} probabilities must be finite and within [0,1]")
    votes = draws >= 0.5
    positive_fraction = votes.mean(axis=1)
    mean = draws.mean(axis=1)
    variance = draws.var(axis=1)
    quantiles = np.quantile(draws, [0.1, 0.5, 0.9], axis=1)
    # xlogy handles probabilities exactly 0 or 1 without clipping.
    from scipy.special import xlogy

    return {
        "current_probability": current.copy(),
        "current_entropy": -xlogy(current, current) - xlogy(1-current, 1-current),
        "completion_mean": mean,
        "completion_variance": variance,
        "completion_std": np.sqrt(variance),
        "completion_q10": quantiles[0],
        "completion_q50": quantiles[1],
        "completion_q90": quantiles[2],
        "positive_vote_fraction": positive_fraction,
        "hard_class": (positive_fraction >= 0.5).astype(int),
        "hard_confidence": np.maximum(positive_fraction, 1-positive_fraction),
        "soft_confidence": np.maximum(mean, 1-mean),
        "current_action_agreement": (votes == (current[:, None] >= 0.5)).mean(axis=1),
    }
