"""Reasons are observed original fields with positive, meaningful contributions."""

import numpy as np


def validate_attributions(attributions, hidden_mask, feature_names):
    phi = np.asarray(attributions, dtype=float)
    mask = np.asarray(hidden_mask)
    if (phi.ndim != 1 or mask.shape != phi.shape or len(feature_names) != len(phi)
            or mask.dtype != np.dtype(bool) or len(set(feature_names)) != len(phi)):
        raise ValueError("Expected aligned one-dimensional attributions, boolean mask and unique names")
    if not np.isfinite(phi).all():
        raise ValueError("Attributions must be finite")
    return phi, mask


def extract_reasons(attributions, hidden_mask, feature_names, k: int = 3,
                    min_attribution: float = 0.01) -> tuple[str, ...]:
    """Return exactly k reasons or (); ties break by the supplied schema order."""
    phi, mask = validate_attributions(attributions, hidden_mask, feature_names)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(min_attribution) or min_attribution < 0:
        raise ValueError("Minimum attribution must be finite and nonnegative")
    eligible = np.flatnonzero(~mask & (phi > min_attribution))
    if len(eligible) < k:
        return ()
    ranked = eligible[np.argsort(-phi[eligible], kind="stable")][:k]
    return tuple(feature_names[i] for i in ranked)
