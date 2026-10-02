"""Material reason revision among the features observed before verification."""

import numpy as np
from scipy.stats import spearmanr

from financepaper.explanations.reasons import validate_attributions


def revision_event(reasons, restored_attributions, hidden_mask, feature_names,
                   k: int = 3, rank_tolerance: int = 0,
                   attribution_tolerance: float = 1e-6) -> bool | None:
    """None = withheld; otherwise any failed sign or tolerant top-k membership.

    For each released field j, fail if phi*_j <= epsilon, or if at least
    k + rank_tolerance originally observed fields exceed phi*_j + epsilon.
    A restored hidden field is never a competitor. Internal rank swaps within
    the accepted set are diagnostics, not revision events.
    """
    phi, mask = validate_attributions(restored_attributions, hidden_mask, feature_names)
    if (isinstance(k, bool) or not isinstance(k, int) or k < 1
            or isinstance(rank_tolerance, bool) or not isinstance(rank_tolerance, int)
            or rank_tolerance < 0 or not np.isfinite(attribution_tolerance)
            or attribution_tolerance < 0):
        raise ValueError("Invalid revision tolerances or k")
    if not reasons:
        return None
    if len(reasons) != k or len(set(reasons)) != k:
        raise ValueError("Released reasons must contain exactly k distinct features")
    indices = {name: i for i, name in enumerate(feature_names)}
    if any(name not in indices or mask[indices[name]] for name in reasons):
        raise ValueError("Released reasons must belong to the originally observed features")
    for name in reasons:
        value = phi[indices[name]]
        if value <= attribution_tolerance:
            return True
        if np.count_nonzero(phi[~mask] > value + attribution_tolerance) >= k + rank_tolerance:
            return True
    return False


def reason_diagnostics(before, after, hidden_mask, feature_names, k: int = 3) -> dict:
    """Observed-only diagnostics; undefined empty/constant cases return None.

    Overlap divides by min(k, n_observed) and ranks by signed contribution,
    including nonpositive fields. This diagnostic is distinct from eligibility.
    """
    a, mask = validate_attributions(before, hidden_mask, feature_names)
    b, _ = validate_attributions(after, hidden_mask, feature_names)
    a, b = a[~mask], b[~mask]
    keys = ("observed_attribution_mae", "observed_sign_agreement", "top_k_overlap", "rank_correlation")
    if len(a) == 0:
        return dict.fromkeys(keys)
    size = min(k, len(a))
    rank_a = np.argsort(-a, kind="stable")[:size]
    rank_b = np.argsort(-b, kind="stable")[:size]
    correlation = None
    if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
        correlation = float(spearmanr(a, b).statistic)
    return dict(zip(keys, [
        float(np.mean(np.abs(a - b))), float(np.mean(np.sign(a) == np.sign(b))),
        len(set(rank_a) & set(rank_b)) / size, correlation,
    ], strict=True))
