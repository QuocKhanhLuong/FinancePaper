"""Current-only features and offline checks for the preregistered round-4 audit.

This is a stronger baseline comparison, not a new revision method. Labels and
verification arrays are deliberately absent from the feature-builder interface.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from financepaper.data.schema import FEATURE_NAMES
from financepaper.reliability.prediction_distribution import prediction_distribution_features
from financepaper.reliability.validation import TAIWAN_GROUPS, aggregate_groups, revision_arrays


CURRENT_KEYS = (
    "record_ids", "hidden", "natural", "artificial", "phi", "probability", "valid",
    "completion_phi", "completion_probability", "completion_valid", "completions",
    "donor_ids", "partial_values",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(hashes: dict[str, str], root: Path) -> None:
    for name, expected in hashes.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Artifact/source drift: {name}")


def split_roles(ids, evaluation_ids, *, seed=20261005, fit_count=400):
    ids, evaluation_ids = np.asarray(ids), np.asarray(evaluation_ids)
    if len(np.unique(ids)) != len(ids) or len(np.unique(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("Duplicate customer ID")
    if np.intersect1d(ids, evaluation_ids).size or not 0 < fit_count < len(ids):
        raise ValueError("Customer roles overlap or invalid fit count")
    shuffled = np.random.default_rng(seed).permutation(np.sort(ids))
    return {"fit": shuffled[:fit_count], "calibration": shuffled[fit_count:],
            "evaluation": np.sort(evaluation_ids)}


def validate_current(current, expected_ids, training_ids) -> None:
    """Check cache contract without opening any restored data or labels."""
    missing = set(CURRENT_KEYS) - current.keys()
    if missing:
        raise ValueError(f"Missing current keys: {sorted(missing)}")
    ids = current["record_ids"]
    n, d = len(ids), len(FEATURE_NAMES)
    if len(np.unique(ids)) != n or not np.array_equal(np.sort(ids), np.sort(expected_ids)):
        raise ValueError("Unexpected/duplicate customer IDs")
    if np.intersect1d(ids, training_ids).size:
        raise ValueError("Predictor/customer leakage")
    shapes = {"hidden": (n,d), "natural": (n,d), "artificial": (n,d), "phi": (n,d),
              "partial_values": (n,d), "probability": (n,), "valid": (n,),
              "completion_phi": (n,16,d), "completion_probability": (n,16),
              "completion_valid": (n,16), "completions": (n,16,d), "donor_ids": (n,16)}
    for key, shape in shapes.items():
        if current[key].shape != shape:
            raise ValueError(f"Invalid shape for {key}: {current[key].shape}")
    for key in ("hidden", "natural", "artificial", "valid", "completion_valid"):
        if current[key].dtype != bool:
            raise ValueError(f"{key} must have boolean dtype")
    h, nat, art = (current[key] for key in ("hidden", "natural", "artificial"))
    if nat.any() or (nat & art).any() or not np.array_equal(h, nat | art):
        raise ValueError("Taiwan artificial/natural mask contract violated")
    x, completed = current["partial_values"], current["completions"]
    if not np.isnan(x[h]).all() or not np.isfinite(x[~h]).all():
        raise ValueError("Hidden truth not erased or observed input invalid")
    if not np.isfinite(completed).all() or not np.all(np.where(h[:,None,:], True, completed == x[:,None,:])):
        raise ValueError("Completions changed observed values or are nonfinite")
    if not np.isin(current["donor_ids"], training_ids).all():
        raise ValueError("Non-training donor")
    if not np.isfinite(current["phi"]).all() or not np.isfinite(current["completion_phi"]).all():
        raise ValueError("Nonfinite attribution")
    prediction_distribution_features(current["probability"], current["completion_probability"])


def current_features(probability, completion_probability, hidden, phi, completion_phi,
                     valid, completion_valid):
    """Build two matrices with no restored/outcome argument or global data read.

    P preserves all eight sorted prediction draws. P+E adds four frozen grouped
    explanation summaries. Candidate eligibility is a common evaluation rule,
    not a predictor feature. Historical all-16 validity remains unchanged.
    """
    draws = np.asarray(completion_probability)[:, :8]
    if draws.shape[1] != 8 or np.asarray(completion_phi).shape[1] < 8:
        raise ValueError("Eight completions required")
    controls = prediction_distribution_features(probability, draws)
    h = np.asarray(hidden, bool)
    columns = [value for name, value in controls.items() if name != "hard_class"]
    p_features = np.column_stack([*columns, np.sort(draws, axis=1), h.astype(float), h.mean(1)])
    before, group_mask = aggregate_groups(phi, h, FEATURE_NAMES, TAIWAN_GROUPS)
    after, _ = aggregate_groups(np.asarray(completion_phi)[:, :8], h, FEATURE_NAMES, TAIWAN_GROUPS)
    revision = revision_arrays(before, after, group_mask, k=2)
    mc, rank, sign = (revision[name].mean(1) for name in ("event", "rank", "sign"))
    variance = (after.var(1) * ~group_mask).sum(1) / np.maximum((~group_mask).sum(1), 1)
    e_features = np.column_stack([p_features, mc, rank, sign, variance])
    scores = {"entropy": controls["current_entropy"], "prediction_variance": controls["completion_variance"],
              "hard_uncertainty": 1-controls["hard_confidence"],
              "action_disagreement": 1-controls["current_action_agreement"],
              "missing_fraction": h.mean(1), "mc8": mc, "rank_instability": rank}
    eligible = revision["eligible"] & np.asarray(valid, bool) & np.asarray(completion_valid, bool).all(1)
    if not np.isfinite(e_features).all():
        raise ValueError("Nonfinite current evidence")
    return p_features, e_features, scores, eligible


def verification_labels(current_phi, hidden, restored_phi, restored_valid):
    """Offline supervision only. Its output must never enter current_features."""
    before, mask = aggregate_groups(current_phi, hidden, FEATURE_NAMES, TAIWAN_GROUPS)
    after, _ = aggregate_groups(restored_phi, hidden, FEATURE_NAMES, TAIWAN_GROUPS)
    result = revision_arrays(before, after, mask, k=2)
    return result["event"].astype(int), np.asarray(restored_valid, bool)
