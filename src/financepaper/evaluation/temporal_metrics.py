"""Metrics used by the temporal pilot.

These functions keep calibration-bin rows alongside aggregate metrics so that
an apparently good ECE cannot hide an empty or tiny bin.  ECE is descriptive
and bin-dependent; it is not a release guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)


def _prediction_arrays(y: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y)
    probabilities = np.asarray(probability, dtype=float)
    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if probabilities.ndim == 2 and 1 in probabilities.shape:
        probabilities = probabilities.reshape(-1)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("y and probability must be aligned one-dimensional arrays")
    if probabilities.size and (not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("probability must be finite and lie in [0, 1]")
    if labels.size and (not np.isfinite(labels.astype(float)).all() or not np.all(np.isin(labels, [0, 1]))):
        raise ValueError("y must contain only finite binary labels")
    return labels.astype(int, copy=False), probabilities


def calibration_bins(y: Any, probability: Any, bins: int = 10) -> list[dict[str, Any]]:
    """Return fixed-width calibration bins, including empty bins."""

    labels, probabilities = _prediction_arrays(y, probability)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    assignments = np.minimum((probabilities * int(bins)).astype(int), int(bins) - 1)
    rows: list[dict[str, Any]] = []
    for index in range(int(bins)):
        selected = assignments == index
        count = int(selected.sum())
        mean_probability = float(probabilities[selected].mean()) if count else None
        default_fraction = float(labels[selected].mean()) if count else None
        gap = abs(mean_probability - default_fraction) if count else None
        rows.append({
            "bin": index,
            "bin_lower": float(index / bins),
            "bin_upper": float((index + 1) / bins),
            "n": count,
            "mean_probability": mean_probability,
            "default_fraction": default_fraction,
            "absolute_gap": float(gap) if gap is not None else None,
        })
    return rows


def expected_calibration_error(y: Any, probability: Any, bins: int = 10) -> float | None:
    """Compute sample-weighted fixed-width ECE, or ``None`` for no records."""

    labels, probabilities = _prediction_arrays(y, probability)
    if len(labels) == 0:
        return None
    rows = calibration_bins(labels, probabilities, bins)
    return float(sum((row["n"] / len(labels)) * row["absolute_gap"]
                     for row in rows if row["n"]))


def detailed_prediction_metrics(
    y: Any,
    probability: Any,
    threshold: float = 0.5,
    bins: int = 10,
) -> dict[str, Any]:
    """Return discrimination, calibration, and classwise metrics.

    The six primary scalar metrics are ``roc_auc``, ``average_precision``,
    ``recall``, ``f1``, ``brier``, and ``log_loss``.  ROC-AUC/AP are undefined
    for a single-class evaluation subset and are returned as ``None``.  The
    ``classwise`` mapping contains precision, recall, F1, and support for both
    classes even when one has zero support.
    """

    labels, probabilities = _prediction_arrays(y, probability)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and lie in [0, 1]")
    if len(labels) == 0:
        empty_classwise = {
            "0": {"precision": None, "recall": None, "f1": None, "support": 0},
            "1": {"precision": None, "recall": None, "f1": None, "support": 0},
        }
        return {
            "n": 0,
            "roc_auc": None,
            "average_precision": None,
            "recall": None,
            "f1": None,
            "brier": None,
            "log_loss": None,
            "classwise": empty_classwise,
            "ece": None,
            "calibration_bins": calibration_bins(labels, probabilities, bins),
        }

    prediction = (probabilities >= threshold).astype(int)
    both_classes = np.unique(labels).size == 2
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1], zero_division=0
    )
    classwise = {
        str(label): {
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        }
        for label in (0, 1)
    }
    rows = calibration_bins(labels, probabilities, bins)
    return {
        "n": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if both_classes else None,
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "classwise": classwise,
        "ece": float(sum((row["n"] / len(labels)) * row["absolute_gap"]
                          for row in rows if row["n"])),
        "calibration_bins": rows,
    }


def reliability_curve(
    score: Any,
    event: Any,
    eligible: Any,
    total_n: int | None = None,
) -> list[dict[str, Any]]:
    """Return lower-score-first selective reliability points.

    ``score`` is interpreted as a risk/uncertainty score where lower means
    more reliable (for example, missing fraction).  Only finite eligible rows
    with binary event labels are ranked.  Equal scores are included as one
    group, so a tie cannot be broken using the event label.  The output is a
    descriptive curve and must not be reported as a calibrated release rule.
    """

    scores = np.asarray(score, dtype=float)
    events = np.asarray(event)
    eligible_array = np.asarray(eligible, dtype=bool)
    if scores.ndim == 2 and 1 in scores.shape:
        scores = scores.reshape(-1)
    if events.ndim == 2 and 1 in events.shape:
        events = events.reshape(-1)
    if scores.ndim != 1 or events.ndim != 1 or eligible_array.ndim != 1:
        raise ValueError("score, event, and eligible must be one-dimensional arrays")
    if not (len(scores) == len(events) == len(eligible_array)):
        raise ValueError("score, event, and eligible must have equal length")
    if total_n is None:
        total_n = len(scores)
    if isinstance(total_n, bool) or not isinstance(total_n, (int, np.integer)) or total_n < 1:
        raise ValueError("total_n must be a positive integer")
    if int(total_n) < int(eligible_array.sum()):
        raise ValueError("total_n cannot be smaller than the eligible count")

    event_numeric = np.asarray(events, dtype=float)
    valid = eligible_array & np.isfinite(scores) & np.isfinite(event_numeric)
    valid &= np.isin(event_numeric, [0, 1])
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    order = indices[np.argsort(scores[indices], kind="stable")]
    rows: list[dict[str, Any]] = []
    selected = 0
    revised = 0
    position = 0
    while position < len(order):
        score_value = scores[order[position]]
        end = position + 1
        while end < len(order) and scores[order[end]] == score_value:
            end += 1
        group = order[position:end]
        selected += len(group)
        revised += int(event_numeric[group].sum())
        rows.append({
            "score_threshold": float(score_value),
            "n_released": int(selected),
            "coverage": float(selected / int(total_n)),
            "revision_rate": float(revised / selected),
            "n_eligible": int(len(indices)),
            "n_total": int(total_n),
        })
        position = end
    return rows


__all__ = [
    "calibration_bins",
    "expected_calibration_error",
    "detailed_prediction_metrics",
    "reliability_curve",
]
