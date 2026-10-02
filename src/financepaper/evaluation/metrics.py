"""Prediction and restoration summaries with explicit evaluation denominators."""

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, recall_score, roc_auc_score)


def prediction_metrics(y, probability, threshold: float) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    if not len(y):
        return {"n": 0, **dict.fromkeys(("roc_auc", "average_precision", "recall", "f1", "brier", "log_loss"))}
    both = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "roc_auc": float(roc_auc_score(y, probability)) if both else None,
        "average_precision": float(average_precision_score(y, probability)) if both else None,
        "recall": float(recall_score(y, probability >= threshold, zero_division=0)),
        "f1": float(f1_score(y, probability >= threshold, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def calibration_rows(y, probability, bins: int) -> list[dict]:
    """Uniform bins include empty bins; probabilities are deliberately uncalibrated."""
    y, probability = np.asarray(y), np.asarray(probability)
    assignments = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        selected = assignments == i
        rows.append({
            "bin_lower": i / bins, "bin_upper": (i + 1) / bins,
            "n": int(selected.sum()),
            "mean_probability": float(probability[selected].mean()) if selected.any() else None,
            "default_fraction": float(y[selected].mean()) if selected.any() else None,
        })
    return rows


def summarize_records(records, threshold: float) -> dict:
    eligible = records["eligible"].to_numpy(dtype=bool)
    released = records.loc[eligible]
    revised = released["revision_event"].astype(bool)
    stable_released = released["prediction_stable"].astype(bool)
    stable_revised = int((revised & stable_released).sum())
    result = {
        "n_test": len(records), "n_eligible": len(released),
        "n_withheld": int((~eligible).sum()), "coverage": float(eligible.mean()),
        "n_revised": int(revised.sum()),
        "reason_revision_rate": float(revised.mean()) if len(released) else None,
        "actual_cell_missing_rate": float(records["missing_fraction"].mean()),
        "mean_absolute_probability_shift": float(records["absolute_probability_shift"].mean()),
        "n_stable_prediction_eligible": int(stable_released.sum()),
        "n_stable_prediction_revised": stable_revised,
        "revision_rate_given_stable_prediction_and_eligible": (
            stable_revised / int(stable_released.sum()) if stable_released.any() else None
        ),
        "prediction_all_before": prediction_metrics(records.y, records.probability_before, threshold),
        "prediction_all_restored": prediction_metrics(records.y, records.probability_restored, threshold),
        "prediction_eligible_before": prediction_metrics(released.y, released.probability_before, threshold),
        "prediction_eligible_restored": prediction_metrics(released.y, released.probability_restored, threshold),
    }
    for key in ("observed_attribution_mae", "observed_sign_agreement", "top_k_overlap", "rank_correlation"):
        values = records[key].dropna()
        result[f"mean_{key}"] = float(values.mean()) if len(values) else None
    return result
