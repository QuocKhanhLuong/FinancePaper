import pandas as pd

from financepaper.evaluation.metrics import summarize_records


def test_withheld_cases_are_excluded_from_revision_denominator():
    records = pd.DataFrame({
        "eligible": [True, False, True], "revision_event": [True, None, False],
        "prediction_stable": [True, True, False], "missing_fraction": [0.3] * 3,
        "absolute_probability_shift": [0.01, 0.01, 0.1], "y": [1, 0, 0],
        "probability_before": [0.5, 0.4, 0.3], "probability_restored": [0.51, 0.41, 0.4],
        "observed_attribution_mae": [0.02] * 3, "observed_sign_agreement": [1.0] * 3,
        "top_k_overlap": [1.0] * 3, "rank_correlation": [1.0] * 3,
    })
    summary = summarize_records(records, 0.5)
    assert summary["n_eligible"] == 2
    assert summary["reason_revision_rate"] == 0.5
    assert summary["prediction_all_before"]["n"] == 3
    assert summary["prediction_eligible_before"]["n"] == 2
    assert summary["revision_rate_given_stable_prediction_and_eligible"] == 1.0
    records["eligible"] = False
    records["revision_event"] = None
    summary = summarize_records(records, 0.5)
    assert summary["reason_revision_rate"] is None
    assert summary["coverage"] == 0
    assert summary["prediction_eligible_before"]["n"] == 0
