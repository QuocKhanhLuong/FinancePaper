"""Synthetic engineering tests. Fixtures are invented, never Freddie records."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from financepaper.data.freddie import (
    FEATURES, GROUPS, ORIGINATION_COLUMNS, PERFORMANCE_COLUMNS, RELEASE,
    FreddieDataError, artificial_mask, month_number, month_text, prepare_vintage,
    restore_artificial, sha256, validate_analysis_freeze, validate_partition_metadata,
    validate_receipt, verification_target,
)


def source_rows():
    orig = dict.fromkeys(ORIGINATION_COLUMNS, "")
    orig.update(loan_id="F00Q10000001", first_payment_date="200011", fico="700",
                amortization_type="FRM", original_upb="100000", original_rate="6",
                original_term="360", original_dti="999", original_ltv="80", original_cltv="80",
                mi_percentage="0", units="1", borrowers="2", occupancy="P", channel="R",
                property_type="SF", loan_purpose="P", first_time_homebuyer="Y", harp_indicator="N")
    perf = []
    for i in range(19):
        row = dict.fromkeys(PERFORMANCE_COLUMNS, "")
        row.update(loan_id=orig["loan_id"], period=month_text(month_number("200011") + i),
                   current_upb="99000", current_rate="6", delinquency_status="00", loan_age=str(i))
        perf.append(row)
    return orig, perf


def archive(tmp_path, orig, perf, *, old_layout=False):
    path = tmp_path / "sample_2000.zip"
    with zipfile.ZipFile(path, "w") as z:
        values = [orig[c] for c in ORIGINATION_COLUMNS] + ([""] if old_layout else [])
        z.writestr("sample_orig_2000.txt", "|".join(values) + "\n")
        z.writestr("sample_perf_2000.txt", "\n".join("|".join(r[c] for c in PERFORMANCE_COLUMNS) for r in perf) + "\n")
    return path


def test_receipt_fails_closed(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(RELEASE))
    with pytest.raises(FreddieDataError, match="attested"):
        validate_receipt(receipt)
    receipt.write_text(json.dumps(RELEASE | {"researcher_accepted_terms": True,
        "applicable_clickthrough_reviewed": True, "clickthrough_conflicts_with_audit": False,
        "download_date": "2026-08-01"}))
    assert validate_receipt(receipt)["release"] == 47


def test_schema_features_and_future_isolation(tmp_path):
    orig, perf = source_rows()
    original = prepare_vintage(archive(tmp_path, orig, perf), 2000, expected_loans=1)
    assert len(ORIGINATION_COLUMNS) == 31 and len(PERFORMANCE_COLUMNS) == 35
    assert original.targets.iloc[0].label == 0
    assert original.targets.iloc[0].landmark == "200104"
    assert original.targets.iloc[0].horizon_end == "200204"
    assert np.isnan(original.features.iloc[0].original_dti)
    assert sorted(sum((list(v) for v in GROUPS.values()), [])) == sorted(FEATURES)
    for i in range(6, len(perf)):
        perf[i].update(current_upb="999999", current_rate="99", actual_loss="777777",
                       modification_flag="Y", loan_age="0")
    perf[17]["delinquency_status"] = "03"  # t+12 is in horizon.
    changed = prepare_vintage(archive(tmp_path, orig, perf), 2000, expected_loans=1)
    pd.testing.assert_frame_equal(original.features, changed.features)
    assert changed.targets.iloc[0].label == 1
    perf[17]["delinquency_status"] = "00"
    perf[18]["delinquency_status"] = "03"  # t+13 excluded.
    late = prepare_vintage(archive(tmp_path, orig, perf), 2000, expected_loans=1)
    assert late.targets.iloc[0].label == 0


@pytest.mark.parametrize("change,reason,label", [
    ("gap", "incomplete_followup", None),
    ("xx", "incomplete_followup", None),
    ("ra", "reo_without_numeric_event", None),
    ("payoff", "competing_voluntary_payoff", 0),
    ("other", "terminal_incomplete_or_other", None),
    ("gap_then_positive", "observed_90plus", 1),
    ("gap_then_payoff", "terminal_incomplete_or_other", None),
    ("payoff_then_positive", "competing_voluntary_payoff", 0),
])
def test_censoring(change, reason, label):
    s, z, p = np.zeros(12), np.zeros(12, int), np.ones(12, bool)
    if "gap" in change:
        p[0] = False
    if change == "xx":
        s[0] = np.nan
    if change == "ra":
        s[0] = 100
    if "payoff" in change:
        z[2] = 1
    if change == "other":
        z[2] = 9
    if "positive" in change:
        s[3] = 3
    result = verification_target(s, z, p)
    assert result.label == label and result.reason == reason


def test_invalid_schema_duplicate_and_termination(tmp_path):
    orig, perf = source_rows()
    with pytest.raises(FreddieDataError, match="Layout"):
        prepare_vintage(archive(tmp_path, orig, perf, old_layout=True), 2000, expected_loans=1)
    with pytest.raises(FreddieDataError, match="Duplicate"):
        prepare_vintage(archive(tmp_path, orig, perf + [perf[0]]), 2000, expected_loans=1)
    perf[6].update(zero_balance_code="01", zero_balance_date=perf[7]["period"])
    with pytest.raises(FreddieDataError, match="alignment"):
        prepare_vintage(archive(tmp_path, orig, perf), 2000, expected_loans=1)


def test_natural_missing_cannot_be_verified():
    full = np.array([[1., np.nan, 3.], [4., 5., 6.]])
    mask = artificial_mask(full, seed=1, rate=1.)
    assert not mask[0, 1]
    partial = np.where(mask, np.nan, full)
    np.testing.assert_equal(restore_artificial(partial, full, mask), full)
    with pytest.raises(ValueError, match="Natural"):
        restore_artificial(partial, full, np.ones_like(mask))


def test_entity_and_outcome_maturation_separation():
    targets = pd.DataFrame({"role": ["train", "development"], "landmark": ["200801", "201101"],
                            "horizon_end": ["200901", "201201"]}, index=["invented_a", "invented_b"])
    validate_partition_metadata(targets)
    targets.loc["invented_a", "horizon_end"] = "201101"
    with pytest.raises(FreddieDataError, match="overlap"):
        validate_partition_metadata(targets)
    targets.index = ["same", "same"]
    with pytest.raises(FreddieDataError, match="crosses"):
        validate_partition_metadata(targets)


def test_assessment_requires_complete_immutable_fit_freeze(tmp_path):
    intake = tmp_path / "intake.json"
    intake.write_text("{}")
    artifact = tmp_path / "synthetic_artifact.txt"
    artifact.write_text("synthetic engineering fixture only")
    keys = ("predictor_lr", "predictor_xgb", "preprocessor", "shap_background",
            "donor_completion", "conditional_completion", "release_policies", "protocol")
    freeze = {"release": 47, "development_intake_sha256": sha256(intake),
              "artifacts": {k: {"path": artifact.name, "sha256": sha256(artifact)} for k in keys}}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    validate_analysis_freeze(path, tmp_path, intake)
    artifact.write_text("changed after freeze")
    with pytest.raises(FreddieDataError, match="checksum"):
        validate_analysis_freeze(path, tmp_path, intake)
