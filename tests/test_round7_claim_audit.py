"""Hand-computable falsifiers, not tests asserting a proposed model must win."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "round7_audit", Path(__file__).resolve().parents[1]/"scripts/audit_round7_claims.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_frozen_aipw_corrects_mean_but_can_increase_variance():
    result = audit.frozen_teacher()
    bad = result["misspecified_q"]
    assert bad["mean"] == pytest.approx(2/3)
    assert bad["variance"] == pytest.approx(941/900)
    assert result["oracle_q"]["variance"] == pytest.approx(5/9)
    assert result["misspecified_q_variance_ratio_vs_fixed100"] == pytest.approx(941/500)


def test_joint_target_unbiasedness_does_not_imply_correct_gradient():
    result = audit.joint_teacher()
    for row in result["rows"]:
        assert row["coherence_loss"] == row["conditional_moment_critic_value"] == 0
        assert row["squared_target_loss"] == pytest.approx(row["theta"]**2/row["pi"])
        assert row["finite_difference_gradient"] == pytest.approx(2*row["theta"]/row["pi"], abs=1e-8)
    assert result["supervised_example"]["naive_theta"] == pytest.approx(1/11)
    assert result["supervised_example"]["naive_outcome_mse"] == pytest.approx(100/121)


def test_quantifier_change_is_not_a_robustness_gain():
    rows = audit.sensitivity()
    assert [r["any_competitor_minimum"] for r in rows] == pytest.approx([4, .8, 0, -4/11])
    assert rows[2]["different_all_competitors_minimax"] == pytest.approx(3)
    assert np.allclose([r["normalization"] for r in rows], 1)


def test_clipping_must_resolve_budget_multiplier():
    rows = audit.allocation()
    at5 = {r["method"]: r for r in rows if r["budget"] == 5}
    assert at5["uniform"]["variance_functional"] == pytest.approx(.1625)
    assert at5["resolve_multiplier"]["variance_functional"] == pytest.approx(.10125)
    at8 = {r["method"]: r for r in rows if r["budget"] == 8}
    assert at8["clip_without_resolving"]["cost"] == pytest.approx(53/9)
    assert at8["resolve_multiplier"]["cost"] == pytest.approx(8)
    assert at8["resolve_multiplier"]["probabilities"] == pytest.approx([1, .6])
    assert at8["resolve_multiplier"]["variance_functional"] < at8["clip_without_resolving"]["variance_functional"]


def test_xor_interventional_shap_contrast_is_zero():
    rows = audit.xor_shap()
    assert [r["phi1"] for r in rows] == [-.25, .25, .25, -.25]
    assert all(r["contrast"] == r["reconstruction_error"] == 0 for r in rows)


@pytest.mark.parametrize("pi", [0, -.1, 1.1, float("nan")])
def test_audit_positivity_guard(pi):
    with pytest.raises(ValueError):
        audit.finite_aipw([.5, 1], [2/3, 1/3], .9, pi)


def test_audit_manifest_and_no_overwrite(tmp_path):
    folder = tmp_path/"run"
    result = audit.write_run(folder)
    assert result["financial_data_accessed"] is False
    assert result["status"] == "complete"
    import hashlib
    assert result["results_sha256"] == hashlib.sha256((folder/"results.json").read_bytes()).hexdigest()
    assert json.loads((folder/"results.json").read_text()) == audit.run_audit() | {
        "xor_shap": [dict(r, x=list(r["x"])) for r in audit.xor_shap()]}
    with pytest.raises(FileExistsError):
        audit.write_run(folder)
