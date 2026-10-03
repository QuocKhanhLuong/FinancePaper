import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import yaml

torch = pytest.importorskip("torch")

from financepaper.data.schema import FEATURE_NAMES, TARGET
from financepaper.data.split import split_indices
from financepaper.data.temporal import TemporalPreprocessor
from financepaper.experiments import robustness_followup as follow
from financepaper.experiments import temporal_pilot as pilot
from financepaper.models.logit_blend import LogitBlend
from financepaper.models.xgboost import make_xgboost


def test_logit_blend_explains_its_actual_function():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    a = LogisticRegression(C=.1).fit(X, y)
    b = LogisticRegression(C=10).fit(X, y)
    model = LogitBlend(a, b, .3)
    reference = X[:20].mean(0)
    phi_a, phi_b = (X-reference)*a.coef_, (X-reference)*b.coef_
    np.testing.assert_allclose(model.combine_attributions(phi_a, phi_b).sum(1),
        model.decision_function(X)-model.decision_function(reference.reshape(1, -1)), atol=1e-12)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], expit(model.decision_function(X)))
    assert not np.allclose(model.predict_proba(X)[:, 1], .7*a.predict_proba(X)[:, 1]+.3*b.predict_proba(X)[:, 1])
    with pytest.raises(ValueError):
        LogitBlend(a, b, np.nan)


def test_training_views_cover_epoch_masks_and_unit_customer_weight(credit_frame):
    X, y = credit_frame
    X, y = X.iloc[:30], y.iloc[:30]
    prep = TemporalPreprocessor().fit(X)
    context = dict(X_train=X, y_train=y.to_numpy(), preprocessor=prep, train_batch=prep.transform(X))
    base = dict(missingness=dict(augmentation_rates=[0., .1, .2, .3]))
    values, labels, weights, ids = follow.training_views(context, base, 42, 3)
    np.testing.assert_allclose(pd.Series(weights).groupby(ids).sum(), 1., atol=1e-7)
    assert set(ids) == set(X.index)
    np.testing.assert_array_equal(labels, np.tile(y, 3))
    callback = pilot._make_augmentation_callback(X, prep, [0., .1, .2, .3], 700042)
    for e in range(3):
        np.testing.assert_array_equal(values[e*30:(e+1)*30], pilot._flat(callback(e))[0])
    np.testing.assert_array_equal(values, follow.training_views(context, base, 42, 3)[0])


def test_additive_tree_observed_attributions_survive_restoration(credit_frame):
    X, y = credit_frame
    prep = TemporalPreprocessor().fit(X.iloc[:180])
    train = prep.transform(X.iloc[:180])
    flat, origins = pilot._flat(train)
    tree = make_xgboost(42, max_depth=1, n_estimators=60, min_child_weight=1).fit(flat, y.iloc[:180])
    observed = X.iloc[180:190]
    mask = pd.DataFrame(False, index=observed.index, columns=observed.columns)
    mask.loc[:, ["PAY_0", "BILL_AMT3", "AGE"]] = True
    before = pilot._tree_shap(tree, prep.transform(observed, mask), flat[:16], origins)
    full = pilot._tree_shap(tree, prep.transform(observed), flat[:16], origins)
    assert before["valid"].all() and full["valid"].all()
    np.testing.assert_allclose(before["original"][~mask.to_numpy()], full["original"][~mask.to_numpy()], atol=1e-7)
    assert np.max(abs(full["original"])) > .01


def test_joint_selection_rejects_stability_gaming_and_failed_coverage():
    table = pd.DataFrame([
        dict(model="best", average_precision=.6, roc_auc=.8, brier=.14, log_loss=.44),
        dict(model="joint", average_precision=.599, roc_auc=.801, brier=.14, log_loss=.441),
        dict(model="constant", average_precision=.22, roc_auc=.5, brier=.18, log_loss=.5)])
    matches = pd.DataFrame([dict(model=name, condition=c, target_coverage=.5, actual_coverage=.5, revision_rate=r)
        for name, r in [("best", .3), ("joint", .1), ("constant", 0.)] for c in ("mcar10", "mcar30")])
    cfg = dict(selection=dict(ap_tolerance=.005, auc_tolerance=.005, brier_tolerance=.002, log_loss_tolerance=.005))
    choice = follow.select_candidates(table, matches, cfg)
    assert choice["predictive_choice"] == "best" and choice["joint_choice"] == "joint"
    assert "constant" not in choice["joint_qualified"]
    matches.actual_coverage = .49
    choice = follow.select_candidates(table, matches, cfg)
    assert not choice["joint_gate_passed"] and choice["joint_choice"] == "best"


def test_matched_coverage_cannot_ignore_an_absent_model():
    rows = [dict(model=name, condition="mcar30", record_id=i, eligible=True,
                 attribution_valid=True, reason_strength=float(i), revision_event=False)
            for name in ("a", "b") for i in range(8)]
    records = pd.DataFrame(rows)
    result = follow.matched_coverage(records, ("a", "b", "absent"), (.5,))
    assert (result.common_n == 0).all() and (result.selected_n == 0).all()
    assert result.revision_rate.isna().all()


def test_dominance_preserves_losses_and_undefined_restarts():
    metrics, shifts, reasons, matching = [], [], [], []
    for seed in (42, 43):
        for name in follow.EVALUATED:
            for condition in pilot.ALL_CONDITIONS:
                row = dict(seed=seed, model=name, condition=condition, probability="calibrated",
                    roc_auc=.8, average_precision=.5, recall=.5, f1=.5, brier=.15, log_loss=.45, ece=.01)
                row.update({f"class_{c}_{key}": .5 for c in (0, 1) for key in ("precision", "recall", "f1")})
                if name in follow.CHOICES:
                    row["average_precision"] = .6
                    row["brier"] = .16  # AP improvement cannot hide calibration harm.
                    if seed == 43:
                        row["roc_auc"] = np.nan
                metrics.append(row)
                shifts.append(dict(seed=seed, model=name, condition=condition, absolute_calibrated_probability_shift=.04))
                reasons.append(dict(seed=seed, model=name, condition=condition, coverage=.9,
                                    top_k_overlap=.9, observed_sign_agreement=.9, rank_correlation=.9))
                matching.append(dict(seed=seed, model=name, condition=condition, target_coverage=.5,
                                     actual_coverage=.5, revision_rate=.1))
    table = follow.dominance_table(*map(pd.DataFrame, (metrics, shifts, reasons, matching)))
    assert (table.loc[table.metric == "average_precision", "verdict"] == "win").all()
    assert (table.loc[table.metric == "brier", "verdict"] == "loss").all()
    assert (table.loc[table.metric == "roc_auc", "verdict"] == "undefined").all()


def _synthetic_config(tmp_path, credit_frame):
    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / "configs/robustness_followup.yaml").read_text())
    base = yaml.safe_load((root / "configs/temporal_pilot.yaml").read_text())
    base.update(background_size=8, hidden_size=8, static_hidden=4, fusion_hidden=8, device="cpu")
    base["training"].update(epochs=1, patience=1, batch_size=64)
    base["xgboost"]["n_estimators"] = 8
    base["explanations"].update(test_records=8, development_records=8, ig_steps=4, ig_max_steps=8, ig_batch_size=8)
    X, y = credit_frame
    data = X.copy()
    data.insert(0, "ID", X.index)
    data[TARGET] = y
    source = tmp_path / "data.csv"
    data.to_csv(source, index=False)
    base["data_path"] = str(source)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    cfg.update(base_config=str(base_path), restart_seeds=[42], augmentation_views=1, regularized_estimators=8)
    path = tmp_path / "followup.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, source, data


def test_frozen_followup_pipeline_and_heldout_mutation(tmp_path, credit_frame, monkeypatch):
    config, source, data = _synthetic_config(tmp_path, credit_frame)
    original_test = pilot._test_context
    def forbidden_test(*args, **kwargs):
        raise AssertionError("test opened during fit")
    monkeypatch.setattr(pilot, "_test_context", forbidden_test)
    first = follow.fit_all(config, tmp_path / "first", "cpu")
    frozen = json.loads((first / "seed_42/frozen_selection.json").read_text())
    assert frozen["test_opened"] is False
    split = split_indices(credit_frame[1], 42)
    positions = np.concatenate([split["test"], split["risk_calibration"]])
    altered = data.copy()
    altered.loc[altered.index[positions], ["AGE", "BILL_AMT3"]] += 1000
    altered.to_csv(source, index=False)
    second = follow.fit_all(config, tmp_path / "second", "cpu")
    other = json.loads((second / "seed_42/frozen_selection.json").read_text())
    # Timings are not scientific selections; all scientific fields must agree.
    for selection in (frozen, other):
        for item in selection["models"].values():
            item["fit"].pop("elapsed_seconds", None)
    assert frozen == other
    for name in follow.NEURAL_BASELINES:
        a = torch.load(first / f"seed_42/{name}.pt", weights_only=True)
        b = torch.load(second / f"seed_42/{name}.pt", weights_only=True)
        assert all(torch.equal(a[k], b[k]) for k in a)
    # Evaluation fails closed on altered workbook identity.
    with pytest.raises(ValueError, match="dataset differs"):
        follow.evaluate(first, "cpu")
    data.to_csv(source, index=False)
    def checked_test(context, settings, output):
        assert (first / "fit_manifest.json").exists()
        assert (first / "seed_42/frozen_selection.json").exists()
        return original_test(context, settings, output)
    monkeypatch.setattr(pilot, "_test_context", checked_test)
    follow.evaluate(first, "cpu")
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["unused_splits"] == ["risk_calibration"]
    assert manifest["explanation_n"] == 8 and not manifest["independent_worker_review"]
    reasons = pd.read_csv(first / "evaluation/seed_42/reasons.csv")
    assert len(reasons) == len(follow.EVALUATED)*4*8
    assert not reasons.loc[reasons.condition == "complete", "revision_event"].dropna().any()
    assert reasons.loc[~reasons.eligible, "revision_event"].isna().all()
    assert "fit_manifest.json" in manifest["artifact_sha256"]
    with pytest.raises(FileExistsError):
        follow.fit_all(config, first, "cpu")
    with pytest.raises(FileExistsError):
        follow.evaluate(first, "cpu")
