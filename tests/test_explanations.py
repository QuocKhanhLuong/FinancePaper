import joblib
import numpy as np
import pytest
from xgboost.callback import EarlyStopping

from financepaper.data.schema import FEATURE_NAMES
from financepaper.explanations.reasons import extract_reasons
from financepaper.explanations.shap_values import FixedShapExplainer, aggregate_shap
from financepaper.evaluation.revision import revision_event
from financepaper.missingness.mcar import apply_mask, mcar_mask
from financepaper.models.logistic import make_logistic
from financepaper.models.xgboost import make_xgboost
from financepaper.preprocessing.pipelines import make_preprocessor, encoded_feature_origins


def test_train_only_statistics_and_unknown_categories(credit_frame):
    X, _ = credit_frame
    pre = make_preprocessor()
    pre.fit(X.iloc[:160])
    before = joblib.hash(pre)
    heldout = X.iloc[160:].copy()
    heldout["LIMIT_BAL"] = np.nan
    heldout["PAY_0"] = 999
    transformed = pre.transform(heldout)
    assert np.isfinite(transformed).all()
    assert joblib.hash(pre) == before
    assert 999 not in pre.named_transformers_["categorical"].named_steps["encoder"].categories_[3]
    numeric = pre.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric.statistics_[0] == np.median(X.LIMIT_BAL.iloc[:160])
    assert len(encoded_feature_origins(pre)) == transformed.shape[1]
    with pytest.raises(ValueError, match="without ID or target"):
        pre.fit(X.assign(ID=X.index))


def test_signed_aggregation_conserves_additivity():
    values = np.array([[1.0, -0.8, 2.0], [-1.0, 0.8, 3.0]])
    aggregated = aggregate_shap(values, ["category", "category", "amount"], ["amount", "category"])
    np.testing.assert_allclose(aggregated, [[2, 0.2], [3, -0.2]])
    np.testing.assert_allclose(aggregated.sum(axis=1), values.sum(axis=1))


def test_lr_observed_attributions_invariant_after_restoration(credit_frame):
    X, y = credit_frame
    pre = make_preprocessor()
    encoded = pre.fit_transform(X.iloc[:160])
    model = make_logistic().fit(encoded, y.iloc[:160])
    explanation = FixedShapExplainer(model, encoded[:32], encoded_feature_origins(pre))
    test = X.iloc[160:]
    mask = mcar_mask(test, 0.3, 123)
    before = explanation.explain(pre.transform(apply_mask(test, mask)))
    after = explanation.explain(pre.transform(test))
    np.testing.assert_allclose((after - before)[~mask.to_numpy()], 0, atol=1e-12)
    np.testing.assert_allclose(after.sum(axis=1) + explanation.expected_value,
                               model.decision_function(pre.transform(test)), atol=1e-10)
    events = []
    for phi, restored, hidden in zip(before, after, mask.to_numpy(), strict=True):
        reasons = extract_reasons(phi, hidden, FEATURE_NAMES, k=1)
        events.append(revision_event(reasons, restored, hidden, FEATURE_NAMES, k=1))
    assert False in events
    assert True not in events


def test_tree_shap_same_early_stopped_predictor_and_interactions(credit_frame):
    X, y = credit_frame
    pre = make_preprocessor(scale_numeric=False)
    train, dev = pre.fit_transform(X.iloc[:160]), pre.transform(X.iloc[160:200])
    model = make_xgboost(n_estimators=40, max_depth=3, min_child_weight=1,
                         callbacks=[EarlyStopping(rounds=4, save_best=True)])
    model.fit(train, y.iloc[:160], eval_set=[(dev, y.iloc[160:200])], verbose=False)
    assert model.get_booster().num_boosted_rounds() == model.best_iteration + 1
    explainer = FixedShapExplainer(model, train[:128], encoded_feature_origins(pre))
    np.testing.assert_array_equal(explainer.tree_explainer.data, train[:128])
    test = X.iloc[200:]
    mask = mcar_mask(test, 0.3, 42)
    before = explainer.explain(pre.transform(apply_mask(test, mask)))
    after = explainer.explain(pre.transform(test))
    np.testing.assert_allclose(after.sum(axis=1) + explainer.expected_value,
                               model.predict(pre.transform(test), output_margin=True), atol=2e-5)
    assert np.max(np.abs(after - before)[~mask.to_numpy()]) > 1e-4


def test_untrimmed_early_stopping_model_is_rejected(credit_frame):
    X, y = credit_frame
    pre = make_preprocessor(False)
    encoded = pre.fit_transform(X)
    model = make_xgboost(n_estimators=20, learning_rate=0.5, min_child_weight=1,
                         early_stopping_rounds=2)
    model.fit(encoded[:160], y.iloc[:160], eval_set=[(encoded[160:], 1-y.iloc[160:])], verbose=False)
    assert model.get_booster().num_boosted_rounds() > model.best_iteration + 1
    with pytest.raises(ValueError, match="identical trees"):
        FixedShapExplainer(model, encoded[:32], encoded_feature_origins(pre))
