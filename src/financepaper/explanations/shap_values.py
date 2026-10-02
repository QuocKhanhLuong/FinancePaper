"""Fixed-reference interventional attributions, with signed dummy aggregation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from financepaper.data.schema import FEATURE_NAMES


def aggregate_shap(values, feature_origins, feature_names=FEATURE_NAMES) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_origins):
        raise ValueError("SHAP matrix and encoded feature mapping disagree")
    if len(set(feature_names)) != len(feature_names) or not set(feature_origins) <= set(feature_names):
        raise ValueError("Invalid original-feature mapping")
    if not np.isfinite(values).all():
        raise ValueError("Attributions must be finite")
    output = np.zeros((len(values), len(feature_names)))
    indices = {name: i for i, name in enumerate(feature_names)}
    for column, name in enumerate(feature_origins):
        output[:, indices[name]] += values[:, column]
    return output


class FixedShapExplainer:
    """Positive values increase predicted default log-odds (class 1).

    LR uses the exact independent-background linear SHAP formula. TreeSHAP uses
    an explicitly interventional background; this is a model attribution
    convention and makes no causal claim about the financial features.
    """

    def __init__(self, model, background, feature_origins, feature_names=FEATURE_NAMES):
        self.model = model
        self.background = np.array(background, dtype=float, copy=True)
        self.feature_origins = tuple(feature_origins)
        self.feature_names = tuple(feature_names)
        if (self.background.ndim != 2 or len(self.background) == 0
                or self.background.shape[1] != len(feature_origins)
                or not np.isfinite(self.background).all()):
            raise ValueError("Expected finite nonempty encoded training background")
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Expected binary default target with classes [0, 1]")
        self.background.setflags(write=False)
        if isinstance(model, LogisticRegression):
            self.mean = self.background.mean(axis=0)
            self.expected_value = float(model.intercept_[0] + self.mean @ model.coef_[0])
            self.tree_explainer = None
        elif isinstance(model, XGBClassifier):
            best = getattr(model, "best_iteration", None)
            if best is not None and model.get_booster().num_boosted_rounds() != best + 1:
                raise ValueError("Use EarlyStopping(save_best=True) so prediction and SHAP use identical trees")
            import shap

            # SHAP otherwise silently subsamples backgrounds larger than 100.
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            self.tree_explainer = shap.TreeExplainer(
                model, data=masker, model_output="raw",
                feature_perturbation="interventional",
            )
            self.expected_value = float(self.tree_explainer.expected_value)
        else:
            raise TypeError("Only the LR and XGBoost pilot models are supported")

    def explain_encoded(self, encoded_X) -> np.ndarray:
        X = np.asarray(encoded_X, dtype=float)
        if (X.ndim != 2 or X.shape[1] != self.background.shape[1]
                or not np.isfinite(X).all()):
            raise ValueError("Expected a finite matrix aligned with the encoded background")
        if self.tree_explainer is None:
            phi = (X - self.mean) * self.model.coef_[0]
            margin = self.model.decision_function(X)
        else:
            phi = np.asarray(self.tree_explainer.shap_values(X, check_additivity=True))
            margin = self.model.predict(X, output_margin=True)
        if not np.allclose(phi.sum(axis=1) + self.expected_value, margin, atol=2e-5, rtol=2e-5):
            raise RuntimeError("Attributions do not reconstruct the same model's raw margin")
        return phi

    def explain(self, encoded_X) -> np.ndarray:
        return aggregate_shap(self.explain_encoded(encoded_X), self.feature_origins, self.feature_names)
