"""Median/mode imputation and reversible encoded-to-original feature mapping."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financepaper.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES


def _validate_features(X):
    if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
        raise ValueError("Expected a DataFrame with unique original feature names")
    if set(X.columns) != set(FEATURE_NAMES):
        raise ValueError("Predictors must be exactly the 23 schema fields, without ID or target")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Predictors cannot contain infinity")


class FeaturePreprocessor(ColumnTransformer):
    def fit_transform(self, X, y=None, **params):
        _validate_features(X)
        # sklearn Pipeline supplies y, but imputation/encoding must never use it.
        return super().fit_transform(X, y=None, **params)

    def transform(self, X, **params):
        _validate_features(X)
        return super().transform(X, **params)


def make_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    return FeaturePreprocessor([
        ("numeric", Pipeline(numeric), list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", sparse_threshold=0)


def encoded_feature_origins(preprocessor: ColumnTransformer) -> list[str]:
    """Derive mapping from fitted encoder metadata, never parse dummy names."""
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    origins = list(NUMERIC_FEATURES)
    for name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
        origins.extend([name] * len(categories))
    if len(origins) != len(preprocessor.get_feature_names_out()):
        raise ValueError("Encoded feature mapping does not match transformer output")
    return origins
