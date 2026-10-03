"""Training-only conditional leaf sampling; experimental second completion family.

This is neither a posterior sampler nor a guaranteed coherent joint distribution.
It is deliberately separate from the frozen historical neighbor/donor estimator.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


class TrainingConditionalForest:
    def __init__(self, feature_names, categorical, *, seed=20261003, max_rows=50000,
                 trees=16, depth=6, min_leaf=40):
        self.names = tuple(feature_names)
        self.categorical = tuple(categorical)
        if len(set(self.names)) != len(self.names) or len(self.names) < 2:
            raise ValueError("Explicit unique predictor schema required")
        if not set(self.categorical) <= set(self.names):
            raise ValueError("Categorical fields outside predictor schema")
        self.seed, self.max_rows = seed, max_rows
        self.trees, self.depth, self.min_leaf = trees, depth, min_leaf

    def _frame(self, frame):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != self.names:
            raise ValueError("Only the frozen predictor columns, in schema order, are allowed")
        if np.isinf(frame.to_numpy(float)).any():
            raise ValueError("Infinity is not a missing-value code")
        return frame.copy()

    @staticmethod
    def _conditioning(frame, columns, missing=None):
        design = frame[columns].copy()
        flags = frame[columns].isna() if missing is None else missing[columns]
        for c in columns:
            design[f"{c}__missing"] = flags[c].to_numpy(dtype=float)
        return design

    def fit(self, train_features, *, partition):
        if partition != "train":
            raise ValueError("Completion models may only fit training features")
        frame = self._frame(train_features)
        if frame.index.has_duplicates or len(frame) < 2 * self.min_leaf:
            raise ValueError("Insufficient independent training rows")
        chosen = np.sort(np.random.default_rng(self.seed).choice(len(frame), min(len(frame), self.max_rows), replace=False))
        frame = frame.iloc[chosen]
        if frame.notna().sum().min() < 2 * self.min_leaf:
            raise ValueError("Insufficient genuinely observed targets for a conditional field")
        self.training_ids_ = frame.index.to_numpy().copy()
        self.initial_ = {c: float(frame[c].mode().iloc[0] if c in self.categorical else frame[c].median()) for c in self.names}
        self.models_ = {}
        for j, field in enumerate(self.names):
            columns = [c for c in self.names if c != field]
            categories = [c for c in columns if c in self.categorical]
            numeric = [c for c in columns if c not in self.categorical] + [f"{c}__missing" for c in columns]
            transformers = []
            if numeric:
                transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
            if categories:
                transformers.append(("categorical", make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categories))
            encoder = ColumnTransformer(transformers, remainder="drop")
            observed = frame[field].notna().to_numpy()
            # Fit only training statistics. Target-field exclusion is structural.
            encoded = encoder.fit_transform(self._conditioning(frame, columns))
            values = frame[field].to_numpy()[observed]
            model_class = ExtraTreesClassifier if field in self.categorical else ExtraTreesRegressor
            model = model_class(n_estimators=self.trees, max_depth=self.depth,
                                min_samples_leaf=self.min_leaf, n_jobs=1, random_state=self.seed + j)
            model.fit(encoded[observed], values)
            leaves = model.apply(encoded[observed])
            pools = [{int(leaf): values[leaves[:, t] == leaf] for leaf in np.unique(leaves[:, t])}
                     for t in range(self.trees)]
            self.models_[field] = (columns, encoder, model, pools)
        return self

    def sample(self, partial, artificial_mask, *, k=8, seed=0):
        """Accept current partial inputs only; natural NaNs are never returned filled.

        artificial_mask identifies verification-eligible hidden cells. Every other
        NaN is natural missingness. No hidden true value is an inference argument.
        """
        if not hasattr(self, "models_"):
            raise ValueError("Completion model is not fitted")
        frame = self._frame(partial)
        mask = np.asarray(artificial_mask)
        x = frame.to_numpy(float)
        if k not in (1, 2, 4, 8, 16) or mask.shape != x.shape or mask.dtype != bool:
            raise ValueError("Invalid completion count or artificial mask")
        if np.any(mask & np.isfinite(x)):
            raise ValueError("Artificially hidden values must not enter completion inference")
        missing = ~np.isfinite(x)
        working = pd.DataFrame(np.repeat(x, k, axis=0), columns=self.names)
        missing_repeated = np.repeat(missing, k, axis=0)
        missing_frame = pd.DataFrame(missing_repeated, columns=self.names)
        for field in self.names:
            working[field] = working[field].fillna(self.initial_[field])
        rng = np.random.default_rng(seed)
        for _ in range(2):
            for j, field in enumerate(self.names):
                rows = np.flatnonzero(missing_repeated[:, j])
                if not len(rows):
                    continue
                columns, encoder, model, pools = self.models_[field]
                design = encoder.transform(self._conditioning(working.loc[rows], columns, missing_frame.loc[rows]))
                trees = rng.integers(self.trees, size=len(rows))
                all_leaves = model.apply(design)
                sampled = np.empty(len(rows))
                for tree in np.unique(trees):
                    indices = np.flatnonzero(trees == tree)
                    leaves = all_leaves[indices, tree]
                    for leaf in np.unique(leaves):
                        local = indices[leaves == leaf]
                        sampled[local] = rng.choice(pools[tree][int(leaf)], len(local), replace=True)
                working.loc[rows, field] = sampled
        completed = working.to_numpy().reshape(len(x), k, len(self.names))
        # Expose only artificial completions; preserve all observed values exactly
        # and reset naturally missing latent variables to NaN at the endpoint.
        return np.where(mask[:, None, :], completed, x[:, None, :])
