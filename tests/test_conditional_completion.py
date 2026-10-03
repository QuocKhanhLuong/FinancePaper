import numpy as np
import pandas as pd
import pytest

from financepaper.reliability.conditional_completion import TrainingConditionalForest


def test_train_only_schema_and_reproducible_valid_support():
    training = pd.DataFrame({"amount": np.arange(80, dtype=float),
                             "category": np.tile([0., 1.], 40), "balance": np.arange(80) * 2.})
    training.loc[0, "amount"] = np.nan
    model = TrainingConditionalForest(training.columns, ["category"], trees=2, min_leaf=2)
    with pytest.raises(ValueError, match="training"):
        model.fit(training, partition="test")
    with pytest.raises(ValueError, match="schema"):
        model.fit(training.assign(target=1), partition="train")
    model.fit(training, partition="train")
    current = pd.DataFrame({"amount": [np.nan, 9.], "category": [np.nan, np.nan], "balance": [30., 18.]})
    artificial = np.array([[True, False, False], [False, True, False]])
    a = model.sample(current, artificial, seed=3)
    b = model.sample(current, artificial, seed=3)
    np.testing.assert_equal(a, b)
    assert a.shape == (2, 8, 3)
    assert np.isnan(a[0, :, 1]).all()  # Natural missingness has no restored truth.
    assert (a[0, :, 2] == 30.).all() and (a[1, :, 0] == 9.).all()
    assert set(a[1, :, 1]) <= {0., 1.}
    assert set(a[0, :, 0]) <= set(training.amount.dropna())
    with pytest.raises(ValueError, match="must not enter"):
        model.sample(current.fillna(0), artificial)


def test_missing_training_target_is_not_fabricated():
    frame = pd.DataFrame({"a": np.ones(80), "b": np.full(80, np.nan)})
    with pytest.raises(ValueError, match="observed targets"):
        TrainingConditionalForest(frame.columns, [], min_leaf=2).fit(frame, partition="train")


def test_original_missing_flags_survive_latent_filling():
    current = pd.DataFrame({"a": [1., np.nan], "b": [np.nan, 2.]})
    latent = current.fillna(5.)
    design = TrainingConditionalForest._conditioning(latent, ["a", "b"], current.isna())
    np.testing.assert_equal(design[["a__missing", "b__missing"]].to_numpy(), [[0., 1.], [1., 0.]])
