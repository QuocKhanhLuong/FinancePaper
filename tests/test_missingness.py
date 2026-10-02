import numpy as np
import pandas as pd
import pytest

from financepaper.missingness.mcar import apply_mask, mcar_mask, restore_hidden


def test_value_independent_reproducible_mcar(credit_frame):
    X, _ = credit_frame
    mask = mcar_mask(X, 0.3, 42)
    pd.testing.assert_frame_equal(mask, mcar_mask(X * 100 + 6, 0.3, 42))
    assert not mask.equals(mcar_mask(X, 0.3, 43))
    assert abs(mask.to_numpy().mean() - 0.3) < 0.03
    assert not mcar_mask(X, 0, 42).any().any()
    assert mcar_mask(X, 1, 42).all().all()


def test_exact_restoration_and_no_source_mutation(credit_frame):
    X, _ = credit_frame
    saved = X.copy(deep=True)
    mask = mcar_mask(X, 0.3, 42)
    observed = apply_mask(X, mask)
    assert observed.isna().equals(mask)
    pd.testing.assert_frame_equal(restore_hidden(observed, X, mask), X)
    pd.testing.assert_frame_equal(X, saved)


def test_invalid_restoration_or_mask_fails(credit_frame):
    X, _ = credit_frame
    mask = mcar_mask(X, 0.3, 42)
    with pytest.raises(ValueError):
        apply_mask(X, mask.iloc[::-1])
    with pytest.raises(ValueError):
        apply_mask(X, mask.astype(int))
    with pytest.raises(ValueError):
        restore_hidden(X, X, mask)
    observed = apply_mask(X, mask)
    bad_truth = X.copy()
    i, j = np.argwhere(~mask.to_numpy())[0]
    bad_truth.iloc[i, j] += 1
    with pytest.raises(ValueError, match="Observed values"):
        restore_hidden(observed, bad_truth, mask)
    for rate in (-1, 1.1, np.nan):
        with pytest.raises(ValueError):
            mcar_mask(X, rate, 42)
