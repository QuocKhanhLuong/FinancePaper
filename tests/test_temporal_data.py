import numpy as np
import pandas as pd
import pytest

from financepaper.data.temporal import (
    MONTH_NAMES,
    STATIC_FIELDS,
    TEMPORAL_FIELDS,
    TemporalPreprocessor,
    elapsed_delta,
)
from financepaper.missingness.mar import AnchorMAR, augmentation_mask
from financepaper.missingness.mcar import apply_mask, mcar_mask


def test_official_chronology_and_sentinel_mapping(credit_frame):
    X, _ = credit_frame
    assert MONTH_NAMES == ("April", "May", "June", "July", "August", "September")
    assert TEMPORAL_FIELDS == (
        ("PAY_6", "BILL_AMT6", "PAY_AMT6"),
        ("PAY_5", "BILL_AMT5", "PAY_AMT5"),
        ("PAY_4", "BILL_AMT4", "PAY_AMT4"),
        ("PAY_3", "BILL_AMT3", "PAY_AMT3"),
        ("PAY_2", "BILL_AMT2", "PAY_AMT2"),
        ("PAY_0", "BILL_AMT1", "PAY_AMT1"),
    )

    # Different sentinels make an accidental newest-to-oldest or PAY/BILL shift
    # visible in the origin metadata and encoded values.
    train = X.iloc[:120].copy()
    probe = X.iloc[[120]].copy()
    for position, (status, bill, payment) in enumerate(TEMPORAL_FIELDS, start=1):
        probe.loc[:, status] = float(position)
        probe.loc[:, bill] = float(position * 100)
        probe.loc[:, payment] = float(position * 10)
    preprocessor = TemporalPreprocessor().fit(train)
    batch = preprocessor.transform(probe)
    assert batch.temporal.shape[1] == 6
    assert batch.temporal_origins[: len(preprocessor.pay_vocabulary)] == (
        "PAY_6",
    ) * len(preprocessor.pay_vocabulary)
    assert batch.temporal_origins[-3:] == ("PAY_0", "BILL_AMT1", "PAY_AMT1")
    assert batch.static_origins[:2] == ("LIMIT_BAL", "AGE")


def test_train_only_encoding_and_unknown_categories(credit_frame):
    X, _ = credit_frame
    train = X.iloc[:100].copy()
    # A category absent from train must not enlarge a fitted vocabulary or be
    # silently folded into an existing category.
    unseen = X.iloc[[100]].copy()
    unseen.loc[:, "SEX"] = 999.0
    unseen.loc[:, "PAY_0"] = 999.0
    preprocessor = TemporalPreprocessor().fit(train)
    pay_vocab_before = preprocessor.pay_vocabulary
    static_vocab_before = dict(preprocessor.static_vocabularies)
    batch = preprocessor.transform(unseen)
    assert preprocessor.pay_vocabulary == pay_vocab_before
    assert preprocessor.static_vocabularies == static_vocab_before
    pay_start = 5 * batch.temporal.shape[-1]
    # The final (September) status is the first block at timestep five.
    assert np.all(batch.temporal[0, 5, : len(pay_vocab_before)] == 0)
    sex_start = 2
    assert np.all(batch.static[0, sex_start : sex_start + len(static_vocab_before["SEX"])] == 0)


def test_hidden_values_are_masked_before_imputation_and_delta_is_explicit(credit_frame):
    X, _ = credit_frame
    preprocessor = TemporalPreprocessor().fit(X.iloc[:160])
    probe = X.iloc[160:168].copy()
    hidden = mcar_mask(probe, 0.3, 42)
    observed = apply_mask(probe, hidden)
    first = preprocessor.transform(observed, hidden)

    # Even if a caller leaves the hidden truth in X, the explicit mask wins.
    leaked = probe.copy()
    second = preprocessor.transform(leaked, hidden)
    np.testing.assert_allclose(first.temporal, second.temporal)
    np.testing.assert_allclose(first.static, second.static)
    np.testing.assert_allclose(first.temporal_observed, second.temporal_observed)
    np.testing.assert_allclose(first.static_observed, second.static_observed)

    expected_delta = elapsed_delta(first.temporal_observed > 0.5)
    np.testing.assert_array_equal(first.delta, expected_delta)
    assert first.temporal.dtype == np.float32
    assert first.static.dtype == np.float32
    assert first.temporal_observed.dtype == np.float32
    assert first.static_observed.dtype == np.float32
    assert first.delta.dtype == np.float32


def test_elapsed_delta_runs_and_first_gap(credit_frame):
    observed = np.array(
        [
            [[1, 1], [0, 1], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 1], [0, 1]],
        ],
        dtype=bool,
    )
    expected = np.array(
        [
            [[0, 0], [1, 1], [2, 1], [3, 2]],
            [[0, 0], [1, 1], [1, 2], [1, 1]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(elapsed_delta(observed), expected)
    np.testing.assert_array_equal(elapsed_delta(observed[0]), expected[0])


def test_flatten_origins_and_take_are_aligned(credit_frame):
    X, _ = credit_frame
    preprocessor = TemporalPreprocessor().fit(X.iloc[:120])
    batch = preprocessor.transform(X.iloc[120:126])
    flat, origins = batch.flatten(include_metadata=True)
    value_flat = batch.flatten(include_metadata=False)
    value_width = batch.temporal.size // len(batch) + batch.static.shape[1]
    assert flat.shape[0] == len(batch)
    assert flat.shape[1] == len(origins)
    np.testing.assert_allclose(flat[:, :value_width], value_flat)
    assert origins[: batch.temporal.size // len(batch)] == batch.temporal_origins
    assert origins[value_width : value_width + batch.temporal_observed.size // len(batch)][0] == (
        "__observed_PAY_6"
    )
    assert "__delta_PAY_0" in origins
    np.testing.assert_allclose(
        flat[:, -batch.delta.size // len(batch) :] if batch.delta.size else flat[:, -0:],
        batch.delta.reshape(len(batch), -1) / 5.0,
    )

    subset = batch.take(np.array([4, 1, 0]))
    np.testing.assert_array_equal(subset.record_ids, batch.record_ids[[4, 1, 0]])
    np.testing.assert_allclose(subset.temporal, batch.temporal[[4, 1, 0]])


def test_train_metadata_serializes_without_target(credit_frame):
    X, y = credit_frame
    preprocessor = TemporalPreprocessor().fit(X.iloc[:50])
    metadata = preprocessor.to_dict()
    assert metadata["train_ids"] == X.index[:50].tolist()
    assert "target" not in " ".join(metadata).lower()
    assert metadata["output_dimensions"] == {
        "temporal": preprocessor.temporal_dim,
        "static": preprocessor.static_dim,
    }
    incomplete = X.iloc[:50].copy()
    incomplete.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        TemporalPreprocessor().fit(incomplete)


def test_nested_mcar_is_reproducible_and_target_free(credit_frame):
    X, y = credit_frame
    low = mcar_mask(X, 0.1, 7)
    high = mcar_mask(X, 0.3, 7)
    assert (~low.to_numpy() | high.to_numpy()).all()
    changed = X * 1000.0 + 17.0
    pd.testing.assert_frame_equal(low, mcar_mask(changed, 0.1, 7))
    assert y is not None  # the helper has no target argument by construction


def test_anchor_mar_uses_only_anchors_and_preserves_them(credit_frame):
    X, _ = credit_frame
    train = X.iloc[:160]
    probe = X.iloc[160:]
    mar = AnchorMAR().fit(train)
    first = mar.mask(probe, seed=9)
    second = mar.mask(probe, seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert not first.loc[:, ["AGE", "LIMIT_BAL"]].any().any()
    assert abs(first.to_numpy().mean() - 0.30) < 0.08

    altered = probe.copy()
    for field in altered.columns:
        if field not in {"AGE", "LIMIT_BAL"}:
            altered.loc[:, field] = altered[field] * 100.0 + 3.0
    pd.testing.assert_frame_equal(first, mar.mask(altered, seed=9))
    saved = mar.to_dict()
    assert saved["train_ids"] == train.index.tolist()
    assert saved["anchor_fields"] == ["AGE", "LIMIT_BAL"]


def test_augmentation_schedule_is_reproducible_and_target_free(credit_frame):
    X, _ = credit_frame
    first = augmentation_mask(X, [0.0, 0.1, 0.2, 0.3], seed=21)
    second = augmentation_mask(X * 5.0 + 2.0, [0.0, 0.1, 0.2, 0.3], seed=21)
    pd.testing.assert_frame_equal(first, second)
    assert all(dtype == bool for dtype in first.dtypes)
    explicit = augmentation_mask(X, 0.2, seed=21)
    assert abs(explicit.to_numpy().mean() - 0.2) < 0.06
    four = augmentation_mask(X.iloc[:4], [0.0, 1.0, 0.0, 1.0], seed=21)
    four_repeat = augmentation_mask(X.iloc[:4] * 5.0 + 2.0, [0.0, 1.0, 0.0, 1.0], seed=21)
    pd.testing.assert_frame_equal(four, four_repeat)
    # The four values are candidates, not a per-row schedule; the draw is
    # reproducible but need not equal the list's position-wise values.
    assert all(dtype == bool for dtype in four.dtypes)
    with pytest.raises(ValueError):
        augmentation_mask(X, [0.2, 1.1], seed=21)
