import numpy as np
import pytest

torch = pytest.importorskip("torch")

from financepaper.evaluation.temporal_metrics import (  # noqa: E402
    detailed_prediction_metrics,
    reliability_curve,
)
from financepaper.data.schema import FEATURE_NAMES  # noqa: E402
from financepaper.data.temporal import TemporalPreprocessor  # noqa: E402
from financepaper.explain.integrated_gradients import (  # noqa: E402
    aggregate_original,
    integrated_gradients,
)
from financepaper.training.calibration import (  # noqa: E402
    PositiveSlopePlattCalibrator,
    fit_platt,
)


class LinearRisk(torch.nn.Module):
    """Small structured-output model for exact IG tests."""

    def __init__(self, temporal_weight=2.0, static_weight=3.0):
        super().__init__()
        self.temporal_weight = float(temporal_weight)
        self.static_weight = float(static_weight)

    def forward(self, batch):
        temporal = batch["temporal"].sum(dim=(1, 2)) * self.temporal_weight
        static = batch["static"].sum(dim=1) * self.static_weight
        return {"risk": {"logit": temporal + static}}


class MaskContextRisk(torch.nn.Module):
    """Value attribution must not absorb a fixed mask/delta context."""

    def forward(self, batch):
        value = batch["temporal"].sum(dim=(1, 2))
        observed = batch["temporal_observed"].sum(dim=(1, 2))
        delta = batch["delta"].sum(dim=(1, 2))
        return {"risk": {"logit": value + 10.0 * observed + delta}}


def _batch(n=2, temporal_dim=2, static_dim=3, observed_value=1.0):
    return {
        "temporal": torch.full((n, 6, temporal_dim), observed_value, dtype=torch.float32),
        "static": torch.full((n, static_dim), observed_value, dtype=torch.float32),
        "temporal_observed": torch.ones((n, 6, 3), dtype=torch.float32),
        "static_observed": torch.ones((n, 5), dtype=torch.float32),
        "delta": torch.zeros((n, 6, 3), dtype=torch.float32),
    }


def test_integrated_gradients_linear_completeness_and_encoded_shapes():
    batch = _batch(n=3, temporal_dim=2, static_dim=3, observed_value=0.5)
    result = integrated_gradients(
        LinearRisk(), batch, torch.zeros(6, 2), torch.zeros(3),
        steps=32, max_steps=32, batch_size=2,
    )
    np.testing.assert_allclose(result["temporal"], 2.0 * batch["temporal"].numpy(), atol=1e-6)
    np.testing.assert_allclose(result["static"], 3.0 * batch["static"].numpy(), atol=1e-6)
    np.testing.assert_allclose(result["residual"], 0.0, atol=1e-6)
    assert result["valid"].tolist() == [True, True, True]
    assert result["steps_used"].tolist() == [32, 32, 32]
    assert result["baseline_logit"].shape == (3,)
    assert result["input_logit"].shape == (3,)


def test_integrated_gradients_holds_mask_context_fixed():
    batch = _batch(n=2, temporal_dim=1, static_dim=1, observed_value=0.25)
    batch["temporal_observed"][0].fill_(0.0)
    batch["temporal_observed"][1].fill_(1.0)
    result = integrated_gradients(
        MaskContextRisk(), batch, torch.zeros(6, 1), torch.zeros(1),
        steps=32, max_steps=32,
    )
    # The fixed mask affects the baseline/input logits, but not the value
    # slope along the attribution path.
    np.testing.assert_allclose(result["temporal"][:, :, 0], 0.25, atol=1e-6)
    assert result["input_logit"][1] - result["input_logit"][0] > 50.0
    np.testing.assert_allclose(result["residual"], 0.0, atol=1e-6)


def test_aggregate_original_sums_one_hot_dimensions_and_months():
    temporal = np.array([
        [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]],
        [[-1.0, 0.5, 2.0], [2.0, 1.5, -3.0]],
    ])
    static = np.array([[4.0, 5.0], [-2.0, 0.25]])
    output = aggregate_original(
        temporal,
        static,
        [["pay", "pay", "bill"], ["pay", "pay", "bill"]],
        ["limit", "bill"],
        ["pay", "bill", "limit"],
    )
    np.testing.assert_allclose(output, [[3.0, 7.0, 4.0], [3.0, -0.75, -2.0]])
    np.testing.assert_allclose(output.sum(axis=1), temporal.sum(axis=(1, 2)) + static.sum(axis=1))


def test_aggregate_original_accepts_temporal_preprocessor_flat_origins(credit_frame):
    X, _ = credit_frame
    preprocessor = TemporalPreprocessor().fit(X.iloc[:120])
    batch = preprocessor.transform(X.iloc[120:124])
    temporal = np.arange(batch.temporal.size, dtype=float).reshape(batch.temporal.shape) / 10.0
    static = np.arange(batch.static.size, dtype=float).reshape(batch.static.shape) / 7.0
    output = aggregate_original(
        temporal, static, batch.temporal_origins, batch.static_origins, FEATURE_NAMES
    )
    assert output.shape == (len(batch), len(FEATURE_NAMES))
    np.testing.assert_allclose(
        output.sum(axis=1), temporal.sum(axis=(1, 2)) + static.sum(axis=1)
    )


def test_positive_slope_calibration_is_monotone_finite_and_roundtrips():
    logits = np.array([-3.0, -1.0, 0.5, 1.5, 3.0])
    labels = np.array([0, 0, 1, 1, 1])
    original_logits = logits.copy()
    calibrator = fit_platt(logits, labels)
    probabilities = calibrator.transform(logits)
    assert calibrator.slope > 0
    assert np.all(np.isfinite(probabilities))
    assert np.all(np.diff(probabilities) >= 0)
    np.testing.assert_array_equal(logits, original_logits)
    restored = PositiveSlopePlattCalibrator.from_dict(calibrator.to_dict())
    np.testing.assert_allclose(restored.predict_proba(logits), probabilities)


def test_detailed_metrics_include_classwise_support_and_edge_calibration_bins():
    metrics = detailed_prediction_metrics(
        np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0]), threshold=0.5, bins=2
    )
    assert metrics["n"] == 4
    assert set(metrics["classwise"]) == {"0", "1"}
    assert metrics["classwise"]["0"]["support"] == 2
    assert metrics["classwise"]["1"]["support"] == 2
    assert [row["n"] for row in metrics["calibration_bins"]] == [2, 2]
    assert metrics["ece"] == pytest.approx(0.0)


def test_reliability_curve_sorts_lower_scores_and_keeps_ties_together():
    curve = reliability_curve(
        np.array([0.2, 0.1, 0.1, 0.4]),
        np.array([1, 0, 1, 0]),
        np.array([True, True, True, False]),
    )
    assert [row["n_released"] for row in curve] == [2, 3]
    assert [row["score_threshold"] for row in curve] == [0.1, 0.2]
    assert curve[0]["revision_rate"] == pytest.approx(0.5)
