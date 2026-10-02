"""CPU smoke and contract tests for the optional temporal branch."""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from financepaper.models.missing_aware_gru import MissingAwareGRU
from financepaper.models.temporal_gru import TemporalGRU
from financepaper.training.fit import fit_model, predict_logits
from financepaper.training.losses import binary_loss, positive_class_weight


def _batch(n: int = 12, temporal_dim: int = 4, static_dim: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    observed = np.ones((n, 6, 3), dtype=np.float32)
    observed[:, 2, 1] = 0
    observed[:, 4, 0] = rng.integers(0, 2, size=n)
    static_observed = np.ones((n, 5), dtype=np.float32)
    static_observed[:, 2] = rng.integers(0, 2, size=n)
    delta = np.zeros_like(observed)
    for step in range(1, 6):
        delta[:, step] = 1 + (1 - observed[:, step - 1]) * delta[:, step - 1]
    return {
        "temporal": rng.normal(size=(n, 6, temporal_dim)).astype(np.float32),
        "static": rng.normal(size=(n, static_dim)).astype(np.float32),
        "temporal_observed": observed,
        "static_observed": static_observed,
        "delta": delta,
    }


def test_nested_output_and_debug_contract():
    batch = _batch()
    model = MissingAwareGRU(temporal_dim=4, static_dim=6)
    output = model(batch, debug=True)
    assert output["risk"]["logit"].shape == (len(batch["temporal"]),)
    assert output["risk"]["prob_raw"].shape == (len(batch["temporal"]),)
    assert output["risk"]["prob_calibrated"] is None
    assert output["temporal"]["hidden_states"].shape == (len(batch["temporal"]), 6, 64)
    assert output["embeddings"]["static"].shape == (len(batch["temporal"]), 32)
    assert output["embeddings"]["fused"].shape == (len(batch["temporal"]), 64)
    assert output["missingness"]["per_month"].shape == (len(batch["temporal"]), 6)
    assert output["explanation"] is None
    assert output["uncertainty"] is None


def test_vanilla_gru_logs_but_does_not_use_temporal_mask_or_delta():
    batch = _batch()
    model = TemporalGRU(temporal_dim=4, static_dim=6).eval()
    altered = copy.deepcopy(batch)
    altered["temporal_observed"][:] = 0
    altered["delta"][:] = 5
    with torch.no_grad():
        original = model(batch)["risk"]["logit"]
        changed = model(altered)["risk"]["logit"]
    assert torch.equal(original, changed)
    assert not torch.equal(
        model(batch)["missingness"]["fraction"], model(altered)["missingness"]["fraction"]
    )


def test_missing_aware_ablation_input_width_and_branch_options():
    batch = _batch()
    model = MissingAwareGRU(
        temporal_dim=4,
        static_dim=6,
        use_mask=False,
        use_delta=False,
        use_static=False,
        use_temporal=True,
    )
    assert model.temporal_cell.input_size == 4
    output = model(batch)
    assert output["risk"]["logit"].shape == (len(batch["temporal"]),)

    static_only = MissingAwareGRU(
        temporal_dim=4,
        static_dim=6,
        use_temporal=False,
        use_static=True,
    )
    assert static_only(batch)["risk"]["logit"].shape == (len(batch["temporal"]),)


def test_all_structural_ablations_require_canonical_metadata():
    batch = _batch()
    model = MissingAwareGRU(temporal_dim=4, static_dim=6, use_temporal=False)
    for field in ("temporal", "static", "temporal_observed", "static_observed", "delta"):
        incomplete = dict(batch)
        incomplete.pop(field)
        with pytest.raises(KeyError, match=field):
            model(incomplete)

    wrong_steps = dict(batch)
    wrong_steps["temporal"] = wrong_steps["temporal"][:, :5]
    with pytest.raises(ValueError, match="canonical"):
        model(wrong_steps)

    invalid_mask = dict(batch)
    invalid_mask["temporal_observed"] = invalid_mask["temporal_observed"].copy()
    invalid_mask["temporal_observed"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        model(invalid_mask)

    invalid_delta = dict(batch)
    invalid_delta["delta"] = invalid_delta["delta"].copy()
    invalid_delta["delta"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        model(invalid_delta)


def test_augmentation_must_preserve_training_record_ids():
    batch = _batch(8)
    batch["record_ids"] = np.arange(100, 108)
    labels = np.asarray([0, 1] * 4, dtype=np.float32)
    model = MissingAwareGRU(temporal_dim=4, static_dim=6)
    dev = {"complete": batch, "mcar10": batch, "mcar30": batch}
    config = {"epochs": 1, "patience": 1, "batch_size": 8}

    with pytest.raises(ValueError, match="record_ids"):
        fit_model(
            model,
            batch,
            labels,
            dev,
            {name: labels for name in dev},
            lambda _epoch: {key: value for key, value in batch.items() if key != "record_ids"},
            config,
            seed=3,
            device="cpu",
        )

    wrong_order = dict(batch)
    wrong_order["record_ids"] = batch["record_ids"][::-1].copy()
    with pytest.raises(ValueError, match="record_ids"):
        fit_model(
            model,
            batch,
            labels,
            dev,
            {name: labels for name in dev},
            lambda _epoch: wrong_order,
            config,
            seed=3,
            device="cpu",
        )


def test_losses_and_training_cpu_smoke():
    batch = _batch(24)
    labels = np.asarray([0, 1] * 12, dtype=np.float32)
    pos_weight, positive, negative = positive_class_weight(labels)
    assert pos_weight == 1.0
    assert positive == negative == 12
    logits = torch.tensor([-2.0, 1.0, 0.0, 3.0], requires_grad=True)
    for kind in ("bce", "weighted_bce", "focal"):
        value = binary_loss(
            logits,
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            kind=kind,
            pos_weight=pos_weight if kind == "weighted_bce" else None,
        )
        assert torch.isfinite(value)
        value.backward(retain_graph=True)

    model = MissingAwareGRU(temporal_dim=4, static_dim=6)
    dev = {"complete": batch, "mcar10": batch, "mcar30": batch}
    callback_epochs: list[int] = []

    def augment(epoch: int):
        callback_epochs.append(epoch)
        return copy.deepcopy(batch)

    report = fit_model(
        model,
        batch,
        labels,
        dev,
        {name: labels for name in dev},
        augment,
        {
            "epochs": 3,
            "patience": 2,
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "loss": "bce",
        },
        seed=7,
        device="cpu",
    )
    assert report["best_epoch"] in {1, 2, 3}
    assert report["history"]
    assert report["positive_count"] == 12
    assert report["negative_count"] == 12
    assert callback_epochs == list(range(len(callback_epochs)))
    predictions = predict_logits(model, batch, device="cpu", batch_size=5)
    assert predictions.shape == (24,)
    assert torch.isfinite(predictions).all()
