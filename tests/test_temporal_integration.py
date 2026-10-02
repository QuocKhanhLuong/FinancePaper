"""Cross-module scientific contracts for the experimental temporal branch."""

import numpy as np
import pandas as pd
import pytest
import copy
import json
from pathlib import Path
import yaml

torch = pytest.importorskip("torch")

from financepaper.experiments.temporal_pilot import EXPLANATION_MODELS, _matched_coverage
from financepaper.data.temporal import TemporalPreprocessor
from financepaper.explain.integrated_gradients import integrated_gradients
from financepaper.models.missing_aware_gru import MissingAwareGRU
from financepaper.training.losses import binary_loss
from financepaper.data.schema import TARGET
from financepaper.data.split import split_indices
from financepaper.experiments import temporal_pilot as pilot
from financepaper.training.calibration import PositiveSlopePlattCalibrator


def test_matched_coverage_intersects_each_condition_and_all_models():
    # Different eligible customer sets in each condition; unioning across
    # conditions would manufacture eligibility for some model/customer pairs.
    available = {
        "mcar10": ({1, 2, 3}, {2, 3, 4}, {2, 3, 5}),
        "mcar30": ({1, 2}, {1, 3}, {1, 4}),
    }
    rows = []
    for condition, sets in available.items():
        for model, ids in zip(EXPLANATION_MODELS, sets, strict=True):
            for record_id in range(1, 6):
                rows.append(dict(model=model, condition=condition, record_id=record_id,
                                 eligible=record_id in ids, attribution_valid=True,
                                 reason_strength=float(record_id), revision_event=False))
    records = pd.DataFrame(rows)
    result = _matched_coverage(records, [.25, .5, 1.], denominator=5)
    for condition, count in [("mcar10", 2), ("mcar30", 1)]:
        group = result[result.condition == condition]
        assert (group.common_n == count).all()
        # Each target is a fraction of all five records, capped by common
        # eligibility. It is not a fraction of the already selected subset.
        assert sorted(group.selected_n.tolist()) == sorted(
            [min(int(np.floor(target * 5)), count) for target in [.25, .5, 1.]] * 3
        )
        np.testing.assert_allclose(group.actual_coverage, group.selected_n / 5)
    # An entire missing/failed model means no shared comparison is possible.
    records.loc[records.model == EXPLANATION_MODELS[-1], "eligible"] = False
    empty = _matched_coverage(records, [1.], denominator=5)
    assert (empty.common_n == 0).all()
    assert (empty.selected_n == 0).all()
    assert empty.revision_rate.isna().all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_actual_mps_forward_backward_and_attribution(credit_frame):
    X, y = credit_frame
    preprocessor = TemporalPreprocessor().fit(X.iloc[:100])
    batch = preprocessor.transform(X.iloc[100:108])
    torch.manual_seed(45)
    cpu = MissingAwareGRU(batch.temporal_dim, batch.static_dim,
                          hidden_size=8, static_hidden=4, fusion_hidden=8).eval()
    mps = copy.deepcopy(cpu).to("mps")
    cpu_input, gpu_input = batch.to_torch("cpu"), batch.to_torch("mps")
    reference = cpu(cpu_input)["risk"]["logit"].detach().numpy()
    actual = mps(gpu_input)["risk"]["logit"]
    np.testing.assert_allclose(actual.detach().cpu().numpy(), reference, atol=2e-5, rtol=2e-5)
    binary_loss(actual, y.iloc[100:108].to_numpy(dtype=np.float32)).backward()
    gradients = [p.grad for p in mps.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    result = integrated_gradients(
        mps, gpu_input, batch.temporal.mean(0), batch.static.mean(0),
        steps=32, max_steps=128, batch_size=8,
    )
    assert result["valid"].all()
    assert next(mps.parameters()).device.type == "mps"


def test_offline_temporal_pipeline_freezes_before_test_and_preserves_leakage_boundaries(
    tmp_path, credit_frame, monkeypatch,
):
    X, y = credit_frame
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/temporal_pilot.yaml").read_text())
    config.update(background_size=8, hidden_size=8, static_hidden=4, fusion_hidden=8)
    config["training"].update(epochs=1, patience=1, batch_size=64)
    config["xgboost"]["n_estimators"] = 8
    config["explanations"].update(test_records=8, development_records=8, ig_steps=4,
                                   ig_max_steps=8, ig_batch_size=8, debug_tensors=True)
    data = X.copy()
    data.insert(0, "ID", X.index)
    data[TARGET] = y
    source = tmp_path / "synthetic.csv"
    data.to_csv(source, index=False)
    config["data_path"] = str(source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    split = split_indices(y, config["seed"])
    train_ids = set(X.iloc[split["train"]].index)
    fit_original = TemporalPreprocessor.fit
    calibrate_original = PositiveSlopePlattCalibrator.fit
    test_original = pilot._test_context
    fit_counts, calibration_counts = [], []

    def checked_fit(self, frame, *args, **kwargs):
        assert set(frame.index) == train_ids
        fit_counts.append(len(frame))
        return fit_original(self, frame, *args, **kwargs)

    def checked_calibration(self, logits, labels):
        np.testing.assert_array_equal(labels, np.tile(y.iloc[split["probability_calibration"]], 3))
        calibration_counts.append(len(labels))
        return calibrate_original(self, logits, labels)

    def checked_test(context, settings, output):
        frozen = json.loads((output / "frozen_selection.json").read_text())
        assert frozen["test_opened"] is False
        assert len(frozen["models"]) == 10 and len(frozen["reason_rules"]) == 3
        return test_original(context, settings, output)

    monkeypatch.setattr(TemporalPreprocessor, "fit", checked_fit)
    monkeypatch.setattr(PositiveSlopePlattCalibrator, "fit", checked_calibration)
    monkeypatch.setattr(pilot, "_test_context", checked_test)
    first = pilot.run_temporal_pilot(config_path, output_dir=tmp_path / "first", device="cpu")
    predictions = pd.read_csv(first / "predictions_test.csv")
    assert len(predictions) == 10 * 4 * len(split["test"])
    complete = predictions[predictions.condition == "complete"]
    assert (complete.absolute_raw_probability_shift_from_complete == 0).all()
    explanations = pd.read_csv(first / "explanations.csv")
    assert len(explanations) == 3 * 4 * 8
    complete_e = explanations[explanations.condition == "complete"]
    assert not complete_e.revision_event.dropna().astype(bool).any()
    assert explanations.loc[~explanations.eligible, "revision_event"].isna().all()
    assert (first / "temporal_calibration.png").is_file()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "completed" and not manifest["official_workbook"]
    assert manifest["unused_splits"] == ["risk_calibration"]
    assert manifest["artifact_sha256"]
    with pytest.raises(FileExistsError):
        pilot.run_temporal_pilot(config_path, output_dir=first, device="cpu")

    # Neither held-out test features nor the reserved reason-risk partition may
    # change training, model selection, calibration or explanation thresholds.
    unused = np.concatenate([split["test"], split["risk_calibration"]])
    data.iloc[unused, data.columns.get_loc("LIMIT_BAL")] += 30
    data.to_csv(source, index=False)
    second = pilot.run_temporal_pilot(config_path, output_dir=tmp_path / "changed", device="cpu")

    def scientific_selection(path):
        frozen = json.loads((path / "frozen_selection.json").read_text())
        for entry in frozen["models"].values():
            entry["fit"].pop("elapsed_seconds", None)
        return frozen

    assert scientific_selection(first) == scientific_selection(second)
    for name in pilot.MODEL_NAMES:
        if name.startswith(("vanilla", "mask_delta")):
            a = torch.load(first / "models" / f"{name}.pt", weights_only=True)["state_dict"]
            b = torch.load(second / "models" / f"{name}.pt", weights_only=True)["state_dict"]
            assert all(torch.equal(a[key], b[key]) for key in a)
    assert len(fit_counts) == 2 and len(calibration_counts) == 20
