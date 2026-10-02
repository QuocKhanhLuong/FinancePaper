import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from financepaper.data.schema import TARGET
from financepaper.data.split import split_indices
from financepaper.experiments.config import load_config
from financepaper.experiments.pilot import run_pilot
from financepaper.preprocessing.pipelines import FeaturePreprocessor


def test_offline_pilot_reproducibility_and_leakage_boundaries(tmp_path, credit_frame, monkeypatch):
    X, y = credit_frame
    data = X.copy()
    data.insert(0, "ID", X.index)
    data[TARGET] = y
    source = tmp_path / "synthetic.csv"
    data.to_csv(source, index=False)
    config = load_config(Path(__file__).parents[1] / "configs/pilot.yaml")
    config["data_path"] = str(source)
    config["background_size"] = 12
    config["logistic"]["c_values"] = [0.1, 1.0]
    config["logistic"]["cv_folds"] = 2
    config["xgboost"]["max_depths"] = [2]
    config["xgboost"]["n_estimators"] = 12
    config["xgboost"]["early_stopping_rounds"] = 3
    config_file = tmp_path / "pilot.yaml"
    config_file.write_text(yaml.safe_dump(config))
    split = split_indices(y, config["seed"])
    training_ids = set(X.iloc[split["train"]].index)
    observed_fits = []
    original_fit = FeaturePreprocessor.fit_transform

    def spy(self, values, *args, **kwargs):
        observed_fits.append(set(values.index))
        assert set(values.index) <= training_ids
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(FeaturePreprocessor, "fit_transform", spy)
    first = run_pilot(config_file, output_dir=tmp_path / "first")
    second = run_pilot(config_file, output_dir=tmp_path / "second")
    assert observed_fits and any(len(ids) < len(training_ids) for ids in observed_fits)
    for name in ("records.csv", "summary.json", "frozen_selection.json", "test_masks.csv", "split_assignments.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert not manifest["official_workbook"]
    assert manifest["unused_splits"] == ["probability_calibration", "risk_calibration"]
    assert set(pd.read_csv(first / "background_ids.csv").record_id) <= training_ids
    records = pd.read_csv(first / "records.csv")
    assert len(records) == 4 * len(split["test"])
    assert records.loc[~records.eligible, "revision_event"].isna().all()
    complete = records[records.condition == "complete"]
    assert not complete.loc[complete.eligible, "revision_event"].astype(bool).any()
    lr = records[records.model == "logistic"]
    assert not lr.loc[lr.eligible, "revision_event"].astype(bool).any()
    with pytest.raises(FileExistsError):
        run_pilot(config_file, output_dir=first)
    # Alter all unused calibration and test predictors. Selection must stay frozen.
    heldout = np.concatenate([split[key] for key in ("probability_calibration", "risk_calibration", "test")])
    data.iloc[heldout, data.columns.get_loc("LIMIT_BAL")] += 999
    data.to_csv(source, index=False)
    changed = run_pilot(config_file, output_dir=tmp_path / "heldout_changed")
    assert (first / "frozen_selection.json").read_bytes() == (changed / "frozen_selection.json").read_bytes()


@pytest.mark.parametrize("section,key,value", [
    (None, "missing_rate", 1.1), (None, "seed", True),
    ("reasons", "min_attribution_grid", [0.0]),
    ("reasons", "rank_tolerance", -1), ("logistic", "c_values", []),
    ("xgboost", "learning_rate", float("nan")),
])
def test_invalid_config_fails(tmp_path, section, key, value):
    config = load_config(Path(__file__).parents[1] / "configs/pilot.yaml")
    target = config if section is None else config[section]
    target[key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)
