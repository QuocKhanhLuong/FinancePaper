"""Fail-closed configuration for the deliberately bounded first pilot."""

from pathlib import Path
import math

import yaml


def load_config(path: Path) -> dict:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    expected = {
        "seed", "data_path", "output_dir", "missing_rate", "background_size",
        "logistic", "xgboost", "reasons", "diagnostics",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Configuration must have exactly these keys: {sorted(expected)}")

    def number(value, lower, upper=math.inf, integer=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper
                or (integer and not isinstance(value, int))):
            raise ValueError(f"Invalid numeric configuration value: {value!r}")

    def keys(section, names):
        if not isinstance(section, dict) or set(section) != set(names.split()):
            raise ValueError(f"Expected section keys: {names}")

    def grid(values, lower, upper=math.inf, integer=False):
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError("Parameter grids must be nonempty and unique")
        for value in values:
            number(value, lower, upper, integer)

    number(config["seed"], 0, 2**32 - 1000, integer=True)
    number(config["missing_rate"], 0, 1)
    number(config["background_size"], 1, integer=True)
    for key in ("data_path", "output_dir"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty path string")
    lr, xgb, reasons, diagnostics = (config[key] for key in ("logistic", "xgboost", "reasons", "diagnostics"))
    keys(lr, "c_values cv_folds max_iter")
    grid(lr["c_values"], 1e-12)
    number(lr["cv_folds"], 2, integer=True)
    number(lr["max_iter"], 1, integer=True)
    keys(xgb, "max_depths n_estimators learning_rate early_stopping_rounds min_child_weight subsample colsample_bytree reg_lambda")
    grid(xgb["max_depths"], 1, 16, integer=True)
    for key in ("n_estimators", "early_stopping_rounds"):
        number(xgb[key], 1, integer=True)
    for key in ("learning_rate", "subsample", "colsample_bytree"):
        number(xgb[key], 1e-12, 1)
    for key in ("min_child_weight", "reg_lambda"):
        number(xgb[key], 0)
    keys(reasons, "k min_attribution_grid development_coverage_retention attribution_tolerance rank_tolerance")
    number(reasons["k"], 1, 23, integer=True)
    number(reasons["rank_tolerance"], 0, integer=True)
    number(reasons["attribution_tolerance"], 0)
    grid(reasons["min_attribution_grid"], 0)
    if min(reasons["min_attribution_grid"]) <= reasons["attribution_tolerance"]:
        raise ValueError("Reason magnitude grid must exceed the numerical attribution tolerance")
    number(reasons["development_coverage_retention"], 1e-12, 1)
    keys(diagnostics, "stable_probability_delta calibration_bins examples_per_model")
    number(diagnostics["stable_probability_delta"], 0, 1)
    number(diagnostics["calibration_bins"], 2, integer=True)
    number(diagnostics["examples_per_model"], 0, integer=True)
    return config
