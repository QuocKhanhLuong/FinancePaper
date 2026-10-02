"""Bounded temporal pilot runner.

This module owns the experiment orchestration for the optional PyTorch branch.
The model, preprocessing, missingness, attribution, calibration and metric
implementations live in their focused modules; this file only fixes their
ordering and persists the provenance needed to audit a run.  The test masks are
created after ``frozen_selection.json`` is written.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import yaml

from financepaper.data.schema import FEATURE_NAMES
from financepaper.data.split import split_indices
from financepaper.data.taiwan import SOURCE_URL, WORKBOOK_SHA256, load_taiwan
from financepaper.data.temporal import MONTH_NAMES, TEMPORAL_FIELDS, TemporalBatch, TemporalPreprocessor
from financepaper.evaluation.revision import reason_diagnostics, revision_event
from financepaper.evaluation.temporal_metrics import (
    calibration_bins,
    detailed_prediction_metrics,
    reliability_curve,
)
from financepaper.explanations.reasons import extract_reasons
from financepaper.missingness.mar import AnchorMAR, augmentation_mask
from financepaper.models.logistic import make_logistic
from financepaper.models.xgboost import make_xgboost
from financepaper.training.calibration import PositiveSlopePlattCalibrator


SELECTION_CONDITIONS = ("complete", "mcar10", "mcar30")
ALL_CONDITIONS = ("complete", "mcar10", "mcar30", "mar30")
EXPLANATION_MODELS = ("vanilla_aug_bce", "mask_delta_aug_bce", "xgb_aug")
MODEL_NAMES = (
    "lr_complete",
    "lr_aug",
    "xgb_complete",
    "xgb_aug",
    "vanilla_complete",
    "vanilla_aug_bce",
    "mask_delta_complete",
    "mask_delta_aug_bce",
    "mask_delta_aug_weighted_bce",
    "mask_delta_aug_focal",
)


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar values and non-finite floats to JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Temporal configuration must be a mapping")
    required = {"seed", "data_path", "output_dir", "background_size", "training", "missingness", "explanations", "calibration", "xgboost"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Temporal configuration is missing keys: {missing}")
    if isinstance(config.get("seed"), bool) or not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    config.setdefault("device", "auto")
    config.setdefault("hidden_size", 64)
    config.setdefault("static_hidden", 32)
    config.setdefault("fusion_hidden", 64)
    config.setdefault("dropout", 0.1)
    for key in ("background_size", "hidden_size", "static_hidden", "fusion_hidden"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= float(config["dropout"]) < 1:
        raise ValueError("dropout must lie in [0, 1)")
    training = config["training"]
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping")
    for key in ("epochs", "patience", "batch_size"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    missingness = config["missingness"]
    if not isinstance(missingness, dict):
        raise ValueError("missingness must be a mapping")
    rates = tuple(float(item) for item in missingness.get("augmentation_rates", (0.0, 0.1, 0.2, 0.3)))
    if not rates or any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("augmentation_rates must contain rates in [0, 1]")
    missingness["augmentation_rates"] = list(rates)
    if list(missingness.get("mcar_rates", [])) != [.1, .3]:
        raise ValueError("This pilot fixes MCAR conditions at 10% and 30%")
    if float(missingness.get("mar_rate", .3)) != .3:
        raise ValueError("This pilot fixes MAR at 30% expected all-field missingness")
    explanations = config["explanations"]
    if not isinstance(explanations, dict):
        raise ValueError("explanations must be a mapping")
    if int(explanations.get("k", 0)) < 1:
        raise ValueError("explanations.k must be positive")
    if float(explanations.get("development_coverage_retention", 0)) <= 0:
        raise ValueError("development_coverage_retention must be positive")
    epsilon = float(explanations.get("attribution_tolerance", 1e-6))
    grid = np.asarray(explanations.get("min_attribution_grid", []), dtype=float)
    if (not np.isfinite(epsilon) or epsilon < 0 or grid.ndim != 1 or not len(grid)
            or not np.isfinite(grid).all() or np.any(grid <= epsilon)):
        raise ValueError("Reason magnitudes must exceed the finite revision epsilon")
    if not 0 < float(explanations["development_coverage_retention"]) <= 1:
        raise ValueError("development_coverage_retention must lie in (0,1]")
    for key in ("test_records", "development_records", "ig_batch_size", "ig_steps", "ig_max_steps"):
        if int(explanations.get(key, 0)) <= 0:
            raise ValueError(f"explanations.{key} must be positive")
    return config


def _select_device(requested: str | None = None):
    from financepaper.training.device import select_device

    return select_device(requested)


def print_device_info(device: str | None = None) -> dict[str, Any]:
    """Print device diagnostics without reading a dataset."""

    from financepaper.models.missing_aware_gru import MissingAwareGRU
    from financepaper.training.device import device_info

    # Nominal dimensions verified on the default seed-42 Taiwan training split.
    # Custom encodings/configurations have their exact count in frozen_selection.
    model = MissingAwareGRU(temporal_dim=13, static_dim=15)
    print("Default Taiwan encoding: temporal=13, static=15 (custom runs log exact counts)")
    return device_info(model, device, print_info=True)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _empty_mask(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(False, index=X.index, columns=X.columns, dtype=bool)


def _nested_masks(X: pd.DataFrame, seed: int, mar: AnchorMAR) -> dict[str, pd.DataFrame]:
    """Create complete, nested MCAR and target-independent MAR masks."""

    uniforms = np.random.default_rng(int(seed)).random(X.shape)
    masks = {
        "complete": _empty_mask(X),
        "mcar10": pd.DataFrame(uniforms < 0.10, index=X.index, columns=X.columns, dtype=bool),
        "mcar30": pd.DataFrame(uniforms < 0.30, index=X.index, columns=X.columns, dtype=bool),
        "mar30": mar.mask(X, int(seed) + 1),
    }
    if not np.all(masks["mcar10"].to_numpy() <= masks["mcar30"].to_numpy()):
        raise RuntimeError("nested MCAR masks were not nested")
    return masks


def _save_masks(output: Path, partition: str, masks: Mapping[str, pd.DataFrame]) -> None:
    directory = output / "masks"
    directory.mkdir(parents=True, exist_ok=True)
    for condition, mask in masks.items():
        mask.astype(np.int8).to_csv(directory / f"{partition}_{condition}.csv")
    summary_path = output / "missingness_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for condition, mask in masks.items():
        summary[f"{partition}/{condition}"] = {
            "n_records": len(mask), "hidden_cells": int(mask.to_numpy().sum()),
            "actual_cell_missing_rate": float(mask.to_numpy().mean()),
        }
    _write_json(summary_path, summary)


def _mask_from_csv(path: Path, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.Index(pd.to_numeric(frame.index), name=index.name)
    frame = frame.reindex(index=index, columns=columns)
    if frame.isna().any().any():
        raise ValueError(f"mask file is not aligned with source records: {path}")
    return frame.astype(bool)


def _batch_map(preprocessor: TemporalPreprocessor, X: pd.DataFrame,
               masks: Mapping[str, pd.DataFrame]) -> dict[str, TemporalBatch]:
    return {
        condition: preprocessor.transform(X, hidden_mask=mask)
        for condition, mask in masks.items()
    }


def _missing_fraction(batch: TemporalBatch) -> np.ndarray:
    observed = np.concatenate(
        [batch.temporal_observed.reshape(len(batch), -1), batch.static_observed], axis=1
    )
    return 1.0 - observed.mean(axis=1)


def _flat(batch: TemporalBatch) -> tuple[np.ndarray, tuple[str, ...]]:
    values, origins = batch.flatten(include_metadata=True)
    return np.asarray(values, dtype=np.float32), tuple(origins)


def _fit_flat_model(name: str, X: np.ndarray, y: np.ndarray, config: Mapping[str, Any], seed: int):
    if name.startswith("lr"):
        model = make_logistic(seed, C=1.0, max_iter=3000)
    else:
        options = config["xgboost"]
        model = make_xgboost(
            seed,
            max_depth=int(options.get("max_depth", 3)),
            n_estimators=int(options.get("n_estimators", 200)),
            learning_rate=float(options.get("learning_rate", 0.05)),
            min_child_weight=float(options.get("min_child_weight", 5)),
            subsample=float(options.get("subsample", 1.0)),
            colsample_bytree=float(options.get("colsample_bytree", 1.0)),
            reg_lambda=float(options.get("reg_lambda", 1.0)),
        )
    model.fit(X, y)
    return model


def _flat_logits(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        values = model.decision_function(X)
    else:
        values = model.predict(X, output_margin=True)
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError("flat model returned non-finite logits")
    return values


def _make_augmentation_callback(X_train: pd.DataFrame, preprocessor: TemporalPreprocessor,
                                rates: list[float], seed: int):
    def callback(epoch: int) -> TemporalBatch:
        mask = augmentation_mask(X_train, rates, int(seed) + int(epoch))
        return preprocessor.transform(X_train, hidden_mask=mask)

    return callback


def _construct_neural(name: str, preprocessor: TemporalPreprocessor, config: Mapping[str, Any]):
    from financepaper.models.missing_aware_gru import MissingAwareGRU
    from financepaper.models.temporal_gru import TemporalGRU

    kwargs = dict(
        temporal_dim=preprocessor.temporal_dim,
        static_dim=preprocessor.static_dim,
        hidden_size=int(config.get("hidden_size", 64)),
        static_hidden=int(config.get("static_hidden", 32)),
        fusion_hidden=int(config.get("fusion_hidden", 64)),
        dropout=float(config.get("dropout", 0.1)),
    )
    if name.startswith("vanilla"):
        return TemporalGRU(**kwargs)
    if name.startswith("mask_delta"):
        return MissingAwareGRU(**kwargs, use_mask=True, use_delta=True)
    raise ValueError(f"unsupported neural model name: {name}")


def _fit_neural(name: str, preprocessor: TemporalPreprocessor, train_batch: TemporalBatch,
                y_train: np.ndarray, dev_batches: Mapping[str, TemporalBatch],
                y_dev: Mapping[str, np.ndarray], config: Mapping[str, Any],
                seed: int, device: Any, augmentation_callback: Any):
    from financepaper.training.fit import fit_model

    loss = "focal" if name.endswith("focal") else "weighted_bce" if name.endswith("weighted_bce") else "bce"
    _seed_everything(seed)
    model = _construct_neural(name, preprocessor, config)
    training_config = dict(config["training"])
    training_config["loss"] = loss
    fit = fit_model(
        model,
        train_batch,
        y_train,
        dev_batches,
        y_dev,
        augmentation_callback,
        {"training": training_config},
        seed,
        device=device,
    )
    return {"name": name, "family": "neural", "model": fit["model"], "fit": fit,
            "loss": loss, "augmented": augmentation_callback is not None}


def _raw_logits(fit: Mapping[str, Any], batch: TemporalBatch, device: Any,
                batch_size: int) -> np.ndarray:
    if fit["family"] == "flat":
        return _flat_logits(fit["model"], _flat(batch)[0])
    from financepaper.training.fit import predict_logits

    return predict_logits(
        fit["model"], batch, device=device, batch_size=batch_size
    ).detach().cpu().numpy().astype(float)


def _prediction_threshold(y: np.ndarray, probability: np.ndarray,
                         grid: np.ndarray) -> float:
    best = float(grid[0])
    best_score = float("-inf")
    for threshold in grid:
        score = float(f1_score(y, probability >= threshold, zero_division=0))
        if score > best_score + 1e-12:
            best, best_score = float(threshold), score
    return best


def _fit_calibrators_and_selection(
    fits: Mapping[str, Mapping[str, Any]],
    dev_batches: Mapping[str, TemporalBatch],
    cal_batches: Mapping[str, TemporalBatch],
    y_dev: np.ndarray,
    y_cal: np.ndarray,
    config: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    train_options = config["training"]
    batch_size = int(train_options.get("batch_size", 256))
    grid_config = config["calibration"]
    grid = np.arange(
        float(grid_config.get("threshold_grid_min", 0.01)),
        float(grid_config.get("threshold_grid_max", 0.99)) + 0.5 * float(grid_config.get("threshold_grid_step", 0.01)),
        float(grid_config.get("threshold_grid_step", 0.01)),
    )
    selection: dict[str, Any] = {"version": "temporal-pilot-selection-v1", "test_opened": False, "models": {}}
    for name, fit in fits.items():
        cal_logits = np.concatenate([
            _raw_logits(fit, cal_batches[condition], device, batch_size)
            for condition in SELECTION_CONDITIONS
        ])
        cal_y = np.tile(y_cal, len(SELECTION_CONDITIONS))
        calibrator = PositiveSlopePlattCalibrator().fit(cal_logits, cal_y)
        dev_logits = np.concatenate([
            _raw_logits(fit, dev_batches[condition], device, batch_size)
            for condition in SELECTION_CONDITIONS
        ])
        dev_y = np.tile(y_dev, len(SELECTION_CONDITIONS))
        raw_probability = expit(dev_logits)
        calibrated_probability = calibrator.transform(dev_logits)
        selection["models"][name] = {
            "fit": _fit_summary(fit),
            "calibrator": calibrator.to_dict(),
            "raw_threshold": _prediction_threshold(dev_y, raw_probability, grid),
            "calibrated_threshold": _prediction_threshold(dev_y, calibrated_probability, grid),
            "threshold_grid": grid.tolist(),
        }
    return selection


def _fit_summary(fit: Mapping[str, Any]) -> dict[str, Any]:
    result = {"family": fit["family"], "loss": fit.get("loss"), "augmented": fit.get("augmented")}
    if fit["family"] == "neural":
        details = fit["fit"]
        result.update({
            key: details.get(key)
            for key in ("best_epoch", "best_mean_average_precision", "best_mean_log_loss", "elapsed_seconds", "positive_weight", "positive_count", "negative_count", "device", "parameter_count")
        })
    return _jsonable(result)


def _aggregate_flat_value_phi(phi: np.ndarray, origins: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return value-only original attributions and metadata attributions."""

    result = np.zeros((len(phi), len(FEATURE_NAMES)), dtype=float)
    metadata_columns: list[int] = []
    lookup = {field: position for position, field in enumerate(FEATURE_NAMES)}
    for column, origin in enumerate(origins):
        if origin in lookup:
            result[:, lookup[origin]] += phi[:, column]
        else:
            metadata_columns.append(column)
    metadata = phi[:, metadata_columns] if metadata_columns else np.empty((len(phi), 0), dtype=float)
    return result, metadata


def _tree_shap(model: Any, batch: TemporalBatch, background: np.ndarray,
               origins: tuple[str, ...]) -> dict[str, Any]:
    import shap

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.TreeExplainer(
        model, data=masker, model_output="raw", feature_perturbation="interventional"
    )
    encoded = _flat(batch)[0]
    phi = np.asarray(explainer.shap_values(encoded, check_additivity=True), dtype=float)
    logit = _flat_logits(model, encoded)
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    residual = phi.sum(axis=1) - (logit - expected)
    tolerance = 0.002 + 0.001 * np.abs(logit - expected)
    valid = np.isfinite(residual) & (np.abs(residual) <= tolerance)
    original, metadata = _aggregate_flat_value_phi(phi, origins)
    return {
        "temporal": None,
        "static": None,
        "original": original,
        "metadata": metadata,
        "metadata_origins": np.asarray([origin for origin in origins if origin.startswith("__")], dtype=object),
        "baseline_logit": np.full(len(encoded), expected),
        "input_logit": logit,
        "residual": residual,
        "valid": valid,
        "steps_used": np.zeros(len(encoded), dtype=np.int64),
        "encoded": phi,
        "tolerance": tolerance,
    }


def _batch_replace_values(context_batch: TemporalBatch, complete_batch: TemporalBatch) -> TemporalBatch:
    return replace(context_batch, temporal=complete_batch.temporal, static=complete_batch.static)


def _batch_replace_values_with_full_context(
    context_batch: TemporalBatch, complete_batch: TemporalBatch
) -> TemporalBatch:
    """Use incomplete/imputed values with the complete observation metadata.

    This is a diagnostic endpoint.  It keeps the values seen by the model in
    the incomplete record while replacing the mask and elapsed-time context
    with the all-observed context from the corresponding complete record.  It
    is never used for fitting, calibration, threshold selection, or reason
    extraction.
    """

    return replace(
        complete_batch,
        temporal=context_batch.temporal,
        static=context_batch.static,
    )


def _ig_attributions(fit: Mapping[str, Any], batch: TemporalBatch,
                     baseline_temporal: np.ndarray, baseline_static: np.ndarray,
                     preprocessor: TemporalPreprocessor, config: Mapping[str, Any],
                     device: Any) -> dict[str, Any]:
    from financepaper.explain.integrated_gradients import aggregate_original, integrated_gradients

    result = integrated_gradients(
        fit["model"],
        batch.to_torch(device),
        baseline_temporal,
        baseline_static,
        steps=int(config.get("ig_steps", 32)),
        max_steps=int(config.get("ig_max_steps", 128)),
        batch_size=int(config.get("ig_batch_size", 64)),
    )
    original = aggregate_original(
        result["temporal"], result["static"], preprocessor.temporal_origins,
        preprocessor.static_origins, FEATURE_NAMES,
    )
    return {**result, "original": original, "metadata": None}


def _attribute_model(fit: Mapping[str, Any], batch: TemporalBatch,
                     preprocessor: TemporalPreprocessor, background: Mapping[str, Any],
                     config: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if fit["family"] == "flat":
        return _tree_shap(fit["model"], batch, background["flat"], background["origins"])
    return _ig_attributions(fit, batch, background["temporal"], background["static"], preprocessor, config, device)


def _save_debug_forward(
    fit: Mapping[str, Any], batch: TemporalBatch, output: Path,
    name: str, condition: str, device: Any,
) -> None:
    """Persist compact recurrent diagnostics when explicitly requested."""

    if fit["family"] != "neural":
        return
    import torch

    model = fit["model"]
    with torch.no_grad():
        result = model(batch.to_torch(device), debug=True)
    temporal = result.get("temporal") or {}
    embeddings = result.get("embeddings") or {}
    missingness = result.get("missingness") or {}

    def _array(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
        if value is None:
            return np.empty((0,), dtype=dtype)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    np.savez_compressed(
        output / f"debug_{name}_{condition}.npz",
        record_ids=np.asarray(batch.record_ids),
        hidden_states=_array(temporal.get("hidden_states")),
        temporal_summary_embedding=_array(temporal.get("summary_embedding")),
        static_embedding=_array(embeddings.get("static")),
        fused_embedding=_array(embeddings.get("fused")),
        missing_fraction=_array(missingness.get("fraction")),
        temporal_missing_fraction=_array(missingness.get("temporal_fraction")),
        static_missing_fraction=_array(missingness.get("static_fraction")),
        missing_count_per_month=_array(missingness.get("per_month")),
    )


def _select_reason_rule(dev_attributions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                        config: Mapping[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    k = int(config.get("k", 3))
    grid = [float(value) for value in config.get("min_attribution_grid", [0.001, 0.01, 0.025, 0.05])]
    counts: list[dict[str, Any]] = []
    for magnitude in sorted(grid):
        eligible = 0
        total = 0
        for phi, hidden, valid in dev_attributions:
            for row, row_hidden, row_valid in zip(phi, hidden, valid, strict=True):
                total += 1
                if row_valid and len(extract_reasons(row, row_hidden, FEATURE_NAMES, k, magnitude)) == k:
                    eligible += 1
        counts.append({"min_attribution": magnitude, "n_eligible": eligible,
                       "n_total": total, "coverage": eligible / total if total else 0.0})
    base = counts[0]["n_eligible"] if counts else 0
    required = float(config.get("development_coverage_retention", 0.90)) * base
    selected = max(
        (row["min_attribution"] for row in counts if row["n_eligible"] >= required),
        default=max(grid),
    )
    return float(selected), counts


def _reason_mask_from_source(mask: pd.DataFrame) -> np.ndarray:
    return mask.to_numpy(dtype=bool)


def _select_explanation_rules(
    fits: Mapping[str, Mapping[str, Any]], preprocessor: TemporalPreprocessor,
    dev_batches: Mapping[str, TemporalBatch], dev_masks: Mapping[str, pd.DataFrame],
    background: Mapping[str, Any], config: Mapping[str, Any], device: Any,
    output: Path, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    count = min(int(config.get("development_records", 256)), len(dev_batches["complete"]))
    positions = np.sort(np.random.default_rng(seed + 910_001).choice(len(dev_batches["complete"]), count, replace=False))
    _write_json(output / "explanation_ids.json", {
        "development_record_ids": dev_batches["complete"].record_ids[positions].tolist(),
        "development_positions": positions.tolist(),
    })
    rules: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for name in EXPLANATION_MODELS:
        sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        cache[name] = {}
        for condition in ("mcar10", "mcar30"):
            batch = dev_batches[condition].take(positions)
            attr = _attribute_model(fits[name], batch, preprocessor, background, config, device)
            source_mask = _reason_mask_from_source(dev_masks[condition].iloc[positions])
            sets.append((attr["original"], source_mask, attr["valid"]))
            cache[name][condition] = attr
        magnitude, grid = _select_reason_rule(sets, config)
        rules[name] = {
            "min_attribution": magnitude,
            "grid": grid,
            "development_records": int(count),
            "development_record_ids": dev_batches["complete"].record_ids[positions].tolist(),
        }
    return rules, cache, positions


def _reason_row(name: str, condition: str, record_id: Any, y: int, mask: np.ndarray,
                before: Mapping[str, Any], full: Mapping[str, Any], value: Mapping[str, Any],
                rule: Mapping[str, Any], config: Mapping[str, Any],
                probability_before: float, probability_full: float,
                probability_value: float,
                probability_same_imputed_full_context: float) -> dict[str, Any]:
    k = int(config.get("k", 3))
    min_attr = float(rule["min_attribution"])
    valid_before = bool(before["valid"][0])
    valid_full = bool(full["valid"][0])
    valid_value = bool(value["valid"][0])
    # The primary verification pair is incomplete -> fully observed.  The
    # value-only endpoint is a secondary sensitivity diagnostic and must not
    # discard an otherwise valid primary pair.
    attribution_valid = valid_before and valid_full
    reasons: tuple[str, ...] = ()
    restored_reasons: tuple[str, ...] = ()
    event: bool | None = None
    event_value_only: bool | None = None
    if valid_before:
        reasons = extract_reasons(before["original"][0], mask, FEATURE_NAMES, k, min_attr)
    if valid_full:
        restored_reasons = extract_reasons(full["original"][0], mask, FEATURE_NAMES, k, min_attr)
    preverification_eligible = len(reasons) == k and valid_before
    if preverification_eligible:
        if valid_full:
            event = revision_event(
                reasons, full["original"][0], mask, FEATURE_NAMES,
                k=k, rank_tolerance=int(config.get("rank_tolerance", 0)),
                attribution_tolerance=float(config.get("attribution_tolerance", 1e-6)),
            )
        if valid_value:
            event_value_only = revision_event(
                reasons, value["original"][0], mask, FEATURE_NAMES,
                k=k, rank_tolerance=int(config.get("rank_tolerance", 0)),
                attribution_tolerance=float(config.get("attribution_tolerance", 1e-6)),
            )
    eligible = len(reasons) == k and attribution_valid
    diagnostics = reason_diagnostics(
        before["original"][0], full["original"][0], mask, FEATURE_NAMES, k
    ) if attribution_valid else dict.fromkeys(("observed_attribution_mae", "observed_sign_agreement", "top_k_overlap", "rank_correlation"))
    value_diagnostics = reason_diagnostics(
        before["original"][0], value["original"][0], mask, FEATURE_NAMES, k
    ) if valid_before and valid_value else dict.fromkeys(("observed_attribution_mae", "observed_sign_agreement", "top_k_overlap", "rank_correlation"))
    strength = float(min(before["original"][0][FEATURE_NAMES.index(field)] for field in reasons)) if eligible else None
    reason_scores = [float(before["original"][0][FEATURE_NAMES.index(field)]) for field in reasons]
    reason_signs = [int(np.sign(score)) for score in reason_scores]
    observed_order = np.argsort(-before["original"][0][~mask], kind="stable")
    observed_indices = np.flatnonzero(~mask)
    ranks = {
        int(index): int(position + 1)
        for position, index in enumerate(observed_indices[observed_order])
    }
    reason_ranks = [ranks.get(FEATURE_NAMES.index(field)) for field in reasons]
    return {
        "model": name,
        "condition": condition,
        "record_id": int(record_id),
        "y": int(y),
        "missing_fraction": float(mask.mean()),
        "probability_before": float(probability_before),
        "probability_restored_full": float(probability_full),
        "probability_restored_value_only": float(probability_value),
        "probability_same_imputed_full_context": float(probability_same_imputed_full_context),
        "absolute_probability_shift_full": float(abs(probability_before - probability_full)),
        "absolute_probability_shift_value_only": float(abs(probability_before - probability_value)),
        "absolute_probability_shift_same_imputed_full_context": float(
            abs(probability_before - probability_same_imputed_full_context)
        ),
        "prediction_stable_full": bool(abs(probability_before - probability_full) <= float(config.get("stable_probability_delta", 0.02))),
        "attribution_valid_before": valid_before,
        "attribution_valid_full": valid_full,
        "attribution_valid_value_only": valid_value,
        "attribution_valid": attribution_valid,
        "preverification_eligible": preverification_eligible,
        "eligible": bool(eligible),
        "revision_event": event,
        "revision_event_value_only": event_value_only,
        "reason_strength": strength,
        "reason_scores_before": json.dumps(reason_scores),
        "reason_signs_before": json.dumps(reason_signs),
        "reason_ranks_before": json.dumps(reason_ranks),
        "reasons_before": json.dumps(reasons),
        "reasons_restored_full": json.dumps(restored_reasons),
        "hidden_features": json.dumps([FEATURE_NAMES[index] for index in np.flatnonzero(mask)]),
        **diagnostics,
        "ig_residual_before": float(before["residual"][0]),
        "ig_residual_full": float(full["residual"][0]),
        "ig_residual_value_only": float(value["residual"][0]),
        "value_only_observed_attribution_mae": value_diagnostics["observed_attribution_mae"],
        "value_only_observed_sign_agreement": value_diagnostics["observed_sign_agreement"],
        "value_only_top_k_overlap": value_diagnostics["top_k_overlap"],
        "value_only_rank_correlation": value_diagnostics["rank_correlation"],
        "ig_steps_before": int(before["steps_used"][0]),
        "ig_steps_full": int(full["steps_used"][0]),
        "ig_steps_value_only": int(value["steps_used"][0]),
    }


def _matched_coverage(records: pd.DataFrame, targets: list[float], denominator: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if records.empty:
        return pd.DataFrame(rows)
    for condition in sorted(records["condition"].unique()):
        eligible = records[
            (records["condition"] == condition)
            & records["eligible"]
            & records["attribution_valid"]
        ]
        common_sets = [
            set(eligible.loc[eligible["model"] == model, "record_id"].astype(int))
            for model in EXPLANATION_MODELS
        ]
        common = set.intersection(*common_sets) if common_sets else set()
        subset = eligible[(eligible["condition"] == condition) & eligible["record_id"].isin(common)]
        common_condition = set(subset["record_id"].astype(int))
        n_common = len(common_condition)
        for target in targets:
            # The denominator is the predetermined 400-record explanation
            # subset.  A requested coverage is clipped only by the common
            # eligible set, never redefined as coverage over that subset.
            n_select = min(int(np.floor(float(target) * denominator)), n_common)
            actual_coverage = n_select / denominator if denominator else None
            for model in EXPLANATION_MODELS:
                selected_n = 0
                revision_rate = None
                if n_select:
                    model_rows = subset[subset["model"] == model].sort_values(
                        ["reason_strength", "record_id"], ascending=[False, True]
                    ).head(n_select)
                    selected_n = len(model_rows)
                    revision_rate = float(model_rows["revision_event"].astype(bool).mean()) if selected_n else None
                rows.append({
                    "condition": condition,
                    "target_common_coverage": float(target),
                    "model": model,
                    "common_n": n_common,
                    "selected_n": selected_n,
                    "coverage_denominator": int(denominator),
                    "actual_coverage": actual_coverage,
                    "revision_rate": revision_rate,
                })
    return pd.DataFrame(rows)


def _run_explanations(
    fits: Mapping[str, Mapping[str, Any]], preprocessor: TemporalPreprocessor,
    test_batches: Mapping[str, TemporalBatch], test_masks: Mapping[str, pd.DataFrame],
    background: Mapping[str, Any], rules: Mapping[str, Any], config: Mapping[str, Any],
    device: Any, output: Path, y_test: np.ndarray, test_positions: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for name in EXPLANATION_MODELS:
        fit = fits[name]
        print(f"Explaining {name} on {len(test_positions)} shared test records", flush=True)
        # The restored full-information attribution is identical for every
        # missingness condition.  Compute it once per model so the explanation
        # stage does not repeat the most expensive IG/TreeSHAP call.
        complete = test_batches["complete"].take(test_positions)
        full_attr = _attribute_model(fit, complete, preprocessor, background, config, device)
        for condition in ALL_CONDITIONS:
            context = test_batches[condition].take(test_positions)
            value_only = _batch_replace_values(context, complete)
            if condition == "complete":
                before_attr = full_attr
                value_attr = full_attr
            else:
                before_attr = _attribute_model(fit, context, preprocessor, background, config, device)
                # Vanilla GRU receives values only.  Replacing values while
                # retaining its metadata therefore produces the same output
                # as the already-computed complete attribution.
                value_attr = (
                    full_attr
                    if name.startswith("vanilla")
                    else _attribute_model(fit, value_only, preprocessor, background, config, device)
                )
            same_imputed_full_context = _batch_replace_values_with_full_context(context, complete)
            same_imputed_full_context_logits = _raw_logits(
                fit, same_imputed_full_context, device,
                int(config.get("ig_batch_size", 64)),
            )
            if bool(config.get("debug_tensors", False)):
                _save_debug_forward(fit, context, output, name, condition, device)
            source_mask = _reason_mask_from_source(test_masks[condition].iloc[test_positions])
            before_logits = before_attr["input_logit"]
            full_logits = full_attr["input_logit"]
            value_logits = value_attr["input_logit"]
            np.savez_compressed(
                output / f"attributions_{name}_{condition}.npz",
                record_ids=context.record_ids,
                feature_names=np.asarray(FEATURE_NAMES),
                month_names=np.asarray(MONTH_NAMES),
                before_month_scores=np.column_stack([
                    before_attr["original"][:, [FEATURE_NAMES.index(f) for f in fields]].sum(1)
                    for fields in TEMPORAL_FIELDS
                ]),
                restored_full_month_scores=np.column_stack([
                    full_attr["original"][:, [FEATURE_NAMES.index(f) for f in fields]].sum(1)
                    for fields in TEMPORAL_FIELDS
                ]),
                before_original=before_attr["original"],
                restored_full_original=full_attr["original"],
                restored_value_only_original=value_attr["original"],
                before_valid=before_attr["valid"],
                restored_full_valid=full_attr["valid"],
                restored_value_only_valid=value_attr["valid"],
                before_residual=before_attr["residual"],
                restored_full_residual=full_attr["residual"],
                restored_value_only_residual=value_attr["residual"],
                before_logit=before_logits,
                restored_full_logit=full_logits,
                restored_value_only_logit=value_logits,
                same_imputed_full_context_logit=same_imputed_full_context_logits,
                before_baseline_logit=before_attr["baseline_logit"],
                restored_full_baseline_logit=full_attr["baseline_logit"],
                restored_value_only_baseline_logit=value_attr["baseline_logit"],
                before_steps=before_attr["steps_used"],
                restored_full_steps=full_attr["steps_used"],
                restored_value_only_steps=value_attr["steps_used"],
                metadata_before=before_attr["metadata"] if before_attr["metadata"] is not None else np.empty((len(context), 0)),
                metadata_restored_full=full_attr["metadata"] if full_attr["metadata"] is not None else np.empty((len(context), 0)),
                metadata_restored_value_only=value_attr["metadata"] if value_attr["metadata"] is not None else np.empty((len(context), 0)),
                metadata_origins=np.asarray(before_attr.get("metadata_origins", []), dtype=str),
            )
            for position in range(len(context)):
                rows.append(_reason_row(
                    name, condition, context.record_ids[position], y_test[test_positions[position]],
                    source_mask[position],
                    {key: value[position:position + 1] if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(context) else value for key, value in before_attr.items()},
                    {key: value[position:position + 1] if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(context) else value for key, value in full_attr.items()},
                    {key: value[position:position + 1] if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == len(context) else value for key, value in value_attr.items()},
                    rules[name], config,
                    float(expit(before_logits[position])), float(expit(full_logits[position])),
                    float(expit(value_logits[position])),
                    float(expit(same_imputed_full_context_logits[position])),
                ))
    records = pd.DataFrame(rows)
    records.to_csv(output / "explanations.csv", index=False, float_format="%.12g")
    reliability: list[dict[str, Any]] = []
    for (name, condition), group in records.groupby(["model", "condition"], sort=True):
        curve = reliability_curve(
            group["missing_fraction"].to_numpy(float),
            pd.to_numeric(group["revision_event"], errors="coerce").to_numpy(float),
            group["eligible"].to_numpy(bool),
            total_n=len(group),
        )
        for row in curve:
            reliability.append({"model": name, "condition": condition, **row})
    pd.DataFrame(reliability, columns=["model", "condition", "score_threshold", "n_released",
                                     "coverage", "revision_rate", "n_eligible", "n_total"]).to_csv(
        output / "risk_coverage.csv", index=False, float_format="%.12g")
    matched = _matched_coverage(
        records, [float(value) for value in config.get("matched_coverage_targets", [0.25, 0.5, 1.0])], len(test_positions)
    )
    matched.to_csv(output / "matched_coverage.csv", index=False, float_format="%.12g")
    summary: dict[str, Any] = {}
    risk_rows: list[dict[str, Any]] = []
    frozen = json.loads((output / "frozen_selection.json").read_text())
    prediction_table = pd.read_csv(output / "predictions_test.csv")
    for (name, condition), group in records.groupby(["model", "condition"], sort=True):
        eligible_group = group[group["eligible"] & group["revision_event"].notna()]
        revision_labels = eligible_group["revision_event"].astype(bool).to_numpy()
        score_auc = score_ap = None
        if len(eligible_group) and np.unique(revision_labels).size == 2:
            scores = eligible_group["missing_fraction"].to_numpy(dtype=float)
            score_auc = float(roc_auc_score(revision_labels, scores))
            score_ap = float(average_precision_score(revision_labels, scores))
        summary[f"{name}/{condition}"] = {
            "n_records": int(len(group)),
            "n_attribution_valid": int(group["attribution_valid"].sum()),
            "n_eligible": int(group["eligible"].sum()),
            "coverage_over_predetermined_subset": float(group["eligible"].mean()),
            "n_revision_events": int(eligible_group["revision_event"].astype(bool).sum()),
            "reason_revision_rate": float(eligible_group["revision_event"].astype(bool).mean()) if len(eligible_group) else None,
            "missing_fraction_revision_score_roc_auc": score_auc,
            "missing_fraction_revision_score_average_precision": score_ap,
        }
        item = summary[f"{name}/{condition}"]
        stable = eligible_group[eligible_group["prediction_stable_full"]]
        item.update({
            "n_preverification_eligible": int(group["preverification_eligible"].sum()),
            "n_attribution_failed_primary": int((~group["attribution_valid"]).sum()),
            "n_stable_prediction_eligible": len(stable),
            "n_stable_prediction_revised": int(stable["revision_event"].astype(bool).sum()),
            "revision_rate_given_stable_prediction": float(stable["revision_event"].astype(bool).mean()) if len(stable) else None,
            "empirical_coverage_at_revision_risk_le_10pct": max([
                row["coverage"] for row in reliability
                if row["model"] == name and row["condition"] == condition and row["revision_rate"] <= .10
            ], default=0.0),
            "coverage_is_calibrated_guarantee": False,
        })
        for diagnostic in ("observed_attribution_mae", "observed_sign_agreement", "top_k_overlap", "rank_correlation",
                           "absolute_probability_shift_full", "absolute_probability_shift_value_only"):
            values = group[diagnostic].dropna()
            item[f"mean_{diagnostic}"] = float(values.mean()) if len(values) else None
        selected_prediction = prediction_table[
            (prediction_table.model == name) & (prediction_table.condition == condition)
            & prediction_table.record_id.isin(eligible_group.record_id)
        ]
        for scale in ("raw", "calibrated"):
            threshold = frozen["models"][name][f"{scale}_threshold"]
            item[f"prediction_eligible_{scale}_before"] = detailed_prediction_metrics(
                selected_prediction.y, selected_prediction[f"{scale}_probability"], threshold)
        risk_rows.append({
            "model": name, "condition": condition,
            "n_eligible": int(len(eligible_group)),
            "revision_score_roc_auc": score_auc,
            "revision_score_average_precision": score_ap,
        })
    pd.DataFrame(risk_rows).to_csv(output / "risk_coverage_summary.csv", index=False, float_format="%.12g")
    _write_json(output / "explanations_summary.json", summary)
    return summary


def _save_model_artifact(output: Path, name: str, fit: Mapping[str, Any]) -> None:
    directory = output / "models"
    directory.mkdir(parents=True, exist_ok=True)
    if fit["family"] == "flat":
        joblib.dump(fit["model"], directory / f"{name}.joblib")
    else:
        import torch

        torch.save({"state_dict": fit["model"].state_dict(), "name": name}, directory / f"{name}.pt")


def _source_manifest(root: Path) -> dict[str, Any]:
    paths = sorted((root / "src" / "financepaper").rglob("*.py"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += [root / "pyproject.toml", root / "uv.lock", root / "configs" / "temporal_pilot.yaml"]
    paths += [root / "docs" / "TEMPORAL_MODEL_SPEC.md", root / "docs" / "TEMPORAL_MODEL_RESEARCH.md"]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "joblib", "pyyaml", "xlrd", "torch")
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def _artifact_manifest(output: Path) -> dict[str, str]:
    """Hash completed run artifacts, excluding the manifest being written."""

    artifacts: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        artifacts[str(path.relative_to(output))] = sha256(path.read_bytes()).hexdigest()
    return artifacts


def _plot_results_if_available(output: Path) -> None:
    """Render the bounded pilot plots when the evaluation helper is present."""

    from financepaper.experiments.temporal_plots import plot_temporal_results
    plot_temporal_results(output)


def _write_manifest(output: Path, dataset: Any, indices: Mapping[str, np.ndarray],
                    preprocessor: TemporalPreprocessor, config: Mapping[str, Any],
                    device: Any, started: float, status: str) -> None:
    root = Path(__file__).resolve().parents[3]
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
        git_commit = git.stdout.strip() if git.returncode == 0 else None
        git_dirty = bool(dirty.stdout) if dirty.returncode == 0 else None
    except OSError:
        git_commit = git_dirty = None
    _write_json(output / "manifest.json", {
        "status": status,
        "scope": "one-seed temporal pilot; descriptive reason-revision diagnostics; no release guarantee",
        "dataset_path": str(dataset.source_path),
        "dataset_sha256": dataset.sha256,
        "official_workbook": dataset.sha256 == WORKBOOK_SHA256,
        "dataset_source_url": SOURCE_URL,
        "n_records": len(dataset.X),
        "feature_names": list(FEATURE_NAMES),
        "split_sizes": {name: len(values) for name, values in indices.items()},
        "unused_splits": ["risk_calibration"],
        "device": str(device),
        "native_thread_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": _package_versions(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "preprocessor": preprocessor.to_dict(),
        "source_sha256": _source_manifest(root),
        "artifact_sha256": _artifact_manifest(output),
        "elapsed_seconds": time.perf_counter() - started,
    })


def _prepare_context(config: Mapping[str, Any], output: Path, *, persist: bool = True):
    dataset = load_taiwan(Path(config["data_path"]))
    indices = split_indices(dataset.y, int(config["seed"]))
    train = indices["train"]
    X_train, y_train = dataset.X.iloc[train], dataset.y.iloc[train].to_numpy(dtype=int)
    preprocessor = TemporalPreprocessor().fit(X_train, train_ids=X_train.index)
    if persist:
        preprocessor_path = output / "preprocessor.json"
        _write_json(preprocessor_path, preprocessor.to_dict())
        assignments = pd.DataFrame({"record_id": dataset.X.index, "split": ""})
        for name, positions in indices.items():
            assignments.loc[positions, "split"] = name
        assignments.to_csv(output / "split_assignments.csv", index=False)
    mar = AnchorMAR().fit(X_train, rate=float(config["missingness"].get("mar_rate", 0.30)), train_ids=X_train.index)
    if persist:
        _write_json(output / "mar_state.json", mar.to_dict())
    train_batch = preprocessor.transform(X_train)
    partitions = {
        "development": (indices["development"], int(config["seed"]) + 1000),
        "probability_calibration": (indices["probability_calibration"], int(config["seed"]) + 2000),
    }
    condition_batches: dict[str, dict[str, TemporalBatch]] = {}
    condition_masks: dict[str, dict[str, pd.DataFrame]] = {}
    for partition, (positions, mask_seed) in partitions.items():
        X = dataset.X.iloc[positions]
        masks = _nested_masks(X, mask_seed, mar)
        condition_masks[partition] = masks
        if persist:
            _save_masks(output, partition, masks)
        condition_batches[partition] = _batch_map(preprocessor, X, masks)
    return {
        "dataset": dataset,
        "indices": indices,
        "preprocessor": preprocessor,
        "mar": mar,
        "X_train": X_train,
        "y_train": y_train,
        "train_batch": train_batch,
        "dev_batches": condition_batches["development"],
        "cal_batches": condition_batches["probability_calibration"],
        "dev_masks": condition_masks["development"],
        "cal_masks": condition_masks["probability_calibration"],
        "y_dev": dataset.y.iloc[indices["development"]].to_numpy(dtype=int),
        "y_cal": dataset.y.iloc[indices["probability_calibration"]].to_numpy(dtype=int),
    }


def _fit_all(context: Mapping[str, Any], config: Mapping[str, Any], output: Path, device: Any):
    seed = int(config["seed"])
    preprocessor = context["preprocessor"]
    train_batch = context["train_batch"]
    X_train, y_train = context["X_train"], context["y_train"]
    train_flat, origins = _flat(train_batch)
    background_positions = np.sort(np.random.default_rng(seed + 300).choice(len(train_batch), int(config["background_size"]), replace=False))
    background_batch = train_batch.take(background_positions)
    background_flat, _ = _flat(background_batch)
    background = {
        "positions": background_positions,
        "record_ids": background_batch.record_ids,
        "flat": background_flat,
        "origins": origins,
        "temporal": background_batch.temporal.mean(axis=0),
        "static": background_batch.static.mean(axis=0),
    }
    _write_json(output / "background_ids.json", {"record_ids": background_batch.record_ids.tolist(), "positions": background_positions.tolist()})
    rates = list(config["missingness"].get("augmentation_rates", [0.0, 0.1, 0.2, 0.3]))
    first_aug_mask = augmentation_mask(X_train, rates, seed + 700_000)
    first_aug_batch = preprocessor.transform(X_train, hidden_mask=first_aug_mask)
    first_aug_flat, _ = _flat(first_aug_batch)
    first_aug_mask.astype(np.int8).to_csv(output / "training_augmentation_first_mask.csv")
    _save_masks(output, "train", {"complete": _empty_mask(X_train), "augmentation_epoch0": first_aug_mask})
    fits: dict[str, Any] = {}
    flat_specs = (
        ("lr_complete", train_flat), ("lr_aug", first_aug_flat),
        ("xgb_complete", train_flat), ("xgb_aug", first_aug_flat),
    )
    for name, matrix in flat_specs:
        print(f"Fitting {name}", flush=True)
        # Keep random streams matched across flat controls; feature exposure is
        # the intended difference between complete and one-copy augmentation.
        model = _fit_flat_model(name, matrix, y_train, config, seed)
        fits[name] = {"name": name, "family": "flat", "model": model, "augmented": name.endswith("_aug"), "loss": None}
        _save_model_artifact(output, name, fits[name])
    y_dev_map = {condition: context["y_dev"] for condition in context["dev_batches"]}
    callback = _make_augmentation_callback(X_train, preprocessor, rates, seed + 700_000)
    neural_specs = (
        ("vanilla_complete", None), ("vanilla_aug_bce", callback),
        ("mask_delta_complete", None), ("mask_delta_aug_bce", callback),
        ("mask_delta_aug_weighted_bce", callback), ("mask_delta_aug_focal", callback),
    )
    for name, augmentation_callback in neural_specs:
        print(f"Fitting {name} on {device}", flush=True)
        fit = _fit_neural(
            name, preprocessor, train_batch, y_train, context["dev_batches"], y_dev_map,
            config, seed, device, augmentation_callback,
        )
        fits[name] = fit
        print(f"  best epoch {fit['fit']['best_epoch']}, development AP {fit['fit']['best_ap']:.4f}", flush=True)
        _write_json(output / f"{name}_history.json", fit["fit"]["history"])
        _save_model_artifact(output, name, fit)
    return fits, background


def _test_context(context: Mapping[str, Any], config: Mapping[str, Any], output: Path):
    positions = context["indices"]["test"]
    X_test = context["dataset"].X.iloc[positions]
    masks = _nested_masks(X_test, int(config["seed"]) + 3000, context["mar"])
    _save_masks(output, "test", masks)
    batches = _batch_map(context["preprocessor"], X_test, masks)
    return X_test, masks, batches, context["dataset"].y.iloc[positions].to_numpy(dtype=int)


def _run_predictions(
    fits: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any], config: Mapping[str, Any],
    output: Path, device: Any, selection: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, pd.DataFrame], Mapping[str, TemporalBatch], np.ndarray, np.ndarray]:
    X_test, masks, batches, y_test = _test_context(context, config, output)
    batch_size = int(config["training"].get("batch_size", 256))
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows_out: list[dict[str, Any]] = []
    for name, fit in fits.items():
        model_selection = selection["models"][name]
        calibrator = PositiveSlopePlattCalibrator.from_dict(model_selection["calibrator"])
        complete_logits = _raw_logits(fit, batches["complete"], device, batch_size)
        complete_raw = expit(complete_logits)
        complete_calibrated = calibrator.transform(complete_logits)
        for condition in ALL_CONDITIONS:
            logits = _raw_logits(fit, batches[condition], device, batch_size)
            raw = expit(logits)
            calibrated = calibrator.transform(logits)
            value_only_batch = _batch_replace_values(batches[condition], batches["complete"])
            value_only_logits = _raw_logits(fit, value_only_batch, device, batch_size)
            value_only_raw = expit(value_only_logits)
            value_only_calibrated = calibrator.transform(value_only_logits)
            same_imputed_full_context = _batch_replace_values_with_full_context(
                batches[condition], batches["complete"]
            )
            same_imputed_full_context_logits = _raw_logits(
                fit, same_imputed_full_context, device, batch_size
            )
            same_imputed_full_context_raw = expit(same_imputed_full_context_logits)
            same_imputed_full_context_calibrated = calibrator.transform(
                same_imputed_full_context_logits
            )
            missing_fraction = _missing_fraction(batches[condition])
            for index, record_id in enumerate(X_test.index):
                rows.append({
                    "model": name, "condition": condition, "record_id": int(record_id),
                    "y": int(y_test[index]), "raw_logit": float(logits[index]),
                    "raw_probability": float(raw[index]), "calibrated_probability": float(calibrated[index]),
                    "absolute_raw_probability_shift_from_complete": float(abs(raw[index] - complete_raw[index])),
                    "absolute_calibrated_probability_shift_from_complete": float(abs(calibrated[index] - complete_calibrated[index])),
                    "value_only_probability": float(value_only_raw[index]),
                    "value_only_calibrated_probability": float(value_only_calibrated[index]),
                    "absolute_raw_probability_shift_value_only": float(abs(raw[index] - value_only_raw[index])),
                    "absolute_calibrated_probability_shift_value_only": float(abs(calibrated[index] - value_only_calibrated[index])),
                    "same_imputed_full_context_probability": float(same_imputed_full_context_raw[index]),
                    "same_imputed_full_context_calibrated_probability": float(same_imputed_full_context_calibrated[index]),
                    "absolute_raw_probability_shift_same_imputed_full_context": float(
                        abs(raw[index] - same_imputed_full_context_raw[index])
                    ),
                    "absolute_calibrated_probability_shift_same_imputed_full_context": float(
                        abs(calibrated[index] - same_imputed_full_context_calibrated[index])
                    ),
                    "missing_fraction": float(missing_fraction[index]),
                })
            for probability_name, probability, threshold_key in (
                ("raw", raw, "raw_threshold"), ("calibrated", calibrated, "calibrated_threshold")
            ):
                metrics = detailed_prediction_metrics(
                    y_test, probability, threshold=float(model_selection[threshold_key]),
                    bins=int(config["calibration"].get("bins", 10)),
                )
                metric_rows.append({"model": name, "condition": condition, "probability": probability_name, **metrics})
                for row in calibration_bins(y_test, probability, int(config["calibration"].get("bins", 10))):
                    calibration_rows_out.append({"model": name, "condition": condition, "probability": probability_name, **row})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "predictions_test.csv", index=False, float_format="%.12g")
    _write_json(output / "prediction_metrics.json", metric_rows)
    pd.DataFrame(metric_rows).to_csv(output / "prediction_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(calibration_rows_out).to_csv(output / "calibration_test.csv", index=False, float_format="%.12g")
    predictions.groupby(["model", "condition"], sort=True)[
        ["absolute_raw_probability_shift_from_complete", "absolute_calibrated_probability_shift_from_complete",
         "absolute_raw_probability_shift_value_only", "absolute_calibrated_probability_shift_value_only",
         "absolute_raw_probability_shift_same_imputed_full_context",
         "absolute_calibrated_probability_shift_same_imputed_full_context", "missing_fraction"]
    ].mean().reset_index().to_csv(output / "prediction_shift_summary.csv", index=False, float_format="%.12g")
    return {"metrics": metric_rows, "predictions": predictions}, masks, batches, y_test, X_test.index.to_numpy()


def _load_fits_for_explain(output: Path, context: Mapping[str, Any], config: Mapping[str, Any], device: Any):
    fits: dict[str, Any] = {}
    for name in MODEL_NAMES:
        if name.startswith(("lr", "xgb")):
            model = joblib.load(output / "models" / f"{name}.joblib")
            fits[name] = {"name": name, "family": "flat", "model": model,
                          "augmented": name.endswith("_aug"), "loss": None}
        else:
            import torch

            model = _construct_neural(name, context["preprocessor"], config)
            payload = torch.load(output / "models" / f"{name}.pt", map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
            model.to(device).eval()
            fits[name] = {"name": name, "family": "neural", "model": model,
                          "augmented": "_aug_" in name, "loss": "focal" if name.endswith("focal") else "weighted_bce" if name.endswith("weighted_bce") else "bce"}
    train_batch = context["train_batch"]
    background_ids = json.loads((output / "background_ids.json").read_text())
    background_batch = train_batch.take(np.asarray(background_ids["positions"], dtype=int))
    flat, origins = _flat(background_batch)
    return fits, {
        "positions": np.asarray(background_ids["positions"], dtype=int),
        "record_ids": background_batch.record_ids,
        "flat": flat,
        "origins": origins,
        "temporal": background_batch.temporal.mean(axis=0),
        "static": background_batch.static.mean(axis=0),
    }


def _run_new(config_path: Path, config: dict[str, Any], output: Path, device: Any, stage: str) -> Path:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=True))
    context = _prepare_context(config, output)
    fits, background = _fit_all(context, config, output, device)
    rules, _, _ = _select_explanation_rules(
        fits, context["preprocessor"], context["dev_batches"], context["dev_masks"],
        background, config["explanations"], device, output, int(config["seed"]),
    )
    selection = _fit_calibrators_and_selection(
        fits, context["dev_batches"], context["cal_batches"], context["y_dev"],
        context["y_cal"], config, device,
    )
    selection["reason_rules"] = rules
    _write_json(output / "frozen_selection.json", selection)
    if stage == "predict":
        _, _, _, _, test_ids = _run_predictions(fits, context, config, output, device, selection)
        test_count = min(int(config["explanations"].get("test_records", 400)), len(test_ids))
        test_positions = np.sort(np.random.default_rng(int(config["seed"]) + 920_001).choice(len(test_ids), test_count, replace=False))
        _write_json(output / "explanation_ids.json", {
            "development_record_ids": json.loads((output / "explanation_ids.json").read_text())["development_record_ids"],
            "test_record_ids": test_ids[test_positions].tolist(),
            "test_positions": test_positions.tolist(),
        })
        _write_manifest(output, context["dataset"], context["indices"], context["preprocessor"], config, device, started, "prediction_complete")
        return output
    prediction, masks, batches, y_test, test_ids = _run_predictions(fits, context, config, output, device, selection)
    test_count = min(int(config["explanations"].get("test_records", 400)), len(test_ids))
    test_positions = np.sort(np.random.default_rng(int(config["seed"]) + 920_001).choice(len(test_ids), test_count, replace=False))
    _write_json(output / "explanation_ids.json", {
        "development_record_ids": json.loads((output / "explanation_ids.json").read_text())["development_record_ids"],
        "test_record_ids": test_ids[test_positions].tolist(),
        "test_positions": test_positions.tolist(),
    })
    _run_explanations(
        fits, context["preprocessor"], batches, masks, background, rules,
        config["explanations"], device, output, y_test, test_positions,
    )
    _plot_results_if_available(output)
    _write_manifest(output, context["dataset"], context["indices"], context["preprocessor"], config, device, started, "completed")
    return output


def _run_existing(config_path: Path, output: Path, device: Any) -> Path:
    started = time.perf_counter()
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("explain stage requires a completed prediction manifest")
    config_file = output / "resolved_config.yaml"
    config = _load_config(config_file if config_file.exists() else config_path)
    prior_manifest = json.loads(manifest_path.read_text())
    if prior_manifest.get("status") == "completed":
        raise FileExistsError("explain stage is already complete for this output directory")
    if prior_manifest.get("status") != "prediction_complete":
        raise ValueError("Only a completed prediction stage can be resumed")
    for relative, expected in prior_manifest.get("artifact_sha256", {}).items():
        artifact = output / relative
        if not artifact.is_file() or sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prediction artifact changed: {relative}")
    root = Path(__file__).resolve().parents[3]
    recorded_sources = prior_manifest.get("source_sha256")
    if recorded_sources and recorded_sources != _source_manifest(root):
        raise ValueError("source hashes differ from the prediction-stage manifest")
    required = [
        output / "frozen_selection.json", output / "explanation_ids.json",
        output / "background_ids.json", output / "predictions_test.csv",
    ] + [output / "models" / f"{name}.{'joblib' if name.startswith(('lr', 'xgb')) else 'pt'}"
         for name in MODEL_NAMES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"prediction artifacts required for explain stage are missing: {missing}")
    context = _prepare_context(config, output, persist=False)
    if prior_manifest.get("dataset_sha256") not in {None, context["dataset"].sha256}:
        raise ValueError("dataset hash differs from the prediction-stage manifest")
    fits, background = _load_fits_for_explain(output, context, config, device)
    selection = json.loads((output / "frozen_selection.json").read_text())
    masks = {
        condition: _mask_from_csv(output / "masks" / f"test_{condition}.csv", context["dataset"].X.iloc[context["indices"]["test"]].index, context["dataset"].X.columns)
        for condition in ALL_CONDITIONS
    }
    X_test = context["dataset"].X.iloc[context["indices"]["test"]]
    batches = _batch_map(context["preprocessor"], X_test, masks)
    y_test = context["dataset"].y.iloc[context["indices"]["test"]].to_numpy(dtype=int)
    ids = json.loads((output / "explanation_ids.json").read_text())
    test_positions = np.asarray(ids["test_positions"], dtype=int)
    _run_explanations(
        fits, context["preprocessor"], batches, masks, background,
        selection["reason_rules"], config["explanations"], device, output, y_test, test_positions,
    )
    _plot_results_if_available(output)
    _write_manifest(output, context["dataset"], context["indices"], context["preprocessor"], config, device, started, "completed")
    return output


def run_temporal_pilot(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    data_path: Path | None = None,
    device: str | None = None,
    stage: str = "all",
) -> Path:
    """Run the prediction stage and, optionally, the fixed explanation stage."""

    if stage not in {"predict", "explain", "all"}:
        raise ValueError("stage must be predict, explain, or all")
    config = _load_config(Path(config_path))
    if data_path is not None:
        config["data_path"] = str(data_path)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if device is not None:
        config["device"] = device
    output = Path(config["output_dir"])
    selected_device = _select_device(config.get("device"))
    if stage == "explain":
        if not output.exists() or not any(output.iterdir()):
            raise FileNotFoundError("explain stage requires an existing prediction output directory")
        return _run_existing(Path(config_path), output, selected_device)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use a new directory")
    return _run_new(Path(config_path), config, output, selected_device, stage)


__all__ = ["print_device_info", "run_temporal_pilot"]
