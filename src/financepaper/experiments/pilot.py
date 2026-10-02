"""One-seed complete/MCAR pilot, with selection frozen before test evaluation."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost.callback import EarlyStopping
import yaml

from financepaper.data.schema import FEATURE_NAMES
from financepaper.data.split import split_indices
from financepaper.data.taiwan import SOURCE_URL, WORKBOOK_SHA256, load_taiwan
from financepaper.evaluation.metrics import calibration_rows, summarize_records
from financepaper.evaluation.revision import reason_diagnostics, revision_event
from financepaper.explanations.reasons import extract_reasons
from financepaper.explanations.shap_values import FixedShapExplainer
from financepaper.missingness.mcar import apply_mask, mcar_mask, restore_hidden
from financepaper.models.logistic import make_logistic
from financepaper.models.xgboost import make_xgboost
from financepaper.preprocessing.pipelines import encoded_feature_origins, make_preprocessor
from .config import load_config


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_reason_magnitude(phi, mask, config) -> tuple[float, list[dict]]:
    """Development-only, no restoration labels or test information involved."""
    candidates = []
    for magnitude in sorted(config["min_attribution_grid"]):
        count = sum(bool(extract_reasons(row, hidden, FEATURE_NAMES, config["k"], magnitude))
                    for row, hidden in zip(phi, mask, strict=True))
        candidates.append({"min_attribution": magnitude, "n_eligible": count,
                           "coverage": count / len(phi)})
    required = config["development_coverage_retention"] * candidates[0]["n_eligible"]
    selected = max(row["min_attribution"] for row in candidates if row["n_eligible"] >= required)
    return selected, candidates


def select_prediction_threshold(y, probability) -> float:
    grid = np.arange(1, 100) / 100
    scores = [f1_score(y, probability >= threshold, zero_division=0) for threshold in grid]
    return float(grid[np.argmax(scores)])  # Lower threshold wins an exact tie.


def fit_baselines(X_train, y_train, X_dev, y_dev, config):
    """All fitted statistics see train only, including within LR's inner CV."""
    seed, lr_config = config["seed"], config["logistic"]
    lr_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("model", make_logistic(seed, max_iter=lr_config["max_iter"])),
    ])
    search = GridSearchCV(
        lr_pipeline, {"model__C": lr_config["c_values"]}, scoring="neg_log_loss",
        cv=StratifiedKFold(lr_config["cv_folds"], shuffle=True, random_state=seed),
        n_jobs=1, refit=True, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X_train, y_train)
    lr = search.best_estimator_
    lr_selection = {
        "C": float(search.best_params_["model__C"]),
        "training_cv": [
            {"C": float(params["model__C"]), "mean_log_loss": float(-score)}
            for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], strict=True)
        ],
    }
    preprocessor = make_preprocessor(scale_numeric=False)
    train_encoded = preprocessor.fit_transform(X_train)
    dev_encoded = preprocessor.transform(X_dev)
    tree_config = config["xgboost"]
    parameters = {key: value for key, value in tree_config.items()
                  if key not in ("max_depths", "early_stopping_rounds")}
    candidates, best_model, best_loss = [], None, float("inf")
    for depth in sorted(tree_config["max_depths"]):
        model = make_xgboost(
            seed, max_depth=depth, **parameters,
            callbacks=[EarlyStopping(rounds=tree_config["early_stopping_rounds"], save_best=True)],
        )
        model.fit(train_encoded, y_train, eval_set=[(dev_encoded, y_dev)], verbose=False)
        loss = float(log_loss(y_dev, model.predict_proba(dev_encoded)[:, 1]))
        candidates.append({"max_depth": depth, "development_log_loss": loss,
                           "retained_trees": int(model.get_booster().num_boosted_rounds())})
        if loss < best_loss:
            best_loss, best_model = loss, model
    tree = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    return {"logistic": (lr, lr_selection), "xgboost": (tree, {
        "max_depth": int(best_model.max_depth),
        "retained_trees": int(best_model.get_booster().num_boosted_rounds()),
        "development_search": candidates,
    })}


def _record_rows(model_name, condition, y, before_probability, restored_probability,
                 before_phi, restored_phi, mask, magnitude, config):
    reasons_config = config["reasons"]
    hidden = mask.to_numpy()
    rows = []
    for i, record_id in enumerate(mask.index):
        reasons = extract_reasons(before_phi[i], hidden[i], FEATURE_NAMES,
                                  reasons_config["k"], magnitude)
        event = revision_event(
            reasons, restored_phi[i], hidden[i], FEATURE_NAMES,
            k=reasons_config["k"], rank_tolerance=reasons_config["rank_tolerance"],
            attribution_tolerance=reasons_config["attribution_tolerance"],
        )
        shift = float(abs(before_probability[i] - restored_probability[i]))
        restored_reasons = extract_reasons(restored_phi[i], hidden[i], FEATURE_NAMES,
                                           reasons_config["k"], magnitude)
        rows.append({
            "model": model_name, "condition": condition, "record_id": int(record_id),
            "y": int(y.iloc[i]), "missing_fraction": float(hidden[i].mean()),
            "probability_before": float(before_probability[i]),
            "probability_restored": float(restored_probability[i]),
            "absolute_probability_shift": shift,
            "prediction_stable": shift <= config["diagnostics"]["stable_probability_delta"],
            "eligible": bool(reasons), "revision_event": event,
            "reasons_before": json.dumps(reasons), "reasons_restored": json.dumps(restored_reasons),
            **reason_diagnostics(before_phi[i], restored_phi[i], hidden[i], FEATURE_NAMES, reasons_config["k"]),
        })
    return pd.DataFrame(rows)


def _plots(records, calibration, output, stable_delta):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x_limit = max(0.025, float(records.absolute_probability_shift.max()) * 1.05)
    y_limit = max(0.01, float(records.observed_attribution_mae.max()) * 1.05)
    for axis, name in zip(axes, ("logistic", "xgboost"), strict=True):
        subset = records[(records.model == name) & (records.condition == "mcar") & records.eligible]
        revised = subset.revision_event.astype(bool)
        for status, color, label in ((False, "#2563eb", "Reasons retained"), (True, "#d97706", "Reasons revised")):
            points = subset[revised == status]
            axis.scatter(points.absolute_probability_shift, points.observed_attribution_mae,
                         s=8, alpha=0.45, c=color, label=label, rasterized=True)
        axis.axvline(stable_delta, color="black", ls="--", lw=1)
        axis.set(title=name, xlabel="Absolute default-probability shift",
                 ylabel="Mean |SHAP shift| on observed fields (log-odds)",
                 xlim=(0, x_limit), ylim=(0, y_limit))
        axis.legend(fontsize=8)
    fig.savefig(output / "prediction_vs_explanation.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 5), layout="constrained")
    axis.plot([0, 1], [0, 1], "k--", lw=1)
    for (name, condition), group in calibration.groupby(["model", "condition"], sort=True):
        points = group[group.n > 0]
        axis.plot(points.mean_probability, points.default_fraction, "o-", label=f"{name}: {condition}", ms=3)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
             ylabel="Observed default fraction", title="Uncalibrated test probabilities")
    axis.legend(fontsize=8)
    fig.savefig(output / "calibration.png", dpi=160)
    plt.close(fig)


def run_pilot(config_path: Path, *, data_path: Path | None = None, output_dir: Path | None = None) -> Path:
    """Persist an auditable run; refuse to overwrite an existing nonempty run."""
    config = load_config(config_path)
    if data_path is not None:
        config["data_path"] = str(data_path)
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; use --output with a new directory")
    dataset = load_taiwan(Path(config["data_path"]))
    indices = split_indices(dataset.y, config["seed"])
    if config["background_size"] > len(indices["train"]):
        raise ValueError("Background size exceeds the training split")
    output.mkdir(parents=True, exist_ok=True)
    # SHAP imports plotting modules while constructing TreeExplainer, so configure
    # writable headless caches before any explainer is created.
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / "plot_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / "plot_cache"))
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    assignments = pd.DataFrame({"record_id": dataset.X.index, "split": ""})
    for name, positions in indices.items():
        assignments.loc[positions, "split"] = name
    assignments.to_csv(output / "split_assignments.csv", index=False)
    train, dev = indices["train"], indices["development"]
    X_train, y_train = dataset.X.iloc[train], dataset.y.iloc[train]
    X_dev, y_dev = dataset.X.iloc[dev], dataset.y.iloc[dev]
    seed = config["seed"]
    dev_mask = mcar_mask(X_dev, config["missing_rate"], seed + 100)
    X_dev_observed = apply_mask(X_dev, dev_mask)
    dev_mask.astype(int).to_csv(output / "development_masks.csv")
    background_positions = np.sort(np.random.default_rng(seed + 300).choice(len(train), config["background_size"], replace=False))
    background = X_train.iloc[background_positions]
    pd.DataFrame({"record_id": background.index}).to_csv(output / "background_ids.csv", index=False)
    print("Fitting train-only LR CV and compact XGBoost search...", flush=True)
    with threadpool_limits(limits=1):
        fitted = fit_baselines(X_train, y_train, X_dev, y_dev, config)
    explainers, selection = {}, {}
    for name, (pipeline, model_selection) in fitted.items():
        preprocessor, model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
        encoded_background = preprocessor.transform(background)
        explainer = FixedShapExplainer(model, encoded_background, encoded_feature_origins(preprocessor))
        explainers[name] = explainer
        print(f"Selecting development thresholds for {name}...", flush=True)
        phi_dev = explainer.explain(preprocessor.transform(X_dev_observed))
        magnitude, magnitude_grid = select_reason_magnitude(phi_dev, dev_mask.to_numpy(), config["reasons"])
        threshold = select_prediction_threshold(y_dev, pipeline.predict_proba(X_dev_observed)[:, 1])
        selection[name] = {
            **model_selection, "min_attribution": magnitude,
            "reason_magnitude_development_grid": magnitude_grid,
            "prediction_threshold": threshold, "shap_expected_value": explainer.expected_value,
            "encoded_feature_origins": encoded_feature_origins(preprocessor),
        }
        joblib.dump(pipeline, output / f"{name}_pipeline.joblib")
        np.save(output / f"{name}_background.npy", encoded_background)
    # This artifact is written before constructing test masks or computing test predictions.
    write_json(output / "frozen_selection.json", selection)
    print("Selection frozen. Evaluating complete and MCAR test records...", flush=True)
    test = indices["test"]
    X_test, y_test = dataset.X.iloc[test], dataset.y.iloc[test]
    test_mask = mcar_mask(X_test, config["missing_rate"], seed + 200)
    test_mask.astype(int).to_csv(output / "test_masks.csv")
    X_observed = apply_mask(X_test, test_mask)
    all_records, all_calibration, summaries, examples = [], [], {}, []
    for name, (pipeline, _) in fitted.items():
        print(f"Explaining {name}: {len(test)} incomplete and restored records...", flush=True)
        preprocessor = pipeline.named_steps["preprocessor"]
        phi_before = explainers[name].explain(preprocessor.transform(X_observed))
        probability_before = pipeline.predict_proba(X_observed)[:, 1]
        restored = restore_hidden(X_observed, X_test, test_mask)  # Evaluation boundary.
        phi_restored = explainers[name].explain(preprocessor.transform(restored))
        probability_restored = pipeline.predict_proba(restored)[:, 1]
        np.savez_compressed(output / f"{name}_attributions.npz",
                            before=phi_before, restored=phi_restored,
                            record_ids=X_test.index.to_numpy(), feature_names=np.array(FEATURE_NAMES))
        for condition, before, probability, mask in (
            ("complete", phi_restored, probability_restored, test_mask & False),
            ("mcar", phi_before, probability_before, test_mask),
        ):
            records = _record_rows(name, condition, y_test, probability, probability_restored,
                                   before, phi_restored, mask, selection[name]["min_attribution"], config)
            all_records.append(records)
            summaries[f"{name}/{condition}"] = summarize_records(records, selection[name]["prediction_threshold"])
            all_calibration.extend({"model": name, "condition": condition, **row}
                                   for row in calibration_rows(y_test, probability, config["diagnostics"]["calibration_bins"]))
            if condition == "mcar":
                chosen = records[records.eligible & records.prediction_stable & (records.revision_event == True)]
                chosen = chosen.sort_values(["absolute_probability_shift", "record_id"]).head(config["diagnostics"]["examples_per_model"])
                for _, row in chosen.iterrows():
                    position = X_test.index.get_loc(row.record_id)
                    reasons = json.loads(row.reasons_before)
                    examples.append({
                        "model": name, "record_id": int(row.record_id),
                        "probability_before": row.probability_before,
                        "probability_restored": row.probability_restored,
                        "absolute_probability_shift": row.absolute_probability_shift,
                        "hidden_features": [FEATURE_NAMES[j] for j in np.flatnonzero(test_mask.iloc[position])],
                        "reasons_before": reasons, "reasons_restored": json.loads(row.reasons_restored),
                        "released_reason_attributions": [
                            {"feature": field, "before": float(phi_before[position, FEATURE_NAMES.index(field)]),
                             "restored": float(phi_restored[position, FEATURE_NAMES.index(field)])}
                            for field in reasons
                        ],
                    })
    records = pd.concat(all_records, ignore_index=True)
    records.to_csv(output / "records.csv", index=False, float_format="%.12g")
    calibration = pd.DataFrame(all_calibration)
    calibration.to_csv(output / "calibration.csv", index=False, float_format="%.12g")
    write_json(output / "summary.json", summaries)
    write_json(output / "stable_prediction_revised_examples.json", examples)
    _plots(records, calibration, output, config["diagnostics"]["stable_probability_delta"])
    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / "src" / "financepaper").rglob("*.py"))
    source_files += [root / "pyproject.toml", root / "uv.lock"]
    source_hashes = {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                     for path in source_files if path.exists()}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    manifest = {
        "status": "completed", "scope": "single-seed simple-imputation pilot; no calibrated release guarantee",
        "dataset_path": str(dataset.source_path), "dataset_sha256": dataset.sha256,
        "official_workbook": dataset.sha256 == WORKBOOK_SHA256, "dataset_source_url": SOURCE_URL,
        "n_records": len(dataset.X), "feature_names": list(FEATURE_NAMES),
        "split_sizes": {name: len(value) for name, value in indices.items()},
        "unused_splits": ["probability_calibration", "risk_calibration"],
        "training_condition": "complete training records; median/mode fitted on training predictors only",
        "shap_scale": "default-class raw log-odds", "shap_background": "fixed training-only interventional",
        "seeds": {"split_and_model": seed, "development_mask": seed + 100,
                  "test_mask": seed + 200, "background": seed + 300},
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {package: version(package) for package in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "matplotlib", "joblib", "pyyaml", "xlrd")},
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "source_sha256": source_hashes,
        "artifact_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                            for path in sorted(output.iterdir()) if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)  # Completion marker, written last.
    print(json.dumps({key: {metric: row[metric] for metric in (
        "n_eligible", "coverage", "reason_revision_rate", "n_stable_prediction_revised")}
        for key, row in summaries.items()}, indent=2), flush=True)
    return output
