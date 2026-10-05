"""Freeze inputs, then run the bounded exploratory prediction-control audit.

No dataset download, predictor training, new SHAP computation or test loading.
Use --stage freeze before --stage run; hashes reject subsequent drift.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from financepaper.reliability.distribution_audit import (
    CURRENT_KEYS, current_features, sha256, split_roles, validate_current,
    verification_labels, verify_hashes,
)
from financepaper.training.calibration import PositiveSlopePlattCalibrator

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "scripts/run_prediction_distribution_audit.py",
    "src/financepaper/reliability/distribution_audit.py",
    "src/financepaper/reliability/prediction_distribution.py",
    "src/financepaper/reliability/validation.py",
    "src/financepaper/training/calibration.py",
    "src/financepaper/data/schema.py",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def read_current(folder):
    with np.load(folder / "current.npz", allow_pickle=False) as z:
        return {name: z[name] for name in CURRENT_KEYS}


def build(current):
    return current_features(*(current[key] for key in (
        "probability", "completion_probability", "hidden", "phi", "completion_phi", "valid", "completion_valid")))


def freeze(config, config_path, out):
    out.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha256(ROOT / name) for name in [*SOURCE_PATHS, config["protocol"], str(config_path.relative_to(ROOT))]}
    for name, hash_key in (("partitions", "partition_sha256"), ("predictor", "predictor_sha256")):
        hashes[config[name]] = sha256(ROOT / config[name])
        if hashes[config[name]] != config[hash_key]:
            raise ValueError(f"Frozen {name} hash mismatch")
    partitions = json.loads((ROOT / config["partitions"]).read_text())
    parts = list(partitions.values())
    if any(len(set(ids)) != len(ids) for ids in parts):
        raise ValueError("Duplicate original partition ID")
    if sum(map(len, parts)) != len(set(sum(parts, []))):
        raise ValueError("Original fold partitions overlap")
    roles = split_roles(partitions["revision_calibration"], partitions["release_calibration"],
                        seed=config["seed"], fit_count=config["fit_count"])
    preflight = []
    for role in ("revision_calibration", "release_calibration"):
        previous_ids = None
        for condition in config["conditions"]:
            folder = ROOT / config["cache"] / role / condition
            receipt = json.loads((folder / "complete.json").read_text())
            for key, filename in (("current_only", "current.npz"), ("verification_only", "verification_only.npz")):
                name = str((folder / filename).relative_to(ROOT))
                actual = sha256(ROOT / name)  # Bytes only; no restored arrays or labels opened.
                if receipt[key] != {name: actual}:
                    raise ValueError(f"Historical receipt mismatch: {name}")
                hashes[name] = actual
            receipt_name = str((folder / "complete.json").relative_to(ROOT))
            hashes[receipt_name] = sha256(folder / "complete.json")
            current = read_current(folder)
            validate_current(current, partitions[role], partitions["predictor_train"])
            if receipt["customers"] != len(current["record_ids"]):
                raise ValueError("Receipt customer count mismatch")
            if previous_ids is not None and not np.array_equal(previous_ids, current["record_ids"]):
                raise ValueError("Condition row alignment mismatch")
            previous_ids = current["record_ids"]
            small = {name: values[:16] for name, values in current.items()}
            start = time.perf_counter()
            p, e, _, _ = build(small)
            preflight.append({"role": role, "condition": condition, "rows": 16,
                              "p_columns": p.shape[1], "e_columns": e.shape[1],
                              "feature_seconds": time.perf_counter()-start})
    write_json(out / "freeze.json", {"git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")), "hashes": hashes,
        "roles": {key: val.tolist() for key, val in roles.items()}, "preflight": preflight,
        "protocol": config, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "python": platform.python_version(), "machine": platform.machine(),
        "verification_opened": False, "status": "FROZEN_CURRENT_ONLY"})
    print(json.dumps({"status": "FROZEN_CURRENT_ONLY", "preflight_rows": 16,
                      "roles": {key: len(val) for key, val in roles.items()}}))


def score_transform(name, values):
    return values if name in ("entropy", "prediction_variance") else logit(np.clip(values, 1e-6, 1-1e-6))


def run(config, out):
    if (out / "started.json").exists():
        raise ValueError("Run already started; use a new output directory, never overwrite")
    frozen = json.loads((out / "freeze.json").read_text())
    if frozen["protocol"] != config:
        raise ValueError("Configuration changed after freeze")
    verify_hashes(frozen["hashes"], ROOT)
    start = time.perf_counter()
    write_json(out / "started.json", {"git_commit": git("rev-parse", "HEAD"), "freeze_sha256": sha256(out / "freeze.json")})

    def budget():
        if time.perf_counter()-start > config["budget_seconds"]:
            raise TimeoutError("Preregistered CPU wall-clock budget exceeded")

    records, matrices_p, matrices_e, counts = [], [], [], []
    for cache_role in ("revision_calibration", "release_calibration"):
        for condition in config["conditions"]:
            budget()
            folder = ROOT / config["cache"] / cache_role / condition
            current = read_current(folder)
            p, e, scores, eligible = build(current)
            # Only now open offline labels. Never pass these arrays to build().
            with np.load(folder / "verification_only.npz", allow_pickle=False) as z:
                target, target_valid = verification_labels(current["phi"], current["hidden"], z["phi"], z["valid"])
            ids = current["record_ids"]
            keep = eligible & target_valid
            role = np.full(len(ids), "evaluation", dtype="U11")
            if cache_role == "revision_calibration":
                role = np.where(np.isin(ids, frozen["roles"]["fit"]), "fit", "calibration")
            for name in np.unique(role):
                rows = role == name
                counts.append({"role": name, "condition": condition, "total": int(rows.sum()),
                    "current_ineligible": int((rows & ~eligible).sum()),
                    "restored_invalid_among_eligible": int((rows & eligible & ~target_valid).sum()),
                    "eligible": int((rows & keep).sum()), "events": int(target[rows & keep].sum())})
            records.append(pd.DataFrame({"id": ids[keep], "condition": condition,
                "role": role[keep], "target": target[keep], **{k:v[keep] for k,v in scores.items()}}))
            matrices_p.append(p[keep]); matrices_e.append(e[keep])
    frame = pd.concat(records, ignore_index=True)
    xp, xe = np.concatenate(matrices_p), np.concatenate(matrices_e)
    fit, cal, evaluation = (frame.role.to_numpy() == name for name in ("fit", "calibration", "evaluation"))
    y = frame.target.to_numpy()
    primary = evaluation & (frame.condition.to_numpy() == "mcar30")
    primary_ids = frame.loc[primary, "id"].to_numpy()
    if len(np.unique(primary_ids)) != len(primary_ids):
        raise ValueError("Primary bootstrap requires one episode per customer")
    for mask in (fit, cal):
        if np.unique(y[mask]).size != 2:
            raise ValueError("Single-class fit/calibration; stop")
    if min((y[primary] == 0).sum(), (y[primary] == 1).sum()) < 20:
        raise ValueError("Fewer than 20 primary events/non-events; stop")
    training_times = {}
    for name, matrix in (("prediction_distribution", xp), ("prediction_plus_explanation", xe)):
        budget()
        tick = time.perf_counter()
        model = HistGradientBoostingClassifier(**config["detector"]).fit(matrix[fit], y[fit])
        frame[name] = model.predict_proba(matrix)[:, 1]
        training_times[name] = time.perf_counter()-tick
    names = list(records[0].columns[4:]) + ["prediction_distribution", "prediction_plus_explanation"]
    metrics, calibrators = [], {}
    for name in names:
        budget()
        score = frame[name].to_numpy()
        mapping = PositiveSlopePlattCalibrator().fit(score_transform(name, score[cal]), y[cal])
        calibrated = mapping.transform(score_transform(name, score))
        calibrators[name] = mapping.to_dict()
        frame[name+"_calibrated"] = calibrated
        for condition in config["conditions"]:
            rows = evaluation & (frame.condition.to_numpy() == condition)
            yy = y[rows]
            binary = np.unique(yy).size == 2
            metrics.append({"condition": condition, "detector": name, "n": int(rows.sum()),
                "events": int(yy.sum()), "prevalence": float(yy.mean()) if len(yy) else None,
                "ap": float(average_precision_score(yy, score[rows])) if binary else None,
                "auroc": float(roc_auc_score(yy, score[rows])) if binary else None,
                "brier": float(brier_score_loss(yy, calibrated[rows])) if len(yy) else None,
                "log_loss": float(log_loss(yy, calibrated[rows], labels=[0,1])) if len(yy) else None})
    comparisons = [("mc8", "prediction_distribution"), ("prediction_plus_explanation", "prediction_distribution")]
    scores = {name: frame.loc[primary,name].to_numpy() for name in set(sum((list(c) for c in comparisons), []))}
    yy = y[primary]
    rng = np.random.default_rng(config["seed"])
    bootstrap = []
    for draw in range(config["bootstrap_draws"]):
        budget()
        idx = rng.integers(len(yy), size=len(yy))
        if np.unique(yy[idx]).size != 2:
            continue
        ap = {name: average_precision_score(yy[idx], val[idx]) for name,val in scores.items()}
        bootstrap.append([ap[a]-ap[b] for a,b in comparisons])
    differences = []
    for j,(a,b) in enumerate(comparisons):
        point = float(average_precision_score(yy,scores[a])-average_precision_score(yy,scores[b]))
        lo,hi = np.quantile(np.asarray(bootstrap)[:,j], [.025,.975])
        differences.append({"comparison": a+" minus "+b, "ap_difference": point,
                            "ci95_low": float(lo), "ci95_high": float(hi)})
    gate = all(d["ci95_low"] > 0 for d in differences) and differences[0]["ap_difference"] >= .05
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "cohorts.csv", index=False)
    frame.to_csv(out / "row_scores.csv", index=False)  # Ignored, never publish record-level outputs.
    write_json(out / "calibrators.json", calibrators)
    budget()
    result = {"status": "MEASURED_EXPLORATORY_BASELINE_AUDIT", "gate_for_three_restart_audit": bool(gate),
        "method_novelty_established": False, "differences": differences, "bootstrap_retained": len(bootstrap),
        "bootstrap_requested": config["bootstrap_draws"], "training_and_scoring_seconds": training_times,
        "total_seconds": time.perf_counter()-start, "device": "cpu", "threads": 1,
        "new_shap_calls": 0, "financial_predictors_trained": 0, "feature_dimensions": [xp.shape[1],xe.shape[1]],
        "freeze_sha256": sha256(out / "freeze.json"),
        "output_sha256": {p.name: sha256(p) for p in out.iterdir() if p.name != "results.json" and p.is_file()}}
    write_json(out / "results.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "run"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prediction_distribution_audit.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    with threadpool_limits(limits=1):
        if args.stage == "freeze":
            freeze(config, config_path, args.output)
        else:
            run(config, args.output)


if __name__ == "__main__":
    main()
