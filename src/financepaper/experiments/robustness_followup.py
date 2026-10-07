"""Prospectively bounded tree exposure / additive-interaction follow-up.

All candidate search and all restart fits finish before evaluate() opens test.
The previous temporal pilot's already exposed test remains exploratory evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import yaml

from financepaper.data.schema import FEATURE_NAMES
from financepaper.evaluation.revision import reason_diagnostics, revision_event
from financepaper.evaluation.temporal_metrics import detailed_prediction_metrics, reliability_curve
from financepaper.explanations.reasons import extract_reasons
from financepaper.experiments import temporal_pilot as p
from financepaper.models.logit_blend import LogitBlend
from financepaper.models.xgboost import make_xgboost
from financepaper.training.calibration import PositiveSlopePlattCalibrator

NEURAL_BASELINES = ("vanilla_aug_bce", "mask_delta_aug_bce")
BASELINES = ("lr_complete", "lr_aug", "xgb_complete", "xgb_aug") + NEURAL_BASELINES + (
    "xgb_multiview", "tree_d3_multi")
COMPONENTS = ("xgb_multiview", "additive_multi", "tree_d2_multi", "tree_d3_multi")
CHOICES = ("predictive_choice", "joint_choice")
EVALUATED = BASELINES + ("additive_multi",) + CHOICES


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    files = list((root / "src/financepaper").rglob("*.py")) + [root / x for x in (
        "scripts/run_robustness_followup.py", "configs/robustness_followup.yaml",
        "configs/temporal_pilot.yaml", "archive/20261007/docs/ROBUSTNESS_FOLLOWUP_PROTOCOL.md",
        "archive/20261007/docs/TEMPORAL_MODEL_SPEC.md", "pyproject.toml", "uv.lock")]
    return {str(f.relative_to(root)): _digest(f) for f in sorted(files)}


def artifact_hashes(directory):
    return {str(f.relative_to(directory)): _digest(f) for f in sorted(directory.rglob("*"))
            if f.is_file() and f.name != "manifest.json"}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    base = p._load_config(Path(cfg["base_config"]))
    seeds = cfg["restart_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("restart_seeds must be unique integers")
    if seeds[0] != cfg["split_seed"] or cfg["split_seed"] != base["seed"]:
        raise ValueError("screening restart and fixed partition seed must agree")
    if type(cfg["augmentation_views"]) is not int or cfg["augmentation_views"] < 1:
        raise ValueError("augmentation_views must be a positive integer")
    if cfg["augmentation_views"] < base["training"]["epochs"]:
        raise ValueError("tree views must cover every possible neural training epoch")
    weights = np.asarray(cfg["blend_weights"], dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or np.any((weights <= 0) | (weights >= 1)):
        raise ValueError("blend weights must lie strictly between zero and one")
    if len(set(weights)) != len(weights):
        raise ValueError("duplicate blend weights")
    if cfg["reason_min_attribution"] <= base["explanations"]["attribution_tolerance"]:
        raise ValueError("reason magnitude must exceed event epsilon")
    for value in cfg["selection"].values():
        if not np.isfinite(value) or value < 0:
            raise ValueError("selection tolerances must be finite and nonnegative")
    return cfg, base


def training_views(context, base, seed, views):
    """Copies stay in train; each original record has unit total sample weight."""
    callback = p._make_augmentation_callback(context["X_train"], context["preprocessor"],
        base["missingness"]["augmentation_rates"], seed + 700_000)
    matrices = [p._flat(callback(e))[0] for e in range(views)]
    matrix = np.concatenate(matrices)
    labels = np.tile(context["y_train"], views)
    weights = np.full(len(labels), 1.0 / views, dtype=np.float32)
    record_ids = np.tile(context["train_batch"].record_ids, views)
    return matrix, labels, weights, record_ids


def _background(context, base):
    positions = np.sort(np.random.default_rng(base["seed"] + 300).choice(
        len(context["train_batch"]), base["background_size"], replace=False))
    batch = context["train_batch"].take(positions)
    matrix, origins = p._flat(batch)
    return dict(flat=matrix, origins=origins, temporal=batch.temporal.mean(0),
                static=batch.static.mean(0), record_ids=batch.record_ids)


def _flat_fit(name, model):
    return dict(name=name, family="flat", model=model, augmented=True, loss=None)


def fit_components(context, base, cfg, seed, directory):
    fits = {}
    started = time.perf_counter()
    matrix, labels, weights, ids = training_views(context, base, seed, cfg["augmentation_views"])
    n = len(context["y_train"])
    complete = p._flat(context["train_batch"])[0]
    for name in BASELINES[:4]:
        X = matrix[:n] if name.endswith("aug") else complete
        fits[name] = _flat_fit(name, p._fit_flat_model(name, X, context["y_train"], base, seed))
        fits[name]["augmented"] = name.endswith("aug")
    for name in COMPONENTS:
        if name == "xgb_multiview":
            params = dict(base["xgboost"])
        else:
            depth = {"additive_multi": 1, "tree_d2_multi": 2, "tree_d3_multi": 3}[name]
            params = dict(max_depth=depth, n_estimators=cfg["regularized_estimators"],
                          learning_rate=.05, min_child_weight=cfg["regularized_min_child_weight"],
                          reg_lambda=cfg["regularized_lambda"], subsample=1., colsample_bytree=1.)
        print(f"  seed={seed} fit {name}", flush=True)
        model = make_xgboost(seed, **params)
        model.fit(matrix, labels, sample_weight=weights)
        fits[name] = _flat_fit(name, model)
    p._write_json(directory / "exposure.json", dict(n_original=n, n_rows=len(labels),
        views=cfg["augmentation_views"], total_weight=float(weights.sum()),
        record_id_sha256=sha256(ids.tobytes()).hexdigest(),
        train_ids=context["train_batch"].record_ids.tolist(),
        view_mask_seeds=[seed + 700_000 + e for e in range(cfg["augmentation_views"])],
        tree_elapsed_seconds=time.perf_counter() - started))
    return fits


def candidate_fits(components, cfg):
    fits = {name: components[name] for name in COMPONENTS}
    recipes = {name: dict(component=name) for name in COMPONENTS}
    for interaction in ("xgb_multiview", "tree_d2_multi", "tree_d3_multi"):
        for weight in cfg["blend_weights"]:
            name = f"blend_{interaction}_{weight:g}"
            model = LogitBlend(components["additive_multi"]["model"],
                               components[interaction]["model"], weight)
            fits[name] = _flat_fit(name, model)
            recipes[name] = dict(additive="additive_multi", interaction=interaction, weight=weight)
    return fits, recipes


def attribution(fit, batch, context, background, base, device, cache):
    model = fit["model"]
    key = id(model)
    if key in cache:
        return cache[key]
    if isinstance(model, LogitBlend):
        a = attribution(_flat_fit("a", model.additive), batch, context, background, base, device, cache)
        b = attribution(_flat_fit("b", model.interaction), batch, context, background, base, device, cache)
        encoded = model.combine_attributions(a["encoded"], b["encoded"])
        reference = (1-model.weight)*a["baseline_logit"] + model.weight*b["baseline_logit"]
        logits = model.decision_function(p._flat(batch)[0])
        result = _encoded_attribution(encoded, reference, logits, background["origins"])
        result["valid"] &= a["valid"] & b["valid"]
    elif hasattr(model, "coef_"):
        matrix = p._flat(batch)[0]
        mean = background["flat"].mean(0, dtype=np.float64)
        encoded = (matrix - mean) * model.coef_.reshape(1, -1)
        reference = np.full(len(matrix), model.decision_function(mean.reshape(1, -1))[0])
        result = _encoded_attribution(encoded, reference, model.decision_function(matrix), background["origins"])
    else:
        result = p._attribute_model(fit, batch, context["preprocessor"], background,
                                    base["explanations"], device)
    cache[key] = result
    return result


def _encoded_attribution(encoded, reference, logits, origins):
    original, metadata = p._aggregate_flat_value_phi(encoded, origins)
    residual = encoded.sum(1) - (logits-reference)
    tolerance = .002 + .001*np.abs(logits-reference)
    valid = np.isfinite(residual) & (np.abs(residual) <= tolerance)
    return dict(original=original, encoded=encoded, metadata=metadata,
                baseline_logit=reference, input_logit=logits, residual=residual,
                valid=valid, steps_used=np.zeros(len(logits), dtype=int))


def reason_records(name, condition, ids, hidden, before, full, magnitude, base):
    rows = []
    options = base["explanations"]
    k = options["k"]
    for i, record_id in enumerate(ids):
        valid = bool(before["valid"][i] and full["valid"][i])
        reasons = extract_reasons(before["original"][i], hidden[i], FEATURE_NAMES, k, magnitude) if before["valid"][i] else ()
        preeligible = len(reasons) == k
        eligible = preeligible and valid
        event = revision_event(reasons, full["original"][i], hidden[i], FEATURE_NAMES,
            k=k, rank_tolerance=options["rank_tolerance"], attribution_tolerance=options["attribution_tolerance"]) if eligible else None
        diagnostics = reason_diagnostics(before["original"][i], full["original"][i], hidden[i], FEATURE_NAMES, k) if valid else {}
        scores = [float(before["original"][i, FEATURE_NAMES.index(f)]) for f in reasons]
        raw, restored = expit(before["input_logit"][i]), expit(full["input_logit"][i])
        order = np.flatnonzero(~hidden[i])
        order = order[np.argsort(-before["original"][i, order], kind="stable")]
        ranks = {FEATURE_NAMES[j]: r+1 for r, j in enumerate(order)}
        rows.append(dict(model=name, condition=condition, record_id=int(record_id),
            preverification_eligible=preeligible, eligible=eligible, attribution_valid=valid,
            revision_event=event, reason_strength=min(scores) if preeligible else None,
            reasons=json.dumps(reasons), reason_scores=json.dumps(scores),
            reason_signs=json.dumps([int(np.sign(s)) for s in scores]),
            reason_ranks=json.dumps([ranks[f] for f in reasons]),
            missing_fraction=float(hidden[i].mean()), raw_probability=float(raw),
            restored_raw_probability=float(restored), absolute_raw_probability_shift=float(abs(raw-restored)),
            prediction_stable=bool(abs(raw-restored) <= options["stable_probability_delta"]),
            residual_before=float(before["residual"][i]), residual_full=float(full["residual"][i]),
            **diagnostics))
    return rows


def explain(fits, batches, masks, positions, context, background, base, cfg, device,
            directory=None, conditions=p.ALL_CONDITIONS):
    selected = {c: batches[c].take(positions) for c in set(conditions) | {"complete"}}
    caches = {c: {} for c in selected}
    rows = []
    for name, fit in fits.items():
        full = attribution(fit, selected["complete"], context, background, base, device, caches["complete"])
        for condition in conditions:
            before = attribution(fit, selected[condition], context, background, base, device, caches[condition])
            hidden = masks[condition].iloc[positions].to_numpy(dtype=bool)
            rows.extend(reason_records(name, condition, selected[condition].record_ids,
                                       hidden, before, full, cfg["reason_min_attribution"], base))
            if directory is not None:
                arrays = dict(record_ids=selected[condition].record_ids, hidden=hidden,
                    original_before=before["original"], original_full=full["original"],
                    valid_before=before["valid"], valid_full=full["valid"],
                    baseline_before=before["baseline_logit"], baseline_full=full["baseline_logit"],
                    logits_before=before["input_logit"], logits_full=full["input_logit"],
                    residual_before=before["residual"], residual_full=full["residual"],
                    steps_before=before["steps_used"], steps_full=full["steps_used"],
                    feature_names=np.asarray(FEATURE_NAMES))
                for endpoint, value in (("before", before), ("full", full)):
                    if value.get("encoded") is not None:
                        arrays[f"encoded_{endpoint}"] = value["encoded"]
                np.savez_compressed(directory / f"{name}_{condition}_attributions.npz", **arrays)
    return pd.DataFrame(rows)


def matched_coverage(records, names, targets=(.25, .5, 1.)):
    rows = []
    for condition, frame in records.groupby("condition"):
        ids = set(frame.record_id)
        common = ids.copy()
        for name in names:
            subset = frame[(frame.model == name) & frame.eligible & frame.attribution_valid]
            common &= set(subset.record_id)
        for name in names:
            available = frame[(frame.model == name) & frame.record_id.isin(common)].sort_values(
                ["reason_strength", "record_id"], ascending=[False, True])
            for target in targets:
                count = min(int(np.floor(len(ids)*target)), len(common))
                selected = available.iloc[:count]
                rows.append(dict(model=name, condition=condition, target_coverage=target,
                    denominator=len(ids), common_n=len(common), selected_n=count,
                    actual_coverage=count/len(ids), revised=int(selected.revision_event.sum()),
                    revision_rate=float(selected.revision_event.mean()) if count else None))
    return pd.DataFrame(rows)


def prediction_metrics(fits, batches, y, selection, base, device):
    rows, records = [], []
    for name, fit in fits.items():
        rule = selection["models"][name]
        calibrator = PositiveSlopePlattCalibrator.from_dict(rule["calibrator"])
        complete_logits = p._raw_logits(fit, batches["complete"], device, base["training"]["batch_size"])
        full_cal = calibrator.transform(complete_logits)
        for condition, batch in batches.items():
            logits = p._raw_logits(fit, batch, device, base["training"]["batch_size"])
            raw, calibrated = expit(logits), calibrator.transform(logits)
            for kind, probability in (("raw", raw), ("calibrated", calibrated)):
                threshold = rule[f"{kind}_threshold"]
                metric = detailed_prediction_metrics(y, probability, threshold, base["calibration"]["bins"])
                rows.append(dict(model=name, condition=condition, probability=kind, **metric))
            records.append(pd.DataFrame(dict(model=name, condition=condition, record_id=batch.record_ids,
                y=y, raw_logit=logits, raw_probability=raw, calibrated_probability=calibrated,
                absolute_raw_probability_shift=abs(raw-expit(complete_logits)),
                absolute_calibrated_probability_shift=abs(calibrated-full_cal))))
    return rows, pd.concat(records, ignore_index=True)


def select_candidates(leaderboard, matching, cfg):
    ordered = leaderboard.sort_values(["average_precision", "log_loss", "model"], ascending=[False, True, True])
    predictive = ordered.iloc[0]
    options = cfg["selection"]
    eligible = leaderboard[
        (leaderboard.average_precision >= leaderboard.average_precision.max()-options["ap_tolerance"])
        & (leaderboard.roc_auc >= leaderboard.roc_auc.max()-options["auc_tolerance"])
        & (leaderboard.brier <= predictive.brier+options["brier_tolerance"])
        & (leaderboard.log_loss <= predictive.log_loss+options["log_loss_tolerance"])]
    matched = matching[matching.target_coverage == .5]
    feasible = set(matched.groupby("model").filter(
        lambda x: len(x) == 2 and (x.actual_coverage >= .5).all()).model)
    eligible = eligible[eligible.model.isin(feasible)].copy()
    reason = matched.groupby("model").revision_rate.mean()
    eligible["revision_rate"] = eligible.model.map(reason)
    eligible = eligible.dropna(subset=["revision_rate"])
    if len(eligible):
        joint = eligible.sort_values(["revision_rate", "average_precision", "model"], ascending=[True, False, True]).iloc[0].model
    else:
        joint = predictive.model
    return dict(predictive_choice=predictive.model, joint_choice=joint,
                joint_gate_passed=bool(len(eligible)), joint_qualified=eligible.model.tolist())


def _save_fit(directory, name, fit):
    if fit["family"] == "neural":
        import torch
        torch.save({k: v.detach().cpu() for k, v in fit["model"].state_dict().items()}, directory / f"{name}.pt")
        p._write_json(directory / f"{name}_history.json", fit["fit"]["history"])
    else:
        joblib.dump(fit["model"], directory / f"{name}.joblib")


def fit_all(config_path, output, device=None):
    cfg, base = load_config(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = source_hashes()
    device = p._select_device(device or base["device"])
    p._write_json(output / "resolved_config.json", dict(followup=cfg, base=base, device=str(device)))
    context = p._prepare_context(base, output)
    background = _background(context, base)
    p._write_json(output / "background_ids.json", background["record_ids"])
    dev_positions = np.sort(np.random.default_rng(base["seed"]+910_001).choice(
        len(context["y_dev"]), min(base["explanations"]["development_records"], len(context["y_dev"])), replace=False))
    p._write_json(output / "development_explanation_ids.json", context["dev_batches"]["complete"].record_ids[dev_positions])
    choices = None
    for seed in cfg["restart_seeds"]:
        directory = output / f"seed_{seed}"
        directory.mkdir()
        print(f"Training restart {seed}", flush=True)
        components = fit_components(context, base, cfg, seed, directory)
        candidates, recipes = candidate_fits(components, cfg)
        if choices is None:
            selection = p._fit_calibrators_and_selection(candidates, context["dev_batches"], context["cal_batches"],
                context["y_dev"], context["y_cal"], base, device)
            dev_batches = {c: context["dev_batches"][c] for c in p.SELECTION_CONDITIONS}
            metrics, _ = prediction_metrics(candidates, dev_batches, context["y_dev"], selection, base, device)
            p._write_json(output / "search_development_metrics.json", metrics)
            table = pd.DataFrame([m for m in metrics if m["probability"] == "calibrated"])
            leaderboard = table.groupby("model")[["average_precision", "roc_auc", "brier", "log_loss"]].mean().reset_index()
            attr = explain(candidates, context["dev_batches"], context["dev_masks"], dev_positions, context,
                           background, base, cfg, device, conditions=("mcar10", "mcar30"))
            matching = matched_coverage(attr, tuple(candidates))
            choices = select_candidates(leaderboard, matching, cfg)
            choices["recipes"] = {alias: recipes[choices[alias]] for alias in CHOICES}
            leaderboard.to_csv(output / "development_leaderboard.csv", index=False)
            attr.to_csv(output / "development_reasons.csv", index=False)
            matching.to_csv(output / "development_matched_coverage.csv", index=False)
            p._write_json(output / "search_selection.json", choices)
            print(f"Frozen recipes: {choices}", flush=True)
        fits = {name: components[name] for name in BASELINES[:4] + (
            "additive_multi", "xgb_multiview", "tree_d3_multi")}
        for alias in CHOICES:
            fits[alias] = dict(candidates[choices[alias]], name=alias)
        callback = p._make_augmentation_callback(context["X_train"], context["preprocessor"],
            base["missingness"]["augmentation_rates"], seed+700_000)
        for name in NEURAL_BASELINES:
            print(f"  seed={seed} fit {name} on {device}", flush=True)
            fits[name] = p._fit_neural(name, context["preprocessor"], context["train_batch"], context["y_train"],
                context["dev_batches"], {c: context["y_dev"] for c in context["dev_batches"]}, base, seed, device, callback)
        selection = p._fit_calibrators_and_selection(fits, context["dev_batches"], context["cal_batches"],
            context["y_dev"], context["y_cal"], base, device)
        selection.update(restart_seed=seed, split_seed=base["seed"], choices=choices,
                         reason_min_attribution=cfg["reason_min_attribution"], test_opened=False)
        p._write_json(directory / "frozen_selection.json", selection)
        for name, fit in fits.items():
            _save_fit(directory, name, fit)
        del fits, components, candidates
    if source_hashes() != sources:
        raise RuntimeError("source changed during fitting; do not evaluate this snapshot")
    p._write_json(output / "fit_manifest.json", dict(status="fitted", test_opened=False,
        data_sha256=context["dataset"].sha256, source_sha256=sources,
        artifact_sha256=artifact_hashes(output), training_seconds=time.perf_counter()-started,
        device=str(device), versions=p._package_versions(), choices=choices,
        scope="five training restarts on a fixed, previously exposed test cohort"))
    return output


def _load_fits(directory, context, base, device):
    import torch
    fits = {}
    for name in EVALUATED:
        if name in NEURAL_BASELINES:
            model = p._construct_neural(name, context["preprocessor"], base)
            model.load_state_dict(torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True))
            fits[name] = dict(name=name, family="neural", model=model.to(device).eval())
        else:
            fits[name] = _flat_fit(name, joblib.load(directory / f"{name}.joblib"))
    return fits


def _flatten_metrics(metrics):
    rows = []
    for item in metrics:
        row = {k: v for k, v in item.items() if k not in {"classwise", "calibration_bins"}}
        for cls, values in item["classwise"].items():
            row.update({f"class_{cls}_{k}": v for k, v in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _reason_summary(records):
    rows = []
    for (seed, name, condition), frame in records.groupby(["seed", "model", "condition"]):
        eligible = frame[frame.eligible]
        stable = eligible[eligible.prediction_stable]
        rows.append(dict(seed=seed, model=name, condition=condition, n=len(frame),
            eligible_n=len(eligible), revised=int(eligible.revision_event.sum()),
            revision_rate=float(eligible.revision_event.mean()) if len(eligible) else None,
            coverage=len(eligible)/len(frame), preverification_coverage=float(frame.preverification_eligible.mean()),
            invalid_n=int((~frame.attribution_valid).sum()), stable_n=len(stable),
            stable_revised=int(stable.revision_event.sum()),
            **{key: float(frame[key].mean()) for key in ("top_k_overlap", "observed_sign_agreement", "rank_correlation")}))
    return pd.DataFrame(rows)


def dominance_table(metrics, shifts, reasons, matching):
    """Every declared metric must be present; undefined values are not wins."""
    cal = metrics[metrics.probability == "calibrated"]
    keys = {k: 1 for k in ("roc_auc", "average_precision", "recall", "f1")}
    keys.update({k: -1 for k in ("brier", "log_loss", "ece")})
    keys.update({f"class_{c}_{k}": 1 for c in (0, 1) for k in ("precision", "recall", "f1")})
    complete_mean = lambda s: s.mean() if s.notna().all() else np.nan
    means = cal.groupby(["model", "condition"])[list(keys)].agg(complete_mean)
    reason_means = reasons.groupby(["model", "condition"])[["coverage", "top_k_overlap", "observed_sign_agreement", "rank_correlation"]].agg(complete_mean)
    shift_means = shifts.groupby(["model", "condition"]).absolute_calibrated_probability_shift.agg(complete_mean)
    matched = matching[matching.target_coverage == .5]
    matched_means = matched.groupby(["model", "condition"]).revision_rate.agg(complete_mean)
    coverage_means = matched.groupby(["model", "condition"]).actual_coverage.min()
    rows = []
    for choice in CHOICES:
        for baseline in BASELINES:
            for condition in p.ALL_CONDITIONS:
                values = [(key, means.loc[(choice, condition), key], means.loc[(baseline, condition), key], sign) for key, sign in keys.items()]
                if condition != "complete":
                    values.extend((key, reason_means.loc[(choice, condition), key], reason_means.loc[(baseline, condition), key], 1) for key in reason_means.columns)
                    enough = min(coverage_means.loc[(choice, condition)], coverage_means.loc[(baseline, condition)]) >= .5
                    values += [("revision_at_50pct", matched_means.loc[(choice, condition)] if enough else np.nan,
                                matched_means.loc[(baseline, condition)] if enough else np.nan, -1),
                               ("absolute_probability_shift", shift_means.loc[(choice, condition)], shift_means.loc[(baseline, condition)], -1)]
                for key, a, b, direction in values:
                    delta = direction*(a-b)
                    verdict = "undefined" if not np.isfinite(delta) else "win" if delta > 1e-8 else "loss" if delta < -1e-8 else "tie"
                    rows.append(dict(candidate=choice, baseline=baseline, condition=condition, metric=key,
                                     candidate_mean=a, baseline_mean=b, oriented_delta=delta, verdict=verdict))
    return pd.DataFrame(rows)


def evaluate(output, device=None):
    started = time.perf_counter()
    manifest = json.loads((output / "fit_manifest.json").read_text())
    if manifest["status"] != "fitted" or manifest["source_sha256"] != source_hashes():
        raise ValueError("fit status or source hashes differ from frozen fit")
    for relative, expected in manifest["artifact_sha256"].items():
        if _digest(output / relative) != expected:
            raise ValueError(f"frozen artifact changed: {relative}")
    resolved = json.loads((output / "resolved_config.json").read_text())
    cfg, base = resolved["followup"], resolved["base"]
    device = p._select_device(device or resolved["device"])
    evaluation = output / "evaluation"
    if evaluation.exists() and any(evaluation.iterdir()):
        raise FileExistsError("evaluation directory is not empty")
    evaluation.mkdir(exist_ok=True)
    context = p._prepare_context(base, output, persist=False)
    if context["dataset"].sha256 != manifest["data_sha256"]:
        raise ValueError("dataset differs from frozen fit")
    background = _background(context, base)
    # The first access that constructs test batches happens only here, after
    # every restart has persisted its model and selection.
    X_test, masks, batches, y_test = p._test_context(context, base, evaluation)
    positions = np.sort(np.random.default_rng(base["seed"]+920_001).choice(
        len(y_test), min(base["explanations"]["test_records"], len(y_test)), replace=False))
    p._write_json(evaluation / "explanation_ids.json", X_test.index[positions].tolist())
    all_metrics, all_reasons, all_matching, all_shifts = [], [], [], []
    for seed in cfg["restart_seeds"]:
        print(f"Evaluate frozen restart {seed}", flush=True)
        directory = evaluation / f"seed_{seed}"
        directory.mkdir()
        fits = _load_fits(output / f"seed_{seed}", context, base, device)
        selection = json.loads((output / f"seed_{seed}/frozen_selection.json").read_text())
        metrics, predictions = prediction_metrics(fits, batches, y_test, selection, base, device)
        metrics = [dict(seed=seed, **m) for m in metrics]
        all_metrics.extend(metrics)
        predictions.to_csv(directory / "predictions.csv", index=False, float_format="%.12g")
        shifts = predictions.groupby(["model", "condition"])[["absolute_raw_probability_shift", "absolute_calibrated_probability_shift"]].mean().reset_index()
        shifts["seed"] = seed
        all_shifts.append(shifts)
        reasons = explain(fits, batches, masks, positions, context, background, base, cfg, device, directory)
        reasons["seed"] = seed
        reasons.to_csv(directory / "reasons.csv", index=False)
        all_reasons.append(reasons)
        matching = matched_coverage(reasons, EVALUATED)
        matching["seed"] = seed
        all_matching.append(matching)
        curves = []
        for (name, condition), frame in reasons.groupby(["model", "condition"]):
            for row in reliability_curve(frame.missing_fraction, frame.revision_event.astype(float), frame.eligible, len(frame)):
                curves.append(dict(model=name, condition=condition, **row))
        pd.DataFrame(curves).to_csv(directory / "missing_fraction_risk_coverage.csv", index=False)
    metrics = _flatten_metrics(all_metrics)
    reasons = pd.concat(all_reasons, ignore_index=True)
    matching = pd.concat(all_matching, ignore_index=True)
    shifts = pd.concat(all_shifts, ignore_index=True)
    summary = _reason_summary(reasons)
    p._write_json(evaluation / "prediction_metrics.json", all_metrics)
    metrics.to_csv(evaluation / "prediction_metrics.csv", index=False)
    summary.to_csv(evaluation / "reason_summary.csv", index=False)
    matching.to_csv(evaluation / "matched_coverage.csv", index=False)
    shifts.to_csv(evaluation / "probability_shifts.csv", index=False)
    for name, frame, group in (("prediction", metrics, ["model", "condition", "probability"]),
                               ("reason", summary, ["model", "condition"])):
        columns = [c for c in frame.select_dtypes(include="number").columns if c != "seed"]
        frame.groupby(group)[columns].agg(["mean", "std", "min", "max"]).to_csv(evaluation / f"{name}_restart_summary.csv")
    dominance = dominance_table(metrics, shifts, summary, matching)
    dominance.to_csv(evaluation / "dominance_by_metric.csv", index=False)
    totals = []
    for (candidate, baseline), frame in dominance.groupby(["candidate", "baseline"]):
        counts = frame.verdict.value_counts().to_dict()
        totals.append(dict(candidate=candidate, baseline=baseline, **{k: counts.get(k, 0) for k in ("win", "tie", "loss", "undefined")},
            strict_all_wins=bool((frame.verdict == "win").all()),
            weak_pareto_dominance=bool(frame.verdict.isin(["win", "tie"]).all() and (frame.verdict == "win").any())))
    p._write_json(evaluation / "dominance_summary.json", totals)
    p._write_json(output / "manifest.json", dict(status="completed", choices=manifest["choices"],
        data_sha256=manifest["data_sha256"], source_sha256=source_hashes(),
        artifact_sha256=artifact_hashes(output), training_seconds=manifest["training_seconds"],
        evaluation_seconds=time.perf_counter()-started, device=str(device),
        versions=p._package_versions(), restart_seeds=cfg["restart_seeds"], fixed_split_seed=base["seed"],
        test_n=len(y_test), explanation_n=len(positions), unused_splits=["risk_calibration"],
        independent_worker_review=False, scope=manifest["scope"], dominance=totals))
    return output
