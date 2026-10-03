"""Development-only diagnostic study; historical test is excluded by construction."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import time
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import yaml

from financepaper.data.schema import FEATURE_NAMES
from financepaper.data.split import split_indices
from financepaper.data.taiwan import load_taiwan
from financepaper.data.temporal import TemporalPreprocessor
from financepaper.missingness.mar import AnchorMAR
from financepaper.evaluation.revision_audit import audit_grid, normalized_shift
from financepaper.experiments import temporal_pilot as p
from financepaper.experiments import robustness_followup as r
from financepaper.models.xgboost import make_xgboost
from financepaper.models.logistic import make_logistic
from financepaper.training.calibration import PositiveSlopePlattCalibrator

CONDITIONS = ("complete", "mcar10", "mcar20", "mcar30", "mar30")
CORE = ("lr", "xgb25", "additive", "vanilla", "mask_delta")
NEURAL = ("vanilla", "mask_delta", "mask_only", "delta_only", "static_only", "temporal_only")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    base = p._load_config(Path(cfg["base_config"]))
    if cfg["folds"] != 3 or len(cfg["restart_seeds"]) != cfg["folds"]:
        raise ValueError("this prospective pilot fixes three outer folds")
    if sum(cfg["inner_sizes"].values()) != 14000:
        raise ValueError("inner partitions must cover 14000 customers")
    return cfg, base


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    files = list((root / "src/financepaper").rglob("*.py"))
    files += [root/x for x in ("configs/revision_study.yaml", "configs/temporal_pilot.yaml",
              "scripts/run_revision_study.py", "docs/REVISION_STUDY_PROTOCOL.md", "uv.lock")]
    return {str(f.relative_to(root)): sha256(f.read_bytes()).hexdigest() for f in sorted(files)}


def make_partitions(dataset, cfg):
    old = split_indices(dataset.y, 42)
    pool = np.sort(np.concatenate([old[k] for k in ("train", "development", "probability_calibration")]))
    excluded = np.concatenate([old["test"], old["risk_calibration"]])
    if np.intersect1d(pool, excluded).size:
        raise AssertionError("historical test/reserve contamination")
    y = dataset.y.to_numpy()
    folds = []
    splitter = StratifiedKFold(cfg["folds"], shuffle=True, random_state=cfg["partition_seed"])
    for fold, (inside, outside) in enumerate(splitter.split(pool, y[pool])):
        remaining = pool[inside]
        pieces = {}
        for offset, (name, size) in enumerate(cfg["inner_sizes"].items()):
            if size == len(remaining):
                selected = remaining
            else:
                selected, remaining = train_test_split(remaining, train_size=size,
                    random_state=cfg["partition_seed"]+100*fold+offset, stratify=y[remaining])
            pieces[name] = np.sort(selected)
        pieces["outer"] = np.sort(pool[outside])
        joined = np.concatenate(list(pieces.values()))
        if len(np.unique(joined)) != len(pool) or set(joined) != set(pool):
            raise AssertionError("customer split overlap or omission")
        folds.append(pieces)
    return folds, excluded


def masks_for(X, mar, seed):
    masks = p._nested_masks(X, seed, mar)
    u = np.random.default_rng(seed).random(X.shape)
    masks["mcar20"] = pd.DataFrame(u < .2, index=X.index, columns=X.columns)
    return {c: masks[c] for c in CONDITIONS}


def partition_batches(dataset, indices, preprocessor, mar, seed):
    X = dataset.X.iloc[indices]
    masks = masks_for(X, mar, seed)
    return p._batch_map(preprocessor, X, masks), masks, dataset.y.iloc[indices].to_numpy()


def fit_fold(dataset, parts, cfg, base, fold, out, device):
    from financepaper.models.temporal_gru import TemporalGRU
    from financepaper.models.missing_aware_gru import MissingAwareGRU
    from financepaper.training.fit import fit_model
    seed = cfg["restart_seeds"][fold]
    X = dataset.X.iloc[parts["predictor_train"]]
    y = dataset.y.iloc[parts["predictor_train"]].to_numpy()
    prep = TemporalPreprocessor().fit(X)
    mar = AnchorMAR().fit(X)
    context = dict(preprocessor=prep, mar=mar, X_train=X, y_train=y, train_batch=prep.transform(X))
    db, dm, yd = partition_batches(dataset, parts["early_stopping"], prep, mar, seed+1000)
    cb, cm, yc = partition_batches(dataset, parts["default_calibration"], prep, mar, seed+2000)
    context.update(dev_batches=db, dev_masks=dm, cal_batches=cb, cal_masks=cm, y_dev=yd, y_cal=yc)
    background = r._background(context, dict(base, seed=seed))
    callback = p._make_augmentation_callback(X, prep, base["missingness"]["augmentation_rates"], seed+700000)
    matrix, labels, weights, ids = r.training_views(context, base, seed, cfg["augmentation_views"])
    fits, summaries = {}, {}
    for name in ("lr", "xgb25", "additive"):
        started = time.perf_counter()
        model = make_logistic(seed) if name == "lr" else make_xgboost(seed,
            **(base["xgboost"] if name == "xgb25" else dict(max_depth=1,n_estimators=600,min_child_weight=20,reg_lambda=10)))
        model.fit(matrix, labels, sample_weight=weights)
        fits[name] = r._flat_fit(name, model)
        summaries[name] = dict(device="cpu", elapsed_seconds=time.perf_counter()-started,
            parameter_count=int(model.coef_.size+model.intercept_.size) if name == "lr" else None,
            trees=None if name == "lr" else model.n_estimators, views=cfg["augmentation_views"])
        print(f"fold {fold}: fitted {name}", flush=True)
    del matrix, labels, weights
    for name in NEURAL:
        p._seed_everything(seed)
        kw = dict(temporal_dim=prep.temporal_dim, static_dim=prep.static_dim,
            hidden_size=base["hidden_size"], static_hidden=base["static_hidden"],
            fusion_hidden=base["fusion_hidden"], dropout=base["dropout"])
        if name == "vanilla":
            model = TemporalGRU(**kw)
        else:
            model = MissingAwareGRU(**kw, use_mask=name != "delta_only", use_delta=name != "mask_only",
                use_static=name != "temporal_only", use_temporal=name != "static_only")
        fit = fit_model(model, context["train_batch"], y,
            {c:db[c] for c in p.SELECTION_CONDITIONS}, yd, callback, base, seed, device=device)
        fits[name] = dict(name=name, family="neural", model=fit["model"], fit=fit, augmented=True, loss="bce")
        summaries[name] = p._fit_summary(fits[name])
        summaries[name]["epochs_run"] = len(fit["history"])
        print(f"fold {fold}: fitted {name}, epoch {fit['best_epoch']}", flush=True)
    selection = p._fit_calibrators_and_selection(fits, db, cb, yd, yc, base, device)
    p._write_json(out/"fit_summary.json", summaries)
    p._write_json(out/"predictor_selection.json", selection)
    p._write_json(out/"preprocessor.json", prep.to_dict())
    p._write_json(out/"background_ids.json", background["record_ids"])
    # Persist only train data, fitted models and transform state. No restored
    # values from other partitions are part of an inference artifact.
    bundle = dict(context=context, background=background, fits=fits, selection=selection, seed=seed)
    joblib.dump(bundle, out/"predictors.joblib")
    return bundle


def get_attr(fit, batch, bundle, base, device):
    return r.attribution(fit, batch, bundle["context"], bundle["background"], base, device, {})


def save_pair(path, batch, hidden, before, full):
    np.savez_compressed(path, record_ids=batch.record_ids, hidden=hidden,
        original_before=before["original"], original_full=full["original"],
        valid_before=before["valid"], valid_full=full["valid"],
        logits_before=before["input_logit"], logits_full=full["input_logit"],
        residual_before=before["residual"], residual_full=full["residual"],
        baseline_before=before["baseline_logit"], baseline_full=full["baseline_logit"])


def audit_explanations(bundle, batches, masks, base, cfg, out, device):
    grids, rows, cached = [], [], {}
    pos = np.arange(min(cfg["audit_records"], len(batches["complete"])))
    # Diagnostic rows were randomly partitioned; choose positions by RNG anyway
    # so workbook order cannot determine who gets costly attributions.
    pos = np.sort(np.random.default_rng(bundle["seed"]+910).choice(len(batches["complete"]), len(pos), replace=False))
    small = {c:b.take(pos) for c,b in batches.items()}
    for name in CORE:
        cached[name] = {}
        for c in CONDITIONS:
            attr = get_attr(bundle["fits"][name], small[c], bundle, base, device)
            cached[name][c] = attr
            h = masks[c].iloc[pos].to_numpy(bool)
            full = cached[name]["complete"]
            frame = audit_grid(attr["original"], full["original"], h, attr["valid"] & full["valid"])
            frame["model"], frame["condition"] = name, c
            grids.append(frame)
            rr = r.reason_records(name, c, small[c].record_ids, h, attr, full, .01, base)
            for i, row in enumerate(rr):
                row["normalized_shift"] = normalized_shift(attr["original"][i], full["original"][i], h[i])
            rows += rr
            save_pair(out/f"{name}_{c}.npz", small[c], h, attr, full)
        print(f"  audited {name}", flush=True)
    pd.concat(grids).to_csv(out/"metric_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(out/"reason_records.csv", index=False)
    return small, pos, cached


def functional_diagnostics(bundle, small, masks, positions, cached, base, cfg, out, device):
    from financepaper.explain.permutation import permutation_attributions
    fits, prep = bundle["fits"], bundle["context"]["preprocessor"]
    n = cfg["shared_explainer_records"]
    shared, contexts, interactions = [], [], []
    refpos = np.sort(np.random.default_rng(bundle["seed"]+930).choice(len(bundle["context"]["train_batch"]), 4, replace=False))
    refs = bundle["context"]["train_batch"].take(refpos)
    for name in ("xgb25", "additive", "vanilla", "mask_delta"):
        fit = fits[name]
        predict = lambda b: p._raw_logits(fit, b, device, 256)
        full = small["complete"].take(np.arange(n))
        altfull = permutation_attributions(full, refs, predict, permutations=cfg["shared_permutations"], seed=441)
        for c in ("mcar10", "mcar30", "mar30"):
            batch = small[c].take(np.arange(n))
            h = masks[c].iloc[positions[:n]].to_numpy(bool)
            alt = permutation_attributions(batch, refs, predict, permutations=cfg["shared_permutations"], seed=441)
            for row in r.reason_records(name,c,batch.record_ids,h,alt,altfull,.01,base):
                row["explainer"] = "conditional_original_field_permutation"
                shared.append(row)
        # Functional pair perturbation: fixed availability, encoded training
        # mean reference. A second difference vanishes for additive logits.
        batch = small["mcar30"].take(np.arange(min(128,len(small["mcar30"]))))
        origins = np.array(batch.temporal_origins).reshape(batch.temporal.shape[1:])
        def ablate(fields):
            t, s = batch.temporal.copy(), batch.static.copy()
            for f in fields:
                t[:, origins == f] = bundle["background"]["temporal"][origins == f]
                sm = np.array(batch.static_origins) == f
                s[:, sm] = bundle["background"]["static"][sm]
            return replace(batch,temporal=t,static=s)
        z = predict(batch)
        for a,b in (("PAY_0","PAY_2"),("BILL_AMT1","LIMIT_BAL"),("BILL_AMT1","PAY_AMT1"),
                    ("PAY_6","PAY_0"),("AGE","LIMIT_BAL"),("BILL_AMT2","BILL_AMT1")):
            diff = z-predict(ablate([a]))-predict(ablate([b]))+predict(ablate([a,b]))
            for rid, val in zip(batch.record_ids, diff):
                interactions.append(dict(model=name,record_id=rid,pair=f"{a}/{b}",absolute_second_difference=abs(val)))
    for name in ("vanilla", "mask_delta"):
        fit = fits[name]
        for c in ("mcar30", "mar30"):
            batch, full = small[c], small["complete"]
            before = cached[name][c]
            h = masks[c].iloc[positions].to_numpy(bool)
            variants = {
                "same_values_all_observed": p._batch_replace_values_with_full_context(batch,full),
                "restored_values_original_context": p._batch_replace_values(batch,full),
                "literal_zero_observed_mask": replace(batch,temporal_observed=np.zeros_like(batch.temporal_observed),static_observed=np.zeros_like(batch.static_observed)),
                "zero_delta": replace(batch,delta=np.zeros_like(batch.delta)),
            }
            for variant, changed in variants.items():
                after = get_attr(fit,changed,bundle,base,device)
                rr = r.reason_records(name,c,batch.record_ids,h,before,after,.01,base)
                for row in rr:
                    row["context_variant"] = variant
                    contexts.append(row)
        # Expected IG: average complete line integrals over four empirical
        # references, holding endpoint context fixed. Not stochastic GradientSHAP.
        alts = {}
        for c in ("complete","mcar10","mcar30","mar30"):
            batch = small[c].take(np.arange(n))
            attrs = [p._ig_attributions(fit,batch,refs.temporal[j],refs.static[j],prep,base["explanations"],device) for j in range(4)]
            attr = {key:np.mean([a[key] for a in attrs],axis=0) for key in ("original","input_logit","baseline_logit","residual","steps_used")}
            attr["valid"] = np.logical_and.reduce([a["valid"] for a in attrs])
            alts[c] = attr
            h = masks[c].iloc[positions[:n]].to_numpy(bool)
            for row in r.reason_records(name,c,batch.record_ids,h,attr,alts["complete"],.01,base):
                row["explainer"] = "four_reference_expected_ig"
                shared.append(row)
    pd.DataFrame(shared).to_csv(out/"explainer_sensitivity.csv",index=False)
    pd.DataFrame(contexts).to_csv(out/"context_sensitivity.csv",index=False)
    pd.DataFrame(interactions).to_csv(out/"pair_interactions.csv",index=False)


def run_audit(config_path, output, device=None):
    cfg, base = load_config(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("new audit requires an empty output directory")
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    device = p._select_device(device or base["device"])
    sources = source_hashes()
    dataset = load_taiwan(Path(base["data_path"]))
    folds, excluded = make_partitions(dataset,cfg)
    p._write_json(output/"protocol_freeze.json",dict(config=cfg,base=base,sources=sources,
        dataset_sha256=dataset.sha256, device=str(device), versions=p._package_versions(),
        excluded_ids=dataset.X.index[excluded].tolist(), outer_opened=False))
    for fold, parts in enumerate(folds):
        directory = output/f"fold_{fold}"
        directory.mkdir()
        p._write_json(directory/"partition_ids.json",{k:dataset.X.index[v].tolist() for k,v in parts.items()})
        bundle = fit_fold(dataset,parts,cfg,base,fold,directory,device)
        batches,masks,y = partition_batches(dataset,parts["diagnostic"],bundle["context"]["preprocessor"],bundle["context"]["mar"],bundle["seed"]+3000)
        metrics, preds = r.prediction_metrics(bundle["fits"],batches,y,bundle["selection"],base,device)
        p._write_json(directory/"diagnostic_prediction_metrics.json",metrics)
        preds.to_csv(directory/"diagnostic_predictions.csv",index=False)
        p._save_masks(directory,"diagnostic",masks)
        small,pos,cached = audit_explanations(bundle,batches,masks,base,cfg,directory,device)
        functional_diagnostics(bundle,small,masks,pos,cached,base,cfg,directory,device)
        del bundle
    p._write_json(output/"audit_manifest.json",dict(status="diagnostics_complete",outer_opened=False,
        elapsed_seconds=time.perf_counter()-started,source_sha256=sources,
        artifact_sha256=r.artifact_hashes(output)))
