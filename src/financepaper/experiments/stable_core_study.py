"""Frozen-predictor Stable-Core extension; historical caches are read-only."""
from dataclasses import asdict
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from financepaper.data.schema import FEATURE_NAMES, CATEGORICAL_FEATURES
from financepaper.data.polish import load_polish
from financepaper.experiments.decisive_validation import (
    taiwan_adapter, polish_adapter, hashes, validate_hashes, write_json, TW_ENV, PL_ENV,
)
from financepaper.reliability.validation import TAIWAN_GROUPS, POLISH_GROUPS, POLISH_NAMES, aggregate_groups
from financepaper.reliability.conditional_completion import TrainingConditionalForest
from financepaper.reliability.reason_sets import (
    METHODS, evidence, concatenate, verified, make_grids, fit_policy, ReasonSetPolicy,
    strongest, matched_random, inference_output,
)


def units():
    return [("taiwan", f) for f in range(3)] + [("polish", 0)]


def folder(root, dataset, fold):
    return root/dataset/f"fold_{fold}" if dataset == "taiwan" else root/dataset


def spec(dataset):
    return (FEATURE_NAMES, TAIWAN_GROUPS, TW_ENV) if dataset == "taiwan" else (POLISH_NAMES, POLISH_GROUPS, PL_ENV)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def freeze(config="configs/stable_core_study.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    root, historic = Path(cfg["output"]), Path(cfg["historical_root"])
    if root.exists():
        raise FileExistsError("Use a new output root; never overwrite a frozen study")
    paths = list(Path("src/financepaper").rglob("*.py"))
    paths += [Path(config), Path("scripts/run_stable_core_study.py"), Path("uv.lock"),
              Path("docs/STABLE_CORE_STUDY_PROTOCOL.md"), Path("docs/STABLE_CORE_RESEARCH_UPDATE.md")]
    old = json.loads((historic/"protocol_freeze.json").read_text())
    validate_hashes(old["sources"])
    for dataset, fold in units():
        source = folder(historic, dataset, fold)
        predictor = (Path("outputs/revision_study")/f"fold_{fold}/predictors.joblib"
                     if dataset == "taiwan" else source/"predictors.joblib")
        paths += [predictor, source/"donor_reference.json"]
        if dataset == "polish":
            paths += [source/"partition_ids.json", Path("data/raw/polish/5year.arff")]
        for split in ("revision_calibration", "release_calibration", "outer"):
            for condition in spec(dataset)[2]:
                p = source/split/condition
                receipt = json.loads((p/"complete.json").read_text())
                validate_hashes(receipt["current_only"]); validate_hashes(receipt["verification_only"])
                paths += [p/"current.npz", p/"verification_only.npz", p/"complete.json"]
    root.mkdir(parents=True)
    write_json(root/"protocol_freeze.json", {"config": cfg, "sources": hashes(paths),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "evidence_status": "exploratory extension of previously inspected Taiwan/Polish; Freddie NOT RUN"})
    print("Stable-Core v2 protocol, inputs and sources frozen.", flush=True)


def load_study(root):
    frozen = json.loads((root/"protocol_freeze.json").read_text())
    validate_hashes(frozen["sources"])
    return frozen["config"], Path(frozen["config"]["historical_root"])


def context(dataset, fold, historical):
    if dataset == "taiwan":
        b = joblib.load(Path("outputs/revision_study")/f"fold_{fold}/predictors.joblib")
        return b["context"]["X_train"], taiwan_adapter(b), b["seed"]
    source = folder(historical, dataset, fold)
    b = joblib.load(source/"predictors.joblib")
    X, _, _ = load_polish("data/raw/polish/5year.arff")
    ids = json.loads((source/"partition_ids.json").read_text())["train"]
    return X.loc[ids], polish_adapter(b), b["seed"]


def conditional_cache(model, adapter, current, names, *, seed, destination):
    if (destination/"complete.json").exists():
        receipt = json.loads((destination/"complete.json").read_text())
        validate_hashes(receipt["files"])
        return load_npz(destination/"current.npz")
    destination.mkdir(parents=True, exist_ok=True)
    partial = pd.DataFrame(current["partial_values"], index=current["record_ids"], columns=names)
    started = time.perf_counter()
    completions = model.sample(partial, current["artificial"], k=8, seed=seed)
    generation = time.perf_counter()-started
    if not np.array_equal(np.where(current["artificial"][:,None,:], np.nan, completions),
                          np.repeat(partial.to_numpy()[:,None,:], 8, axis=1), equal_nan=True):
        raise RuntimeError("STOP: conditional completion altered observed/natural cells")
    phis, probabilities, valid, durations = [], [], [], []
    for k in range(8):
        start = time.perf_counter()
        a = adapter.attribute(pd.DataFrame(completions[:,k], columns=names, index=partial.index))
        durations.append(time.perf_counter()-start)
        phis.append(a["phi"]); probabilities.append(a["probability"]); valid.append(a["valid"])
    result = {"completion_phi": np.stack(phis,1), "completion_probability": np.stack(probabilities,1),
              "completion_valid": np.stack(valid,1), "completions": completions,
              "record_ids": current["record_ids"]}
    if not result["completion_valid"].all():
        raise RuntimeError("STOP: invalid conditional attribution; document incident before continuing")
    np.savez_compressed(destination/"current.npz", **result)
    write_json(destination/"complete.json", {"files": hashes([destination/"current.npz"]),
        "customers": len(partial), "completion_seconds": generation, "attribution_seconds": sum(durations),
        "seed": seed, "K": 8, "verification_accessed": False})
    return result


def grouped(current, conditional, dataset):
    names, groups, _ = spec(dataset)
    before, hidden = aggregate_groups(current["phi"], current["hidden"], names, groups)
    donor, _ = aggregate_groups(current["completion_phi"][:,:8], current["hidden"], names, groups)
    other, _ = aggregate_groups(conditional["completion_phi"][:,:8], current["hidden"], names, groups)
    if not current["valid"].all() or not current["completion_valid"][:,:8].all():
        raise RuntimeError("STOP: invalid frozen attribution")
    return evidence(before, donor, other, hidden, current["probability"])


def cluster_ids(dataset, ids):
    if dataset == "taiwan":
        return ids
    X, _, clusters = load_polish("data/raw/polish/5year.arff")
    return pd.Series(clusters, index=X.index).loc[ids].to_numpy()


def calibration(root):
    cfg, historical = load_study(root)
    if (root/"calibration_freeze.json").exists():
        raise FileExistsError("Policies already frozen; use evaluate")
    artifacts = []
    for dataset, fold in units():
        dest, source = folder(root,dataset,fold), folder(historical,dataset,fold)
        dest.mkdir(parents=True, exist_ok=True)
        if (dest/"calibration_complete.json").exists():
            saved = json.loads((dest/"calibration_complete.json").read_text())
            validate_hashes(saved["artifacts"]); artifacts += list(saved["artifacts"])
            continue
        train, adapter, seed = context(dataset,fold,historical)
        names, groups, environments = spec(dataset)
        current_design = []
        for condition in environments:
            cur = load_npz(source/"revision_calibration"/condition/"current.npz")
            # Continuous baseline grids use donor evidence only; no conditional
            # sampling or restoration is needed in this unlabelled design pool.
            current_design.append(grouped(cur,cur,dataset))
        grids = make_grids(concatenate(current_design))
        start = time.perf_counter()
        model = TrainingConditionalForest(names, CATEGORICAL_FEATURES if dataset=="taiwan" else (), seed=seed).fit(train,partition="train")
        fit_seconds = time.perf_counter()-start
        joblib.dump(model,dest/"conditional.joblib")
        blocks, restored, envs, ids = [], [], [], []
        for cno, condition in enumerate(environments):
            cur = load_npz(source/"release_calibration"/condition/"current.npz")
            if np.intersect1d(train.index,cur["record_ids"]).size:
                raise RuntimeError("Training and release calibration entities overlap")
            cond = conditional_cache(model,adapter,cur,names,seed=cfg["seed"]+fold*100+cno,
                                     destination=dest/"release_calibration"/condition)
            blocks.append(grouped(cur,cond,dataset))
            truth = load_npz(source/"release_calibration"/condition/"verification_only.npz")
            if not truth["valid"].all():
                raise RuntimeError("Invalid verification attribution")
            restored.append(aggregate_groups(truth["phi"],cur["hidden"],names,groups)[0])
            envs.extend([condition]*len(cur["record_ids"])); ids.extend(cur["record_ids"])
            print(f"Calibration evidence {dataset}/fold{fold}/{condition}",flush=True)
        ev, full = concatenate(blocks), np.concatenate(restored)
        clusters = cluster_ids(dataset,np.asarray(ids))
        policies, audit = {}, []
        for target in cfg["targets"]:
            survival = verified(full,ev["hidden"],target)
            for method in METHODS:
                for alpha in cfg["alphas"]:
                    for conservative in (False,True):
                        key = (target,method,alpha,conservative)
                        policy = (ReasonSetPolicy(method,target,alpha,conservative,()) if method=="release_all" else
                                  fit_policy(ev,survival,envs,clusters,method=method,grid=grids[method],target=target,
                                             alpha=alpha,conservative=conservative))
                        policies[key] = policy
                        release = policy.release(ev)
                        for condition in environments:
                            ix = np.asarray(envs)==condition
                            count = int(release[ix].sum()); failures=int((release[ix]&~survival[ix]).sum())
                            audit.append(dict(target=target,method=method,alpha=alpha,conservative=conservative,
                                condition=condition,reasons=count,failures=failures,risk=failures/count if count else None,
                                customers=int(release[ix].any(1).sum()),n=int(ix.sum()),parameters=policy.parameters))
        joblib.dump(dict(policies=policies,grids=grids),dest/"policies.joblib")
        pd.DataFrame(audit).to_csv(dest/"calibration_metrics.csv",index=False)
        write_json(dest/"fit_receipt.json",dict(fit_seconds=fit_seconds,training_rows=len(train),
            training_ids=train.index.tolist(),seed=seed,device="cpu",predictor_refit=False))
        paths=[dest/f for f in ("conditional.joblib","policies.joblib","fit_receipt.json","calibration_metrics.csv")]
        artifacts+=paths
        write_json(dest/"calibration_complete.json",dict(artifacts=hashes(paths)))
        print(f"Policies frozen for {dataset}/fold{fold}",flush=True)
    write_json(root/"calibration_freeze.json",dict(artifacts=hashes(artifacts),
        protocol=hashes([root/"protocol_freeze.json"]),timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat()))


def evaluate(root):
    cfg,historical=load_study(root)
    gate=json.loads((root/"calibration_freeze.json").read_text());validate_hashes(gate["artifacts"]);validate_hashes(gate["protocol"])
    if (root/"assessment_complete.json").exists():
        raise FileExistsError("Assessment complete; no retuning/replacement")
    write_json(root/"assessment_opened.json",dict(timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                                                freeze=hashes([root/"calibration_freeze.json"])))
    artifacts=[]
    for dataset,fold in units():
        dest,source=folder(root,dataset,fold),folder(historical,dataset,fold)
        train,adapter,seed=context(dataset,fold,historical)
        model=joblib.load(dest/"conditional.joblib")
        bundle=joblib.load(dest/"policies.joblib")
        names,groups,environments=spec(dataset)
        for cno,condition in enumerate(environments):
            out=dest/"outer"/condition
            cur=load_npz(source/"outer"/condition/"current.npz")
            if np.intersect1d(train.index,cur["record_ids"]).size:
                raise RuntimeError("Training/assessment entities overlap")
            cond=conditional_cache(model,adapter,cur,names,seed=cfg["seed"]+10000+fold*100+cno,destination=out)
            ev=grouped(cur,cond,dataset)
            decisions={}
            for key,policy in bundle["policies"].items():
                target,method,alpha,conservative=key
                label=f"{target}|{method}|{alpha}|{int(conservative)}"
                selected=policy.release(ev);decisions[label]=selected
                if method.startswith("stable_"):
                    counts=selected.sum(1)
                    decisions[label.replace(method,method+"_matched_strength")]=strongest(ev,counts)
                    decisions[label.replace(method,method+"_matched_random")]=matched_random(ev,counts,cur["record_ids"])
            np.savez_compressed(out/"decisions.npz",**decisions)
            np.savez_compressed(out/"reason_evidence.npz",**ev,record_ids=cur["record_ids"],
                                cluster_ids=cluster_ids(dataset,cur["record_ids"]))
            # Verification loaded only after all release decisions are serialized.
            truth=load_npz(source/"outer"/condition/"verification_only.npz")
            full=aggregate_groups(truth["phi"],cur["hidden"],names,groups)[0]
            if not truth["valid"].all():raise RuntimeError("Invalid verification attribution")
            np.savez_compressed(out/"verification_only.npz",restored_phi=full,
                meaningful=verified(full,ev["hidden"],"meaningful"),ranked=verified(full,ev["hidden"],"ranked"),
                probability_shift=abs(expit(cur["logits"])-expit(truth["logits"])))
            policy=bundle["policies"][("meaningful","stable_both",.1,False)]
            outputs=inference_output(ev,policy,list(groups),expit(cur["logits"]),cur["probability"])
            with (out/"inference.jsonl").open("w") as f:
                for item in outputs:f.write(json.dumps(item)+"\n")
            artifacts += [out/f for f in ("current.npz","decisions.npz","reason_evidence.npz","verification_only.npz","inference.jsonl")]
            print(f"Assessment {dataset}/fold{fold}/{condition}; no adaptation",flush=True)
    write_json(root/"assessment_complete.json",dict(artifacts=hashes(artifacts),
        calibration_freeze=hashes([root/"calibration_freeze.json"]),Freddie="NOT RUN"))
