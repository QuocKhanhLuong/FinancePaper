"""Freeze all per-fold policies before opening any outer assessment."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json,time
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from financepaper.experiments import revision_study as s
from financepaper.experiments import revision_selection as q
from financepaper.experiments import temporal_pilot as p
from financepaper.experiments import robustness_followup as r
from financepaper.reliability.current import MASK_FEATURES
from financepaper.reliability.policy import fit_policy,one_condition_per_customer
from financepaper.training.calibration import PositiveSlopePlattCalibrator
from financepaper.data.taiwan import load_taiwan

BASELINES=("entropy","max_probability","prediction_variance","missing_fraction","missing_temporal",
           "attribution_variance","rank_instability","sign_instability","mc_revision")


def raw_scores(features,selectors):
    scores={n:selector["model"].predict_proba(features[selector["columns"]])[:,1] for n,selector in selectors.items()}
    for name in BASELINES:
        if name=="max_probability": value=1-np.maximum(features.probability,1-features.probability)
        elif name=="entropy": value=features.entropy/np.log(2)
        else: value=features[name]
        scores[name]=np.asarray(value,float)
    return scores


def calibrate_scores(raw,labels):
    z=logit(np.clip(raw,1e-7,1-1e-7))
    if len(np.unique(labels))<2:
        # No invented slope information for a degenerate pool.
        prevalence=(np.sum(labels)+.5)/(len(labels)+1)
        return PositiveSlopePlattCalibrator(slope=1e-12,intercept=float(logit(prevalence)))
    return PositiveSlopePlattCalibrator().fit(z,labels)


def score_with_bundle(features,bundle):
    raw=raw_scores(features,bundle["selectors"])
    return {name:dict(raw=value,calibrated=bundle["calibrators"][name].transform(logit(np.clip(value,1e-7,1-1e-7)))) for name,value in raw.items()}


def run_freeze(config_path,output,device=None):
    cfg,base=s.load_config(config_path); output=Path(output)
    if not (output/"baseline_manifest.json").exists(): raise RuntimeError("complete diagnostic gate first")
    if (output/"assessment_freeze.json").exists(): raise FileExistsError("policies are already frozen")
    if "**GO with one strategy" not in Path("docs/NEXT_MODEL_DECISION.md").read_text(): raise RuntimeError("document GO decision first")
    started=time.perf_counter(); sources=s.source_hashes()
    dataset=load_taiwan(Path(base["data_path"])); folds,_=s.make_partitions(dataset,cfg)
    device=p._select_device(device or base["device"])
    metrics=pd.read_csv(output/"revision_baseline_metrics.csv")
    decisions={}
    for fold,parts in enumerate(folds):
        d=output/f"fold_{fold}"; bundle=joblib.load(d/"predictors.joblib")
        selectors=joblib.load(d/"revision_baselines.joblib")
        # Predeclared anti-shortcut control; no validation tuning of its parameters.
        ft=pd.read_csv(d/"revision_train/current.csv"); vt=pd.read_csv(d/"revision_train/verification.csv")
        good=q.valid_rows(ft,vt)
        mask=q.make_selector("boosted",bundle["seed"])
        columns=["missing_fraction"]+MASK_FEATURES
        mask.fit(ft.loc[good,columns],vt.loc[good,"revision_event"].astype(int))
        selectors["mask_only_boosted"]=dict(model=mask,columns=columns)
        candidates=metrics[(metrics.fold==fold)&(metrics.condition=="pooled")&~metrics.method.str.startswith("prediction_")]
        chosen=candidates.sort_values(["average_precision","brier","method"],ascending=[False,True,True]).iloc[0].method
        fc,vc=q.collect(dataset,parts["revision_calibration"],bundle,cfg,base,device,bundle["seed"]+5000,d/"revision_calibration")
        fr,vr=q.collect(dataset,parts["release_calibration"],bundle,cfg,base,device,bundle["seed"]+6000,d/"release_calibration")
        ic=one_condition_per_customer(fc,bundle["seed"]+50)&q.valid_rows(fc,vc)
        ir=one_condition_per_customer(fr,bundle["seed"]+60)
        rawc=raw_scores(fc,selectors); rawr=raw_scores(fr,selectors)
        calibrators={}; policies={}
        for name,value in rawc.items():
            cal=calibrate_scores(value[ic],vc.loc[ic,"revision_event"].astype(int))
            calibrators[name]=cal
            probability=cal.transform(logit(np.clip(rawr[name][ir],1e-7,1-1e-7)))
            policies[name]={kind:fit_policy(probability,fr.loc[ir,"eligible_current"],vr.loc[ir,"revision_event"],
                conservative=kind=="conservative",risk_budget=cfg["risk_budget"]) for kind in ("empirical","conservative")}
        inference=dict(selectors=selectors,chosen=chosen,calibrators=calibrators,policies=policies)
        joblib.dump(inference,d/"release_model.joblib")
        decisions[str(fold)]=dict(chosen=chosen,calibration_eligible=int(ic.sum()),
            policies={n:{k:asdict(v) for k,v in pv.items()} for n,pv in policies.items()},
            calibration={n:v.to_dict() for n,v in calibrators.items()})
        print(f"fold {fold}: selected {chosen}; calibrated and froze policies",flush=True)
    p._write_json(output/"release_decisions.json",decisions)
    artifact_paths=[output/f"fold_{f}"/name for f in range(3) for name in ("predictors.joblib","release_model.joblib","partition_ids.json")]
    artifact_paths += [output/"release_decisions.json",Path("docs/NEXT_MODEL_DECISION.md"),Path("docs/REVISION_AWARE_MODEL_SPEC.md")]
    p._write_json(output/"assessment_freeze.json",dict(outer_opened=False,sources=sources,
        elapsed_seconds=time.perf_counter()-started,artifacts={str(path):sha256(path.read_bytes()).hexdigest() for path in artifact_paths}))


def run_assessment(config_path,output,device=None):
    cfg,base=s.load_config(config_path); output=Path(output)
    freeze=json.loads((output/"assessment_freeze.json").read_text())
    if (output/"assessment_manifest.json").exists(): raise FileExistsError("assessment is already completed")
    for file,expected in freeze["artifacts"].items():
        if sha256(Path(file).read_bytes()).hexdigest()!=expected: raise RuntimeError(f"frozen artifact changed: {file}")
    for file,expected in freeze["sources"].items():
        if sha256(Path(file).read_bytes()).hexdigest()!=expected: raise RuntimeError(f"source changed after freeze: {file}")
    started=time.perf_counter(); device=p._select_device(device or base["device"])
    dataset=load_taiwan(Path(base["data_path"])); folds,_=s.make_partitions(dataset,cfg)
    allmetrics=[];policies=[]; scores=[]
    for fold,parts in enumerate(folds):
        d=output/f"fold_{fold}"; target=d/"outer"; target.mkdir(exist_ok=True)
        bundle=joblib.load(d/"predictors.joblib"); release=joblib.load(d/"release_model.joblib")
        batches,masks,y=s.partition_batches(dataset,parts["outer"],bundle["context"]["preprocessor"],bundle["context"]["mar"],bundle["seed"]+7000)
        metric,preds=r.prediction_metrics({n:bundle["fits"][n] for n in s.CORE},batches,y,bundle["selection"],base,device)
        p._write_json(target/"prediction_metrics.json",metric);preds.to_csv(target/"predictions.csv",index=False)
        cfg_outer=dict(cfg,audit_records=cfg["outer_explanation_records"])
        _,positions,_=s.audit_explanations(bundle,batches,masks,base,cfg_outer,target,device)
        # Same people; q.collect gets the same masks by regenerating full outer
        # masks then subsetting, rather than drawing masks on a shorter frame.
        indices=parts["outer"][positions]
        f,v=collect_outer_same_masks(dataset,indices,bundle,cfg,base,device,batches,masks,positions,target)
        ss=score_with_bundle(f,release)
        eligible=q.valid_rows(f,v); current=f.eligible_current.to_numpy(bool)
        pooled=one_condition_per_customer(f,bundle["seed"]+70)
        for name,values in ss.items():
            for c in ("pooled",*s.CONDITIONS):
                ix=pooled if c=="pooled" else f.condition.eq(c).to_numpy()
                good=ix&eligible
                for kind in ("raw","calibrated"):
                    allmetrics.append(dict(fold=fold,method=name,condition=c,probability=kind,
                        **q.metric(v.loc[good,"revision_event"],values[kind][good])))
                for mode,policy in release["policies"][name].items():
                    emitted=policy.release(values["calibrated"],current)&ix
                    known=emitted&v.revision_event.notna().to_numpy()
                    events=float(v.loc[known,"revision_event"].sum()); n=int(emitted.sum())
                    policies.append(dict(fold=fold,method=name,selected=name==release["chosen"],condition=c,policy=mode,
                        total=int(ix.sum()),released=n,events=int(events),unknown=int((emitted&~known).sum()),
                        coverage=n/int(ix.sum()),risk=events/int(known.sum()) if known.sum() else None,
                        worst_case_risk=(events+(emitted&~known).sum())/n if n else None))
            sf=f[["record_id","condition","eligible_current"]].copy()
            sf["fold"]=fold;sf["method"]=name;sf["selected"]=name==release["chosen"]
            sf["revision_event"]=v.revision_event;sf["raw_score"]=values["raw"];sf["calibrated_score"]=values["calibrated"]
            sf["pooled"]=pooled
            for mode,policy in release["policies"][name].items(): sf[f"release_{mode}"]=policy.release(values["calibrated"],current)
            scores.append(sf)
        print(f"fold {fold}: outer assessment completed",flush=True)
    pd.DataFrame(allmetrics).to_csv(output/"outer_revision_metrics.csv",index=False)
    pd.DataFrame(policies).to_csv(output/"outer_release_metrics.csv",index=False)
    pd.concat(scores).to_csv(output/"outer_revision_scores.csv",index=False)
    p._write_json(output/"assessment_manifest.json",dict(status="complete",outer_opened=True,
        elapsed_seconds=time.perf_counter()-started,freeze_sha256=sha256((output/"assessment_freeze.json").read_bytes()).hexdigest(),
        artifacts=r.artifact_hashes(output)))


def collect_outer_same_masks(dataset,indices,bundle,cfg,base,device,batches,masks,positions,target):
    """Use precomputed matched masks. No generator monkey-patching/global state."""
    from financepaper.reliability.donors import ConditionalDonors
    from financepaper.reliability.current import CurrentEvidence,current_features
    prep=bundle["context"]["preprocessor"]; fit=bundle["fits"]["xgb25"]
    cal=PositiveSlopePlattCalibrator.from_dict(bundle["selection"]["models"]["xgb25"]["calibrator"])
    donor=ConditionalDonors(bundle["context"]["X_train"],neighbours=cfg["completion_neighbours"],max_reference=cfg["completion_donors"],seed=bundle["seed"])
    fs=[]
    v=pd.read_csv(target/"reason_records.csv");v=v[v.model=="xgb25"].reset_index(drop=True)
    for c in s.CONDITIONS:
        b=batches[c].take(positions); h=masks[c].iloc[positions]
        a=np.load(target/f"xgb25_{c}.npz")
        partial=dataset.X.iloc[indices].mask(h)
        cp=[];ca=[]
        for completed in donor.complete(partial,draws=cfg["completion_draws"],seed=bundle["seed"]+7013):
            ac=s.get_attr(fit,prep.transform(completed),bundle,base,device)
            cp.append(cal.transform(ac["input_logit"]));ca.append(ac["original"])
        f=current_features(CurrentEvidence(cal.transform(a["logits_before"]),h.to_numpy(bool),a["original_before"],np.stack(cp,1),np.stack(ca,1)))
        f.insert(0,"record_id",b.record_ids);f.insert(1,"condition",c)
        f["eligible_current"]=v[v.condition==c].preverification_eligible.to_numpy(bool)
        fs.append(f)
    f=pd.concat(fs,ignore_index=True)
    q.valid_rows(f,v)
    f.to_csv(target/"current.csv",index=False);v.to_csv(target/"verification.csv",index=False)
    return f,v
