"""Separate revision supervision from strictly current-only inference evidence."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from financepaper.data.taiwan import load_taiwan
from financepaper.evaluation.revision_audit import normalized_shift
from financepaper.reliability.current import CurrentEvidence, current_features, RECIPES
from financepaper.reliability.donors import ConditionalDonors
from financepaper.experiments import revision_study as s
from financepaper.experiments import temporal_pilot as p
from financepaper.experiments import robustness_followup as r
from financepaper.training.calibration import PositiveSlopePlattCalibrator


def collect(dataset, indices, bundle, cfg, base, device, seed, directory, *, completions=True):
    """Offline supervision producer. Output keeps features and verification separate."""
    directory=Path(directory)
    if (directory/"complete.json").exists():
        return pd.read_csv(directory/"current.csv"),pd.read_csv(directory/"verification.csv")
    directory.mkdir(parents=True,exist_ok=True)
    prep=bundle["context"]["preprocessor"]
    batches,masks,y=s.partition_batches(dataset,indices,prep,bundle["context"]["mar"],seed)
    fit=bundle["fits"]["xgb25"]
    cal=PositiveSlopePlattCalibrator.from_dict(bundle["selection"]["models"]["xgb25"]["calibrator"])
    donors=ConditionalDonors(bundle["context"]["X_train"],neighbours=cfg["completion_neighbours"],
        max_reference=cfg["completion_donors"],seed=bundle["seed"])
    full=s.get_attr(fit,batches["complete"],bundle,base,device)
    features,labels=[],[]
    for c,b in batches.items():
        before=full if c=="complete" else s.get_attr(fit,b,bundle,base,device)
        h=masks[c].to_numpy(bool)
        # Erase hidden truth before constructing the donor input.
        partial=dataset.X.iloc[indices].mask(masks[c])
        draws=donors.complete(partial,draws=cfg["completion_draws"],seed=seed+13)
        cp,ca=[],[]
        for completed in draws:
            if c=="complete":
                a=before
            elif completions:
                a=s.get_attr(fit,prep.transform(completed),bundle,base,device)
            else:
                a={"input_logit":p._raw_logits(fit,prep.transform(completed),device,256)}
            cp.append(cal.transform(a["input_logit"]))
            if completions: ca.append(a["original"])
        evidence=CurrentEvidence(cal.transform(before["input_logit"]),h,before["original"],
            np.stack(cp,axis=1),np.stack(ca,axis=1) if completions else None)
        f=current_features(evidence)
        f.insert(0,"record_id",b.record_ids)
        f.insert(1,"condition",c)
        rr=r.reason_records("xgb25",c,b.record_ids,h,before,full,.01,base)
        # Only current validity/eligibility are allowed in release inference.
        f["eligible_current"]=[row["preverification_eligible"] for row in rr]
        for i,row in enumerate(rr):
            row["normalized_shift"]=normalized_shift(before["original"][i],full["original"][i],h[i])
            row["y"]=y[i]
            row["calibrated_probability"]=evidence.probability[i]
            row["calibrated_full"]=cal.transform(full["input_logit"])[i]
        labels.extend(rr)
        features.append(f)
        print(f"  collected {directory.name}/{c}: {len(b)} customers",flush=True)
    f,v=pd.concat(features,ignore_index=True),pd.DataFrame(labels)
    f.to_csv(directory/"current.csv",index=False)
    v.to_csv(directory/"verification.csv",index=False)
    p._write_json(directory/"complete.json",dict(n_customers=len(indices),seed=seed,completions=completions))
    return f,v


def metric(y,score):
    y,score=np.asarray(y,int),np.clip(np.asarray(score,float),1e-9,1-1e-9)
    if not len(y): return dict(n=0,events=0,roc_auc=None,average_precision=None,brier=None,log_loss=None)
    return dict(n=len(y),events=int(y.sum()),roc_auc=float(roc_auc_score(y,score)) if len(set(y))==2 else None,
        average_precision=float(average_precision_score(y,score)) if y.sum() else None,
        brier=float(brier_score_loss(y,score)),log_loss=float(log_loss(y,score,labels=[0,1])))


def valid_rows(features,verification):
    if not np.array_equal(features[["record_id","condition"]].to_numpy(),verification[["record_id","condition"]].to_numpy()):
        raise AssertionError("features and supervision are misaligned")
    return features.eligible_current.to_numpy(bool)&verification.revision_event.notna().to_numpy()


def make_selector(kind,seed):
    if kind=="logistic": return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed))
    return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=40,
        learning_rate=.05,l2_regularization=5,early_stopping=False,random_state=seed)


def run_baselines(config_path,output,device=None):
    """Experiment B; do not read any outer assessment or fit release policies."""
    cfg,base=s.load_config(config_path)
    output=Path(output)
    if not (output/"audit_manifest.json").exists(): raise RuntimeError("finish metric audit first")
    if (output/"baseline_manifest.json").exists(): raise FileExistsError("baseline stage already complete")
    sources=s.source_hashes(); started=time.perf_counter()
    p._write_json(output/"baseline_freeze.json",dict(sources=sources,recipes=RECIPES,outer_opened=False))
    dataset=load_taiwan(Path(base["data_path"]))
    folds,_=s.make_partitions(dataset,cfg)
    device=p._select_device(device or base["device"])
    rows=[]
    for fold,parts in enumerate(folds):
        d=output/f"fold_{fold}"; bundle=joblib.load(d/"predictors.joblib")
        ft,vt=collect(dataset,parts["revision_train"],bundle,cfg,base,device,bundle["seed"]+4000,d/"revision_train")
        fd,vd=collect(dataset,parts["diagnostic"],bundle,cfg,base,device,bundle["seed"]+3000,d/"revision_diagnostic")
        it,idv=valid_rows(ft,vt),valid_rows(fd,vd)
        yt=vt.loc[it,"revision_event"].astype(int)
        fitted={}; pred=fd[["record_id","condition","eligible_current"]].copy()
        pred["revision_event"]=vd.revision_event
        for recipe,columns in RECIPES.items():
            for kind in ("logistic","boosted"):
                name=f"{recipe}_{kind}"
                model=make_selector(kind,bundle["seed"])
                model.fit(ft.loc[it,columns],yt)
                score=model.predict_proba(fd[columns])[:,1]
                pred[name]=score
                fitted[name]=dict(model=model,columns=columns)
                for c in ("pooled",*s.CONDITIONS[1:]):
                    choose=idv & (True if c=="pooled" else fd.condition.eq(c).to_numpy())
                    rows.append(dict(fold=fold,method=name,condition=c,**metric(vd.loc[choose,"revision_event"],score[choose])))
        joblib.dump(fitted,d/"revision_baselines.joblib")
        pred.to_csv(d/"revision_diagnostic_scores.csv",index=False)
        print(f"fold {fold}: revision baselines fit",flush=True)
    pd.DataFrame(rows).to_csv(output/"revision_baseline_metrics.csv",index=False)
    p._write_json(output/"baseline_manifest.json",dict(outer_opened=False,elapsed_seconds=time.perf_counter()-started,
        sources=sources,artifacts=r.artifact_hashes(output)))
