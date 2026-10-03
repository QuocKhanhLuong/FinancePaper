"""Outcome-only analysis of pre-frozen validation; no fits or policy selection."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from financepaper.data.schema import FEATURE_NAMES,CATEGORICAL_FEATURES
from financepaper.data.polish import load_polish
from financepaper.experiments.decisive_validation import (read_pair,write_json,TW_ENV,PL_ENV,POLICY_VARIANTS,hashes)
from financepaper.reliability.validation import (TAIWAN_GROUPS,POLISH_GROUPS,POLISH_NAMES,aggregate_groups,revision_arrays)
from financepaper.evaluation.temporal_metrics import reliability_curve,detailed_prediction_metrics


def metric(y,p):
    y,p=np.asarray(y,int),np.asarray(p,float);q=np.clip(p,1e-9,1-1e-9)
    return dict(n=len(y),events=int(y.sum()),prevalence=y.mean() if len(y) else np.nan,
        roc_auc=roc_auc_score(y,p) if len(set(y))==2 else np.nan,
        average_precision=average_precision_score(y,p) if y.sum() else np.nan,
        brier=brier_score_loss(y,q) if len(y) else np.nan,
        log_loss=log_loss(y,q,labels=[0,1]) if len(y) else np.nan)


def cluster_map(dataset,external_path):
    if dataset=="taiwan":
        return None
    X,_,clusters=load_polish(external_path)
    return dict(zip(X.index,clusters))


def weights(frame,cluster_lookup,draws=1000):
    cluster=frame.record_id.map(cluster_lookup).to_numpy() if cluster_lookup is not None else frame.record_id.to_numpy()
    ids,inverse=np.unique(cluster,return_inverse=True)
    # Preserve outer folds: each Taiwan record belongs to one outer fold.
    folds=frame.groupby(pd.Series(cluster,index=frame.index)).fold.first().reindex(ids).to_numpy()
    rng=np.random.default_rng(781);result=np.zeros((draws,len(ids)),dtype=np.float64)
    for fold in np.unique(folds):
        positions=np.flatnonzero(folds==fold)
        for r in range(draws):
            result[r,positions]=np.bincount(rng.integers(len(positions),size=len(positions)),minlength=len(positions))
    return result,inverse


def ratio_ci(frame,numerator,denominator,lookup):
    w,inv=weights(frame,lookup)
    n=np.bincount(inv,weights=np.asarray(numerator,float),minlength=w.shape[1])
    d=np.bincount(inv,weights=np.asarray(denominator,float),minlength=w.shape[1])
    draws=w@n;den=w@d
    values=np.divide(draws,den,out=np.full_like(draws,np.nan),where=den>0)
    finite=values[np.isfinite(values)]
    bounds=np.quantile(finite,[.025,.975]).tolist() if len(finite) else [np.nan,np.nan]
    return (float(n.sum()/d.sum()) if d.sum() else np.nan, bounds[0], bounds[1])


def decoupling(frame,dataset,model="xgb25"):
    rows=[]
    for (variant,condition),g in frame.groupby(["variant","condition"]):
        eligible=g[g.eligible & g.event.notna()]
        eventcols=["event"]+[c for c in eligible if c.startswith("event_gap_")]
        for eventcol in eventcols:
            for cutoff in (.01,.02,.05):
                stable=eligible.prob_shift<=cutoff;e=eligible[eventcol].eq(1)
                counts=[int((stable&~e).sum()),int((stable&e).sum()),int((~stable&~e).sum()),int((~stable&e).sum())]
                rows.append(dict(dataset=dataset,model=model,variant=variant,condition=condition,tolerance=eventcol,cutoff=cutoff,
                    n_total=len(g),eligible=len(eligible),eligibility=len(eligible)/len(g),overall_revision=e.mean(),stable_n=int(stable.sum()),
                    revision_among_stable=e[stable].mean() if stable.any() else np.nan,
                    **dict(zip("ABCD",counts)),**{f"{r}_percent":100*c/len(eligible) if len(eligible) else np.nan for r,c in zip("ABCD",counts)}))
    return pd.DataFrame(rows)


def control_tables(root,historical):
    rows=[]
    for fold in range(3):
        for model in ("lr","additive","vanilla","mask_delta"):
            for condition in ("mcar10","mcar30","mar30"):
                a=np.load(historical/f"fold_{fold}/outer/{model}_{condition}.npz")
                before,full,h=a["original_before"],a["original_full"],a["hidden"]
                valid=a["valid_before"]&a["valid_full"]
                for variant in ("feature3","group1","group2","group3"):
                    k=int(variant[-1]);x,z,mask=before,full,h
                    if variant.startswith("group"):
                        x,mask=aggregate_groups(before,h,FEATURE_NAMES,TAIWAN_GROUPS)
                        z,_=aggregate_groups(full,h,FEATURE_NAMES,TAIWAN_GROUPS)
                    result=revision_arrays(x,z,mask,k=k)
                    rows.append(pd.DataFrame(dict(record_id=a["record_ids"],fold=fold,model=model,variant=variant,condition=condition,
                        eligible=result["eligible"]&valid,event=np.where(result["eligible"]&valid,result["event"].astype(float),np.nan),
                        prob_shift=abs(expit(a["logits_before"])-expit(a["logits_full"])),
                        shift=(abs(x-z)*~mask).sum(1)/np.maximum(((abs(x)+abs(z))*~mask).sum(1),1e-12))))
    return pd.concat(rows,ignore_index=True)


def paired_detection(scores,dataset,lookup):
    rows=[]
    for variant in ("feature3","group1","group2","group3"):
        for condition in (("mcar10","mcar30","mar30") if dataset=="taiwan" else PL_ENV):
            g=scores[(scores.variant==variant)&(scores.condition==condition)&scores.eligible].dropna(subset=["event"])
            wide=g.pivot(index=["fold","record_id","event"],columns="method",values="calibrated_score").reset_index()
            w,inv=weights(wide,lookup);y=wide.event.to_numpy(int)
            for other in ("prediction_only","learned_selector","prediction_variance","attribution_variance","rank_instability","mc4","mc16"):
                a,b=wide.mc8.to_numpy(),wide[other].to_numpy()
                diffs=[]
                for weight in w:
                    ww=weight[inv]
                    if not (ww*y).sum() or not (ww*(1-y)).sum():
                        continue
                    diffs.append(average_precision_score(y,a,sample_weight=ww)-average_precision_score(y,b,sample_weight=ww))
                if diffs:
                    lo,hi=np.quantile(diffs,[.025,.975])
                    rows.append(dict(dataset=dataset,variant=variant,condition=condition,first="mc8",second=other,metric="AP",
                        difference=average_precision_score(y,a)-average_precision_score(y,b),low=lo,high=hi,draws=len(diffs)))
    return pd.DataFrame(rows)


def completion_audit(root,dataset,folders):
    names,groups=(FEATURE_NAMES,TAIWAN_GROUPS) if dataset=="taiwan" else (POLISH_NAMES,POLISH_GROUPS)
    categories=set(CATEGORICAL_FEATURES) if dataset=="taiwan" else set()
    rows=[];case_rows=[];cell_rows=[]
    for fold,folder in folders:
        for condition in (TW_ENV if dataset=="taiwan" else PL_ENV):
            c,v=read_pair(folder/"outer"/condition);art=c["artificial"];draw=c["completions"][:,:8];truth=v["truth"]
            rr=revision_arrays(c["phi"],v["phi"],c["hidden"])
            mc=revision_arrays(c["phi"],c["completion_phi"][:,:8],c["hidden"])["event"].mean(1)
            shift=abs(expit(c["logits"])-expit(v["logits"]));ok=rr["eligible"]&c["valid"]&v["valid"]
            hit=np.zeros_like(truth,dtype=float);pct=np.zeros_like(truth,dtype=float)
            for j,name in enumerate(names):
                observed_truth=art[:,j]
                values=draw[:,:,j]
                if name in categories:
                    support=(values==truth[:,j,None]).any(1);percentile=np.full(len(c["phi"]),np.nan)
                else:
                    support=(truth[:,j]>=np.nanmin(values,1))&(truth[:,j]<=np.nanmax(values,1))
                    percentile=((values<truth[:,j,None]).mean(1)+(values==truth[:,j,None]).mean(1)*.5)
                hit[:,j]=support;pct[:,j]=percentile
                for i in np.flatnonzero(observed_truth):
                    cell_rows.append(dict(dataset=dataset,fold=fold,condition=condition,record_id=int(c["record_ids"][i]),feature=name,
                        categorical=name in categories,truth_supported=bool(support[i]),truth_percentile=percentile[i],
                        truth=float(truth[i,j]),completion_min=float(np.min(values[i])),completion_max=float(np.max(values[i]))))
            supported=(hit*art).sum(1)/np.maximum(art.sum(1),1)
            gap=[]
            for i in range(len(truth)):
                obs=np.sort(c["phi"][i,~c["hidden"][i]])[::-1]
                gap.append(obs[2]-obs[3] if len(obs)>3 else np.nan)
            unique=np.array([len(set(ids)) for ids in c["donor_ids"][:,:8]])
            types={"high_MC_no_revision":ok&(mc>=.5)&~rr["event"],
                   "low_MC_revision":ok&(mc<=.125)&rr["event"],
                   "stable_prediction_revised":ok&(shift<=.02)&rr["event"],
                   "unstable_prediction_stable_reasons":ok&(shift>.02)&~rr["event"]}
            for name,choose in types.items():
                rows.append(dict(dataset=dataset,fold=fold,condition=condition,case_type=name,n=int(choose.sum()),
                    mean_truth_support=supported[choose].mean() if choose.any() else np.nan,
                    near_tie_fraction=(np.asarray(gap)[choose]<=.01).mean() if choose.any() else np.nan,
                    unique_donors=unique[choose].mean() if choose.any() else np.nan))
                ids=np.flatnonzero(choose);ids=ids[np.argsort(c["record_ids"][ids])][:5]
                for i in ids:
                    rid=int(c["record_ids"][i]);r=rr["reasons"][i]
                    case_rows.append(dict(dataset=dataset,fold=fold,condition=condition,case_type=name,record_id=rid,
                        prob_partial=expit(c["logits"][i]),prob_restored=expit(v["logits"][i]),prob_shift=shift[i],
                        event=bool(rr["event"][i]),mc8=mc[i],rank_gap=gap[i],truth_support=supported[i],unique_donors=unique[i],
                        reasons=json.dumps([names[j] for j in r]),before_scores=json.dumps(c["phi"][i,r].tolist()),
                        restored_scores=json.dumps(v["phi"][i,r].tolist()),hidden_features=json.dumps([names[j] for j in np.flatnonzero(art[i])]),
                        true_hidden=json.dumps(truth[i,art[i]].tolist()),completion_min=json.dumps(np.min(draw[i,:,art[i]],axis=1).tolist()),
                        completion_max=json.dumps(np.max(draw[i,:,art[i]],axis=1).tolist())))
            for group,members in groups.items():
                idx=[names.index(n) for n in members];has=art[:,idx].any(1)&ok
                rows.append(dict(dataset=dataset,fold=fold,condition=condition,case_type="missing_group_"+group,n=int(has.sum()),
                    mean_mc=mc[has].mean() if has.any() else np.nan,verified_revision=rr["event"][has].mean() if has.any() else np.nan,
                    mean_truth_support=supported[has].mean() if has.any() else np.nan,unique_donors=unique[has].mean() if has.any() else np.nan))
    return pd.DataFrame(rows),pd.DataFrame(case_rows),pd.DataFrame(cell_rows)


def savefig(out,name):
    plt.tight_layout();plt.savefig(out/f"{name}.png",dpi=180);plt.savefig(out/f"{name}.pdf");plt.close()


def main(root):
    out=root/"analysis";out.mkdir(exist_ok=True)
    cfg=json.loads((root/"protocol_freeze.json").read_text())["config"]
    alltables={};allmetrics=[];allpolicies=[];allcases=[];allpaired=[];allcorrelations=[];allconditional=[];allcompletion=[];allrepresentative=[];allcells=[]
    for dataset in ("taiwan","polish"):
        folders=[(f,root/"taiwan"/f"fold_{f}") for f in range(3)] if dataset=="taiwan" else [(0,root/"polish")]
        table=pd.concat([pd.read_csv(p/"outer_derived.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        scores=pd.concat([pd.read_csv(p/"scores.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        released=pd.concat([pd.read_csv(p/"releases.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        alltables[dataset]=(table,scores,released)
        lookup=cluster_map(dataset,cfg["external_path"])
        allcases.append(decoupling(table,dataset,"xgb25" if dataset=="taiwan" else "xgb"))
        for (fold,variant,condition,method),g in scores.groupby(["fold","variant","condition","method"]):
            g=g[g.eligible & g.event.notna()]
            if len(g):
                for kind in ("raw_score","calibrated_score"):
                    allmetrics.append(dict(dataset=dataset,fold=fold,variant=variant,condition=condition,method=method,probability=kind,**metric(g.event,g[kind])))
        for (variant,method,alpha,family,conservative),g in released.groupby(["variant","method","alpha","family","conservative"]):
            conditions=TW_ENV if dataset=="taiwan" else PL_ENV
            for condition in ("overall",*conditions):
                f=g[g.condition.isin(conditions)] if condition=="overall" else g[g.condition==condition]
                a=f.released.to_numpy(bool);e=f.event.fillna(1).to_numpy(float)
                # Paired customer CIs for all main policies and K ablations.
                if method.startswith("mc") or method in ("prediction_only","learned_selector"):
                    cov,cl,ch=ratio_ci(f,a,np.ones(len(f)),lookup)
                    risk,rl,rh=ratio_ci(f,a*e,a,lookup)
                else:
                    cov=a.mean();risk=e[a].mean() if a.any() else np.nan;cl=ch=rl=rh=np.nan
                allpolicies.append(dict(dataset=dataset,variant=variant,method=method,alpha=alpha,family=family,conservative=conservative,
                    condition=condition,n=len(f),released=int(a.sum()),events=int((e*a).sum()),coverage=cov,coverage_low=cl,coverage_high=ch,
                    risk=risk,risk_low=rl,risk_high=rh))
        print(f"{dataset}: policy intervals complete",flush=True)
        allpaired.append(paired_detection(scores,dataset,lookup))
        for (variant,condition),g in table[table.variant.isin(POLICY_VARIANTS)&table.eligible].groupby(["variant","condition"]):
            cols=["score_mc8","score_rank_instability","score_sign_instability","score_attribution_variance","score_prediction_variance","missing_fraction"]
            corr=g[cols].corr(method="spearman")["score_mc8"]
            for name,value in corr.items():
                allcorrelations.append(dict(dataset=dataset,variant=variant,condition=condition,comparator=name,spearman=value))
            for count,sub in g.groupby(np.rint(g.missing_fraction*(23 if dataset=="taiwan" else 64)).astype(int)):
                if len(sub)<30 or sub.event.nunique()<2:
                    continue
                for method in ("mc8","rank_instability","attribution_variance","prediction_variance","prediction_only"):
                    allconditional.append(dict(dataset=dataset,variant=variant,condition=condition,missing_count=count,method=method,**metric(sub.event,sub[f"score_{method}"])))
        diagnostics,cases,cells=completion_audit(root,dataset,folders)
        allcompletion.append(diagnostics);allrepresentative.append(cases);allcells.append(cells)
        print(f"{dataset}: paired detection and completion audit complete",flush=True)
    controls=control_tables(root,Path(cfg["historical_root"]))
    for model,g in controls.groupby("model"):
        allcases.append(decoupling(g,"taiwan",model))
    # External LR control shares frozen raw-logit explanation definition.
    records=[]
    for condition in PL_ENV:
        a=np.load(root/"polish/outer"/f"lr_{condition}.npz")
        for variant in POLICY_VARIANTS:
            k=int(variant[-1]);x,z,h=a["original_before"],a["original_full"],a["hidden"]
            if variant.startswith("group"):
                x,mask=aggregate_groups(x,h,POLISH_NAMES,POLISH_GROUPS);z,_=aggregate_groups(z,h,POLISH_NAMES,POLISH_GROUPS);h=mask
            rr=revision_arrays(x,z,h,k=k);valid=a["valid_before"]&a["valid_full"]
            records.append(pd.DataFrame(dict(record_id=a["record_ids"],variant=variant,condition=condition,eligible=rr["eligible"]&valid,
                event=np.where(rr["eligible"]&valid,rr["event"].astype(float),np.nan),prob_shift=abs(expit(a["logits_before"])-expit(a["logits_full"])))))
    allcases.append(decoupling(pd.concat(records),"polish","lr"))
    phenomena=pd.concat(allcases);metrics=pd.DataFrame(allmetrics);policies=pd.DataFrame(allpolicies)
    phenomena.to_csv(out/"A_phenomenon_sensitivity.csv",index=False)
    metrics.to_csv(out/"B_detector_metrics.csv",index=False)
    metrics.groupby(["dataset","variant","condition","method","probability"])[["roc_auc","average_precision","brier","log_loss"]].agg(["mean","std"]).to_csv(out/"B_detector_mean_sd.csv")
    metrics[metrics.method.str.match(r"mc\d+$")].to_csv(out/"C_K_ablation.csv",index=False)
    policies.to_csv(out/"D_policy_intervals.csv",index=False)
    pd.concat(allpaired).to_csv(out/"paired_AP_differences.csv",index=False)
    pd.DataFrame(allcorrelations).to_csv(out/"MC_correlations.csv",index=False)
    pd.DataFrame(allconditional).to_csv(out/"fixed_missing_count_detection.csv",index=False)
    pd.concat(allcompletion).to_csv(out/"completion_failure_groups.csv",index=False)
    pd.concat(allrepresentative).to_csv(out/"representative_cases.csv",index=False)
    pd.concat(allcells).to_csv(out/"verification_only_completion_support.csv",index=False)
    pred=pd.read_csv(root/"polish/predictions.csv");pm=[]
    for (model,condition),g in pred.groupby(["model","condition"]):
        for prob in ("prob_raw","prob_calibrated"):
            result=detailed_prediction_metrics(g.y,g[prob],.5)
            pm.append(dict(model=model,condition=condition,probability=prob,**{k:v for k,v in result.items() if k not in ("classwise","calibration_bins")},
                **{f"class_{c}_{k}":v for c,values in result["classwise"].items() for k,v in values.items()}))
    pd.DataFrame(pm).to_csv(out/"E_external_prediction.csv",index=False)
    figures(out,alltables,phenomena,metrics,policies)
    write_json(out/"report_manifest.json",dict(bootstrap_draws=1000,unit="original customer; Polish exact-feature duplicate clusters",
        artifacts=hashes([p for p in out.iterdir() if p.suffix in (".csv",".png",".pdf")]),method_changes=False))
    print("All validation tables and figures written.",flush=True)


def figures(out,tables,phenomena,metrics,policies):
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,dataset in enumerate(("taiwan","polish")):
        for j,variant in enumerate(("feature3","group2")):
            g=tables[dataset][0];g=g[(g.condition=="mcar30")&(g.variant==variant)&g.eligible]
            b=(g.prob_shift<=.02)&g.event.eq(1);ax=axes[i,j]
            ax.scatter(g.prob_shift,g["shift"],s=6,color="gray",alpha=.25)
            ax.scatter(g.loc[b,"prob_shift"],g.loc[b,"shift"],s=10,color="firebrick",alpha=.7,label=f"Region B: {b.sum()}")
            ax.axvline(.02,ls="--",color="black",lw=.7);ax.set(title=f"{dataset}: {variant}, MCAR30",xlabel="Absolute raw probability shift",ylabel="Observed normalized attribution shift");ax.legend()
    savefig(out,"01_decoupling")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset in zip(axes,("taiwan","polish")):
        g=phenomena[(phenomena.dataset==dataset)&phenomena.model.isin(["xgb25","xgb"])&(phenomena.cutoff==.02)&(phenomena.tolerance=="event")&phenomena.variant.isin(POLICY_VARIANTS)]
        g=g[g.condition.isin(TW_ENV if dataset=="taiwan" else PL_ENV)]
        g.pivot(index="condition",columns="variant",values="overall_revision").plot.bar(ax=ax)
        ax.set(title=dataset,ylabel="Verified revision among eligible",xlabel="Missing environment")
    savefig(out,"02_feature_group_revision")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for dataset in ("taiwan","polish"):
        g=metrics[(metrics.dataset==dataset)&(metrics.variant=="feature3")&(metrics.condition=="mcar30")&(metrics.probability=="calibrated_score")&metrics.method.str.match(r"mc\d+$")].copy()
        g["K"]=g.method.str[2:].astype(int)
        for ax,col in zip(axes[:2],("average_precision","roc_auc")):
            m=g.groupby("K")[col].mean();ax.plot(m.index,m,marker="o",label=dataset);ax.set(xlabel="K",ylabel=col);ax.legend()
    if (out/"runtime_repetitions.csv").exists():
        t=pd.read_csv(out/"runtime_repetitions.csv");t=t[(t.batch_size==128)&t["mode"].ne("learned")].copy();t["K"]=t["mode"].astype(int)
        q=t.groupby("K").seconds.agg(["mean","std"]);axes[2].errorbar(q.index,q["mean"],yerr=q["std"],marker="o",capsize=3)
    axes[2].set(xlabel="K",ylabel="Seconds / batch128, mean ± SD")
    savefig(out,"03_K_compute")
    curves=[];fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,dataset in enumerate(("taiwan","polish")):
        for j,variant in enumerate(("feature3","group2")):
            for method in ("prediction_only","learned_selector","mc2","mc4","mc8","mc16"):
                s=tables[dataset][1];s=s[(s.variant==variant)&(s.condition=="mcar30")&(s.method==method)]
                curve=pd.DataFrame(reliability_curve(s.calibrated_score,s.event,s.eligible,len(s)))
                if len(curve):
                    axes[i,j].plot(curve.coverage,curve.revision_rate,label=method)
                    curves.append(curve.assign(dataset=dataset,variant=variant,condition="mcar30",method=method))
            axes[i,j].axhline(.1,ls="--",color="black",lw=.7);axes[i,j].set(title=f"{dataset}: {variant}",xlabel="Coverage / all customers",ylabel="Observed revision risk");axes[i,j].legend(fontsize=7)
    savefig(out,"04_selective_explanation");pd.concat(curves).to_csv(out/"risk_coverage_curves.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,dataset in enumerate(("taiwan","polish")):
        for j,variant in enumerate(("feature3","group2")):
            for family in ("pooled","stratified","robust"):
                g=policies[(policies.dataset==dataset)&(policies.variant==variant)&(policies.method=="mc8")&(policies.alpha==.1)&(policies.family==family)&policies.conservative&policies.condition.ne("overall")]
                axes[i,j].scatter(g.coverage,g.risk,label=family)
                for row in g.itertuples():
                    if np.isfinite(row.risk):
                        axes[i,j].annotate(row.condition,(row.coverage,row.risk),fontsize=6)
                    else:
                        axes[i,j].text(.02,.96,f"{family}/{row.condition}: no release",transform=axes[i,j].transAxes,va="top",fontsize=7)
            axes[i,j].axhline(.1,color="black",ls="--",lw=.7);axes[i,j].set(title=f"{dataset}: {variant}, conservative",xlabel="Coverage",ylabel="Observed revision risk");axes[i,j].legend(fontsize=7)
    savefig(out,"05_policy_comparison")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,dataset in zip(axes,("taiwan","polish")):
        g=phenomena[(phenomena.dataset==dataset)&phenomena.model.isin(["xgb25","xgb"])&(phenomena.cutoff==.02)&(phenomena.tolerance=="event")&(phenomena.condition=="mcar30")&phenomena.variant.isin(POLICY_VARIANTS)]
        q=g.set_index("variant")[[f"{c}_percent" for c in "ABCD"]].rename(columns={
            "A_percent":"A: both stable", "B_percent":"B: prediction stable, reasons revised",
            "C_percent":"C: prediction shifts, reasons stable", "D_percent":"D: both change"})
        q.plot.bar(stacked=True,ax=ax);ax.set(title=f"{dataset}: separate target, MCAR30",ylabel="Percent of eligible customers",xlabel="Reason definition")
    savefig(out,"06_external_regions")


if __name__=="__main__":
    main(Path("outputs/decisive_validation"))
