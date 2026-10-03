"""Predeclared sensitivity summaries and unselected baseline; no method changes."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import runpy,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score


def main(root):
    report=runpy.run_path("scripts/report_decisive_validation.py")
    out=root/"analysis";cfg=json.loads((root/"protocol_freeze.json").read_text())["config"]
    baseline=[];phenomenon=[];natural=[];common=[];differences=[];policydiff=[];smallest=[]
    for dataset in ("taiwan","polish"):
        folders=[(f,root/"taiwan"/f"fold_{f}") for f in range(3)] if dataset=="taiwan" else [(0,root/"polish")]
        frame=pd.concat([pd.read_csv(p/"outer_derived.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        scores=pd.concat([pd.read_csv(p/"scores.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        releases=pd.concat([pd.read_csv(p/"releases.csv").assign(fold=f) for f,p in folders],ignore_index=True)
        lookup=report["cluster_map"](dataset,cfg["external_path"])
        for (variant,condition),g in frame[frame.variant.isin(["feature3","group1","group2","group3"])&frame.condition.ne("complete")].groupby(["variant","condition"]):
            a=g.eligible.to_numpy(bool);e=g.event.fillna(1).to_numpy(float)
            cov,cl,ch=report["ratio_ci"](g,a,np.ones(len(g)),lookup)
            risk,rl,rh=report["ratio_ci"](g,a*e,a,lookup)
            baseline.append(dict(dataset=dataset,variant=variant,condition=condition,coverage=cov,coverage_low=cl,coverage_high=ch,risk=risk,risk_low=rl,risk_high=rh))
            for cutoff in (.01,.02,.05):
                for probability in ("prob_shift","calibrated_shift"):
                    stable=a&(g[probability].to_numpy()<=cutoff)
                    rate,lo,hi=report["ratio_ci"](g,stable*e,stable,lookup)
                    phenomenon.append(dict(dataset=dataset,variant=variant,condition=condition,cutoff=cutoff,probability_scale=probability,
                        stable_n=int(stable.sum()),B=int((stable*e).sum()),B_given_stable=rate,low=lo,high=hi))
            if dataset=="polish":
                for has,g2 in g.groupby(g.natural_fraction>0):
                    eligible=g2[g2.eligible];stable=eligible[eligible.prob_shift<=.02]
                    natural.append(dict(variant=variant,condition=condition,has_natural_missing=has,n=len(g2),eligible=len(eligible),
                        revision=eligible.event.mean(),stable_n=len(stable),B=int(stable.event.sum()),B_given_stable=stable.event.mean()))
            if variant!="feature3":
                f=frame[(frame.variant=="feature3")&(frame.condition==condition)]
                pair=g.merge(f,on=["fold","record_id","condition"],suffixes=("_group","_feature"))
                pair=pair[pair.eligible_group & pair.eligible_feature]
                common.append(dict(dataset=dataset,variant=variant,condition=condition,common_eligible=len(pair),
                    feature_revision=pair.event_feature.mean(),group_revision=pair.event_group.mean(),
                    group_only_events=int(((pair.event_group==1)&(pair.event_feature==0)).sum()),
                    feature_only_events=int(((pair.event_group==0)&(pair.event_feature==1)).sum())))
        for variant in ("feature3","group1","group2","group3"):
            for condition in (("mcar10","mcar30","mar30") if dataset=="taiwan" else ("mcar10","mcar30")):
                g=scores[(scores.variant==variant)&(scores.condition==condition)&scores.eligible].dropna(subset=["event"])
                wide=g.pivot(index=["fold","record_id","event"],columns="method",values="calibrated_score").reset_index()
                w,inv=report["weights"](wide,lookup)
                for other in ("prediction_only","learned_selector","rank_instability","mc4","mc16"):
                    for metric,fn in (("AP",average_precision_score),("AUROC",roc_auc_score)):
                        estimates=[];draw_values=[]
                        for _,part in wide.groupby("fold"):
                            idx=part.index.to_numpy();y=part.event.to_numpy(int);a=part.mc8.to_numpy();b=part[other].to_numpy()
                            if len(set(y))<2:
                                continue
                            estimates.append(fn(y,a)-fn(y,b));vals=[]
                            for weight in w:
                                ww=weight[inv[idx]]
                                vals.append(fn(y,a,sample_weight=ww)-fn(y,b,sample_weight=ww) if (ww*y).sum() and (ww*(1-y)).sum() else np.nan)
                            draw_values.append(vals)
                        if estimates:
                            values=np.nanmean(draw_values,axis=0);lo,hi=np.nanquantile(values,[.025,.975])
                            differences.append(dict(dataset=dataset,variant=variant,condition=condition,first="mc8",second=other,metric=metric,
                                difference=np.mean(estimates),low=lo,high=hi,aggregation="mean within-fold metric difference",draws=1000))
        # Same bootstrap clusters for policy contrasts; never compare independent resamples.
        g=releases[(releases.method=="mc8")&releases.conservative&releases.alpha.eq(.1)]
        for (variant,condition),part in g.groupby(["variant","condition"]):
            if condition in ("complete","mcar20"):
                continue
            wide=part.pivot(index=["fold","record_id","event"],columns="family",values="released").reset_index()
            w,inv=report["weights"](wide,lookup);e=wide.event.fillna(1).to_numpy();n=len(wide)
            for other in ("pooled","stratified"):
                a=wide.robust.to_numpy(float);b=wide[other].to_numpy(float)
                totals=w@np.bincount(inv,minlength=w.shape[1])
                an=w@np.bincount(inv,weights=a,minlength=w.shape[1]);bn=w@np.bincount(inv,weights=b,minlength=w.shape[1])
                ae=w@np.bincount(inv,weights=a*e,minlength=w.shape[1]);be=w@np.bincount(inv,weights=b*e,minlength=w.shape[1])
                for met,point,vals in (("coverage",(a-b).mean(),(an-bn)/totals),
                    ("risk",np.sum(a*e)/a.sum()-np.sum(b*e)/b.sum() if a.sum() and b.sum() else np.nan,
                     np.divide(ae,an,out=np.full_like(ae,np.nan),where=an>0)-np.divide(be,bn,out=np.full_like(be,np.nan),where=bn>0))):
                    finite=vals[np.isfinite(vals)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
                    policydiff.append(dict(dataset=dataset,variant=variant,condition=condition,first="robust",second=other,metric=met,difference=point,low=lo,high=hi))
        print(dataset,"supplement complete",flush=True)
    pd.DataFrame(baseline).to_csv(out/"release_all_eligible.csv",index=False)
    pd.DataFrame(phenomenon).to_csv(out/"region_B_intervals.csv",index=False)
    pd.DataFrame(natural).to_csv(out/"external_natural_missingness_strata.csv",index=False)
    pd.DataFrame(common).to_csv(out/"common_eligibility_grouping.csv",index=False)
    pd.DataFrame(differences).to_csv(out/"paired_fold_mean_differences.csv",index=False)
    pd.DataFrame(policydiff).to_csv(out/"paired_policy_differences.csv",index=False)
    d=pd.read_csv(out/"B_detector_metrics.csv")
    f=d[(d.dataset=="taiwan")&d.variant.eq("feature3")&d.condition.isin(["mcar10","mcar30","mar30"])&d.method.str.match(r"mc\d+$")&d.probability.eq("calibrated_score")]
    b=f[f.method.eq("mc16")][["fold","condition","average_precision","prevalence"]].rename(columns={"average_precision":"ap16","prevalence":"base"})
    f=f.merge(b,on=["fold","condition"]);f["relative_gain"]=(f.average_precision-f.base)/(f.ap16-f.base)
    f.to_csv(out/"K_relative_gain.csv",index=False)
    # Append-only supplementary analysis provenance, no changes to experiment freezes.
    report["write_json"](out/"supplement_manifest.json",dict(method_changed=False,draws=1000,
        analysis="baseline release-all, common eligibility, probability scale, natural missingness, paired fold metrics/policies",
        artifacts=report["hashes"]([out/n for n in ("release_all_eligible.csv","region_B_intervals.csv","external_natural_missingness_strata.csv","common_eligibility_grouping.csv","paired_fold_mean_differences.csv","paired_policy_differences.csv","K_relative_gain.csv")])))


if __name__=="__main__":
    main(Path("outputs/decisive_validation"))
