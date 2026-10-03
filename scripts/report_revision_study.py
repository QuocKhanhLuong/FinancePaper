"""Deterministic tables/figures from frozen assessment; never fits a model."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score
from financepaper.evaluation.temporal_metrics import reliability_curve,calibration_bins


def concat(root,pattern):
    return pd.concat([pd.read_csv(p).assign(fold=int(p.parts[-3 if p.parent.name=="outer" else -2].split('_')[-1])) for p in sorted(root.glob(pattern))],ignore_index=True)


def bootstrap_difference(frame, first, second, metric="ap", draws=1000, seed=42):
    """Original-customer weights retain all repeated masks/folds of a customer."""
    frame=frame.dropna(subset=["revision_event",first,second])
    ids, inverse=np.unique(frame.record_id,return_inverse=True)
    y=frame.revision_event.to_numpy(int);a=frame[first].to_numpy(float);b=frame[second].to_numpy(float)
    rng=np.random.default_rng(seed)
    fn=average_precision_score if metric=="ap" else roc_auc_score
    point=float(fn(y,a)-fn(y,b)); values=[]
    for _ in range(draws):
        weight=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[inverse]
        if (weight*y).sum()==0 or (weight*(1-y)).sum()==0: continue
        values.append(fn(y,a,sample_weight=weight)-fn(y,b,sample_weight=weight))
    low,high=np.quantile(values,[.025,.975])
    return dict(difference=point,low=float(low),high=float(high),customers=len(ids),rows=len(frame),draws=len(values))


def case_b(records):
    out=[]
    for (model,c),g in records.groupby(["model","condition"]):
        g=g[g.eligible].copy();e=g.revision_event.astype(float).eq(1)
        for cutoff in (.01,.02,.05):
            stable=g.absolute_raw_probability_shift<=cutoff
            out.append(dict(model=model,condition=c,cutoff=cutoff,n=len(g),
                A=int((stable&~e).sum()),B=int((stable&e).sum()),C=int((~stable&~e).sum()),D=int((~stable&e).sum()),
                stable_n=int(stable.sum()),B_given_stable=float(e[stable].mean()) if stable.any() else None))
    return pd.DataFrame(out)


def savefig(directory,name):
    plt.tight_layout()
    plt.savefig(directory/f"{name}.png",dpi=180)
    plt.savefig(directory/f"{name}.pdf")
    plt.close()


def main(root):
    out=root/"analysis";out.mkdir(exist_ok=True)
    diagnostic=concat(root,"fold_*/reason_records.csv")
    if not (out/"diagnostic_case_b.csv").exists():
        case_b(diagnostic).to_csv(out/"diagnostic_case_b.csv",index=False)
    audit=concat(root,"fold_*/metric_audit.csv")
    if not (out/"metric_audit_counts.csv").exists():
        audit.groupby(["model","condition","variant"])[["n","pair_valid","eligible","revised","sign_events"]].sum().to_csv(out/"metric_audit_counts.csv")
    dev=concat(root,"fold_*/revision_diagnostic_scores.csv")
    intervals=[]
    for condition in ("mcar10","mcar30","mar30"):
        frame=dev[(dev.condition==condition)&dev.eligible_current]
        for first in ("explanation_boosted","mask_explanation_boosted"):
            for metric in ("ap","auc"):
                intervals.append(dict(stage="diagnostic",condition=condition,first=first,second="prediction_boosted",metric=metric,
                    **bootstrap_difference(frame,first,"prediction_boosted",metric)))
    if not (root/"assessment_manifest.json").exists():
        pd.DataFrame(intervals).to_csv(out/"paired_intervals.csv",index=False);return
    records=concat(root,"fold_*/outer/reason_records.csv")
    case_b(records).to_csv(out/"outer_case_b.csv",index=False)
    pred=concat(root,"fold_*/outer/predictions.csv")
    metrics=pd.DataFrame([dict(fold=int(path.parent.parent.name.split('_')[-1]),**row)
        for path in root.glob("fold_*/outer/prediction_metrics.json") for row in json.loads(path.read_text())])
    metrics.drop(columns=["classwise","calibration_bins"]).to_csv(out/"prediction_metrics.csv",index=False)
    measures=["average_precision","roc_auc","recall","f1","brier","log_loss","ece"]
    metrics[metrics.probability=="calibrated"].groupby(["model","condition"])[measures].agg(["mean","std"]).to_csv(out/"prediction_mean_sd.csv")
    rs=records[records.eligible].copy();rs["revision_event"]=rs.revision_event.astype(float)
    rs.groupby(["model","condition"]).agg(eligible=("revision_event","size"),events=("revision_event","sum"),
        revision=("revision_event","mean"),normalized_shift=("normalized_shift","mean")).to_csv(out/"revision_summary.csv")
    conditions=["complete","mcar10","mcar20","mcar30"]
    for columns,title in ((["average_precision","roc_auc"],"01_prediction_missing_rate"),(["brier","ece"],"02_calibration_missing_rate")):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,col in zip(axes,columns):
            for model,g in metrics[metrics.probability=="calibrated"].groupby("model"):
                agg=g.groupby("condition")[col].agg(["mean","std"]).reindex(conditions)
                ax.errorbar([0,10,20,30],agg["mean"],yerr=agg["std"],marker="o",label=model,capsize=2)
            ax.set(xlabel="MCAR rate (%)",ylabel=col,title="Fold mean ± SD; internal assessment")
        axes[-1].legend(fontsize=8);savefig(out,title)
    plt.figure(figsize=(7,4))
    for model,g in rs.groupby("model"):
        per=g.groupby(["fold","condition"]).revision_event.mean().reset_index()
        agg=per.groupby("condition").revision_event.agg(["mean","std"]).reindex(conditions)
        plt.errorbar([0,10,20,30],agg["mean"],yerr=agg["std"],label=model,marker="o",capsize=2)
    plt.xlabel("MCAR rate (%)");plt.ylabel("Revision among eligible valid pairs");plt.legend(fontsize=8);savefig(out,"03_revision_missing_rate")
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
    for ax,model in zip(axes,["xgb25","vanilla","mask_delta"]):
        g=rs[(rs.model==model)&(rs.condition=="mcar30")]
        highlight=(g.absolute_raw_probability_shift<=.02)&g.revision_event.eq(1)
        ax.scatter(g.absolute_raw_probability_shift,g.normalized_shift,s=5,alpha=.2,color="grey")
        ax.scatter(g.loc[highlight,"absolute_raw_probability_shift"],g.loc[highlight,"normalized_shift"],s=8,alpha=.7,color="darkred",label="Stable + revised")
        ax.axvline(.02,ls="--",color="black",lw=.8);ax.set(title=model,xlabel="Absolute raw probability shift")
    axes[0].set_ylabel("Observed normalized attribution shift");axes[-1].legend(fontsize=8);savefig(out,"04_prediction_explanation_shift")
    scores=pd.read_csv(root/"outer_revision_scores.csv")
    selected=scores[scores.selected].copy();selected["method"]="selected"
    scores=pd.concat([scores,selected],ignore_index=True)
    methods=["entropy","prediction_boosted","mask_only_boosted","mc_revision","selected"]
    plt.figure(figsize=(7,4));curve_rows=[]
    for method in methods:
        g=scores[(scores.method==method)&(scores.condition=="mcar30")]
        curve=pd.DataFrame(reliability_curve(g.calibrated_score,g.revision_event,g.eligible_current,total_n=len(g)))
        plt.plot(curve.coverage,curve.revision_rate,label=method)
        curve["method"]=method;curve_rows.append(curve)
    plt.axhline(.1,ls="--",color="black",lw=.8);plt.ylim(0,.3);plt.xlabel("Explanation coverage (all sampled customers)");plt.ylabel("Observed revision risk");plt.legend(fontsize=8);savefig(out,"06_selective_explanation")
    pd.concat(curve_rows).to_csv(out/"explanation_risk_coverage.csv",index=False)
    plt.figure(figsize=(7,4))
    for model in ("xgb25","vanilla","mask_delta"):
        ids=rs[(rs.model==model)&(rs.condition=="mcar30")][["record_id","fold"]]
        # Show same explanation cohort, including ineligible customers.
        ids=records[(records.model==model)&(records.condition=="mcar30")][["record_id","fold"]]
        g=pred[(pred.model==model)&(pred.condition=="mcar30")].merge(ids,on=["record_id","fold"])
        threshold={fold:json.loads((root/f"fold_{fold}/predictor_selection.json").read_text())["models"][model]["calibrated_threshold"] for fold in range(3)}
        error=(g.calibrated_probability>=g.fold.map(threshold))!=g.y
        pr=np.clip(g.calibrated_probability,1e-9,1-1e-9)
        entropy=-(pr*np.log(pr)+(1-pr)*np.log1p(-pr))
        curve=pd.DataFrame(reliability_curve(entropy,error.astype(int),np.ones(len(g),bool)))
        plt.plot(curve.coverage,curve.revision_rate,label=model)
    plt.xlabel("Prediction coverage");plt.ylabel("Classification error (dev-selected threshold)");plt.legend();savefig(out,"05_selective_prediction")
    calibration=[];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axes,["default","revision"]):
        if kind=="default":
            groups=[(model,g.y,g.calibrated_probability) for model,g in pred[pred.condition=="mcar30"].groupby("model")]
        else:
            groups=[]
            for method in ("selected","mc_revision","prediction_boosted"):
                g=scores[(scores.method==method)&(scores.condition=="mcar30")&scores.eligible_current].dropna(subset=["revision_event"])
                for prob in ("raw_score","calibrated_score"):
                    groups.append((method+"/"+prob,g.revision_event,np.clip(g[prob],0,1)))
        for label,y,pr in groups:
            bins=pd.DataFrame(calibration_bins(y,pr));bins["kind"]=kind;bins["method"]=label;calibration.append(bins)
            nonempty=bins[bins.n>0];ax.plot(nonempty.mean_probability,nonempty.default_fraction,marker=".",label=label)
        ax.plot([0,1],[0,1],ls="--",color="black",lw=.7);ax.set(xlabel="Mean estimated probability",ylabel="Observed event fraction",title=kind);ax.legend(fontsize=6)
    savefig(out,"07_calibration_curves");pd.concat(calibration).to_csv(out/"calibration_bins.csv",index=False)
    for condition in ("mcar10","mcar30","mar30"):
        g=scores[(scores.condition==condition)&scores.eligible_current]
        wide=g.pivot(index=["fold","record_id","revision_event"],columns="method",values="calibrated_score").reset_index()
        for other in ("prediction_boosted","mc_revision","mask_only_boosted"):
            for metric in ("ap","auc"):
                intervals.append(dict(stage="outer",condition=condition,first="selected",second=other,metric=metric,
                    **bootstrap_difference(wide,"selected",other,metric)))
    pd.DataFrame(intervals).to_csv(out/"paired_intervals_assessment.csv",index=False)
    fixed=[]
    current=concat(root,"fold_*/outer/current.csv")
    for condition in ("mcar30","mar30"):
        g=scores[scores.condition==condition].merge(current[["record_id","fold","condition","missing_fraction","missing_temporal","missing_static"]],on=["record_id","fold","condition"])
        g=g[g.eligible_current].dropna(subset=["revision_event"])
        g["missing_count"]=np.rint(g.missing_fraction*23).astype(int)
        for count in (5,6,7,8):
            for method,part in g[(g.missing_count==count)&g.method.isin(methods)].groupby("method"):
                if part.revision_event.nunique()<2:continue
                fixed.append(dict(condition=condition,count=count,method=method,n=len(part),events=int(part.revision_event.sum()),
                    auc=roc_auc_score(part.revision_event.astype(int),part.calibrated_score),ap=average_precision_score(part.revision_event.astype(int),part.calibrated_score)))
    pd.DataFrame(fixed).to_csv(out/"fixed_missing_count.csv",index=False)
    # Oracle restoration is used only to define evaluation strata, never scores.
    critical=[]
    xgb=records[records.model=="xgb25"][["record_id","fold","condition","absolute_raw_probability_shift"]]
    joined=scores.merge(xgb,on=["record_id","fold","condition"])
    for condition in ("mcar10","mcar30","mar30"):
        for cutoff in (.01,.02,.05):
            g=joined[(joined.condition==condition)&(joined.absolute_raw_probability_shift<=cutoff)&joined.eligible_current].dropna(subset=["revision_event"])
            for method,part in g[g.method.isin(methods)].groupby("method"):
                if part.revision_event.nunique()<2:continue
                critical.append(dict(condition=condition,cutoff=cutoff,method=method,n=len(part),events=int(part.revision_event.sum()),
                    auc=roc_auc_score(part.revision_event.astype(int),part.calibrated_score),ap=average_precision_score(part.revision_event.astype(int),part.calibrated_score)))
    pd.DataFrame(critical).to_csv(out/"critical_case_detection.csv",index=False)
    print(f"Reports and seven plot pairs written to {out}")


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",default="outputs/revision_study")
    main(Path(ap.parse_args().output))
