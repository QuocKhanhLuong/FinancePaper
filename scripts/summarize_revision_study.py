"""Customer-paired policy intervals, prediction differences and Pareto accounting."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def main(root):
    out=root/"analysis";out.mkdir(exist_ok=True)
    scores=pd.read_csv(root/"outer_revision_scores.csv")
    selected=scores[scores.selected].copy();selected["method"]="selected"
    scores=pd.concat([scores,selected],ignore_index=True)
    rows=[];rng=np.random.default_rng(1942)
    for condition in ("pooled","mcar30","mar30"):
        g=scores[scores.pooled] if condition=="pooled" else scores[scores.condition==condition]
        events=g[g.method=="selected"].set_index("record_id").revision_event.astype(float)
        for mode in ("empirical","conservative"):
            releases=g.pivot(index="record_id",columns="method",values=f"release_{mode}").reindex(events.index)
            e=events.to_numpy(); e=np.where(np.isfinite(e),e,1)
            for method in ("selected","mc_revision","prediction_boosted","missing_fraction"):
                a=releases[method].to_numpy(bool);n=len(a)
                cover=[];risks=[]
                for _ in range(1000):
                    idx=rng.integers(n,size=n);chosen=a[idx]
                    cover.append(chosen.mean());risks.append(e[idx][chosen].mean() if chosen.any() else np.nan)
                rows.append(dict(condition=condition,policy=mode,method=method,coverage=a.mean(),
                    coverage_low=np.quantile(cover,.025),coverage_high=np.quantile(cover,.975),
                    risk=e[a].mean() if a.any() else None,risk_low=np.nanquantile(risks,.025),risk_high=np.nanquantile(risks,.975)))
    pd.DataFrame(rows).to_csv(out/"policy_intervals.csv",index=False)
    diffs=[]
    for condition in ("mcar30","mar30"):
        frames=[]
        for path in sorted(root.glob("fold_*/outer/predictions.csv")):
            frame=pd.read_csv(path);frame=frame[(frame.condition==condition)&frame.model.isin(["xgb25","vanilla","mask_delta"])]
            frames.append(frame.pivot(index=["record_id","y"],columns="model",values="calibrated_probability").reset_index())
        for model in ("vanilla","mask_delta"):
            ap=[];brier=[]
            for _ in range(1000):
                av=[];bv=[]
                for frame in frames:
                    ix=rng.integers(len(frame),size=len(frame));y=frame.y.to_numpy()[ix]
                    a=frame[model].to_numpy()[ix];b=frame.xgb25.to_numpy()[ix]
                    av.append(average_precision_score(y,a)-average_precision_score(y,b));bv.append(np.mean((a-y)**2-(b-y)**2))
                ap.append(np.mean(av));brier.append(np.mean(bv))
            for name,vals in (("AP",ap),("Brier",brier)):
                point=np.mean([average_precision_score(f.y,f[model])-average_precision_score(f.y,f.xgb25) if name=="AP" else np.mean((f[model]-f.y)**2-(f.xgb25-f.y)**2) for f in frames])
                diffs.append(dict(condition=condition,model=model,reference="xgb25",metric=name,difference=point,low=np.quantile(vals,.025),high=np.quantile(vals,.975)))
    pd.DataFrame(diffs).to_csv(out/"prediction_paired_intervals.csv",index=False)
    # No arbitrary combined score. Dominance is only descriptive in these axes.
    m=pd.read_csv(root/"outer_revision_metrics.csv")
    m=m[(m.probability=="calibrated")&m.condition.isin(["mcar30","mar30"])]
    pivot=m.groupby(["method","condition"])[["roc_auc","average_precision","brier","log_loss"]].mean()
    frontier=[]
    for c in ("mcar30","mar30"):
        for ref in ("prediction_boosted","mc_revision"):
            a=pivot.loc[("mask_explanation_boosted",c)];b=pivot.loc[(ref,c)]
            wins=[a.roc_auc>=b.roc_auc,a.average_precision>=b.average_precision,a.brier<=b.brier,a.log_loss<=b.log_loss]
            frontier.append(dict(condition=c,reference=ref,predictor_unchanged=True,selector_detection_calibration_dominates=all(wins),
                coverage_risk_and_compute="separate dimensions; inspect policy intervals and latency, no scalar score"))
    (out/"pareto_accounting.json").write_text(json.dumps(frontier,indent=2)+"\n")
    print("Paired customer intervals and Pareto accounting saved.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="outputs/revision_study")
    main(Path(parser.parse_args().output))
