"""Export common current diagnostics and a separate verification-only audit file."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from financepaper.data.schema import FEATURE_NAMES
from financepaper.reliability.current import MONTHS


def main(root):
    scores=pd.read_csv(root/"outer_revision_scores.csv")
    for fold in range(3):
        d=root/f"fold_{fold}/outer"
        reasons=pd.read_csv(d/"reason_records.csv")
        preds=pd.read_csv(d/"predictions.csv").set_index(["model","condition","record_id"])
        current=pd.read_csv(d/"current.csv").set_index(["condition","record_id"])
        chosen=scores[(scores.fold==fold)&scores.selected].set_index(["condition","record_id"])
        with (d/"current_diagnostics.jsonl").open("w") as normal,(d/"verification_only.jsonl").open("w") as verification:
            for (model,condition),g in reasons.groupby(["model","condition"]):
                a=np.load(d/f"{model}_{condition}.npz")
                for i,row in enumerate(g.itertuples(index=False)):
                    rid=int(row.record_id); key=(condition,rid)
                    pr=preds.loc[(model,condition,rid)]
                    h=a["hidden"][i];p=float(pr.calibrated_probability)
                    reliability=dict(revision_score=None,revision_probability=None,release_decision=None)
                    variance=None
                    if model=="xgb25":
                        cs=chosen.loc[key];variance=float(current.loc[key,"prediction_variance"])
                        reliability=dict(revision_score=float(cs.raw_score),revision_probability=float(cs.calibrated_score),
                            release_decision=bool(cs.release_empirical),release_conservative=bool(cs.release_conservative))
                    record=dict(model=model,condition=condition,record_id=rid,
                        prediction=dict(logit=float(pr.raw_logit),prob_raw=float(pr.raw_probability),prob_calibrated=p),
                        missingness=dict(fraction_total=float(h.mean()),fraction_temporal=float(h[5:].mean()),fraction_static=float(h[:5].mean()),
                            features_missing=[f for f,m in zip(FEATURE_NAMES,h) if m],months_missing=[int(h[list(j)].sum()) for j in MONTHS]),
                        uncertainty=dict(predictive_entropy=float(-p*np.log(max(p,1e-12))-(1-p)*np.log(max(1-p,1e-12))),
                            prediction_variance=variance,ensemble_or_mc_std=float(np.sqrt(variance)) if variance is not None else None,
                            kind="donor_completion_sensitivity" if variance is not None else "entropy_only"),
                        explanation=dict(top_features=json.loads(row.reasons),top_scores=json.loads(row.reason_scores),
                            signs=json.loads(row.reason_signs),rank=json.loads(row.reason_ranks),eligible=bool(row.preverification_eligible)),
                        reliability=reliability)
                    normal.write(json.dumps(record,allow_nan=False)+"\n")
                    full=a["original_full"][i]
                    indices=[FEATURE_NAMES.index(f) for f in json.loads(row.reasons)]
                    count=sum(full[j]<=1e-6 or np.sum(full[~h]>full[j]+1e-6)>=3 for j in indices)
                    verification.write(json.dumps(dict(model=model,condition=condition,record_id=rid,
                        verification_only=dict(prob_full=float(row.restored_raw_probability),prob_shift=float(row.absolute_raw_probability_shift),
                            revision_event=bool(row.revision_event) if pd.notna(row.revision_event) else None,
                            num_revised_reasons=int(count) if row.eligible else None)),allow_nan=False)+"\n")
    print("Common schema exported; verification information is in separate files.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="outputs/revision_study")
    main(Path(parser.parse_args().output))
