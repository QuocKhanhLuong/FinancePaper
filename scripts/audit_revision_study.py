"""Independent root-run recomputation of recorded splits, probabilities and events."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import argparse,json
from pathlib import Path
from hashlib import sha256
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from financepaper.data.schema import FEATURE_NAMES


def main(root):
    freeze=json.loads((root/"assessment_freeze.json").read_text())
    normalizations=[]
    for file,expected in freeze["sources"].items():
        actual=Path(file).read_bytes()
        if sha256(actual).hexdigest()!=expected:
            archived=root/"source_at_measurement"/file
            assert archived.exists() and sha256(archived.read_bytes()).hexdigest()==expected,file
            assert actual.rstrip()==archived.read_bytes().rstrip(),file
            normalizations.append(file)
    for file,expected in freeze["artifacts"].items():
        assert sha256(Path(file).read_bytes()).hexdigest()==expected,file
    manifest=json.loads((root/"assessment_manifest.json").read_text())
    for file,expected in manifest["artifacts"].items():
        path=root/file
        assert sha256(path.read_bytes()).hexdigest()==expected,file
    old=json.loads((root/"protocol_freeze.json").read_text())
    excluded=set(old["excluded_ids"]);outer=[];count=0;predn=0
    for fold in range(3):
        d=root/f"fold_{fold}";parts=json.loads((d/"partition_ids.json").read_text())
        flattened=sum(parts.values(),[])
        assert len(flattened)==21000 and len(set(flattened))==21000
        assert not excluded.intersection(flattened)
        outer+=parts["outer"]
        for directory in (d,d/"outer"):
            rows=pd.read_csv(directory/"reason_records.csv")
            for (model,condition),g in rows.groupby(["model","condition"]):
                a=np.load(directory/f"{model}_{condition}.npz")
                assert np.array_equal(g.record_id,a["record_ids"])
                for i,row in enumerate(g.itertuples(index=False)):
                    before=a["original_before"][i];full=a["original_full"][i];hidden=a["hidden"][i]
                    candidates=np.flatnonzero((~hidden)&(before>.01))
                    ranked=candidates[np.argsort(-before[candidates],kind="stable")][:3]
                    pre=len(ranked)==3 and a["valid_before"][i]
                    eligible=pre and a["valid_full"][i]
                    assert row.preverification_eligible==pre and row.eligible==eligible
                    if eligible:
                        event=any(full[j]<=1e-6 or np.sum(full[~hidden]>full[j]+1e-6)>=3 for j in ranked)
                        assert bool(row.revision_event)==event
                        assert json.loads(row.reasons)==[FEATURE_NAMES[j] for j in ranked]
                    else: assert pd.isna(row.revision_event)
                    count+=1
        predictions=pd.read_csv(d/"outer/predictions.csv")
        metrics=json.loads((d/"outer/prediction_metrics.json").read_text())
        for row in metrics:
            g=predictions[(predictions.model==row["model"])&(predictions.condition==row["condition"])]
            pr=g[row["probability"]+"_probability"];y=g.y
            expected=dict(average_precision=average_precision_score(y,pr),roc_auc=roc_auc_score(y,pr),brier=brier_score_loss(y,pr),log_loss=log_loss(y,pr))
            for key,val in expected.items(): assert np.isclose(row[key],val,rtol=1e-10,atol=1e-12),(fold,key)
        predn+=len(predictions)
        current=pd.read_csv(d/"outer/current.csv")
        verification=pd.read_csv(d/"outer/verification.csv")
        assert np.array_equal(current[["record_id","condition"]],verification[["record_id","condition"]])
        forbidden={"revision_event","restored_raw_probability","normalized_shift","absolute_raw_probability_shift","y"}
        assert not forbidden.intersection(current.columns)
    assert len(outer)==21000 and len(set(outer))==21000
    result=dict(status="passed",historical_excluded_customers=len(excluded),outer_customers=len(outer),
        reason_rows_recomputed=count,prediction_rows_recomputed=predn,
        frozen_sources=len(freeze["sources"]),frozen_artifacts=len(freeze["artifacts"]),
        result_artifacts=len(manifest["artifacts"]),review="root-run recomputation, not independent peer review")
    result["publication_eof_whitespace_only"]=normalizations
    (root/"audit_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="outputs/revision_study")
    main(Path(parser.parse_args().output))
