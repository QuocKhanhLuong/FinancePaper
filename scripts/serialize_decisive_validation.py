"""Export compact current-only explanations separately from verification labels."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from financepaper.experiments.decisive_validation import read_pair,write_json,hashes
from financepaper.reliability.validation import TAIWAN_GROUPS,POLISH_GROUPS,POLISH_NAMES,aggregate_groups,revision_arrays
from financepaper.data.schema import FEATURE_NAMES


def main(root):
    receipts={}
    for dataset in ("taiwan","polish"):
        folders=[root/"taiwan"/f"fold_{f}" for f in range(3)] if dataset=="taiwan" else [root/"polish"]
        names,groups=(FEATURE_NAMES,TAIWAN_GROUPS) if dataset=="taiwan" else (POLISH_NAMES,POLISH_GROUPS)
        for folder in folders:
            scores=pd.read_csv(folder/"scores.csv");scores=scores[scores.method.eq("mc8")].set_index(["condition","variant","record_id"])
            releases=pd.read_csv(folder/"releases.csv")
            releases=releases[releases.method.eq("mc8")&releases.family.eq("robust")&releases.conservative&releases.alpha.eq(.1)].set_index(["condition","variant","record_id"])
            count=0
            currentpath=folder/"current_diagnostics.jsonl";verifypath=folder/"verification_diagnostics.jsonl"
            with currentpath.open("w") as currentstream,verifypath.open("w") as verifystream:
                for cache in sorted((folder/"outer").glob("*/current.npz")):
                    condition=cache.parent.name;c,v=read_pair(cache.parent)
                    for variant in ("feature3","group1","group2","group3"):
                        phi,full,h,labels=c["phi"],v["phi"],c["hidden"],names;k=int(variant[-1])
                        if variant.startswith("group"):
                            phi,h=aggregate_groups(phi,c["hidden"],names,groups)
                            full,_=aggregate_groups(full,c["hidden"],names,groups);labels=tuple(groups)
                        rr=revision_arrays(phi,full,h,k=k)
                        for i,rid in enumerate(c["record_ids"]):
                            key=(condition,variant,int(rid));s=scores.loc[key];release=releases.loc[key]
                            order=rr["reasons"][i] if rr["eligible"][i] else np.array([],int)
                            p=float(c["probability"][i]);cp=c["completion_probability"][i,:8]
                            current=dict(record_id=int(rid),condition=condition,reason_definition=variant,
                                prediction=dict(logit=float(c["logits"][i]),prob_raw=float(expit(c["logits"][i])),prob_calibrated=p),
                                missingness=dict(fraction_total=float(c["hidden"][i].mean()),fraction_natural=float(c["natural"][i].mean()),
                                    fraction_artificial=float(c["artificial"][i].mean()),features_missing=[n for n,b in zip(names,c["hidden"][i]) if b]),
                                uncertainty=dict(predictive_entropy=float(-p*np.log(max(p,1e-12))-(1-p)*np.log(max(1-p,1e-12))),
                                    prediction_variance=float(cp.var()),completion_std=float(cp.std()),kind="completion sensitivity, not posterior uncertainty"),
                                explanation=dict(top_features=[labels[j] for j in order],top_scores=phi[i,order].tolist(),signs=np.sign(phi[i,order]).astype(int).tolist(),rank=list(range(1,len(order)+1))),
                                reliability=dict(revision_score=float(s.raw_score),revision_probability=float(s.calibrated_score),
                                    release_decision=bool(release.released),policy="robust_simulated_environments/conservative/alpha0.10",K=8))
                            currentstream.write(json.dumps(current,allow_nan=False)+"\n")
                            verified=dict(record_id=int(rid),condition=condition,reason_definition=variant,verification_only=dict(
                                prob_full=float(expit(v["logits"][i])),prob_shift=float(abs(expit(c["logits"][i])-expit(v["logits"][i]))),
                                revision_event=bool(rr["event"][i]) if rr["eligible"][i] else None,
                                num_revised_reasons=int(rr["num_revised"][i]) if rr["eligible"][i] else None))
                            verifystream.write(json.dumps(verified,allow_nan=False)+"\n");count+=1
            receipts[str(folder)]=dict(rows=count,artifacts=hashes([currentpath,verifypath]))
    write_json(root/"serialization_receipt.json",receipts);print("Current diagnostics and verification-only labels serialized separately.")


if __name__=="__main__":
    main(Path("outputs/decisive_validation"))
