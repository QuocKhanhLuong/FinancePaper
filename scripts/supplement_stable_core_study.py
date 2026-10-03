"""Read-only diagnostics for eligibility and unchanged predictor performance."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from financepaper.experiments.stable_core_study import units,folder,spec,load_npz,load_study
from financepaper.experiments import revision_study as old
from financepaper.data.taiwan import load_taiwan
from financepaper.data.polish import load_polish
from financepaper.evaluation.temporal_metrics import detailed_prediction_metrics
from financepaper.experiments.decisive_validation import hashes,write_json
from financepaper.reliability.reason_sets import rank_membership


def run(root):
    cfg,historical=load_study(root)
    _,base=old.load_config("configs/revision_study.yaml")
    tw=load_taiwan(Path(base["data_path"]));_,py,_=load_polish("data/raw/polish/5year.arff")
    prediction=[];eligibility=[];preexisting=[];support=[]
    for dataset,fold in units():
        source,dest=folder(historical,dataset,fold),folder(root,dataset,fold)
        for condition in spec(dataset)[2]:
            p=dest/"outer"/condition;ev=load_npz(p/"reason_evidence.npz")
            truth=load_npz(p/"verification_only.npz");decisions=load_npz(p/"decisions.npz")
            cur=load_npz(source/"outer"/condition/"current.npz")
            y=(tw.y if dataset=="taiwan" else py).loc[cur["record_ids"]].to_numpy()
            for mode,prob in (("raw",expit(cur["logits"])),("calibrated",cur["probability"])):
                prediction.append(dict(dataset=dataset,fold=fold,condition=condition,probability=mode,
                                       **detailed_prediction_metrics(y,prob)))
            count=ev["candidate"].sum(1)
            already=ev["candidate"]&~rank_membership(ev["phi"],ev["hidden"])[:,0]
            preexisting.append(dict(dataset=dataset,fold=fold,condition=condition,candidates=int(count.sum()),
                current_outside_top2=int(already.sum()),fraction=float(already.sum()/count.sum())))
            for target in ("meaningful","ranked"):
                for method in ("whole_legacy2","whole_mc1","whole_mc2","rank","frequency","stable_donor","stable_conditional","stable_both"):
                    selected=decisions[f"{target}|{method}|0.1|0"]
                    for label,ix in (("exactly_one_candidate",count==1),("at_least_two_candidates",count>=2)):
                        num=int(selected[ix].sum());failed=int((selected[ix]&~truth[target][ix]).sum())
                        eligibility.append(dict(dataset=dataset,fold=fold,condition=condition,target=target,method=method,
                            subset=label,n=int(ix.sum()),customers_ge1=int(selected[ix].any(1).sum()),
                            released=num,failed=failed))
    out=root/"analysis"
    write_json(out/"prediction_metrics.json",prediction)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ("classwise","calibration_bins")} for r in prediction]).to_csv(out/"prediction_metrics.csv",index=False)
    df=pd.DataFrame(eligibility)
    df.groupby(["dataset","target","method","subset"])[["n","customers_ge1","released","failed"]].sum().reset_index().assign(
        customer_coverage=lambda f:f.customers_ge1/f.n,
        reason_risk=lambda f:np.where(f.released>0,f.failed/f.released,np.nan)).to_csv(out/"common_eligibility.csv",index=False)
    pd.DataFrame(preexisting).to_csv(out/"current_rank_exclusions.csv",index=False)
    write_json(out/"supplement_receipt.json",dict(sources=hashes([Path(__file__)]),
        scope="postassessment descriptive diagnostics, no policy modification",
        outputs=hashes([out/f for f in ("prediction_metrics.json","prediction_metrics.csv","common_eligibility.csv","current_rank_exclusions.csv")])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="outputs/stable_core_study")
    run(Path(p.parse_args().output))
