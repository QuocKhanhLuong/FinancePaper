"""Five interleaved end-to-end repetitions; caches are not substituted for SHAP."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from financepaper.experiments.stable_core_study import context,spec,folder,load_npz,grouped,load_study
from financepaper.experiments.decisive_validation import hashes,write_json,validate_hashes
from financepaper.reliability.validation import ValidationDonors
from financepaper.data.schema import CATEGORICAL_FEATURES


def run(root):
    cfg,historical=load_study(root)
    frozen=json.loads((root/"calibration_freeze.json").read_text());validate_hashes(frozen["artifacts"])
    rows=[]
    methods=["whole_legacy2","rank","strength","stable_donor","stable_conditional","stable_both"]
    for dataset in ("taiwan","polish"):
        train,adapter,seed=context(dataset,0,historical)
        names,_,_=spec(dataset);dest=folder(root,dataset,0)
        donor=ValidationDonors(train.dropna(),CATEGORICAL_FEATURES if dataset=="taiwan" else (),seed=seed)
        conditional=joblib.load(dest/"conditional.joblib")
        policies=joblib.load(dest/"policies.joblib")["policies"]
        cached=load_npz(folder(historical,dataset,0)/"outer/mcar30/current.npz")
        n=16;mask=cached["artificial"][:n];hidden=cached["hidden"][:n]
        partial=pd.DataFrame(cached["partial_values"][:n],columns=names,index=cached["record_ids"][:n])
        for rep in range(-1,5):
            order=np.random.default_rng(20261005+rep).permutation(methods)
            for method in order:
                start=time.perf_counter();a=adapter.attribute(partial);current_seconds=time.perf_counter()-start
                cur=dict(phi=a["phi"],valid=a["valid"],hidden=hidden,probability=a["probability"])
                generation=attribution=scoring=0.;families={}
                needed=([] if method=="strength" else ["donor","conditional"] if method=="stable_both"
                        else ["conditional"] if method=="stable_conditional" else ["donor"])
                for family in needed:
                    t=time.perf_counter()
                    draws=(donor.sample(partial,mask,k=8,seed=123)[0] if family=="donor" else
                           conditional.sample(partial,mask,k=8,seed=123))
                    generation+=time.perf_counter()-t;t=time.perf_counter();phi=[];valid=[]
                    for k in range(8):
                        b=adapter.attribute(pd.DataFrame(draws[:,k],columns=names,index=partial.index))
                        phi.append(b["phi"]);valid.append(b["valid"])
                    attribution+=time.perf_counter()-t
                    families[family]=dict(completion_phi=np.stack(phi,1),completion_valid=np.stack(valid,1))
                # Unused-family statistics are placeholders used only by the common
                # scoring interface; no measured completion is invented or logged.
                fallback=dict(completion_phi=np.repeat(a["phi"][:,None,:],8,axis=1),completion_valid=np.repeat(a["valid"][:,None],8,axis=1))
                aa=families.get("donor",families.get("conditional",fallback));bb=families.get("conditional",aa)
                cur.update(aa);t=time.perf_counter();ev=grouped(cur,bb,dataset)
                policies[("meaningful",method,.1,False)].release(ev)
                scoring=time.perf_counter()-t;total=time.perf_counter()-start
                if rep>=0:
                    rows.append(dict(dataset=dataset,method=method,repeat=rep,customers=n,
                        current_explanation_seconds=current_seconds,completion_seconds=generation,
                        completion_shap_seconds=attribution,scoring_seconds=scoring,total_seconds=total,
                        shap_batch_calls=1+8*len(needed),completion_calls=len(needed),device="cpu"))
    out=root/"analysis";out.mkdir(exist_ok=True);df=pd.DataFrame(rows)
    df.to_csv(out/"runtime_repetitions.csv",index=False)
    summary=df.groupby(["dataset","method"])[["total_seconds","current_explanation_seconds","completion_seconds","completion_shap_seconds","scoring_seconds"]].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns];summary.to_csv(out/"runtime_summary.csv")
    write_json(out/"runtime_receipt.json",dict(source=hashes([Path(__file__)]),repetitions=5,batch_size=16,
        warmup_per_method=1,interleaved_order=True,includes_current_shap=True,cache_substitution=False,
        artifacts=hashes([out/"runtime_repetitions.csv",out/"runtime_summary.csv"])))
    print(summary[["total_seconds_mean","total_seconds_std"]].to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="outputs/stable_core_study")
    run(Path(p.parse_args().output))
