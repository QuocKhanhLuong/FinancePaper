"""Warm CPU inference timing on a fixed inner diagnostic subset, no tuning."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import argparse,time,json
from pathlib import Path
import joblib,numpy as np
from financepaper.experiments import revision_study as s
from financepaper.experiments import temporal_pilot as p
from financepaper.data.taiwan import load_taiwan
from financepaper.reliability.current import CurrentEvidence,current_features
from financepaper.reliability.donors import ConditionalDonors
from financepaper.training.calibration import PositiveSlopePlattCalibrator


def main(root):
    cfg,base=s.load_config("configs/revision_study.yaml")
    data=load_taiwan(Path(base["data_path"]));parts,_=s.make_partitions(data,cfg)
    bundle=joblib.load(root/"fold_0/predictors.joblib");release=joblib.load(root/"fold_0/release_model.joblib")
    ids=parts[0]["diagnostic"][:128];prep=bundle["context"]["preprocessor"];fit=bundle["fits"]["xgb25"]
    batches,masks,_=s.partition_batches(data,ids,prep,bundle["context"]["mar"],bundle["seed"]+3000)
    partial=data.X.iloc[ids].mask(masks["mcar30"]);batch=batches["mcar30"]
    donor=ConditionalDonors(bundle["context"]["X_train"],seed=bundle["seed"])
    cal=PositiveSlopePlattCalibrator.from_dict(bundle["selection"]["models"]["xgb25"]["calibrator"])
    times=[]
    for repetition in range(6):
        for mode in ("learned_current","mc_revision"):
            started=time.perf_counter()
            a=s.get_attr(fit,batch,bundle,base,"cpu")
            cp=[];ca=[]
            for completed in donor.complete(partial,draws=8,seed=114):
                b=prep.transform(completed)
                if mode=="mc_revision":
                    ac=s.get_attr(fit,b,bundle,base,"cpu");logits=ac["input_logit"];ca.append(ac["original"])
                else: logits=p._raw_logits(fit,b,"cpu",256)
                cp.append(cal.transform(logits))
            f=current_features(CurrentEvidence(cal.transform(a["input_logit"]),masks["mcar30"].to_numpy(bool),a["original"],
                np.stack(cp,1),np.stack(ca,1) if ca else None))
            if mode=="learned_current":
                selector=release["selectors"][release["chosen"]];selector["model"].predict_proba(f[selector["columns"]])
            else: f.mc_revision.to_numpy()
            elapsed=time.perf_counter()-started
            if repetition:times.append(dict(mode=mode,repetition=repetition,batch_size=128,seconds=elapsed))
    (root/"analysis/inference_timing.json").write_text(json.dumps(dict(device="cpu",warmup_repetitions=1,
        includes="current SHAP, donors, completion predictions, score; excludes disk loading/default calibration scalar transform",timings=times),indent=2)+"\n")
    for mode in ("learned_current","mc_revision"):
        values=[r["seconds"] for r in times if r["mode"]==mode];print(mode,np.mean(values),np.std(values,ddof=1))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="outputs/revision_study")
    main(Path(parser.parse_args().output))
