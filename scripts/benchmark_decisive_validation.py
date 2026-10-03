"""Five-repeat warm inference timing, separate stages and explicit call counts."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import json,time,platform
import joblib,numpy as np,pandas as pd
from scipy.special import logit
from financepaper.experiments import revision_study as old
from financepaper.experiments.decisive_validation import taiwan_adapter,validate_hashes,write_json
from financepaper.reliability.validation import ValidationDonors,revision_arrays
from financepaper.reliability.current import CurrentEvidence,current_features
from financepaper.data.schema import CATEGORICAL_FEATURES
from financepaper.data.taiwan import load_taiwan


def main(root):
    freeze=json.loads((root/"protocol_freeze.json").read_text());validate_hashes(freeze["sources"])
    hist=Path(freeze["config"]["historical_root"])
    cfg,base=old.load_config("configs/revision_study.yaml")
    data=load_taiwan(Path(base["data_path"]));parts,_=old.make_partitions(data,cfg)
    bundle=joblib.load(hist/"fold_0/predictors.joblib");release=joblib.load(hist/"fold_0/release_model.joblib")
    adapter=taiwan_adapter(bundle);donor=ValidationDonors(bundle["context"]["X_train"],CATEGORICAL_FEATURES,seed=bundle["seed"])
    selector=release["selectors"][release["chosen"]]
    X=data.X.iloc[parts[0]["diagnostic"][:128]]
    h=old.masks_for(X,bundle["context"]["mar"],bundle["seed"]+3000)["mcar30"]
    timings=[]
    for size in (1,128):
        partial=X.iloc[:size].mask(h.iloc[:size]);hidden=h.iloc[:size].to_numpy()
        for rep in range(6):
            # Rotate execution order to avoid always assigning warmest position to K16.
            modes=[1,2,4,8,16,"learned"]
            modes=modes[rep%6:]+modes[:rep%6]
            for mode in modes:
                stages=dict(transform=0.,prediction=0.,shap=0.,completion=0.,scoring=0.)
                overall=time.perf_counter()
                t=time.perf_counter();matrix=adapter.encode(partial);stages["transform"]+=time.perf_counter()-t
                t=time.perf_counter();z=adapter.predict_encoded(matrix);p=adapter.calibrator.transform(z);stages["prediction"]+=time.perf_counter()-t
                t=time.perf_counter();phi,_=adapter.explain_encoded(matrix);stages["shap"]+=time.perf_counter()-t
                k=8 if mode=="learned" else mode
                t=time.perf_counter();draws,_=donor.sample(partial,hidden,k=k,seed=114);stages["completion"]+=time.perf_counter()-t
                cp,ca=[],[]
                for d in range(k):
                    t=time.perf_counter();frame=pd.DataFrame(draws[:,d],index=partial.index,columns=partial.columns);m=adapter.encode(frame);stages["transform"]+=time.perf_counter()-t
                    t=time.perf_counter();cp.append(adapter.calibrator.transform(adapter.predict_encoded(m)));stages["prediction"]+=time.perf_counter()-t
                    if mode!="learned":
                        t=time.perf_counter();ca.append(adapter.explain_encoded(m)[0]);stages["shap"]+=time.perf_counter()-t
                t=time.perf_counter()
                if mode=="learned":
                    features=current_features(CurrentEvidence(p,hidden,phi,np.stack(cp,1)))
                    raw=selector["model"].predict_proba(features[selector["columns"]])[:,1]
                    cal=release["calibrators"][release["chosen"]]
                    eligible=((phi>.01)&~hidden).sum(1)>=3
                else:
                    result=revision_arrays(phi,np.stack(ca,1),hidden)
                    raw=result["event"].mean(1);eligible=result["eligible"]
                    cal=release["calibrators"]["mc_revision"]
                # Calibration/policy invocation is timed; this is a timing-only
                # reference, not a K-specific accuracy or deployment selection.
                calibrated=cal.transform(logit(np.clip(raw,1e-7,1-1e-7)))
                release["policies"]["mc_revision"]["empirical"].release(calibrated,eligible)
                stages["scoring"]+=time.perf_counter()-t
                elapsed=time.perf_counter()-overall
                if rep:
                    timings.append(dict(mode=str(mode),batch_size=size,repetition=rep,seconds=elapsed,
                        **{f"seconds_{s}":v for s,v in stages.items()},shap_calls=1 if mode=="learned" else k+1,
                        prediction_calls=k+1,completion_calls=1,draws=k))
    out=root/"analysis";out.mkdir(exist_ok=True)
    frame=pd.DataFrame(timings);frame.to_csv(out/"runtime_repetitions.csv",index=False)
    frame.groupby(["batch_size","mode"]).agg({c:["mean","std"] for c in frame if c.startswith("seconds")}).to_csv(out/"runtime_summary.csv")
    write_json(out/"runtime_metadata.json",dict(cpu=platform.processor(),machine=platform.machine(),system=platform.platform(),
        warmups=1,repetitions=5,threads=1,excludes="disk/model loading; pre-fitted explainers reused",includes="input transform/current SHAP/donors/completion prediction+SHAP/scoring/default+revision calibration/policy",
        calls="explicit batched calls; TreeSHAP internal additivity operations included in SHAP time"))
    print(frame.groupby(["batch_size","mode"]).seconds.agg(["mean","std"]).to_string())


if __name__=="__main__":
    main(Path("outputs/decisive_validation"))
