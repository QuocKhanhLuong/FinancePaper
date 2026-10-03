"""Independent recomputation and provenance checks for the frozen validation."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
import json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from financepaper.data.taiwan import load_taiwan
from financepaper.data.polish import load_polish
from financepaper.data.schema import FEATURE_NAMES,CATEGORICAL_FEATURES
from financepaper.reliability.validation import TAIWAN_GROUPS,POLISH_GROUPS,POLISH_NAMES
from financepaper.experiments.decisive_validation import validate_hashes,write_json,hashes


def main(root):
    frozen=json.loads((root/"protocol_freeze.json").read_text());validate_hashes(frozen["sources"])
    ext=json.loads((root/"polish/assessment_freeze.json").read_text());validate_hashes(ext["artifacts"])
    opened=json.loads((root/"polish/assessment_opened.json").read_text())
    assert pd.Timestamp(ext["timestamp_utc"])<pd.Timestamp(opened["timestamp_utc"])
    hist=Path(frozen["config"]["historical_root"])
    taiwan=load_taiwan(Path("data/raw/default of credit card clients.xls")).X
    polish,_,clusters=load_polish(frozen["config"]["external_path"])
    audit=dict(cache_batches=0,current_rows=0,completion_rows=0,events_recomputed=0,
        invalid_current=0,invalid_restored=0,invalid_completion=0,policy_releases_recomputed=0,
        source_files=len(frozen["sources"]),external_frozen_artifacts=len(ext["artifacts"]))
    for dataset in ("taiwan","polish"):
        X=taiwan if dataset=="taiwan" else polish
        names=FEATURE_NAMES if dataset=="taiwan" else POLISH_NAMES
        groups=TAIWAN_GROUPS if dataset=="taiwan" else POLISH_GROUPS
        folders=[root/"taiwan"/f"fold_{i}" for i in range(3)] if dataset=="taiwan" else [root/"polish"]
        for folder in folders:
            parts=json.loads(((hist/folder.name/"partition_ids.json") if dataset=="taiwan" else folder/"partition_ids.json").read_text())
            train=set(parts["predictor_train"] if dataset=="taiwan" else parts["train"])
            reference=json.loads((folder/"donor_reference.json").read_text())["ids"]
            assert set(reference)<=train
            assert X.loc[reference].notna().all().all()
            if dataset=="polish":
                seen=set()
                for ids in parts.values():
                    gs=set(clusters[np.array(ids)-1]);assert not seen&gs;seen|=gs
            for marker in sorted(folder.glob("*/*/complete.json")):
                cache=marker.parent; receipt=json.loads(marker.read_text())
                validate_hashes(receipt["current_only"]);validate_hashes(receipt["verification_only"])
                c=np.load(cache/"current.npz");v=np.load(cache/"verification_only.npz")
                ids=c["record_ids"];truth=X.loc[ids].to_numpy();natural=np.isnan(truth);art=c["artificial"]
                assert not set(ids)&train
                assert not np.any(art&natural)
                np.testing.assert_array_equal(natural,c["natural"])
                np.testing.assert_array_equal(natural|art,c["hidden"])
                np.testing.assert_equal(v["truth"],truth)
                assert not {"truth","restored_phi","restored_probability","revision_event","y"}&set(c.files)
                assert set(c["donor_ids"].ravel())<=set(reference)
                donor_values=X.loc[c["donor_ids"].ravel()].to_numpy().reshape(len(ids),16,len(names))
                expected=np.where(art[:,None,:],donor_values,c["partial_values"][:,None,:])
                np.testing.assert_equal(c["completions"],expected)
                assert np.isnan(c["completions"])[np.broadcast_to(natural[:,None,:],c["completions"].shape)].all()
                if dataset=="taiwan":
                    for name in CATEGORICAL_FEATURES:
                        j=names.index(name);assert set(c["completions"][:,:,j].ravel())<=set(X.loc[reference,name])|set(X.loc[ids,name])
                audit["cache_batches"]+=1;audit["current_rows"]+=len(ids);audit["completion_rows"]+=16*len(ids)
                audit["invalid_current"]+=int((~c["valid"]).sum());audit["invalid_restored"]+=int((~v["valid"]).sum());audit["invalid_completion"]+=int((~c["completion_valid"]).sum())
            for split in ("revision_calibration","release_calibration","outer"):
                table=pd.read_csv(folder/f"{split}_derived.csv")
                for condition,f in table.groupby("condition"):
                    c=np.load(folder/split/condition/"current.npz");v=np.load(folder/split/condition/"verification_only.npz")
                    for variant,g in f.groupby("variant",sort=False):
                        np.testing.assert_array_equal(g.record_id,c["record_ids"])
                        k=int(variant[-1]);a=c["phi"];b=v["phi"];h=c["hidden"];ca=c["completion_phi"]
                        if not variant.startswith("feature"):
                            observed=not variant.startswith("whole")
                            av=[];bv=[];cv=[];hv=[]
                            for members in groups.values():
                                idx=[names.index(n) for n in members];o=~h[:,idx]
                                av.append((a[:,idx]*(o if observed else 1)).sum(1))
                                bv.append((b[:,idx]*(o if observed else 1)).sum(1))
                                cv.append((ca[:,:,idx]*(o[:,None,:] if observed else 1)).sum(2))
                                hv.append(h[:,idx].all(1))
                            a,b,h,ca=np.stack(av,1),np.stack(bv,1),np.stack(hv,1),np.stack(cv,2)
                        for i,row in enumerate(g.itertuples()):
                            cand=np.flatnonzero((~h[i])&(a[i]>.01));rr=cand[np.argsort(-a[i,cand],kind="stable")][:k]
                            eligible=len(rr)==k and c["valid"][i] and c["completion_valid"][i].all()
                            assert row.eligible==eligible
                            if eligible and v["valid"][i]:
                                event=any(b[i,j]<=1e-6 or (b[i,~h[i]]>b[i,j]+1e-6).sum()>=k for j in rr)
                                assert row.event==event
                                outcomes=np.array([any(ca[i,d,j]<=1e-6 or (ca[i,d,~h[i]]>ca[i,d,j]+1e-6).sum()>=k for j in rr) for d in range(16)])
                                for budget in (1,2,4,8,16):
                                    assert np.isclose(getattr(row,f"score_mc{budget}"),outcomes[:budget].mean())
                                for eps in (0,.005,.01,.02):
                                    value=any(b[i,j]<=1e-6 or (b[i,~h[i]]>b[i,j]+eps).sum()>=k for j in rr)
                                    assert g.iloc[i][f"event_gap_{eps:g}"]==value
                            else:
                                assert pd.isna(row.event)
                            assert np.isclose(row.prob_shift,abs(expit(c["logits"][i])-expit(v["logits"][i])))
                            audit["events_recomputed"]+=1
            policies=joblib.load(folder/"policies.joblib")["policies"]
            releases=pd.read_csv(folder/"releases.csv");scores=pd.read_csv(folder/"scores.csv")
            for key,group in releases.groupby(["variant","method","alpha","family","conservative"]):
                policy=policies[key]
                s=scores[(scores.variant==key[0])&(scores.method==key[1])]
                np.testing.assert_array_equal(s[["record_id","condition"]],group[["record_id","condition"]])
                expected=np.zeros(len(s),bool)
                for j,t in enumerate(policy.thresholds):
                    if t is not None:
                        expected|=s.eligible.to_numpy()&(s.calibrated_score.to_numpy()<=t)&((s.stratum.to_numpy()==j) if policy.family=="stratified" else True)
                np.testing.assert_array_equal(group.released,expected)
                audit["policy_releases_recomputed"]+=len(group)
    audit.update(status="passed",review="root independent recomputation; not external peer review",
        calibration_limitation="Polish row-level binomial rule is descriptive; exact duplicates and unknown company identity preclude an entity guarantee",
        evaluator_sha256=hashes([Path(__file__)])[str(Path(__file__))])
    write_json(root/"audit_receipt.json",audit);print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main(Path("outputs/decisive_validation"))
