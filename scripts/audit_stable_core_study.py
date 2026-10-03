"""Recompute decisions and endpoints; no fitting or policy selection."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name,"1")
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from financepaper.experiments.stable_core_study import load_study,units,folder,spec,load_npz,grouped,context
from financepaper.experiments.decisive_validation import validate_hashes,hashes,write_json
from financepaper.reliability.reason_sets import verified,strongest,matched_random
from financepaper.reliability.validation import aggregate_groups


def audit(root):
    cfg,historical=load_study(root)
    gate=json.loads((root/"calibration_freeze.json").read_text());validate_hashes(gate["artifacts"]);validate_hashes(gate["protocol"])
    done=json.loads((root/"assessment_complete.json").read_text());validate_hashes(done["artifacts"])
    counts=dict(customer_condition_records=0,recomputed_policy_decisions=0,matched_count_checks=0,
                invalid_attributions=0,observed_cell_changes=0,natural_cell_changes=0)
    support=[]
    for dataset,fold in units():
        source,dest=folder(historical,dataset,fold),folder(root,dataset,fold)
        names,groups,envs=spec(dataset)
        model=joblib.load(dest/"conditional.joblib");policies=joblib.load(dest/"policies.joblib")["policies"]
        train,_,_=context(dataset,fold,historical)
        assert set(model.training_ids_)<=set(train.index)
        for split in ("release_calibration","outer"):
            for condition in envs:
                old=load_npz(source/split/condition/"current.npz")
                new=load_npz(dest/split/condition/"current.npz")
                assert np.array_equal(new["record_ids"],old["record_ids"])
                assert not set(new["record_ids"])&set(model.training_ids_)
                expected=np.repeat(old["partial_values"][:,None,:],8,axis=1)
                keep=np.broadcast_to(~old["artificial"][:,None,:],expected.shape)
                np.testing.assert_equal(new["completions"][keep],expected[keep])
                assert new["completion_valid"].all()
                if split!="outer":continue
                counts["customer_condition_records"]+=len(old["record_ids"])
                ev=grouped(old,new,dataset);decisions=load_npz(dest/split/condition/"decisions.npz")
                truth=load_npz(source/split/condition/"verification_only.npz")
                full=aggregate_groups(truth["phi"],old["hidden"],names,groups)[0]
                saved=load_npz(dest/split/condition/"verification_only.npz")
                for target in ("meaningful","ranked"):
                    np.testing.assert_equal(saved[target],verified(full,ev["hidden"],target))
                for (target,method,alpha,conservative),policy in policies.items():
                    key=f"{target}|{method}|{alpha}|{int(conservative)}"
                    actual=policy.release(ev)
                    np.testing.assert_equal(actual,decisions[key]);assert not (actual&~ev["candidate"]).any()
                    counts["recomputed_policy_decisions"]+=len(actual)
                    if method.startswith("stable_"):
                        for suffix,matched in (("strength",strongest(ev,actual.sum(1))),
                                               ("random",matched_random(ev,actual.sum(1),old["record_ids"]))):
                            recorded=decisions[key.replace(method,method+"_matched_"+suffix)]
                            np.testing.assert_equal(matched,recorded);np.testing.assert_equal(recorded.sum(1),actual.sum(1))
                            counts["matched_count_checks"]+=len(actual)
                # Truth support is diagnostic only and never enters policy evidence.
                for family,draws in (("donor",old["completions"][:,:8]),("conditional",new["completions"])):
                    for j,name in enumerate(names):
                        rows=old["artificial"][:,j]
                        if not rows.any():continue
                        values=truth["truth"][rows,j];sampled=draws[rows,:,j]
                        lo=sampled.min(1);hi=sampled.max(1)
                        support.append(dict(dataset=dataset,fold=fold,condition=condition,family=family,feature=name,
                            hidden_count=int(rows.sum()),inside_range=int(((values>=lo)&(values<=hi)).sum()),
                            exact_sample_match=int((sampled==values[:,None]).any(1).sum()),
                            mean_percentile=float((sampled<=values[:,None]).mean())))
    pd.DataFrame(support).to_csv(root/"analysis/completion_truth_support.csv",index=False)
    write_json(root/"audit_receipt.json",dict(**counts,passed=True,Freddie="NOT RUN",
        method_retrained_on_assessment=False,sources=hashes([Path(__file__)])))
    print(json.dumps(counts,indent=2));print("Stable-Core audit passed; software audit, not independent peer review.")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="outputs/stable_core_study")
    audit(Path(p.parse_args().output))
