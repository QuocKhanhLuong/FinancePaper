"""Export hash-verified compute-gate aggregates; never overwrite a prior report."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil

import pandas as pd

from financepaper.evaluation.decision_value import check_hashes,digest,write_json


def summarize(run,validation,output):
    receipt=json.loads((run/"execution_receipt.json").read_text())
    validate=json.loads((validation/"validation_receipt.json").read_text())
    check_hashes(receipt["frozen"]["sources"])
    check_hashes(receipt["outputs"])
    check_hashes(receipt["task_receipts"])
    assert digest(run/"execution_receipt.json")==validate["real_receipt_sha256"]
    assert all(command["accepted"] for command in validate["commands"])
    config=receipt["frozen"]["config"]
    cells=pd.read_csv(run/"cells.csv")
    costs=pd.read_csv(run/"costs.csv")
    assert len(cells)==receipt["cells"] and len(costs)==receipt["tasks"]
    output.mkdir(parents=True,exist_ok=False)
    exports={}
    for source in [run/name for name in ("cells.csv","costs.csv","manifest.json","execution_receipt.json")]+[validation/"validation_receipt.json"]:
        target=output/source.name
        shutil.copy2(source,target)
        assert digest(source)==digest(target)
        exports[str(target)]={"source":str(source),"sha256":digest(target)}
    methods=["generic","transfer","c6","reference_aware","cantelli","hybrid","branch_upper"]+[f"mc_{n}" for n in config["mc_draws"]]+["oracle"]
    coverage=[]
    for scope,frame in (("enumerated",cells[cells.enumeration_status=="RUN"]),("large_not_enumerated",cells[cells.enumeration_status!="RUN"])):
        for alpha_text in config["release_budgets"]:
            alpha=float(F(alpha_text))
            for method in methods:
                if method=="oracle" and scope!="enumerated":
                    continue
                released=int((frame[method]<=alpha).sum())
                coverage.append(dict(scope=scope,alpha=alpha_text,method=method,cells=len(frame),released=released,
                    coverage=released/len(frame),evidence="finite_sample_binomial_bound" if method.startswith("mc_") else "known_law_population"))
    pd.DataFrame(coverage).to_csv(output/"coverage.csv",index=False,float_format="%.17g")
    costs_summary=[]
    increments=dict(mean_bundle=None,mean_plus_variance="variance_increment_seconds",branch="branch_increment_seconds",
                    enumeration="enumeration_increment_seconds",**{f"mc_{n}":f"mc_{n}_increment_seconds" for n in config["mc_draws"]})
    for k,frame in costs.groupby("k"):
        for method,increment in increments.items():
            seconds=frame.mean_bundle_seconds.copy()
            if increment:
                seconds=seconds+frame[increment]
            seconds=seconds.dropna()
            if not len(seconds):
                continue
            costs_summary.append(dict(k=int(k),method=method,model_tasks=len(seconds),law_cells_per_task=9,
                median_seconds=float(seconds.median()),q25_seconds=float(seconds.quantile(.25)),q75_seconds=float(seconds.quantile(.75))))
    pd.DataFrame(costs_summary).to_csv(output/"cost_summary.csv",index=False,float_format="%.17g")
    enumerated=cells[cells.enumeration_status=="RUN"]
    largest=costs[costs.k==max(config["enumerated_dimensions"])]
    median_ratio=float(((largest.mean_bundle_seconds+largest.enumeration_increment_seconds)/largest.mean_bundle_seconds).median())
    count=int((enumerated.reference_aware<=.05).sum())
    incremental=int(sum(((cells.c6<=float(F(a)))&(cells.reference_aware>float(F(a)))).sum() for a in config["release_budgets"]))
    assert incremental==0
    events=[json.loads(line) for line in (run/"progress.jsonl").read_text().splitlines()]
    summary=dict(tasks=receipt["tasks"],cells=len(cells),enumerated_cells=len(enumerated),large_not_enumerated_cells=len(cells)-len(enumerated),
        total_enumerated_states=int(costs.enumerated_states.sum()),branch_nodes=int(costs.branch_nodes.sum()),
        branch_model_law_tasks=len(cells.drop_duplicates(["task_index","q"])),
        completed_branch_model_law_tasks=int(cells.drop_duplicates(["task_index","q"]).branch_complete.sum()),
        mc_draws=len(cells)*sum(config["mc_draws"]),mc_bounds=len(cells)*len(config["mc_draws"]),
        mc_misses_on_enumerated={str(n):int(cells[f"mc_{n}_miss"].sum()) for n in config["mc_draws"]},
        maximum_moment_error=float(cells.moment_error.max()),task_seconds=receipt["task_seconds"],warnings=receipt["warnings"],
        c6_extra_releases=incremental,reference_strictly_tighter_cells=int((cells.reference_aware<cells.c6-1e-12).sum()),
        cheap_certificate_gate=dict(alpha=.05,released=count,cells=len(enumerated),coverage=count/len(enumerated),
            largest_enumerated_dimension=max(config["enumerated_dimensions"]),median_enumeration_to_mean_cost_ratio=median_ratio,
            pass_gate=count/len(enumerated)>=.25 and median_ratio>=10),
        incremental_c6_gate="NO_GO_reference_mean_available",pytest_summary=validate["pytest_summary"],
        progress_milestones=[e for e in events if e["event"] in ("start","intentional_pause","complete","time_budget_stop")],
        financial_evaluation="NOT_RUN",human_study="NOT_RUN")
    write_json(output/"summary.json",summary)
    write_json(output/"artifact_export.json",dict(copies=exports,summary_script_sha256=digest(__file__),
        derived={str(output/name):digest(output/name) for name in ("coverage.csv","cost_summary.csv","summary.json")},
        local_progress=str(run/"progress.jsonl"),local_progress_sha256=digest(run/"progress.jsonl")))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.run,args.validation,args.output)
