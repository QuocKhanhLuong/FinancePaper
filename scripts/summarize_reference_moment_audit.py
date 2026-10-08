"""Export the complete rerun and retain the first numerical failure as evidence."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil

import pandas as pd

from financepaper.evaluation.completion_rank import rank_frontier
from financepaper.evaluation.reference_moments import parameters,range_markov_bound
from financepaper.evaluation.decision_value import check_hashes,digest,write_json
from run_reference_moment_audit import generic_moment_lp as first_lp
from run_reference_moment_audit_v2 import generic_moment_lp as repaired_lp


def summarize(run,validation,first_run,first_validation,output):
    receipt=json.loads((run/"execution_receipt.json").read_text())
    verify=json.loads((validation/"validation_receipt.json").read_text())
    first=json.loads((first_run/"manifest.json").read_text())
    for manifest in (receipt,first):
        check_hashes(manifest["frozen"]["sources"])
    assert first["frozen"]["config"]==receipt["frozen"]["config"]
    check_hashes(receipt["outputs"])
    check_hashes(receipt["task_receipts"])
    assert digest(run/"execution_receipt.json")==verify["real_receipt_sha256"]
    assert all(c["accepted"] for c in verify["commands"])
    original_rows=[]
    for path in sorted(first_run.glob("*_complete.json")):
        task=json.loads(path.read_text())
        check_hashes(task["outputs"])
        original_rows.extend(json.loads(Path(p).read_text()) for p in task["outputs"])
    assert len(original_rows)==272
    assert not (first_run/"execution_receipt.json").exists()
    output.mkdir(parents=True,exist_ok=False)
    exports={}
    for source in [run/name for name in ("profile.csv","c5_attainer.csv","manifest.json","execution_receipt.json")]+[validation/"validation_receipt.json"]:
        target=output/source.name
        shutil.copy2(source,target)
        assert digest(source)==digest(target)
        exports[str(target)]=dict(source=str(source),sha256=digest(target))
    shutil.copy2(first_run/"manifest.json",output/"first_run_manifest.json")
    pd.DataFrame(original_rows).to_csv(output/"first_run_partial.csv",index=False,float_format="%.17g")
    info=parameters(16,F(9,10),[0]*15)
    assert info["tau"]-sum(a*mu for a,mu in zip(info["a"],info["mu"],strict=True))==0
    old_value,new_value=first_lp(info),repaired_lp(info)
    failure=dict(first_run=str(first_run),completed_tasks=272,remaining_tasks=64,status="STOPPED_NUMERICAL_DISCREPANCY",
        failed_case=dict(m=16,p="9/10",moments=["0"]*15),exact_optimum="1",original_lp=old_value,
        original_error=abs(1-old_value),v2_lp=new_value,v2_error=abs(1-new_value),acceptance_tolerance=1e-9,
        exact_primal_certificate=dict(z="1",s=["1/2"]*15,event_residual="0",objective_upper_bound="1"),
        failure_log=str(first_validation/"complete_resume.log"),failure_log_sha256=digest(first_validation/"complete_resume.log"),
        frozen_source_snapshot=str(first_run/"frozen_source_snapshot"))
    assert failure["original_error"]>1e-9 and failure["v2_error"]<1e-9
    write_json(output/"first_run_failure.json",failure)
    profiles=pd.read_csv(run/"profile.csv")
    attainers=pd.read_csv(run/"c5_attainer.csv")
    rows=profiles.to_dict("records")
    coverage=[]
    for alpha_text in receipt["frozen"]["config"]["release_budgets"]:
        alpha=F(alpha_text)
        counts=dict(generic=0,range_markov=0,c5=0,moment_bound=0)
        interior_counts=dict(counts)
        interior_n=0
        for row in rows:
            m,p,t=int(row["m"]),F(row["p"]),F(row["exact_mean"])
            b=tuple(map(F,row["moments"].split(",")))
            decisions=dict(generic=1-t<=alpha,range_markov=range_markov_bound(m,p,b)<=alpha,
                c5=t>=rank_frontier(m,p,alpha)["normalized_mean"],moment_bound=F(row["exact_risk"])<=alpha)
            interior=all(-1<v<1 for v in b)
            interior_n+=interior
            for method,release in decisions.items():
                counts[method]+=int(release)
                interior_counts[method]+=int(interior and release)
        for scope,denom,values in (("all_profiles",len(rows),counts),("strictly_interior_moments",interior_n,interior_counts)):
            coverage.extend(dict(scope=scope,alpha=alpha_text,method=method,profiles=denom,released=count)
                            for method,count in values.items())
    pd.DataFrame(coverage).to_csv(output/"coverage.csv",index=False)
    eps=1e-12
    summary=dict(profile_tasks=len(profiles),attainer_tasks=len(attainers),generic_moment_lp_solves=len(profiles)+len(attainers),
        full_table_profile_cases=int((profiles.full_table_status=="RUN").sum()),full_table_lp_solves=int(profiles.full_lp_solves.sum()),
        full_table_above_bound_probes=int(profiles.full_probe_min_gap.notna().sum()),minimum_rejected_probe_gap=float(profiles.full_probe_min_gap.min()),
        maximum_moment_lp_error=float(max(profiles.moment_lp_error.max(),attainers.moment_lp_error.max())),
        vector_strictly_tighter_than_c5=int((profiles.moment_bound<profiles.c5-eps).sum()),
        vector_strictly_tighter_than_range=int((profiles.moment_bound<profiles.range_markov-eps).sum()),
        c5_strictly_tighter_than_range=int((profiles.c5<profiles.range_markov-eps).sum()),
        range_strictly_tighter_than_c5=int((profiles.range_markov<profiles.c5-eps).sum()),
        zero_risk_profiles=int((profiles.moment_bound==0).sum()),unit_risk_profiles=int((profiles.moment_bound==1).sum()),
        strict_comparison_display_tolerance=eps,task_seconds=receipt["task_seconds"],warnings=receipt["warnings"],
        pytest_summary=verify["pytest_summary"],numerical_rerun=True,config_changed=False,
        first_run_completed_tasks=272,first_run_remaining_tasks=64,full_table_dimension_cap=5,
        universal_free_moment_extraction="NOT_RUN",financial_evaluation="NOT_RUN",human_study="NOT_RUN")
    write_json(output/"summary.json",summary)
    for source in (run/"progress.jsonl",first_run/"progress.jsonl"):
        events=[json.loads(line) for line in source.read_text().splitlines()]
        name="progress_milestones.json" if source.parent==run else "first_progress_milestones.json"
        write_json(output/name,[e for e in events if e["event"] in ("start","intentional_pause","complete","time_budget_stop")])
    derived={str(p):digest(p) for p in output.iterdir() if str(p) not in exports}
    write_json(output/"artifact_export.json",dict(copies=exports,derived=derived,
        summarizer_sha256=digest(__file__),first_log=dict(path=failure["failure_log"],sha256=failure["failure_log_sha256"])))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","validation","first-run","first-validation","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.run,args.validation,args.first_run,args.first_validation,args.output)
