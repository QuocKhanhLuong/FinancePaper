"""Export verified uncertainty audit aggregates; detailed logs remain local."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import time

import pandas as pd

from financepaper.evaluation.decision_value import digest,check_hashes,write_json


def summarize(run,first,validation,output):
    receipt=json.loads((run/"execution_receipt.json").read_text())
    check_hashes(receipt["frozen"]["sources"])
    check_hashes(receipt["outputs"]); check_hashes(receipt["task_receipts"])
    validation_receipt=json.loads((validation/"validation_receipt.json").read_text())
    assert validation_receipt["real_receipt_sha256"]==digest(run/"execution_receipt.json")
    box=pd.read_csv(run/"box.csv",dtype={"radius":str})
    sample=pd.read_csv(run/"sample.csv")
    output.mkdir(parents=True,exist_ok=False)
    summary_rows=[]; pair_rows=[]; cell_rows=[]
    for stage,frame,key,methods in (("box",box,"radius",("vector","c5","markov","generic")),
        ("sample",sample,"n",("vector","c5","markov","generic","direct_c5","direct_generic"))):
        for value,part in frame.groupby(key,sort=False):
            for subset,data in (("all",part),("original_interior",part[part.original_interior])):
                for budget in receipt["frozen"]["config"]["release_budgets"]:
                    for method in methods:
                        summary_rows.append(dict(stage=stage,setting=str(value),subset=subset,budget=budget,
                            method=method,certified=int(data[f"{method}@{budget}"].sum()),denominator=len(data),
                            underbounds=int(data[f"{method}_underbound"].sum()) if stage=="sample" else None))
            if stage=="sample":
                for budget in receipt["frozen"]["config"]["release_budgets"]:
                    v=part[f"vector@{budget}"]; d=part[f"direct_c5@{budget}"]
                    pair_rows.append(dict(n=int(value),budget=budget,vector_only=int((v & ~d).sum()),
                        direct_only=int((d & ~v).sum()),both=int((v & d).sum()),neither=int((~v & ~d).sum()),denominator=len(part)))
                for (m,profile),cell in part.groupby(["m","profile"],sort=False):
                    for method in methods:
                        cell_rows.append(dict(n=int(value),m=int(m),profile=profile,method=method,
                            certified_5pct=int(cell[f"{method}@1/20"].sum()),denominator=len(cell),
                            true_risk=float(cell.true_risk.iloc[0])))
    pd.DataFrame(summary_rows).to_csv(output/"coverage.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(output/"paired_sample.csv",index=False)
    pd.DataFrame(cell_rows).to_csv(output/"sample_cells.csv",index=False)
    old_manifest=json.loads((first/"manifest.json").read_text())
    check_hashes(old_manifest["frozen"]["sources"])
    old=[]
    for path in sorted(first.glob("*_complete.json")):
        record=json.loads(path.read_text()); check_hashes(record["outputs"])
        old.extend(json.loads(Path(next(iter(record["outputs"]))).read_text()))
    assert len(old)==490
    old_frame=pd.DataFrame(old)
    v2_records=[]
    for path in sorted(run.glob("*_box.json"))[:490]:
        v2_records.extend(json.loads(path.read_text()))
    omitted={"lp_max_error","probe_residual"}
    assert [{k:v for k,v in row.items() if k not in omitted} for row in old]==[
        {k:v for k,v in row.items() if k not in omitted} for row in v2_records]
    old_frame.to_csv(output/"first_run_partial.csv",index=False)
    snapshot=first/"frozen_source_snapshot"
    for name,expected in old_manifest["frozen"]["sources"].items():
        target=snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert digest(target)==expected
        else:
            shutil.copyfile(name,target)
    import run_uncertain_reference_moments as original
    solver=original.linprog; diagnostics=[]
    try:
        for method,presolve in (("highs-ds",True),("highs-ipm",True),("highs-ipm",False)):
            def limited(*args,**kwargs):
                kwargs["method"]=method
                kwargs["options"].update(presolve=presolve,time_limit=2.,maxiter=1000)
                return solver(*args,**kwargs)
            original.linprog=limited
            for z in (F(0),F(1,2)):
                tick=time.perf_counter()
                try:
                    value=original.uncertain_lp_residual(4,F(1,10),(F(1),)*3,(F(1),)*3,z)
                    result=dict(value=value,status="success")
                except AssertionError as error:
                    result=dict(status="solver_non_success",message=str(error))
                diagnostics.append(dict(method=method,presolve=presolve,z=str(z),seconds=time.perf_counter()-tick,**result))
    finally:
        original.linprog=solver
    write_json(output/"first_run_failure.json",dict(completed=490,remaining_not_run=1226,
        interrupted_exit_code=130,diagnostics=diagnostics,shared_scientific_rows_identical=490))
    for directory,prefix in ((run,""),(first,"first_")):
        events=[json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()]
        write_json(output/f"{prefix}progress_milestones.json",[e for e in events if e["event"]!="task_verified"])
    for name in ("manifest.json","execution_receipt.json","box.csv"):
        shutil.copyfile(run/name,output/name)
    shutil.copyfile(first/"manifest.json",output/"first_run_manifest.json")
    shutil.copyfile(validation/"validation_receipt.json",output/"validation_receipt.json")
    summary=dict(tasks=receipt["tasks"],box_rows=len(box),sample_rows=len(sample),
        lp_solves=int(box.lp_solves.sum()),lp_max_error=float(box.lp_max_error.max()),
        smallest_probe_residual=float(box.probe_residual.min()),
        vector_coverage_failures=int((~sample.vector_covered).sum()),direct_coverage_failures=int((~sample.direct_covered).sum()),
        underbounds={method:int(sample[f"{method}_underbound"].sum()) for method in ("vector","c5","markov","generic","direct_c5","direct_generic")},
        represented_iid_vectors=int(sample.n.sum()),binomial_count_replicates=int((sample.law_states==2).sum()),
        deterministic_replicates=int((sample.law_states==1).sum()),task_seconds=receipt["task_seconds"],warnings=receipt["warnings"],
        box_strict_improvements=int((box.vector<box.c5-1e-12).sum()),
        sample_vector_tighter_than_direct=int((sample.vector<sample.direct_c5-1e-12).sum()),
        sample_direct_tighter_than_vector=int((sample.direct_c5<sample.vector-1e-12).sum()),
        first_completed=490,shared_scientific_rows_identical=490)
    write_json(output/"summary.json",summary)
    write_json(output/"artifact_export.json",dict(run=str(run),first_run=str(first),validation=str(validation),
        summarizer_sha256=digest(__file__),files={str(p):digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("run","first","validation","output"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    summarize(args.run,args.first,args.validation,args.output)
