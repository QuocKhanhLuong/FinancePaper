"""Run the frozen synthetic compute/coverage gate with per-model resume."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"

import argparse
from fractions import Fraction as F
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.disclosure_compute import (
    make_polynomial,exact_means,exact_second_moments,enumerate_laws,
    branch_interval,monte_carlo,reference_aware_bound,cantelli_bound,
)
from financepaper.evaluation.law_robust_disclosure import law_risk_upper,transfer_risk_upper
from financepaper.evaluation.decision_value import digest,check_hashes,write_json
from run_completion_novelty_audit import utc,log


def mean_bundle(poly,config):
    qs=list(map(F,config["reference_plus_probabilities"]))
    means=exact_means(poly,qs)
    rows=[]
    for qi,(q,b) in enumerate(zip(qs,means,strict=True)):
        for ei,e_text in enumerate(config["contamination"]):
            e=F(e_text)
            t=(1-e/2)*b
            rows.append(dict(q_index=qi,e_index=ei,q=str(q),epsilon=e_text,b_exact=str(b),t_exact=str(t),
                reference_mean=float(b),mean=float(t),generic=float(1-t),
                transfer=float(transfer_risk_upper(t,e)),c6=float(law_risk_upper(t,e)),
                reference_aware=float(reference_aware_bound(t,b,e))))
    return means,rows


def variance_bundle(poly,config,means):
    seconds,uniform=exact_second_moments(poly,list(map(F,config["reference_plus_probabilities"])))
    bounds=[]
    for qi,b in enumerate(means):
        for e_text in config["contamination"]:
            e=F(e_text)
            variance=((1-e)*seconds[qi]+e*uniform-((1-e)*b)**2)/4
            bounds.append(dict(variance=float(variance),cantelli=float(cantelli_bound((1-e/2)*b,variance))))
    return seconds,uniform,bounds


def timed(operation,repeats=1):
    times=[]
    answer=None
    for _ in range(repeats):
        tick=time.perf_counter()
        answer=operation()
        times.append(time.perf_counter()-tick)
    return answer,dict(median=statistics.median(times),repeats=times)


def task(config,k,family,seed,task_index,total_cells):
    start=time.perf_counter()
    poly=make_polynomial(k,family,seed)
    qs=list(map(F,config["reference_plus_probabilities"]))
    # Deterministic dependency values are shared; recorded method timings below
    # separately include the work required by each information bundle.
    means=exact_means(poly,qs)
    delta=config["mc_family_error"]/(total_cells*len(config["mc_draws"]))
    operations={
        "mean":lambda:mean_bundle(poly,config),
        "variance":lambda:variance_bundle(poly,config,means),
        "branch":lambda:[branch_interval(poly,q,b,config["branch_node_budget"]) for q,b in zip(qs,means,strict=True)],
    }
    if k in config["enumerated_dimensions"]:
        operations["enumeration"]=lambda:enumerate_laws(poly,qs,means,config["enumeration_chunk"])
    for draws in config["mc_draws"]:
        def mc_operation(draws=draws):
            return [monte_carlo(poly,q,F(e),b,draws,delta,
                    config["schedule_seed"]+100000*task_index+100*qi+ei)
                for qi,(q,b) in enumerate(zip(qs,means,strict=True))
                for ei,e in enumerate(config["contamination"])]
        operations[f"mc_{draws}"]=mc_operation
    order=list(operations)
    np.random.default_rng(config["schedule_seed"]+task_index).shuffle(order)
    results,timing={},{}
    for name in order:
        results[name],timing[name]=timed(operations[name],config["timing_repeats"] if name in ("mean","variance") else 1)
    means,rows=results["mean"]
    seconds,uniform,variance=results["variance"]
    tolerance=config["tolerance"]
    for index,row in enumerate(rows):
        qi=row["q_index"]
        e=float(F(row["epsilon"]))
        branch=results["branch"][qi]
        row.update(variance[index])
        row["hybrid"]=min(row["reference_aware"],row["cantelli"])
        row["branch_lower"]=(1-e)*branch["lower_r"]+e*branch["lower_u"]
        row["branch_upper"]=(1-e)*branch["upper_r"]+e*branch["upper_u"]
        row["branch_unresolved"]=row["branch_upper"]-row["branch_lower"]
        row["branch_nodes"]=branch["nodes"]
        row["branch_complete"]=branch["complete"]
        row.update(k=k,family=family,model_seed=seed,task_index=task_index,terms=len(poly.masks),
            enumeration_status="RUN" if "enumeration" in results else "NOT_RUN_prespecified_scaling")
        assert row["reference_aware"]<=row["c6"]+tolerance
        if "enumeration" in results:
            oracle=results["enumeration"]
            reference=oracle["laws"][qi]
            assert abs(reference["probability_sum"]-1)<tolerance
            error=max(abs(reference["mean_r"]-float(means[qi])),abs(reference["second_r"]-float(seconds[qi])),
                      abs(oracle["mean_u"]),abs(oracle["second_u"]-float(uniform)))
            assert error<tolerance
            risk=(1-e)*reference["risk_r"]+e*reference["risk_u"]
            row.update(oracle=risk,moment_error=error)
            for method in ("generic","transfer","c6","reference_aware","cantelli","hybrid","branch_upper"):
                assert risk<=row[method]+tolerance,(k,family,seed,row,method)
            assert row["branch_lower"]<=risk+tolerance
        else:
            row.update(oracle=None,moment_error=None)
        for draws in config["mc_draws"]:
            sample=results[f"mc_{draws}"][index]
            row[f"mc_{draws}"]=sample["upper"]
            row[f"mc_{draws}_failures"]=sample["failures"]
            row[f"mc_{draws}_miss"]=None if row["oracle"] is None else sample["upper"]+tolerance<row["oracle"]
        row["mc_delta_per_bound"]=delta
    costs=dict(k=k,family=family,model_seed=seed,task_index=task_index,terms=len(poly.masks),
        support_states=2**k,enumerated_states=2**k if "enumeration" in results else 0,
        branch_nodes=sum(b["nodes"] for b in results["branch"]),
        mean_bundle_seconds=timing["mean"]["median"],
        variance_increment_seconds=timing["variance"]["median"],
        branch_increment_seconds=timing["branch"]["median"],
        enumeration_increment_seconds=timing["enumeration"]["median"] if "enumeration" in results else None,
        total_task_seconds=time.perf_counter()-start)
    for draws in config["mc_draws"]:
        costs[f"mc_{draws}_increment_seconds"]=timing[f"mc_{draws}"]["median"]
    return dict(rows=rows,costs=costs,timing_repeats=timing,method_order=order,
                polynomial=dict(k=k,masks=poly.masks,coefficients=poly.coefficients))


def run(config_path,output,resume=False,stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks<1:
        raise ValueError("Positive stop count required")
    config=json.loads(config_path.read_text())
    sources=[config_path,Path(__file__),Path("src/financepaper/evaluation/disclosure_compute.py"),
        Path("src/financepaper/evaluation/law_robust_disclosure.py"),Path("src/financepaper/evaluation/decision_value.py"),
        Path("scripts/run_completion_novelty_audit.py"),Path("tests/test_disclosure_compute.py"),
        Path("docs/DISCLOSURE_COMPUTE_GATE.md"),Path("pyproject.toml"),Path("uv.lock")]
    frozen=dict(config=config,sources={str(p):digest(p) for p in sources},device="cpu",numeric_threads=1,
        python=platform.python_version(),platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas","tqdm")})
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"]!=frozen:
            raise ValueError("Resume rejected: source/config/dependency drift or missing --resume")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("Need new empty run directory")
        manifest=dict(created_utc=utc(),frozen=frozen,base_git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
        write_json(manifest_path,manifest)
    tasks=[(k,family,seed) for k in config["enumerated_dimensions"]+config["large_dimensions"]
           for family in config["families"] for seed in config["model_seeds"]]
    np.random.default_rng(config["schedule_seed"]).shuffle(tasks)
    total_cells=len(tasks)*len(config["reference_plus_probabilities"])*len(config["contamination"])
    log(output,"start",tasks=len(tasks),cells=total_cells,resume=resume,device="cpu")
    records,costs,receipts=[],[],{}
    new=reused=0
    tick=time.perf_counter()
    for index,(k,family,seed) in enumerate(tqdm(tasks,desc="Disclosure compute gate",unit="model",mininterval=1)):
        name=f"model_k{k}_{family}_s{seed}"
        path,receipt_path=output/f"{name}.json",output/f"{name}_complete.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if set(receipt["outputs"])!={str(path)}:
                raise ValueError("Task output set mismatch")
            check_hashes(receipt["outputs"])
            reused+=1
        else:
            if path.exists():
                raise ValueError("Unreceipted output")
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result=task(config,k,family,seed,index,total_cells)
            write_json(path,result)
            receipt=dict(seconds=time.perf_counter()-start,outputs={str(path):digest(path)},
                         warnings=[str(w.message) for w in caught])
            write_json(receipt_path,receipt)
            new+=1
        result=json.loads(path.read_text())
        records.extend(result["rows"])
        costs.append(result["costs"])
        receipts[str(receipt_path)]=digest(receipt_path)
        log(output,"task_verified",task=name,new=new,reused=reused,seconds=receipt["seconds"],warnings=receipt["warnings"],
            eta_seconds=(time.perf_counter()-tick)/max(new,1)*(len(tasks)-new-reused))
        if (stop_after_tasks and new>=stop_after_tasks) or receipt["seconds"]>config["max_task_seconds"]:
            log(output,"intentional_pause" if stop_after_tasks and new>=stop_after_tasks else "time_budget_stop",
                new=new,reused=reused,remaining=len(tasks)-new-reused)
            return
    check_hashes(frozen["sources"])
    outputs={}
    for name,data in (("cells",records),("costs",costs)):
        path=output/f"{name}.csv"
        temporary=path.with_suffix(".csv.tmp")
        pd.DataFrame(data).to_csv(temporary,index=False,float_format="%.17g")
        if path.exists() and digest(path)!=digest(temporary):
            raise ValueError("Aggregate drift")
        temporary.replace(path)
        outputs[str(path)]=digest(path)
    execution=output/"execution_receipt.json"
    if execution.exists():
        old=json.loads(execution.read_text())
        check_hashes(old["outputs"])
        check_hashes(old["task_receipts"])
    else:
        write_json(execution,dict(**manifest,completed_utc=utc(),tasks=len(tasks),cells=len(records),
            outputs=outputs,task_receipts=receipts,task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
            warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
            financial_evaluation="NOT_RUN",model_fitting="NOT_RUN",human_study="NOT_RUN"))
    log(output,"complete",new=new,reused=reused,seconds=time.perf_counter()-tick)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/disclosure_compute_gate.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-tasks",type=int)
    args=parser.parse_args()
    run(args.config,args.output,args.resume,args.stop_after_tasks)
