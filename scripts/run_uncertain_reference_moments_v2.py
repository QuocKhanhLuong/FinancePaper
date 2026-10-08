"""Frozen deterministic-box and iid-sampling audit, with checked resume."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"

import argparse
from fractions import Fraction as F
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm.auto import tqdm

from financepaper.evaluation.completion_rank import rank_cells,rank_risk_upper
from financepaper.evaluation.reference_moments import parameters,reference_moment_bound,range_markov_bound,moment_witness
from financepaper.evaluation.uncertain_reference_moments import moment_box,interval_risk_bound,hoeffding_radius,c5_certifies
from financepaper.evaluation.decision_value import digest,check_hashes,write_json
from run_reference_moment_audit_v2 import profile_vectors
from run_completion_novelty_audit import utc,log


@lru_cache(maxsize=16384)
def c5_bound(m,p,mean):
    return rank_risk_upper(m,p,max(F(0),mean))


def bounds(m,p,b):
    info=parameters(m,p,b)
    return dict(vector=reference_moment_bound(m,p,b),
        c5=c5_bound(m,p,info["mean"]),markov=range_markov_bound(m,p,b),
        generic=1-max(F(0),info["mean"])),info["mean"]


def uncertain_lp_residual(m,p,lower,upper,z):
    """Independent LP over unknown moments b and bad-event masses s, at fixed z."""
    cells=rank_cells(m,p); n=len(cells)
    rows=[]; rhs=[]
    for k in range(n):
        row=np.zeros(2*n); row[k]=.5; row[n+k]=1
        rows.append(row); rhs.append(.5)
        rows.append(-row); rhs.append(float(F(1,2)-z))
    scale=1e6
    objective=scale*np.array([float(z*(a-b)/2) for a,b in cells]+[-float(b) for _,b in cells])
    result=linprog(objective,A_ub=np.array(rows),b_ub=rhs,
        bounds=[(float(l),float(u)) for l,u in zip(lower,upper,strict=True)]+[(0,float(z))]*n,
        method="highs-ipm",options={"presolve":True,"time_limit":10.,"maxiter":10000,"primal_feasibility_tolerance":1e-10,
        "dual_feasibility_tolerance":1e-10,"ipm_optimality_tolerance":1e-12})
    assert result.success,result.message
    return float(result.fun/scale+float(z)/4)


def box_task(config,m,p,profile,b,radius):
    lower,upper=moment_box(b,radius)
    result,mean=bounds(m,p,lower)
    risk=interval_risk_bound(m,p,lower,upper)
    assert risk==result["vector"] and all(risk<=v for v in result.values())
    witness=moment_witness(m,p,lower)
    assert witness["risk"]==risk
    assert sum(q for q,g in zip(witness["law"],witness["gaps"],strict=True) if g<=0)==risk
    for k,v in enumerate(lower):
        assert sum(q*d[k] for q,d in zip(witness["law"],witness["contrasts"],strict=True))==v
    info=parameters(m,p,lower)
    errors=[]; probe_residual=None
    for z in ([risk,(1+risk)/2] if risk<1 else [risk]):
        exact=z*info["tau"]-sum(a*min(z,mu) for a,mu in zip(info["a"],info["mu"],strict=True))
        measured=uncertain_lp_residual(m,p,lower,upper,z)
        errors.append(abs(measured-float(exact)))
        assert errors[-1]<config["lp_tolerance"],(m,p,profile,radius,z,exact,measured)
        if z>risk:
            assert exact>0 and measured>0
            probe_residual=measured
    row=dict(stage="box",m=m,p=str(p),profile=profile,radius=str(radius),
        center=",".join(map(str,b)),lower=",".join(map(str,lower)),upper=",".join(map(str,upper)),
        original_interior=all(-1<x<1 for x in b),lower_interior=all(-1<x<1 for x in lower),
        exact_vector=str(risk),mean_lower=str(mean),lp_solves=len(errors),
        lp_max_error=max(errors),probe_residual=probe_residual,
        **{method:float(value) for method,value in result.items()})
    for budget in config["release_budgets"]:
        for method,value in result.items():
            row[f"{method}@{budget}"]=c5_certifies(m,p,mean,F(budget)) if method=="c5" else value<=F(budget)
    return [row]


def sample_task(config,m,p,profile,b,n,index):
    cells=rank_cells(m,p); witness=moment_witness(m,p,b)
    actual=witness["risk"]; true_mean=parameters(m,p,b)["mean"]
    vector_radius=hoeffding_radius(n,m-1,F(config["confidence_delta"]))
    scalar_radius=hoeffding_radius(n,1,F(config["confidence_delta"]))
    rng=np.random.default_rng(np.random.SeedSequence([config["seed"],index]))
    rows=[]
    for replicate in range(config["replicates"]):
        # NumPy binomial sufficient counts avoid allocating N repeated vectors.
        count=n if len(witness["law"])==1 else int(rng.binomial(n,float(witness["law"][0])))
        first,last=witness["contrasts"][0],witness["contrasts"][-1]
        estimate=tuple((count*x+(n-count)*y)/n for x,y in zip(first,last,strict=True))
        lower=tuple(max(F(-1),v-vector_radius) for v in estimate)
        result,mean=bounds(m,p,lower)
        direct_mean=max(F(-1),sum(a*v for (a,_),v in zip(cells,estimate,strict=True))-scalar_radius)
        result.update(direct_c5=c5_bound(m,p,direct_mean),direct_generic=1-max(F(0),direct_mean))
        vector_covered=all(l<=v for l,v in zip(lower,b,strict=True))
        direct_covered=direct_mean<=true_mean
        if vector_covered:
            assert all(actual<=result[k] for k in ("vector","c5","markov","generic"))
        if direct_covered:
            assert actual<=result["direct_c5"] and actual<=result["direct_generic"]
        row=dict(stage="sample",m=m,p=str(p),profile=profile,n=n,replicate=replicate,
            first_state_count=count,law_states=len(witness["law"]),true_risk=float(actual),
            exact_true_risk=str(actual),vector_radius=float(vector_radius),scalar_radius=float(scalar_radius),
            original_interior=all(-1<x<1 for x in b),vector_covered=vector_covered,direct_covered=direct_covered,
            **{method:float(value) for method,value in result.items()})
        for method,value in result.items():
            row[f"{method}_underbound"]=value<actual
            for budget in config["release_budgets"]:
                row[f"{method}@{budget}"]=(c5_certifies(m,p,direct_mean if method=="direct_c5" else mean,F(budget))
                    if method in ("c5","direct_c5") else value<=F(budget))
        rows.append(row)
    return rows


def run(config_path,output,resume=False,stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks<1:
        raise ValueError("Positive stop count required")
    config=json.loads(config_path.read_text())
    profiles_path=Path(config["profile_config"])
    profiles=json.loads(profiles_path.read_text())
    paths=[Path("docs/UNCERTAIN_REFERENCE_NUMERICS_V2.md"),Path("tests/test_uncertain_reference_numerics.py"),config_path,profiles_path,Path(__file__),Path("src/financepaper/evaluation/uncertain_reference_moments.py"),
        Path("src/financepaper/evaluation/reference_moments.py"),Path("src/financepaper/evaluation/completion_rank.py"),
        Path("src/financepaper/evaluation/bounded_completion.py"),Path("src/financepaper/evaluation/decision_value.py"),
        Path("scripts/run_reference_moment_audit_v2.py"),Path("scripts/run_completion_novelty_audit.py"),
        Path("tests/test_uncertain_reference_moments.py"),Path("docs/UNCERTAIN_REFERENCE_MOMENTS.md"),Path("pyproject.toml"),Path("uv.lock")]
    frozen=dict(config=config,sources={str(p):digest(p) for p in paths},device="cpu",numeric_threads=1,
        python=platform.python_version(),platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas","tqdm")})
    output.mkdir(parents=True,exist_ok=True); manifest_path=output/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"]!=frozen:
            raise ValueError("Resume rejected: source/config/dependency drift or missing --resume")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("Need new empty run directory")
        manifest=dict(created_utc=utc(),frozen=frozen,base_git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
        write_json(manifest_path,manifest)
    tasks=[]
    for m in profiles["observed_dimensions"]:
        for p_text in profiles["reference_probabilities"]:
            for profile,b in profile_vectors(profiles,m):
                for radius in config["radii"]:
                    tasks.append(("box",(config,m,F(p_text),profile,b,F(radius))))
    for m in config["sample_dimensions"]:
        for profile,b in profile_vectors(profiles,m):
            for n in config["sample_sizes"]:
                tasks.append(("sample",(config,m,F(config["sample_probability"]),profile,b,n,len(tasks))))
    rows=[]; receipts={}; new=reused=0; start=time.perf_counter()
    log(output,"start",tasks=len(tasks),resume=resume,device="cpu")
    for index,(stage,args) in enumerate(tqdm(tasks,desc="Uncertain moments",unit="task",mininterval=1)):
        name=f"{index:05d}_{stage}"; path=output/f"{name}.json"; receipt_path=output/f"{name}_complete.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if set(receipt["outputs"])!={str(path)}:
                raise ValueError("Task output set mismatch")
            check_hashes(receipt["outputs"]); reused+=1
        else:
            if path.exists():
                raise ValueError("Unreceipted output")
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                payload=(box_task if stage=="box" else sample_task)(*args)
            write_json(path,payload)
            receipt=dict(seconds=time.perf_counter()-tick,outputs={str(path):digest(path)},warnings=[str(w.message) for w in caught])
            write_json(receipt_path,receipt); new+=1
        receipts[str(receipt_path)]=digest(receipt_path); rows.extend(json.loads(path.read_text()))
        log(output,"task_verified",task=name,new=new,reused=reused,seconds=receipt["seconds"],warnings=receipt["warnings"],
            eta_seconds=(time.perf_counter()-start)/max(1,new)*(len(tasks)-new-reused))
        if (stop_after_tasks and new>=stop_after_tasks) or receipt["seconds"]>config["max_task_seconds"]:
            log(output,"intentional_pause" if stop_after_tasks and new>=stop_after_tasks else "time_budget_stop",new=new,reused=reused)
            return
    check_hashes(frozen["sources"]); outputs={}
    for stage,frame in pd.DataFrame(rows).groupby("stage"):
        path=output/f"{stage}.csv"; temporary=path.with_suffix(".csv.tmp")
        frame.dropna(axis=1,how="all").to_csv(temporary,index=False,float_format="%.17g")
        if path.exists() and digest(path)!=digest(temporary):
            raise ValueError("Aggregate drift")
        temporary.replace(path); outputs[str(path)]=digest(path)
    execution=output/"execution_receipt.json"
    if execution.exists():
        receipt=json.loads(execution.read_text()); check_hashes(receipt["outputs"]); check_hashes(receipt["task_receipts"])
    else:
        write_json(execution,dict(**manifest,completed_utc=utc(),tasks=len(tasks),rows=len(rows),outputs=outputs,task_receipts=receipts,
            task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
            warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
            financial_evaluation="NOT_RUN",training="NOT_RUN",human_study="NOT_RUN"))
    log(output,"complete",new=new,reused=reused,seconds=time.perf_counter()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/uncertain_reference_moments.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-tasks",type=int)
    args=parser.parse_args()
    run(args.config,args.output,args.resume,args.stop_after_tasks)
