"""Numerical repair of the frozen reference-moment audit; see V2 protocol addendum."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"

import argparse
from fractions import Fraction as F
from functools import lru_cache
import importlib.metadata
import json
from math import comb
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm.auto import tqdm

from financepaper.evaluation.bounded_completion import finite_definition_matrix
from financepaper.evaluation.completion_rank import rank_frontier,rank_risk_upper
from financepaper.evaluation.reference_moments import parameters,reference_moment_bound,range_markov_bound,moment_witness,witness_table
from financepaper.evaluation.decision_value import digest,check_hashes,write_json
from run_completion_novelty_audit import log,utc


def generic_moment_lp(info):
    n=len(info["mu"])
    constraints,rhs=[],[]
    for k,mu in enumerate(info["mu"]):
        row=np.zeros(n+1); row[0]=-1; row[k+1]=1
        constraints.append(row); rhs.append(0.)
        row=np.zeros(n+1); row[0]=1; row[k+1]=-1
        constraints.append(row); rhs.append(float(1-mu))
    # Equivalent row scaling prevents relevant tiny beta coefficients from
    # falling below the solver's absolute matrix-entry threshold.
    constraints.append(1e6*np.array([float(info["tau"])]+[-float(a) for a in info["a"]]))
    rhs.append(0.)
    result=linprog([-1.]+[0.]*n,A_ub=np.array(constraints),b_ub=rhs,
        bounds=[(0.,1.)]+[(0.,float(mu)) for mu in info["mu"]],method="highs-ipm",
        options={"presolve":False,"primal_feasibility_tolerance":1e-10,
                 "dual_feasibility_tolerance":1e-10,"ipm_optimality_tolerance":1e-12})
    assert result.success,result.message
    return float(result.x[0])


@lru_cache(maxsize=64)
def definition_basis(m,p):
    # The hidden reference enters coalition expectations affinely. Use two
    # positive laws so the historical definition oracle remains unchanged.
    bases=[]
    for z in (F(1,3),F(2,3)):
        rows=[]
        for h in (0,1):
            matrix=finite_definition_matrix(m,p,(z,1-z),h)
            rows.append(tuple((a-b)/(1-p) for a,b in zip(matrix[0],matrix[1],strict=True)))
        bases.append(tuple(rows))
    return tuple(bases)


def definition_pair(m,p,z):
    left,right=definition_basis(m,p)
    return tuple(tuple((2-3*z)*a+(3*z-1)*b for a,b in zip(x,y,strict=True))
                 for x,y in zip(left,right,strict=True))


def full_table_minimum(m,p,b,z,constant):
    operators=definition_pair(m,p,z)
    rows=np.zeros((m-1,2**m*2))
    for h,mass in enumerate((z,1-z)):
        for state in range(2**(m-2)):
            k=state.bit_count()
            coefficient=float(mass/F(comb(m-2,k)))
            rows[k,h*2**m+4*state+1]+=coefficient
            rows[k,h*2**m+4*state+2]-=coefficient
    bounds=[(0.,1.)]*(2**m*2)
    for h in (0,1):
        bounds[(h+1)*2**m-1]=(float(constant),float(constant))
    result=linprog(np.asarray(operators[0],float),A_eq=rows,b_eq=np.asarray(b,float),bounds=bounds,method="highs",
        options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    assert result.success,result.message
    return float(result.fun)


def audit(config,m,p_text,profile,b,stage="profile",expected_z=None):
    p=F(p_text)
    info=parameters(m,p,b)
    witness=moment_witness(m,p,b)
    z=witness["risk"]
    assert z==reference_moment_bound(m,p,b)
    assert sum(witness["law"])==1 and min(witness["law"])>0
    for k,value in enumerate(b):
        assert sum(q*d[k] for q,d in zip(witness["law"],witness["contrasts"],strict=True))==value
    assert all(-1<=v<=1 for point in witness["contrasts"] for v in point)
    assert sum(q for q,g in zip(witness["law"],witness["gaps"],strict=True) if g<=0)==z
    assert sum(q*g for q,g in zip(witness["law"],witness["gaps"],strict=True))==info["mean"]
    if expected_z is not None:
        assert z==expected_z
    optimum=generic_moment_lp(info)
    lp_error=abs(optimum-float(z))
    assert lp_error<config["tolerance"],(m,p_text,b,z,optimum)
    c5=rank_risk_upper(m,p,info["mean"])
    markov=range_markov_bound(m,p,b)
    assert z<=c5 and z<=markov
    row=dict(stage=stage,m=m,p=p_text,profile=profile,moments=",".join(map(str,b)),
        exact_mean=str(info["mean"]),mean=float(info["mean"]),exact_risk=str(z),moment_bound=float(z),
        c5=float(c5),range_markov=float(markov),generic=float(1-info["mean"]),
        moment_lp=optimum,moment_lp_error=lp_error,exact_witness=True,
        witness_states=len(witness["law"]),full_lp_solves=0,full_table_status="NOT_RUN_DIMENSION_CAP",
        full_boundary_min_gap=None,full_probe_min_gap=None,full_probe_z=None,exact_definition_check=None)
    if stage=="profile" and m<=config["full_table_max_m"]:
        table=witness_table(m,witness,F(config["constant_prediction"]))
        if len(witness["law"])==2:
            operators=definition_pair(m,p,z)
        else:
            matrix=finite_definition_matrix(m,p,witness["law"],0)
            operators=(tuple((a-b)/(1-p) for a,b in zip(matrix[0],matrix[1],strict=True)),)
        for op,gap in zip(operators,witness["gaps"],strict=True):
            assert sum(a*v for a,v in zip(op,table,strict=True))==gap
        assert min(table)>=0 and max(table)<=1
        assert all(table[(h+1)*2**m-1]==F(config["constant_prediction"]) for h in range(len(witness["law"])))
        value=full_table_minimum(m,p,b,z,F(config["constant_prediction"]))
        assert value<config["tolerance"],(m,p_text,profile,z,value)
        row.update(full_boundary_min_gap=value,full_lp_solves=1,full_table_status="RUN",exact_definition_check=True)
        if z<1:
            probe=(1+z)/2
            value=full_table_minimum(m,p,b,probe,F(config["constant_prediction"]))
            assert value>config["tolerance"],(m,p_text,profile,probe,value)
            row.update(full_probe_min_gap=value,full_probe_z=str(probe),full_lp_solves=2)
    elif stage=="c5_attainer":
        row["full_table_status"]="NOT_RUN_attainer_checked_compactly"
    return row


def profile_vectors(config,m):
    n=m-1
    result=[(f"constant_{value}",tuple([F(value)]*n)) for value in config["constant_moments"]]
    for name in config["structured_profiles"]:
        ascending=tuple(F(k,n-1) if n>1 else F(1,2) for k in range(n))
        values={"ascending":ascending,"descending":tuple(1-v for v in ascending),"alternating":tuple(F(k%2) for k in range(n))}
        result.append((name,values[name]))
    for seed in config["random_seeds"]:
        rng=np.random.default_rng(seed)
        grid=list(map(F,config["random_moment_values"]))
        result.append((f"random_{seed}",tuple(grid[int(i)] for i in rng.integers(0,len(grid),n))))
    return result


def run(config_path,output,resume=False,stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks<1:
        raise ValueError("Positive stop count required")
    config=json.loads(config_path.read_text())
    paths=[config_path,Path(__file__),Path("src/financepaper/evaluation/reference_moments.py"),
        Path("src/financepaper/evaluation/completion_rank.py"),Path("src/financepaper/evaluation/bounded_completion.py"),
        Path("src/financepaper/evaluation/decision_value.py"),Path("scripts/run_completion_novelty_audit.py"),
        Path("tests/test_reference_moments.py"),Path("docs/REFERENCE_MOMENT_FRONTIER.md"),Path("docs/REFERENCE_MOMENT_NUMERICS_V2.md"),
        Path("tests/test_reference_moment_numerics.py"),Path("pyproject.toml"),Path("uv.lock")]
    frozen=dict(config=config,sources={str(p):digest(p) for p in paths},device="cpu",numeric_threads=1,
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
    tasks=[]
    for m in config["observed_dimensions"]:
        for p in config["reference_probabilities"]:
            for profile,b in profile_vectors(config,m):
                tasks.append((f"m{m}_p{p}_{profile}".replace("/","_"),(config,m,p,profile,b)))
            for z_text in config["attaining_failure_masses"]:
                z=F(z_text)
                b=tuple(1-2*z*l for l in rank_frontier(m,F(p),z)["loss"])
                tasks.append((f"m{m}_p{p}_attainer_{z_text}".replace("/","_"),(config,m,p,z_text,b,"c5_attainer",z)))
    rows,receipts=[],{}
    new=reused=0
    start=time.perf_counter()
    log(output,"start",tasks=len(tasks),resume=resume,device="cpu")
    for name,args in tqdm(tasks,desc="Reference moment audit",unit="task",mininterval=1):
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
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                row=audit(*args)
            write_json(path,row)
            receipt=dict(seconds=time.perf_counter()-tick,outputs={str(path):digest(path)},warnings=[str(w.message) for w in caught])
            write_json(receipt_path,receipt)
            new+=1
        receipts[str(receipt_path)]=digest(receipt_path)
        rows.append(json.loads(path.read_text()))
        log(output,"task_verified",task=name,new=new,reused=reused,seconds=receipt["seconds"],warnings=receipt["warnings"],
            eta_seconds=(time.perf_counter()-start)/max(1,new)*(len(tasks)-new-reused))
        if (stop_after_tasks and new>=stop_after_tasks) or receipt["seconds"]>config["max_task_seconds"]:
            log(output,"intentional_pause" if stop_after_tasks and new>=stop_after_tasks else "time_budget_stop",new=new,reused=reused)
            return
    check_hashes(frozen["sources"])
    outputs={}
    for stage,frame in pd.DataFrame(rows).groupby("stage"):
        path=output/f"{stage}.csv"
        temporary=path.with_suffix(".csv.tmp")
        frame.to_csv(temporary,index=False,float_format="%.17g")
        if path.exists() and digest(path)!=digest(temporary):
            raise ValueError("Aggregate drift")
        temporary.replace(path)
        outputs[str(path)]=digest(path)
    execution=output/"execution_receipt.json"
    if execution.exists():
        receipt=json.loads(execution.read_text())
        check_hashes(receipt["outputs"]); check_hashes(receipt["task_receipts"])
    else:
        write_json(execution,dict(**manifest,completed_utc=utc(),tasks=len(tasks),rows=len(rows),outputs=outputs,task_receipts=receipts,
            task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
            warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
            financial_evaluation="NOT_RUN",training="NOT_RUN",human_study="NOT_RUN"))
    log(output,"complete",new=new,reused=reused,seconds=time.perf_counter()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/reference_moment_audit.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-tasks",type=int)
    args=parser.parse_args()
    run(args.config,args.output,args.resume,args.stop_after_tasks)
