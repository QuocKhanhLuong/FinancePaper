"""Finite definition/LP audit of C6; no fitted models or financial outcomes."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
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

from financepaper.evaluation.completion_moments import FiniteMarginal, coalition_oracle
from financepaper.evaluation.law_robust_disclosure import law_frontier, law_risk_upper, law_witness, transfer_risk_upper
from financepaper.evaluation.decision_value import check_hashes, digest, write_json
from run_completion_novelty_audit import log, utc


@lru_cache(maxsize=128)
def operators(p, reference):
    marginal = FiniteMarginal((0., 1.), (float(1-p), float(p)))
    hidden = FiniteMarginal(tuple(map(float, range(len(reference)))), tuple(map(float, reference)))
    answer = np.zeros((len(reference), 4*len(reference)))
    for col in range(answer.shape[1]):
        def predict(batch):
            codes = batch[:, 0]+2*batch[:, 1]+4*batch[:, 2]
            return (codes == col).astype(float)
        for h in range(len(reference)):
            phi = coalition_oracle(predict, [marginal, marginal, hidden], [1., 1., float(h)], [0, 1])
            answer[h, col] = (phi[0]-phi[1])/float(1-p)
    return answer


def full_lp(matrix, completion, bad, constant):
    bounds = [(0., 1.)]*matrix.shape[1]
    for h in range(len(completion)):
        bounds[4*h+3] = (float(constant), float(constant))
    objective = np.asarray(completion, float)@matrix
    result = linprog(-objective, A_ub=matrix[bad], b_ub=np.zeros(len(bad)), bounds=bounds, method="highs")
    assert result.success, result.message
    return -float(result.fun)


def witness_case(config, p_text, z_text, epsilon_text):
    p, z, epsilon = map(F, (p_text, z_text, epsilon_text))
    rows = []
    for c_text in config["constant_predictions"]:
        witness = law_witness(z, epsilon, F(c_text))
        reference, completion = witness["reference"], witness["completion"]
        matrix = operators(p, reference)
        gaps = matrix@np.asarray(witness["table"], float)
        error = float(np.max(np.abs(gaps-np.asarray(witness["normalized_gaps"], float))))
        assert error < config["tolerance"]
        exact_mean = sum(a*b for a,b in zip(completion,witness["normalized_gaps"], strict=True))
        expected = law_frontier(z, epsilon)
        assert exact_mean == expected and law_risk_upper(exact_mean, epsilon) == z
        optimum = full_lp(matrix, completion, [0], F(c_text))
        lp_error = abs(optimum-float(expected))
        assert lp_error < config["tolerance"]
        blind = (1-exact_mean)/(1+exact_mean)
        rows.append(dict(stage="witness", p=p_text, z=z_text, epsilon=epsilon_text, constant=c_text,
            reference_bad=str(reference[0]), actual_tv=str(abs(reference[0]-z)),
            exact_mean=str(exact_mean), normalized_mean=float(exact_mean),
            exact_risk=str(z), risk=float(z), blind_matched_risk=float(blind),
            blind_certificate_violated=blind<z, coalition_error=error, lp_error=lp_error,
            lp_normalized_mean=optimum, zero_reference_mass=reference[0]==0))
    return rows


def simplex(denominator):
    return [tuple(F(v,denominator) for v in (a,b,denominator-a-b))
            for a in range(denominator+1) for b in range(denominator+1-a)]


def three_state_case(config, reference):
    matrix = operators(F(1,2), reference)
    rows = []
    for completion in simplex(config["simplex_denominator"]):
        epsilon = sum(abs(a-b) for a,b in zip(reference,completion, strict=True))/2
        for mask in range(1,7):
            bad = [h for h in range(3) if mask & (1 << h)]
            z = sum(completion[h] for h in bad)
            upper = law_frontier(z,epsilon)
            optimum = full_lp(matrix,completion,bad,F(1,2))
            excess = optimum-float(upper)
            assert excess < config["tolerance"]
            rows.append(dict(stage="three_state", reference=",".join(map(str,reference)),
                completion=",".join(map(str,completion)), mask=mask, z=str(z), epsilon=str(epsilon),
                exact_upper_mean=str(upper), lp_normalized_mean=optimum, excess=excess,
                note="Prescribed bad states; fixed laws need not attain the class-wide frontier"))
    return rows


def baselines(config):
    rows = []
    for mean in config["baseline_means"]:
        for epsilon in config["tv_budgets"]:
            t, e = F(mean), F(epsilon)
            sharp, transfer = law_risk_upper(t,e), transfer_risk_upper(t,e)
            assert sharp <= transfer <= 1-t
            rows.append(dict(stage="baseline", exact_mean=mean, epsilon=epsilon,
                sharp_risk=float(sharp), exact_sharp_risk=str(sharp),
                transfer_risk=float(transfer), generic_risk=float(1-t),
                blind_matched_risk=float((1-t)/(1+t))))
    return rows


def run(config_path, output, resume=False, stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks < 1:
        raise ValueError("Positive stop count required")
    config = json.loads(config_path.read_text())
    sources = [config_path, Path(__file__), Path("scripts/run_completion_novelty_audit.py"),
        Path("src/financepaper/evaluation/completion_moments.py"), Path("src/financepaper/evaluation/decision_value.py"),
        Path("src/financepaper/evaluation/law_robust_disclosure.py"), Path("tests/test_law_robust_disclosure.py"),
        Path("docs/LAW_ROBUST_DISCLOSURE.md"), Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(p): digest(p) for p in sources}, device="cpu", numeric_threads=1,
                  python=platform.python_version(), versions={p:importlib.metadata.version(p) for p in ("numpy","scipy","pandas","tqdm")})
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: source/config/dependency drift or missing --resume")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("Need new empty run directory")
        manifest=dict(created_utc=utc(),frozen=frozen,base_git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
        write_json(manifest_path,manifest)
    tasks=[]
    for p in config["reference_probabilities"]:
        for z in config["failure_masses"]:
            for e in config["tv_budgets"]:
                tasks.append((f"witness_p{p}_z{z}_e{e}".replace("/","_"),witness_case,(config,p,z,e)))
    for index,reference in enumerate(simplex(config["simplex_denominator"])):
        tasks.append((f"three_state_{index}",three_state_case,(config,reference)))
    tasks.append(("baseline",baselines,(config,)))
    records, receipts=[],{}
    new,reused,start=0,0,time.perf_counter()
    log(output,"start",tasks=len(tasks),device="cpu",resume=resume)
    for name,operation,args in tqdm(tasks,desc="Law-robust disclosure",unit="task",mininterval=1):
        path,completion=output/f"{name}.json",output/f"{name}_complete.json"
        if completion.exists():
            receipt=json.loads(completion.read_text())
            if set(receipt["outputs"]) != {str(path)}:
                raise ValueError("Task output set mismatch")
            check_hashes(receipt["outputs"])
            reused+=1
        else:
            if path.exists():
                raise ValueError("Unreceipted output")
            tick=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows=operation(*args)
            write_json(path,rows)
            receipt=dict(seconds=time.perf_counter()-tick,outputs={str(path):digest(path)},warnings=[str(w.message) for w in caught])
            write_json(completion,receipt)
            new+=1
        receipts[str(completion)]=digest(completion)
        records.extend(json.loads(path.read_text()))
        log(output,"task_verified",task=name,new=new,reused=reused,warnings=receipt["warnings"],
            eta_seconds=(time.perf_counter()-start)/max(1,new)*(len(tasks)-new-reused))
        if stop_after_tasks and new>=stop_after_tasks:
            log(output,"intentional_pause",new=new,reused=reused)
            return
    check_hashes(frozen["sources"])
    outputs={}
    for stage,part in pd.DataFrame(records).groupby("stage"):
        path=output/f"{stage}.csv"
        temp=path.with_suffix(".csv.tmp")
        part.dropna(axis=1,how="all").to_csv(temp,index=False,float_format="%.17g")
        if path.exists() and digest(path)!=digest(temp):
            raise ValueError("Aggregate drift")
        temp.replace(path)
        outputs[str(path)]=digest(path)
    path=output/"execution_receipt.json"
    if not path.exists():
        write_json(path,dict(**manifest,completed_utc=utc(),tasks=len(tasks),rows=len(records),outputs=outputs,task_receipts=receipts,
            task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
            warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
            financial_data="NOT_RUN",training="NOT_RUN",human_study="NOT_RUN"))
    else:
        receipt=json.loads(path.read_text())
        check_hashes(receipt["outputs"])
        check_hashes(receipt["task_receipts"])
    log(output,"complete",new=new,reused=reused,seconds=time.perf_counter()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/law_robust_disclosure_audit.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-tasks",type=int)
    args=parser.parse_args()
    run(args.config,args.output,args.resume,args.stop_after_tasks)
