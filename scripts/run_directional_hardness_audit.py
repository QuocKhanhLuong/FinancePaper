"""Exact finite audit of the C4 counting reduction, not a complexity benchmark."""
import argparse
from fractions import Fraction
import importlib.metadata
from math import comb
import json
from pathlib import Path
import platform
import subprocess
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.bounded_completion import completion_kernel, directional_width
from financepaper.evaluation.decision_value import digest, write_json, check_hashes
from run_completion_novelty_audit import log, utc


def alpha(m, r):
    if not 1 <= r < m:
        raise ValueError("Internal subset sizes only")
    return sum((Fraction(comb(r-1, j)*comb(m-r-1, k)*(-1)**k, (j+k+2)*2**(m-1))
                for j in range(r) for k in range(m-r)), Fraction(0))


@lru_cache(maxsize=None)
def kernel(m):
    return completion_kernel(m, Fraction(1, 2))


def audit(weights, target):
    n, m, total = len(weights), len(weights)+2, sum(weights)
    mass = kernel(m)
    for r in range(1, m):
        code = 2**r-1
        assert alpha(m, r) == mass[0][code]-mass[-1][code] > 0
    brute = [0]*(n+1)
    for subset in range(2**n):
        if sum(w for j, w in enumerate(weights) if subset & (1 << j)) == target:
            brute[subset.bit_count()] += 1
    lift = total+target+2
    lifted = [lift+w for w in weights]; lifted_total = sum(lifted)
    rows = []
    for cardinality in range(n+1):
        center = cardinality*lift+target
        queries = []
        for t in (center-1, center, center+1):
            direction = lifted+[-t, -(lifted_total-t)]
            assert sum(direction) == 0
            queries.append(directional_width(mass, direction))
        delta = queries[2]-2*queries[1]+queries[0]
        recovered = delta/(2*(alpha(m, cardinality+1)+alpha(m, n-cardinality+1)))
        assert recovered.denominator == 1 and recovered == brute[cardinality], (weights, target, cardinality, recovered, brute)
        rows.append(dict(cardinality=cardinality, brute_count=brute[cardinality], recovered_count=int(recovered),
                         exact_second_difference=str(delta), oracle_queries=3, exact_match=True))
    return rows


def run(config_path, output, resume=False, stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Unique ignored run directory required")
    if stop_after_tasks is not None and stop_after_tasks < 1:
        raise ValueError("Positive stop count required")
    config = json.loads(config_path.read_text())
    paths = [config_path, Path(__file__), Path("scripts/run_completion_novelty_audit.py"),
             Path("src/financepaper/evaluation/bounded_completion.py"), Path("src/financepaper/evaluation/decision_value.py"),
             Path("docs/DIRECTIONAL_COMPLETION_HARDNESS.md"), Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(p): digest(p) for p in paths}, device="cpu",
                  python=platform.python_version(), dependencies={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "tqdm")})
    output.mkdir(parents=True, exist_ok=True); manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: code/config/dependencies drift or --resume missing")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("New run requires empty new directory")
        manifest = dict(created_utc=utc(), frozen=frozen, platform=platform.platform(),
                        base_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip())
        write_json(manifest_path, manifest)
    tasks = [(n, repeat, kind) for n in config["dimensions"] for repeat in config["replicates"] for kind in config["targets"]]
    records, receipts = [], {}; new, reused = 0, 0; started = time.perf_counter()
    log(output, "start", tasks=len(tasks), resume=resume, device="cpu")
    for n, repeat, kind in tqdm(tasks, desc="Exact counting reduction", unit="instance", mininterval=1):
        name = f"n{n}_r{repeat}_{kind}"; path=output/f"{name}.json"; receipt_path=output/f"{name}_complete.json"
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text()); check_hashes(receipt["outputs"])
            reused+=1; log(output,"resume_verified",task=name)
        else:
            if path.exists():
                raise ValueError("Unreceipted output: inspect/new run")
            tick=time.perf_counter()
            rng=np.random.default_rng(np.random.SeedSequence([config["seed"],n,repeat]))
            weights=[int(v) for v in rng.integers(config["weight_min"],config["weight_max_inclusive"]+1,size=n)]
            target={"zero":0,"half_total":sum(weights)//2,"above_total":sum(weights)+1}[kind]
            rows=[dict(n=n,repeat=repeat,kind=kind,weights=str(weights),target=target,**row) for row in audit(weights,target)]
            write_json(path,rows)
            receipt=dict(seconds=time.perf_counter()-tick,outputs={str(path):digest(path)})
            write_json(receipt_path,receipt); new+=1
            log(output,"task_complete",task=name,seconds=receipt["seconds"],eta_seconds=(time.perf_counter()-started)/new*(len(tasks)-new-reused))
        records.extend(json.loads(path.read_text())); receipts[str(receipt_path)]=digest(receipt_path)
        if stop_after_tasks is not None and new>=stop_after_tasks:
            log(output,"intentional_pause",new=new,reused=reused); return
    check_hashes(frozen["sources"])
    path=output/"counting.csv"; temp=path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(temp,index=False)
    if path.exists() and digest(path)!=digest(temp):
        raise ValueError("Aggregate changed during resume")
    temp.replace(path)
    receipt_path=output/"execution_receipt.json"
    if not receipt_path.exists():
        write_json(receipt_path,dict(**manifest,completed_utc=utc(),instances=len(tasks),cardinality_checks=len(records),
                   oracle_queries=sum(row["oracle_queries"] for row in records),outputs={str(path):digest(path)},task_receipts=receipts,
                   task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
                   model_fitting="NOT_RUN",financial_data="NOT_RUN",human_study="NOT_RUN"))
    else:
        receipt=json.loads(receipt_path.read_text()); check_hashes(receipt["outputs"]); check_hashes(receipt["task_receipts"])
    log(output,"complete",new=new,reused=reused,seconds=time.perf_counter()-started)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/directional_hardness_audit.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-tasks",type=int)
    args=parser.parse_args();run(args.config,args.output,args.resume,args.stop_after_tasks)
