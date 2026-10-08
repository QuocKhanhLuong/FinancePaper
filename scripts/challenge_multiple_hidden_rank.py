"""Exploratory boundary challenge only: C5 is NOT extended by finite LP checks.

Predeclared cases: k=2..8 uniform independent hidden bits; singleton,
coordinate half, Hamming ball <=2, ball <=floor(k/2), codimension-two subcube.
The k=2 ball<=2 is the whole cube and is retained as a zero-mean control.
"""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
import argparse
from fractions import Fraction as F
import importlib.metadata
import json
from math import comb, factorial
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import linprog
from tqdm.auto import tqdm

from financepaper.evaluation.completion_moments import FiniteMarginal, coalition_oracle
from financepaper.evaluation.decision_value import check_hashes, digest, write_json
from run_completion_novelty_audit import log, utc


def challenge(k, name):
    count = 2**k
    # Independent positive coalition expansion, not the agent's alternating
    # polynomial expansion of the integrated noise kernel.
    radial = [sum((F(comb(k-distance, size)*factorial(size)*factorial(k-size),
                     factorial(k+1)*2**(k-size)) for size in range(k-distance+1)), F(0))
              for distance in range(k+1)]
    matrix = np.asarray([[float(radial[(h^x).bit_count()]) for x in range(count)] for h in range(count)])
    assert np.max(np.abs(matrix.sum(axis=1)-1)) < 1e-12
    assert np.max(np.abs(matrix-matrix.T)) == 0
    sets = dict(one=[0], coord_half=[h for h in range(count) if not h & 1],
                low2=[h for h in range(count) if h.bit_count() <= 2],
                low_half=[h for h in range(count) if h.bit_count() <= k//2],
                subcube_codim2=[h for h in range(count) if not h & 3])
    bad = sets[name]
    result = linprog(-np.ones(count)/count, A_ub=matrix[bad], b_ub=np.zeros(len(bad)),
                     bounds=[(-1, 1)]*count, method="highs")
    assert result.success, result.message
    gaps = matrix@result.x
    residual = float(max(0, np.max(gaps[bad])))
    assert residual < 1e-10
    definition_error = None
    if k <= 3:
        def predict(batch):
            hidden = np.asarray(batch[:, 2:], int) @ (2**np.arange(k))
            d = result.x[hidden]
            return np.where(batch[:, 0] > batch[:, 1], (1+d)/2,
                            np.where(batch[:, 0] < batch[:, 1], (1-d)/2, .5))
        marginal = FiniteMarginal((0., 1.), (.5, .5))
        direct = []
        for h in range(count):
            query = [1., 1.]+[float(bool(h & (1 << j))) for j in range(k)]
            phi = coalition_oracle(predict, [marginal]*(k+2), query, [0, 1])
            direct.append(2*(phi[0]-phi[1]))
        definition_error = float(np.max(np.abs(np.asarray(direct)-gaps)))
        assert definition_error < 1e-10
    z = F(len(bad), count)
    conjectured = (1-z)/(1+z)
    maximum = float(-result.fun)
    return dict(k=k, name=name, bad_states=bad, z=str(z), lp_max_normalized_mean=maximum,
                conjectured_upper=str(conjectured), gap=maximum-float(conjectured),
                constraint_residual=residual, coalition_error=definition_error,
                coalition_status="RUN" if k<=3 else "NOT_RUN_DIMENSION_CAP",
                contrast_witness=result.x.tolist(), radial_exact=list(map(str, radial)),
                extension_theorem="UNPROVED")


def run(output, resume, stop_after_tasks):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use ignored run directory")
    sources = [Path(__file__), Path("scripts/run_completion_novelty_audit.py"),
               Path("src/financepaper/evaluation/completion_moments.py"),
               Path("src/financepaper/evaluation/decision_value.py"), Path("uv.lock")]
    frozen = dict(sources={str(p): digest(p) for p in sources}, device="cpu", numeric_threads=1,
                  versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "tqdm")})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume/source/config drift rejected")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("Need new empty run directory")
        manifest = dict(created_utc=utc(), frozen=frozen, status="EXPLORATORY_NOT_A_THEOREM")
        write_json(manifest_path, manifest)
    tasks = [(k, name) for k in range(2, 9) for name in ("one", "coord_half", "low2", "low_half", "subcube_codim2")]
    records, outputs = [], {}
    new, reused, start = 0, 0, time.perf_counter()
    log(output, "start", device="cpu", tasks=len(tasks), resume=resume)
    for k, name in tqdm(tasks, desc="Multiple hidden boundary", unit="LP", mininterval=1):
        path = output/f"k{k}_{name}.json"
        receipt_path = output/f"k{k}_{name}_complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            check_hashes(receipt["outputs"])
            reused += 1
        else:
            if path.exists():
                raise ValueError("Unreceipted task output")
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = challenge(k, name)
            write_json(path, value)
            receipt = dict(seconds=time.perf_counter()-tick, outputs={str(path): digest(path)},
                           warnings=[str(w.message) for w in caught])
            write_json(receipt_path, receipt)
            new += 1
        outputs[str(receipt_path)] = digest(receipt_path)
        records.append(json.loads(path.read_text()))
        log(output, "task_verified", k=k, case=name, new=new, reused=reused,
            eta_seconds=(time.perf_counter()-start)/max(1,new)*(len(tasks)-new-reused), warnings=receipt["warnings"])
        if stop_after_tasks and new >= stop_after_tasks:
            log(output, "intentional_pause", new=new, reused=reused)
            return
    check_hashes(frozen["sources"])
    result_path = output/"results.json"
    if result_path.exists():
        assert json.loads(result_path.read_text()) == records
    else:
        write_json(result_path, records)
    execution = output/"execution_receipt.json"
    if not execution.exists():
        write_json(execution, dict(**manifest, completed_utc=utc(), tasks=len(records), task_receipts=outputs,
                   outputs={str(result_path): digest(result_path)},
                   task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in outputs),
                   warnings=[w for p in outputs for w in json.loads(Path(p).read_text())["warnings"]]))
    else:
        prior = json.loads(execution.read_text())
        check_hashes(prior["outputs"])
        check_hashes(prior["task_receipts"])
    log(output, "complete", new=new, reused=reused, seconds=time.perf_counter()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    run(args.output, args.resume, args.stop_after_tasks)
