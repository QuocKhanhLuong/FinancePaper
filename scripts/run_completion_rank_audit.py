"""Audit the sharp completion rank-tail frontier, with resumable task receipts."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from fractions import Fraction as F
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

from financepaper.evaluation.bounded_completion import finite_definition_matrix
from financepaper.evaluation.completion_rank import (
    rank_cells, rank_frontier, rank_pair_operator, rank_risk_upper,
    rank_witness, universal_risk_upper,
)
from financepaper.evaluation.decision_value import check_hashes, digest, write_json
from run_completion_novelty_audit import log, utc


def dot(a, b):
    return sum((x*y for x, y in zip(a, b, strict=True)), F(0))


def definition_operators(m, p, q):
    operators = []
    for h in range(len(q)):
        matrix = finite_definition_matrix(m, p, q, h)
        pair = tuple(a-b for a, b in zip(matrix[0], matrix[1], strict=True))
        assert pair == rank_pair_operator(m, p, q, h), "Definition/mixture coefficient mismatch"
        operators.append(pair)
    return operators


def check_lp(m, p, q, bad, operators, c, tolerance):
    objective = [sum(q[h]*operators[h][a] for h in range(len(q))) for a in range(2**m*len(q))]
    bounds = [(0., 1.)]*len(objective)
    for h in range(len(q)):
        bounds[(h+1)*2**m-1] = (float(c), float(c))
    result = linprog(-np.asarray(objective, float),
                     A_ub=np.asarray([operators[h] for h in bad], float),
                     b_ub=np.zeros(len(bad)), bounds=bounds, method="highs")
    if not result.success:
        raise AssertionError(result.message)
    z = sum(q[h] for h in bad)
    expected = (1-p)*rank_frontier(m, p, z)["normalized_mean"]
    error = abs(-result.fun-float(expected))
    assert error < tolerance, (m, p, q, bad, c, error)
    return float(-result.fun), error


def binary_case(config, m, p_text, z_text):
    p, z = F(p_text), F(z_text)
    q = (z, 1-z)
    operators = definition_operators(m, p, q)
    frontier = rank_frontier(m, p, z)["normalized_mean"]
    records = []
    for c_text in config["constant_predictions"]:
        table = rank_witness(m, p, z, F(c_text))
        gaps = [dot(op, table) for op in operators]
        mean = dot(q, gaps)
        assert gaps[0] == 0 < gaps[1]
        assert mean == (1-p)*frontier
        assert all(table[(h+1)*2**m-1] == F(c_text) for h in range(2))
        optimum, error = check_lp(m, p, q, [0], operators, F(c_text), config["tolerance"])
        strict = rank_witness(m, p, z, F(c_text), F(config["strict_perturbation"]))
        strict_gaps = [dot(op, strict) for op in operators]
        assert strict_gaps[0] < 0 < strict_gaps[1]
        strict_mean = dot(q, strict_gaps)/(1-p)
        assert strict_mean < frontier
        records.append(dict(stage="binary", m=m, p=p_text, z=z_text, constant=c_text,
            exact_normalized_mean=str(frontier), normalized_mean=float(frontier),
            exact_bad_gap=str(gaps[0]), exact_good_gap=str(gaps[1]),
            lp_mean=optimum, lp_error=error, exact_coefficient_check=True,
            exact_witness_check=True, table_size=len(table), failure_probability=float(z),
            strict_bad_gap=str(strict_gaps[0]), strict_mean=float(strict_mean),
            generic_risk_upper=float(1-frontier), universal_risk_upper=universal_risk_upper(frontier)))
    return records


def multistate_case(config, m, p_text):
    p = F(p_text)
    q = tuple(F(v) for v in config["multi_state_law"])
    operators = definition_operators(m, p, q)
    records = []
    for mask in range(1, 2**len(q)-1):
        bad = [h for h in range(len(q)) if mask & (1 << h)]
        z = sum(q[h] for h in bad)
        frontier = rank_frontier(m, p, z)["normalized_mean"]
        for c_text in config["constant_predictions"]:
            binary = rank_witness(m, p, z, F(c_text))
            table = tuple(v for h in range(len(q)) for v in binary[(0 if h in bad else 1)*2**m:(1 if h in bad else 2)*2**m])
            gaps = [dot(op, table) for op in operators]
            assert all(gaps[h] == 0 if h in bad else gaps[h] > 0 for h in range(len(q)))
            assert dot(q, gaps) == (1-p)*frontier
            optimum, error = check_lp(m, p, q, bad, operators, F(c_text), config["tolerance"])
            records.append(dict(stage="multistate", m=m, p=p_text, z=str(z), constant=c_text,
                event_mask=mask, law=",".join(map(str, q)), normalized_mean=float(frontier),
                exact_normalized_mean=str(frontier), lp_mean=optimum, lp_error=error,
                exact_coefficient_check=True, exact_witness_check=True))
    return records


def curve_case(config, m, p_text):
    p = F(p_text)
    cells = rank_cells(m, p)
    records = []
    for alpha in config["curve_risk_budgets"]:
        z = F(alpha)
        frontier = rank_frontier(m, p, z)
        maximum = float(frontier["normalized_mean"])
        envelope = 1-2*float(z)/(1+float(z)**.5)
        assert maximum <= envelope+config["tolerance"]
        assert maximum < 1-float(z)
        records.append(dict(stage="threshold", m=m, p=p_text, risk_budget=alpha,
            exact_required_mean=str(frontier["normalized_mean"]), required_mean=maximum,
            generic_required_mean=1-float(z), universal_required_mean=envelope,
            envelope_gap=envelope-maximum, cells=len(cells), full_table="NOT_RUN_DIMENSION_CAP" if m>5 else "SEE_BINARY_AUDIT"))
    for mean_text in config["certificate_means"]:
        mean = F(mean_text)
        upper = rank_risk_upper(m, p, mean)
        assert rank_frontier(m, p, upper)["normalized_mean"] <= mean
        assert rank_frontier(m, p, upper-F(1, 2**64))["normalized_mean"] >= mean
        records.append(dict(stage="certificate", m=m, p=p_text, normalized_mean=float(mean),
            exact_risk_upper=str(upper), risk_upper=float(upper),
            universal_risk_upper=universal_risk_upper(mean), generic_risk_upper=float(1-mean),
            cells=len(cells)))
    return records


def law_mismatch_case():
    # The same bounded model/reference; changing only completion Q violates
    # the matched-law guarantee. This is an intended boundary falsifier.
    m, p, z = 2, F(1, 2), F(1, 100)
    reference = (z, 1-z)
    completion = (F(1, 2), F(1, 2))
    table = rank_witness(m, p, z, perturbation=1)
    operators = definition_operators(m, p, reference)
    gaps = [dot(op, table) for op in operators]
    mean = dot(completion, gaps)/(1-p)
    fail = sum(completion[h] for h, gap in enumerate(gaps) if gap <= 0)
    invalid_bound = rank_risk_upper(m, p, mean)
    assert mean == F(49, 100) and fail == F(1, 2) and invalid_bound < fail
    return [dict(stage="law_mismatch", normalized_mean=str(mean), actual_failure=str(fail),
                 invalid_bound=float(invalid_bound), gap_bad=str(gaps[0]), gap_good=str(gaps[1]),
                 verdict="EXPECTED_ASSUMPTION_FAILURE", model_fitting="NOT_RUN")]


def run(config_path, output, resume=False, stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks < 1:
        raise ValueError("Positive stop count required")
    config = json.loads(config_path.read_text())
    sources = [config_path, Path(__file__), Path("scripts/run_completion_novelty_audit.py"),
        Path("src/financepaper/evaluation/bounded_completion.py"),
        Path("src/financepaper/evaluation/completion_rank.py"),
        Path("src/financepaper/evaluation/decision_value.py"),
        Path("tests/test_completion_rank.py"), Path("docs/COMPLETION_RANK_FRONTIER.md"),
        Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(path): digest(path) for path in sources},
        device="cpu", numeric_threads=1, python=platform.python_version(),
        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "tqdm")})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: source/config/dependency drift or missing --resume")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("New run requires empty new directory")
        manifest = dict(created_utc=utc(), frozen=frozen, platform=platform.platform(),
            base_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip())
        write_json(manifest_path, manifest)
    tasks = []
    for m in config["observed_dimensions"]:
        for p in config["reference_probabilities"]:
            for z in config["failure_masses"]:
                tasks.append((f"binary_m{m}_p{p}_z{z}".replace("/", "_"), binary_case, (config, m, p, z)))
            tasks.append((f"multistate_m{m}_p{p}".replace("/", "_"), multistate_case, (config, m, p)))
    for m in config["curve_dimensions"]:
        for p in config["reference_probabilities"]:
            tasks.append((f"curve_m{m}_p{p}".replace("/", "_"), curve_case, (config, m, p)))
    tasks.append(("law_mismatch", law_mismatch_case, ()))
    records, receipts = [], {}
    new, reused, started = 0, 0, time.perf_counter()
    log(output, "start", resume=resume, device="cpu", tasks=len(tasks))
    for name, operation, args in tqdm(tasks, desc="Sharp completion rank frontier", unit="task", mininterval=1):
        path, receipt_path = output/f"{name}.json", output/f"{name}_complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if set(receipt["outputs"]) != {str(path)}:
                raise ValueError("Task receipt output set mismatch")
            check_hashes(receipt["outputs"])
            reused += 1
            log(output, "resume_verified", task=name)
        else:
            if path.exists():
                raise ValueError("Unreceipted output: inspect and use new run")
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows = operation(*args)
            write_json(path, rows)
            receipt = dict(seconds=time.perf_counter()-tick, outputs={str(path): digest(path)},
                           warnings=[str(w.message) for w in caught])
            write_json(receipt_path, receipt)
            new += 1
            log(output, "task_complete", task=name, seconds=receipt["seconds"], warnings=receipt["warnings"],
                eta_seconds=(time.perf_counter()-started)/new*(len(tasks)-new-reused))
        records.extend(json.loads(path.read_text()))
        receipts[str(receipt_path)] = digest(receipt_path)
        if stop_after_tasks is not None and new >= stop_after_tasks:
            log(output, "intentional_pause", new=new, reused=reused)
            return
    check_hashes(frozen["sources"])
    outputs = {}
    for stage, part in pd.DataFrame(records).groupby("stage"):
        path = output/f"{stage}.csv"
        temporary = path.with_suffix(".csv.tmp")
        part.dropna(axis=1, how="all").to_csv(temporary, index=False, float_format="%.17g")
        if path.exists() and digest(path) != digest(temporary):
            raise ValueError("Aggregate changed during resume")
        temporary.replace(path)
        outputs[str(path)] = digest(path)
    receipt_path = output/"execution_receipt.json"
    if not receipt_path.exists():
        write_json(receipt_path, dict(**manifest, completed_utc=utc(), tasks=len(tasks), rows=len(records),
            task_receipts=receipts, outputs=outputs,
            task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
            warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
            model_fitting="NOT_RUN", human_study="NOT_RUN", financial_data="NOT_RUN"))
    else:
        receipt = json.loads(receipt_path.read_text())
        check_hashes(receipt["outputs"])
        check_hashes(receipt["task_receipts"])
    log(output, "complete", new=new, reused=reused, seconds=time.perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/completion_rank_audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.resume, args.stop_after_tasks)
