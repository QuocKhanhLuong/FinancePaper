"""Exact-rational, LP and bounded-tree challenge of the C3 sharp bound."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from fractions import Fraction
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

from financepaper.evaluation.bounded_completion import (
    completion_kernel, definition_matrix, directional_width, scalar_width, sharp_tree,
    fixed_law_factor, finite_definition_matrix,
)
from financepaper.evaluation.completion_moments import (
    FiniteMarginal, attribution_moments, evaluate, point_shapley, tree_leaves,
)
from financepaper.evaluation.decision_value import digest, write_json, check_hashes
from run_completion_novelty_audit import log, utc


def case(config, m, p_text):
    p = Fraction(p_text)
    kernel = completion_kernel(m, p)
    low, high = definition_matrix(m, p, 0), definition_matrix(m, p, 1)
    difference = [[b-a for a, b in zip(l, h, strict=True)] for l, h in zip(low, high, strict=True)]
    for i in range(m):
        assert difference[i] == [-v for v in kernel[i]]+kernel[i], "Exact rational coefficient mismatch"
        assert sum(abs(v) for v in kernel[i][:-1]) == scalar_width(m, p)
    delta = np.asarray(difference, float)
    directions = [(f"coordinate_{i}", [int(j == i) for j in range(m)]) for i in range(m)]
    directions.append(("sum", [1]*m))
    if m > 1:
        directions.append(("first_minus_last", [1]+[0]*(m-2)+[-1]))
    records = []
    for constant in config["constant_predictions"]:
        bounds = [(0., 1.)]*(2**(m+1))
        bounds[2**m-1] = (constant, constant)
        bounds[-1] = (constant, constant)
        for label, direction in directions:
            objective = np.asarray(direction) @ delta
            lp = linprog(-objective, bounds=bounds, method="highs")
            if not lp.success:
                raise AssertionError(lp.message)
            bound = directional_width(kernel, direction)
            error = abs(-lp.fun-float(bound))
            assert error < config["tolerance"], (m, p_text, label, error)
            records.append(dict(stage="directional", observed=m, p=p_text, constant=constant, direction=label,
                                exact_width=str(bound), width=float(bound), lp_maximum=float(-lp.fun),
                                lp_error=float(error), lp_iterations=int(lp.nit), exact_coefficient_check=True))
        for target in range(m):
            leaves = tree_leaves(sharp_tree(m, target, constant))
            assert len(leaves) == 2*m+2 and all(0 <= leaf.value <= 1 for leaf in leaves)
            reference = [FiniteMarginal((0., 1.), (float(1-p), float(p)))]*m+[FiniteMarginal((0., 1.), (.5, .5))]
            completion = [FiniteMarginal((1.,), (1.,))]*m+[reference[-1]]
            x = np.array([[1.]*m+[0.], [1.]*m+[1.]])
            predictions = evaluate(leaves, x)
            phi = point_shapley(leaves, reference, x, [target])[:, 0]
            truth_table = np.array([[float(bool(code & (1 << j))) for j in range(m+1)] for code in range(2**(m+1))])
            output_table = evaluate(leaves, truth_table)
            direct = np.array([np.asarray(matrix[target], float) @ output_table for matrix in (low, high)])
            moments = attribution_moments(leaves, reference, completion, [target])
            width = scalar_width(m, p)
            errs = dict(width_error=abs(float(np.ptp(phi))-float(width)),
                        variance_error=abs(float(moments["covariance"][0, 0])-float(width**2/4)),
                        coalition_error=float(np.max(np.abs(phi-direct))),
                        prediction_error=float(np.max(np.abs(predictions-constant))))
            assert max(errs.values()) < config["tolerance"], errs
            records.append(dict(stage="sharp_tree", observed=m, p=p_text, constant=constant, target=target,
                                exact_width=str(width), width=float(width), exact_variance=str(width**2/4),
                                variance=float(moments["covariance"][0, 0]), leaves=len(leaves),
                                model_min=float(output_table.min()), model_max=float(output_table.max()), **errs))
    for law_id, q_text in enumerate(config["hidden_laws"]):
        q = [Fraction(v) for v in q_text]
        beta, partition = fixed_law_factor(q)
        operators = [finite_definition_matrix(m, p, q, state)[0] for state in range(len(q))]
        # Scalar endpoint tables selected by the exact kernel coefficient signs.
        table = []
        for state in range(len(q)):
            for code in range(2**m):
                value = Fraction(1, 2) if code == 2**m-1 else Fraction(int((kernel[0][code] > 0) == (state in partition)))
                table.append(value)
        phi = [sum((a*b for a, b in zip(op, table, strict=True)), Fraction(0)) for op in operators]
        average = sum((w*v for w, v in zip(q, phi, strict=True)), Fraction(0))
        variance = sum((w*(v-average)**2 for w, v in zip(q, phi, strict=True)), Fraction(0))
        expected = beta*scalar_width(m, p)**2
        assert variance == expected, "Exact fixed-q variance mismatch"
        enumerated, maximum = 0, None
        if m <= 2 and len(q) <= 3:
            free = [i for i in range(len(table)) if i % 2**m != 2**m-1]
            maximum = Fraction(-1)
            for assignment in range(2**len(free)):
                candidate = [Fraction(1, 2)]*len(table)
                for bit, index in enumerate(free):
                    candidate[index] = Fraction(int(bool(assignment & (1 << bit))))
                scores = [sum((a*b for a, b in zip(op, candidate, strict=True)), Fraction(0)) for op in operators]
                center = sum((w*v for w, v in zip(q, scores, strict=True)), Fraction(0))
                value = sum((w*(v-center)**2 for w, v in zip(q, scores, strict=True)), Fraction(0))
                maximum = max(maximum, value); enumerated += 1
            assert maximum == expected, "Exhaustive bounded truth-table variance disagrees"
        records.append(dict(stage="fixed_q", observed=m, p=p_text, law=law_id,
                            probabilities=",".join(q_text), partition=str(partition), beta=str(beta),
                            exact_variance=str(variance), variance=float(variance), exact_definition_check=True,
                            enumerated_endpoint_models=enumerated, exhaustive_maximum=str(maximum) if maximum is not None else "NOT_RUN_DIMENSION_CAP"))
    return records


def run(config_path, output, resume=False, stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use unique ignored run directory")
    if stop_after_tasks is not None and stop_after_tasks < 1:
        raise ValueError("Positive stop count required")
    config = json.loads(config_path.read_text())
    sources = [config_path, Path(__file__), Path("scripts/run_completion_novelty_audit.py"),
               Path("src/financepaper/evaluation/bounded_completion.py"), Path("src/financepaper/evaluation/completion_moments.py"),
               Path("src/financepaper/evaluation/decision_value.py"), Path("tests/test_bounded_completion.py"),
               Path("docs/BOUNDED_COMPLETION_THEOREM.md"), Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(p): digest(p) for p in sources}, device="cpu", numeric_threads=1,
                  python=platform.python_version(), dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "tqdm")})
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
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
    records, receipts = [], {}; new, reused = 0, 0; started = time.perf_counter()
    tasks = [(m, p) for m in config["observed_dimensions"] for p in config["reference_probabilities"]]
    log(output, "start", resume=resume, device="cpu", tasks=len(tasks))
    for m, p in tqdm(tasks, desc="Sharp bounded SHAP", unit="case", mininterval=1):
        name = f"m{m}_p{p.replace('/', '_')}"
        path, receipt_path = output / f"{name}.json", output / f"{name}_complete.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text()); check_hashes(receipt["outputs"])
            reused += 1; log(output, "resume_verified", task=name)
        else:
            if path.exists():
                raise ValueError("Unreceipted output: inspect and use new run")
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always"); rows = case(config, m, p)
            write_json(path, rows)
            receipt = dict(seconds=time.perf_counter()-tick, outputs={str(path): digest(path)}, warnings=[str(w.message) for w in caught])
            write_json(receipt_path, receipt); new += 1
            log(output, "task_complete", task=name, seconds=receipt["seconds"], warnings=receipt["warnings"],
                eta_seconds=(time.perf_counter()-started)/new*(len(tasks)-new-reused))
        records.extend(json.loads(path.read_text())); receipts[str(receipt_path)] = digest(receipt_path)
        if stop_after_tasks is not None and new >= stop_after_tasks:
            log(output, "intentional_pause", new=new, reused=reused); return
    check_hashes(frozen["sources"])
    outputs = {}
    for stage, part in pd.DataFrame(records).groupby("stage"):
        path = output / f"{stage}.csv"; temporary = path.with_suffix(".csv.tmp")
        part.dropna(axis=1, how="all").to_csv(temporary, index=False, float_format="%.17g")
        if path.exists() and digest(path) != digest(temporary):
            raise ValueError("Aggregate changed during resume")
        temporary.replace(path); outputs[str(path)] = digest(path)
    receipt_path = output / "execution_receipt.json"
    if not receipt_path.exists():
        write_json(receipt_path, dict(**manifest, completed_utc=utc(), tasks=len(tasks), rows=len(records),
                   task_receipts=receipts, outputs=outputs,
                   task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
                   warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
                   model_fitting="NOT_RUN", human_study="NOT_RUN", financial_data="NOT_RUN"))
    else:
        receipt = json.loads(receipt_path.read_text()); check_hashes(receipt["outputs"]); check_hashes(receipt["task_receipts"])
    log(output, "complete", new=new, reused=reused, seconds=time.perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bounded_completion_audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.resume, args.stop_after_tasks)
