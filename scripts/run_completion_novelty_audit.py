"""Resumable synthetic audit of completion SHAP moments and finite-law witnesses."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.completion_moments import (
    FiniteMarginal, Leaf, attribution_moments, cell_enumeration_moments,
    coefficient_moments, coalition_oracle, enumeration_moments, evaluate,
    finite_states, joint_law_witness, point_shapley, sample_product, tree_leaves,
)
from financepaper.evaluation.decision_value import digest, write_json, check_hashes


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(output, event, **values):
    row = dict(utc=utc(), event=event, **values)
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")
    tqdm.write(json.dumps(row, allow_nan=False))


def tree(rng, dimensions, depth):
    if depth == 0:
        return float(rng.normal())
    return dict(feature=int(rng.integers(dimensions)), threshold=float(rng.choice([-.5, .5])),
                left=tree(rng, dimensions, depth-1), right=tree(rng, dimensions, depth-1))


def discrepancy(left, right):
    return max(float(np.max(np.abs(left[k]-right[k]), initial=0)) for k in ("mean", "second", "covariance"))


def correctness(config, index):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 1, index]))
    leaves = tree_leaves(tree(rng, 3, 3))+tree_leaves(tree(rng, 3, 2))
    p = [FiniteMarginal((-1., 0., 1.), tuple(rng.dirichlet(np.ones(3)))) for _ in range(3)]
    q = [FiniteMarginal((-1., 0., 1.), tuple(rng.dirichlet(np.ones(3)))) for _ in range(3)]
    truth = enumeration_moments(leaves, p, q, [0, 1, 2], definition_oracle=True)
    errors = dict(quadrature_error=discrepancy(attribution_moments(leaves, p, q, [0, 1, 2]), truth),
                  coefficient_error=discrepancy(coefficient_moments(leaves, p, q, [0, 1, 2]), truth),
                  pair_cell_error=discrepancy(cell_enumeration_moments(leaves, p, q, [0, 1, 2], pairwise=True), truth))
    if max(errors.values()) > config["absolute_correctness_tolerance"]:
        raise AssertionError(errors)
    return [dict(stage="correctness", case=index, leaves=len(leaves), **errors)]


def benchmark(config, index, repeat):
    spec = config["benchmarks"][index]
    depth, count = spec["depth"], spec["leaves"]
    dimensions = depth if spec["layout"] == "shared" else 2+(depth-2)*count
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 2, index]))
    leaves = []
    for j in range(count):
        ids = list(range(depth)) if spec["layout"] == "shared" else [0, 1]+list(range(2+j*(depth-2), 2+(j+1)*(depth-2)))
        bounds = tuple((i, .5, np.inf) if rng.random() > .3 else (i, -np.inf, .5) for i in ids)
        leaves.append(Leaf(float(rng.normal()), bounds))
    p = [FiniteMarginal((0., 1.), (.5, .5))]*dimensions
    q = [FiniteMarginal((1.,), (1.,))]*2+[FiniteMarginal((0., 1.), (.35, .65))]*(dimensions-2)
    exact = attribution_moments(leaves, p, q, [0, 1])
    records = []
    methods = [("quadrature", lambda: attribution_moments(leaves, p, q, [0, 1])),
               ("coefficients", lambda: coefficient_moments(leaves, p, q, [0, 1])),
               ("pair_cells", lambda: cell_enumeration_moments(leaves, p, q, [0, 1], pairwise=True, max_states=config["max_pair_states"]))]
    for draws in config["mc_draws"]:
        def monte_carlo(n=draws):
            generator = np.random.default_rng(np.random.SeedSequence([config["seed"], 3, index, repeat, n]))
            x = sample_product(q, n, generator)
            phi = point_shapley(leaves, p, x, [0, 1])
            mean, second = phi.mean(axis=0), phi.T @ phi/n
            return dict(mean=mean, second=second, covariance=second-np.outer(mean, mean))
        methods.append((f"mc_{draws}", monte_carlo))
    # Fixed rotation avoids always giving one method the first measured position.
    methods = methods[repeat % len(methods):]+methods[:repeat % len(methods)]
    for name, method in methods:
        start = time.perf_counter()
        try:
            got = method()
            extra = dict(status="RUN", discrepancy_from_quadrature=discrepancy(got, exact),
                         covariance_discrepancy=float(np.max(np.abs(got["covariance"]-exact["covariance"]))),
                         max_pair_states=got.get("max_pair_states"), minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(got["covariance"]).min()))
        except OverflowError as exc:
            extra = dict(status="NOT_RUN_CAP", reason=str(exc))
        records.append(dict(stage="benchmark", benchmark=index, repeat=repeat, method=name, dimensions=dimensions,
                            possible_completions=str(2**(dimensions-2)), targets=2, **spec,
                            seconds=time.perf_counter()-start, **extra))
    return records


def law_case(config, m, states, seed):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 4, m, states, seed]))
    desired = rng.normal(size=(states, m))
    weights = rng.dirichlet(np.ones(states))
    leaves, p, q = joint_law_witness(desired, weights)
    x, _ = finite_states(q)
    targets = list(range(m+1))
    direct = np.asarray([coalition_oracle(lambda y: evaluate(leaves, y), p, row, targets) for row in x])
    mu = weights @ desired
    expected_covariance = (desired.T*weights) @ desired-np.outer(mu, mu)
    got = attribution_moments(leaves, p, q, targets)
    checks = dict(attribution_error=float(np.max(np.abs(direct[:, :m]-desired))),
                  covariance_error=float(np.max(np.abs(got["covariance"][:m, :m]-expected_covariance))),
                  hidden_attribution_error=float(np.max(np.abs(direct[:, m]+(desired-mu).sum(axis=1)))),
                  prediction_error=float(np.max(np.abs(evaluate(leaves, x)-2*mu.sum()))))
    if max(checks.values()) > config["absolute_correctness_tolerance"]:
        raise AssertionError(checks)
    return [dict(stage="joint_law", observed=m, states=states, seed=seed, leaves=len(leaves),
                 prediction_range=float(np.ptp(evaluate(leaves, x))),
                 target_covariance_trace=float(np.trace(expected_covariance)), **checks)]


def tail_case(config, n):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 5, n]))
    weights = rng.integers(1, 20, size=n)
    capacity = int(weights.sum()//2)
    c = capacity/2+.25
    leaves = [Leaf(-c, ((0, .5, np.inf),))]+[Leaf(float(w), ((0, .5, np.inf), (j+1, .5, np.inf))) for j, w in enumerate(weights)]
    p = [FiniteMarginal((0.,), (1.,))]*(n+1)
    q = [FiniteMarginal((1.,), (1.,))]+[FiniteMarginal((0., 1.), (.5, .5))]*n
    x, _ = finite_states(q)
    sums = x[:, 1:] @ weights
    phi = point_shapley(leaves, p, x, [0])[:, 0]
    counts = [0]*(int(weights.sum())+1); counts[0] = 1
    for w in weights:
        for total in range(len(counts)-1, int(w)-1, -1):
            counts[total] += counts[total-int(w)]
    expected = 2**n-sum(counts[:capacity+1])
    actual = int((phi > 0).sum())
    assert actual == expected == int((sums >= capacity+1).sum())
    error = float(np.max(np.abs(phi-(sums/2-c))))
    assert error < config["absolute_correctness_tolerance"]
    return [dict(stage="tail", hidden=n, capacity=capacity, states=len(x), sign_count=actual,
                 dp_count=expected, formula_error=error, probability=actual/(2**n))]


def run(config_path, output, *, resume=False, stop_after_tasks=None):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Unique runs must stay inside runs/decision_value_pilot")
    if stop_after_tasks is not None and stop_after_tasks < 1:
        raise ValueError("Stop count must be positive")
    config = json.loads(config_path.read_text())
    paths = [config_path, Path(__file__), Path("src/financepaper/evaluation/completion_moments.py"),
             Path("src/financepaper/evaluation/decision_value.py"), Path("tests/test_completion_moments.py"),
             Path("docs/COMPLETION_NOVELTY_AUDIT_PROTOCOL.md"), Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(p): digest(p) for p in paths}, device="cpu", numeric_threads=1,
                  dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "tqdm")}, python=platform.python_version())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: code/config/dependencies differ or --resume absent")
    else:
        if resume or any(output.iterdir()):
            raise ValueError("New run requires a new empty directory")
        manifest = dict(created_utc=utc(), frozen=frozen, platform=platform.platform(),
                        base_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip())
        write_json(manifest_path, manifest)
    tasks = [(f"correct_{i:03d}", lambda i=i: correctness(config, i)) for i in range(config["correctness_cases"])]
    tasks += [(f"bench_{i}_{r}", lambda i=i, r=r: benchmark(config, i, r))
              for i in range(len(config["benchmarks"])) for r in range(config["timing_repeats"])]
    tasks += [(f"law_{m}_{s}_{seed}", lambda m=m, s=s, seed=seed: law_case(config, m, s, seed))
              for m in config["joint_law_observed_dimensions"] for s in config["joint_law_states"] for seed in config["joint_law_seeds"]]
    tasks += [(f"tail_{n}", lambda n=n: tail_case(config, n)) for n in config["knapsack_dimensions"]]
    start = time.perf_counter(); new, reused = 0, 0
    records, receipts = [], {}
    log(output, "start", resume=resume, tasks=len(tasks), device="cpu")
    with tqdm(total=len(tasks), desc="Completion audit", unit="task", mininterval=1) as bar:
        for name, operation in tasks:
            path, receipt_path = output / f"{name}.json", output / f"{name}_complete.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                check_hashes(receipt["outputs"])
                reused += 1
                log(output, "resume_verified", task=name)
            else:
                if path.exists():
                    raise FileExistsError(f"Unreceipted output requires inspection/new run: {path}")
                tick = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    rows = operation()
                write_json(path, rows)
                receipt = dict(task=name, seconds=time.perf_counter()-tick,
                               outputs={str(path): digest(path)}, warnings=[str(w.message) for w in caught])
                write_json(receipt_path, receipt)
                new += 1
                log(output, "task_complete", task=name, seconds=receipt["seconds"], warnings=receipt["warnings"],
                    completed=bar.n+1, total=len(tasks), eta_seconds=(time.perf_counter()-start)/new*(len(tasks)-bar.n-1))
            receipts[str(receipt_path)] = digest(receipt_path)
            records.extend(json.loads(path.read_text()))
            bar.update(1)
            if stop_after_tasks is not None and new >= stop_after_tasks:
                log(output, "intentional_pause", new=new, reused=reused)
                return
    check_hashes(frozen["sources"])
    frame = pd.DataFrame(records)
    aggregates = {}
    for stage, part in frame.groupby("stage", sort=False):
        path = output / f"{stage}.csv"
        temp = path.with_suffix(".csv.tmp")
        part.dropna(axis=1, how="all").to_csv(temp, index=False, float_format="%.17g")
        if path.exists() and digest(path) != digest(temp):
            raise ValueError("Aggregate changed during resume")
        temp.replace(path)
        aggregates[str(path)] = digest(path)
    if not (output / "execution_receipt.json").exists():
        write_json(output / "execution_receipt.json", dict(**manifest, completed_utc=utc(), tasks=len(tasks),
                   task_receipts=receipts, aggregate_hashes=aggregates,
                   task_seconds=sum(json.loads(Path(p).read_text())["seconds"] for p in receipts),
                   warnings=[w for p in receipts for w in json.loads(Path(p).read_text())["warnings"]],
                   human_study="NOT_RUN", financial_data="NOT_RUN", model_fitting="NOT_RUN"))
    else:
        saved = json.loads((output / "execution_receipt.json").read_text())
        check_hashes(saved["aggregate_hashes"]); check_hashes(saved["task_receipts"])
    log(output, "complete", new=new, reused=reused, invocation_seconds=time.perf_counter()-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/completion_novelty_audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    args = parser.parse_args()
    run(args.config, args.output, resume=args.resume, stop_after_tasks=args.stop_after_tasks)
