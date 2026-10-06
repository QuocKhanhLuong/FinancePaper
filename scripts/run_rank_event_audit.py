"""Run the preregistered round-five falsifier. No financial data are loaded."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import time

# Before importing numpy / BLAS, enforce the registered single-thread budget.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from threadpoolctl import threadpool_limits
from financepaper.explanations.conditional_moments import FiniteJointLaw
from financepaper.explanations.rank_event_audit import (
    FiniteContrastAudit, coalition_shap, toy_compiler, toy_margin,
)
from financepaper.reliability.validation import revision_arrays


def run():
    start = time.perf_counter()
    cpu_start = time.process_time()
    support = np.array([-1., -.5, 0., 1., 2.])
    points = np.column_stack([np.ones((5, 3)), support])
    observed = np.array([True, True, True, False])
    partial = np.array([1., 1., 1., np.nan])
    compiler = toy_compiler()
    phi = compiler.values(points)
    coalition = coalition_shap(toy_margin, points, np.zeros(4))
    analytic = np.column_stack([2+support/2, 2-support/2, np.full(5, 2.125), np.zeros(5)])
    np.testing.assert_allclose(phi, coalition, atol=1e-12, rtol=0)
    np.testing.assert_allclose(phi, analytic, atol=1e-12, rtol=0)
    np.testing.assert_allclose(phi.sum(1), toy_margin(points), atol=1e-12, rtol=0)
    current = compiler.values(points[[0]])[:, :3]
    query = FiniteContrastAudit(compiler, np.eye(4)[:, :3], current[0], np.ones(3, bool))
    frozen = revision_arrays(current, phi[None, :, :3], np.zeros((1, 3), bool), k=2)
    rows, moments = [], []
    laws = {"A": [.5, 0., 0., .5, 0.], "B": [0., .8, 0., 0., .2]}
    for seed in (101, 102, 103):
        w = np.random.default_rng(seed).uniform(.1, 1., len(points))
        laws[f"identity_seed_{seed}"] = (w/w.sum()).tolist()
    for name, weights in laws.items():
        law = FiniteJointLaw(points, weights)
        event = query.events(partial, observed, law)
        np.testing.assert_array_equal(event, frozen["event"][0])
        result = compiler.moments(partial, observed, law)
        mean = law.weights @ phi
        centered = phi-mean
        covariance = (centered.T*law.weights) @ centered
        np.testing.assert_allclose(result.mean, mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result.covariance, covariance, atol=1e-12, rtol=0)
        rows.append(dict(law=name, weights=list(weights), events=event.tolist(),
                         revision_probability=query.probability(partial, observed, law),
                         # Descriptive exact-law versions of frozen completion
                         # controls, not new detector training or K8 estimates.
                         rank_instability=float(law.weights @ frozen["rank"][0]),
                         sign_instability=float(law.weights @ frozen["sign"][0]),
                         observed_attribution_variance=float(np.diag(covariance)[:3].mean()),
                         mean=result.mean.tolist(), covariance=result.covariance.tolist()))
        moments.append(result)
    np.testing.assert_allclose(moments[0].mean, moments[1].mean, atol=1e-12, rtol=0)
    np.testing.assert_allclose(moments[0].covariance, moments[1].covariance, atol=1e-12, rtol=0)
    np.testing.assert_allclose([rows[0]["revision_probability"], rows[1]["revision_probability"]], [.5, .2])
    elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu_start
    if cpu > 60:
        raise RuntimeError("Registered 60 CPU-second budget exceeded")
    return dict(status="VERIFIED_SYNTHETIC", trained_parameters=0, financial_records=0,
                current_candidates=query.candidates.tolist(), point_attributions=phi.tolist(),
                margin_values=toy_margin(points).tolist(), laws=rows,
                max_coalition_error=float(np.abs(phi-coalition).max()),
                matched_mean_error=float(np.abs(moments[0].mean-moments[1].mean).max()),
                matched_covariance_error=float(np.abs(moments[0].covariance-moments[1].covariance).max()),
                rank_probability_gap=rows[0]["revision_probability"]-rows[1]["revision_probability"],
                full_group_channels=3, projected_sign_contrast_channels=6,
                four_atom_fourier_comparator="NOT_APPLICABLE: nonuniform laws; moments cannot identify event",
                wall_seconds=elapsed, cpu_seconds=cpu, budget_cpu_seconds=60, threads=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/method_pivot/round5/falsifier"))
    args = parser.parse_args()
    dest = args.output / "results.json"
    if dest.exists():
        parser.error(f"Refusing to overwrite an existing result: {dest}")
    root = Path(__file__).resolve().parents[1]
    with threadpool_limits(limits=1):
        result = run()
    sources = ["scripts/run_rank_event_audit.py", "src/financepaper/explanations/rank_event_audit.py",
               "src/financepaper/explanations/conditional_moments.py", "src/financepaper/reliability/validation.py",
               "tests/test_rank_event_audit.py", "docs/method_pivot/ROUND5_RANK_INFERENCE_AUDIT.md", "uv.lock"]
    result["provenance"] = dict(
        utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        source_sha256={s: sha256((root/s).read_bytes()).hexdigest() for s in sources},
        preregistration_commit="2a40d1e", seeds=[101, 102, 103],
        seed_purpose="finite-weight identity checks only; no training or selection",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"path": str(dest), "status": result["status"],
                      "rank_probability_gap": result["rank_probability_gap"],
                      "cpu_seconds": result["cpu_seconds"]}))


if __name__ == "__main__":
    main()
