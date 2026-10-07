"""Exact finite counterexamples; NOT a financial model or candidate implementation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.optimize import brentq, linprog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/method_pivot/ROUND7_PREREGISTERED_FALSIFICATION.md"


def finite_aipw(values, probabilities, q, pi):
    """Enumerate independent Bernoulli verification; q and pi are fixed."""
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if (values.ndim != 1 or probabilities.shape != values.shape
            or not np.all(np.isfinite(values)) or not np.isfinite(q)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1) or not 0 < pi <= 1):
        raise ValueError("finite probability distribution and 0 < pi <= 1 required")
    target = np.array([q + a / pi * (v - q) for a in (0, 1) for v in values])
    mass = np.concatenate(((1-pi)*probabilities, pi*probabilities))
    mean = float(mass @ target)
    return {"mean": mean, "variance": float(mass @ (target-mean)**2),
            "second_moment": float(mass @ target**2)}


def frozen_teacher():
    g, p, pi, n = [.5, 1.], [2/3, 1/3], .1, 1000
    mean = float(np.dot(g, p))
    var = float(np.dot(p, (np.array(g)-mean)**2))
    rows = {}
    for label, q in [("misspecified_q", .9), ("oracle_q", mean)]:
        stats = finite_aipw(g, p, q, pi)
        rows[label] = {**stats, "q": q, "mean_standard_error_n1000":
                       float(np.sqrt(stats["variance"]/n))}
    rows["verification_only_fixed_n100"] = {
        "mean": mean, "mean_variance": var/100,
        "mean_standard_error": float(np.sqrt(var/100))}
    rows["misspecified_q_variance_ratio_vs_fixed100"] = (
        rows["misspecified_q"]["variance"]/n/(var/100))
    return rows


def joint_teacher():
    rows = []
    for theta, pi in itertools.product([0., .5, 1.], [.1, .25, .5, 1.]):
        loss = lambda t: finite_aipw([-t, t], [.5, .5], 0., pi)["second_moment"]
        step = 1e-5
        # Unrestricted scalar critic: sup_h 2*h*E[f-g] - h^2.
        residual_mean = -float(np.dot([.5, .5], [-theta, theta]))
        h_star = residual_mean
        rows.append({"theta": theta, "pi": pi, "coherence_loss": residual_mean**2,
                     "squared_target_loss": loss(theta),
                     "finite_difference_gradient": (loss(theta+step)-loss(theta-step))/(2*step),
                     "conditional_moment_critic_value": 2*h_star*residual_mean-h_star**2})
    # E[(theta*H-H)^2]=(theta-1)^2, lambda=1, pi=.1.
    theta_naive = .1/(.1+1)
    return {"rows": rows, "supervised_example": {
        "lambda": 1, "pi": .1, "correct_theta": 1., "naive_theta": theta_naive,
        "correct_outcome_mse": 0., "naive_outcome_mse": (theta_naive-1)**2}}


def sensitivity():
    q = np.full(3, 1/3)
    delta = np.array([[-4., 8., 8.], [8., -4., 8.]])
    rows = []
    for gamma in [1., 1.8, 2., 2.2]:
        solutions = [linprog(q*d, A_eq=[q], b_eq=[1.],
                     bounds=[(1/gamma, gamma)]*3, method="highs") for d in delta]
        if not all(r.success for r in solutions):
            raise RuntimeError("normalized pairwise LP failed")
        # Deliberately DIFFERENT question: minimize maximum competitor margin.
        all_competitors = linprog([0., 0., 0., 1.],
            A_ub=np.column_stack((delta*q, -np.ones(2))), b_ub=np.zeros(2),
            A_eq=[[*q, 0.]], b_eq=[1.],
            bounds=[(1/gamma, gamma)]*3+[(None, None)], method="highs")
        if not all_competitors.success:
            raise RuntimeError("all-competitor diagnostic LP failed")
        rows.append({"gamma": gamma, "competitor_optima": [float(r.fun) for r in solutions],
                     "any_competitor_minimum": min(float(r.fun) for r in solutions),
                     "normalization": [float(q @ r.x) for r in solutions],
                     "different_all_competitors_minimax": float(all_competitors.fun)})
    return rows


def allocation():
    p, sigma, cost = np.array([.5, .5]), np.array([.4, .05]), np.array([10., 10.])
    rows = []
    for budget in [5., 8.]:
        raw_scale = budget/np.sum(p*sigma*np.sqrt(cost))
        rule = lambda scale: np.minimum(1., scale*sigma/np.sqrt(cost))
        corrected_scale = brentq(lambda s: np.sum(p*cost*rule(s))-budget, 0., 100.)
        choices = {"uniform": np.full(2, budget/float(p@cost)),
                   "clip_without_resolving": rule(raw_scale),
                   "resolve_multiplier": rule(corrected_scale)}
        for label, prop in choices.items():
            rows.append({"budget": budget, "method": label, "probabilities": prop.tolist(),
                         "cost": float(np.sum(p*cost*prop)),
                         "variance_functional": float(np.sum(p*sigma**2/prop))})
    return rows


def xor_shap():
    domain = list(itertools.product([0, 1], repeat=2))
    rows = []
    for x in domain:
        def value(subset):
            completed = [tuple(x[j] if j in subset else z[j] for j in range(2))
                         for z in domain]
            return sum(int(a != b) for a, b in completed)/4
        base, first, second, both = [value(s) for s in [(), (0,), (1,), (0, 1)]]
        phi1 = .5*((first-base)+(both-second))
        phi2 = .5*((second-base)+(both-first))
        rows.append({"x": x, "prediction": both, "phi1": phi1, "phi2": phi2,
                     "contrast": phi1-phi2, "reconstruction_error": base+phi1+phi2-both})
    return rows


def run_audit():
    return {"frozen_teacher": frozen_teacher(), "joint_teacher": joint_teacher(),
            "sensitivity": sensitivity(), "allocation": allocation(), "xor_shap": xor_shap(),
            "constant_reason_counterexample": {"nominal_label_vocabulary": 3,
                "target_entropy": 0., "mutual_information": 0., "error": 0.,
                "actual_coverage": 1., "proposed_bound": 0.}}


def write_run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = run_audit()
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
    (output/"results.json").write_text(text)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    manifest = {"status": "complete", "evidence_type": "exact_finite_synthetic_audit",
        "financial_data_accessed": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter()-start, "platform": platform.platform(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "git_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain"),
        "hashes": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), PROTOCOL]},
        "results_sha256": sha(output/"results.json")}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss*1024
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_run(args.output)
    print(json.dumps({"status": manifest["status"], "output": str(args.output),
                      "runtime_seconds": manifest["runtime_seconds"]}))
