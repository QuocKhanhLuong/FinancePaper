"""Exact synthetic baseline check. Not a financial or learned-MVU experiment."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import combinations
from math import factorial
from pathlib import Path
import json
import subprocess

import numpy as np
from scipy.special import expit

from financepaper.reliability.prediction_distribution import prediction_distribution_features


def point_shap(f, x, background):
    phi = np.zeros(len(x))
    for j in range(len(x)):
        others = [i for i in range(len(x)) if i != j]
        for size in range(len(x)):
            weight = factorial(size)*factorial(len(x)-size-1)/factorial(len(x))
            for subset in combinations(others, size):
                z = background.copy()
                z[list(subset)] = x[list(subset)]
                before = f(z)
                z[j] = x[j]
                phi[j] += weight*(f(z)-before)
    return phi


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    f = lambda x: x[0]*x[2]+x[1]*(1-x[2])
    b = np.array([0., 0., 0.25])
    x = np.array([[1., 1., 0.], [1., 1., 1.]])
    phi = np.stack([point_shap(f, z, b) for z in x])
    np.testing.assert_allclose(phi, [[0.125, 0.875, 0], [0.625, 0.375, 0]], atol=1e-14)
    p = expit([f(z) for z in x])
    # Repeated support represents the exact declared masses 3/4 and 1/4.
    stats = prediction_distribution_features(p[:1], np.array([[p[0], p[0], p[0], p[1]]]))
    additive_phi = [point_shap(lambda z: z[0]+z[1], z, b).tolist() for z in x]
    np.testing.assert_allclose(additive_phi, [[1, 1, 0], [1, 1, 0]], atol=1e-14)
    result = {
        "status": "EXACT_SYNTHETIC_ENUMERATION", "financial_validation": False,
        "new_method_claim": False, "hidden_support": [0, 1], "hidden_weights": [0.75, 0.25],
        "phi": phi.tolist(), "probability": p.tolist(), "additive_phi": additive_phi,
        "prediction_variance": float(stats["completion_variance"][0]),
        "hard_confidence": float(stats["hard_confidence"][0]),
        "current_action_agreement": float(stats["current_action_agreement"][0]),
        "observed_reason_revision_probability": float(np.array([0.75, 0.25]) @
            (phi[:, :2].argmax(1) != phi[0, :2].argmax())),
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    root = Path(__file__).resolve().parents[1]
    sources = ["docs/method_pivot/ROUND4_BASELINE_SANITY_PROTOCOL.md",
               "scripts/run_prediction_distribution_sanity.py",
               "src/financepaper/reliability/prediction_distribution.py",
               "tests/test_prediction_distribution.py"]
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "source_hashes": {s: sha256((root/s).read_bytes()).hexdigest() for s in sources},
        "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
