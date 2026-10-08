"""Export verified C5 numeric receipts and publication figures without fitting."""
import argparse
import json
import os
from pathlib import Path
import shutil
import warnings

from financepaper.evaluation.decision_value import check_hashes, digest, write_json


def report(run, validation, output):
    receipt = json.loads((run/"execution_receipt.json").read_text())
    check_hashes(receipt["frozen"]["sources"])
    check_hashes(receipt["outputs"])
    check_hashes(receipt["task_receipts"])
    checks = json.loads((validation/"guard_checks.json").read_text())
    tests = json.loads((validation/"pytest_final.json").read_text())
    assert tests["returncode"] == 0 and all(c["passed"] for c in checks["checks"])
    assert tests["sha256"] == digest(validation/"pytest_final.txt")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("execution_receipt.json", "manifest.json", "binary.csv", "multistate.csv",
                 "threshold.csv", "certificate.csv", "law_mismatch.csv"):
        target = output/name
        if target.exists() and target.read_bytes() != (run/name).read_bytes():
            raise ValueError("Refusing to overwrite a different report receipt")
        shutil.copyfile(run/name, target)
    for name in ("guard_checks.json", "pytest_final.json"):
        target = output/name
        if target.exists() and target.read_bytes() != (validation/name).read_bytes():
            raise ValueError("Refusing to overwrite a different validation receipt")
        shutil.copyfile(validation/name, target)
    cache = validation/"plot_cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    os.environ["XDG_CACHE_HOME"] = str(cache.resolve())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        certificates = pd.read_csv(run/"certificate.csv")
        thresholds = pd.read_csv(run/"threshold.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        colors = ["#0072B2", "#009E73", "#CC79A7"]
        for m, color, marker in zip([2, 5, 100], colors, ["o", "s", "^"]):
            part = certificates[(certificates.p == "1/2") & (certificates.m == m)].sort_values("normalized_mean")
            axes[0].plot(part.normalized_mean, part.risk_upper, marker=marker, color=color, label=f"Sharp frontier, m={m}")
        part = certificates[(certificates.p == "1/2") & (certificates.m == 2)].sort_values("normalized_mean")
        axes[0].plot(part.normalized_mean, part.generic_risk_upper, "--", color="#333333", label="Oscillation only")
        axes[0].plot(part.normalized_mean, part.universal_risk_upper, ":", color="#D55E00", label="Universal envelope")
        axes[0].set(xlabel="Exact normalized mean pair gap", ylabel="Upper bound on nonpositive-gap probability",
                    title="Signed pair disclosure (p=1/2)", ylim=(0, 1), xlim=(0, 1))
        axes[0].legend(fontsize=8)
        for p, color, marker in zip(["1/10", "1/2", "9/10"], colors, ["o", "s", "^"]):
            part = thresholds[(thresholds.p == p) & (thresholds.risk_budget == "1/10")].sort_values("m")
            axes[1].plot(part.m, part.required_mean, color=color, marker=marker, label=f"p={p}")
        axes[1].axhline(.9, linestyle="--", color="#333333", label="Oscillation only")
        axes[1].axhline(float(part.universal_required_mean.iloc[0]), linestyle=":", color="#D55E00", label="Universal envelope")
        axes[1].set(xlabel="Number of observed players m (log scale)", ylabel="Required normalized mean for risk <= 0.10",
                    title="Finite dimensions approach the envelope", xscale="log", ylim=(.8, .91))
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.18)
        for suffix in ("png", "pdf"):
            fig.savefig(output/f"rank_frontier.{suffix}", dpi=180)
        plt.close(fig)
    write_json(output/"figure_receipt.json", dict(
        inputs={str(run/name): digest(run/name) for name in ("certificate.csv", "threshold.csv")},
        outputs={str(output/f"rank_frontier.{suffix}"): digest(output/f"rank_frontier.{suffix}") for suffix in ("png", "pdf")},
        warnings=[str(w.message) for w in caught], synthetic_theory_only=True,
        note="Lines join configured rational-audit points; not financial observations or measured human outcomes."))
    print(json.dumps(dict(tasks=receipt["tasks"], rows=receipt["rows"], task_seconds=receipt["task_seconds"], output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.validation, args.output)
