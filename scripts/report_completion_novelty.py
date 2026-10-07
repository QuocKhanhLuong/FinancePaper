"""Publish one evidence-bounded aggregate report for the completed novelty audit."""
import argparse
from fractions import Fraction
from itertools import product
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from financepaper.evaluation.bounded_completion import completion_kernel, scalar_width
from financepaper.evaluation.decision_value import digest, write_json, check_hashes


def table(frame):
    return "\n".join(["| "+" | ".join(frame.columns)+" |", "| "+" | ".join("---" for _ in frame.columns)+" |"]+
                     ["| "+" | ".join(map(str, row))+" |" for row in frame.itertuples(index=False, name=None)])


def report(moments, bounded, counting, validation, output):
    if not output.resolve().is_relative_to(Path("reports/completion_novelty").resolve()):
        raise ValueError("Publish only in reports/completion_novelty")
    output.mkdir(parents=True, exist_ok=False)
    source_runs = {"moments": moments, "bounded": bounded, "counting": counting}
    receipts = {}
    for label, source in source_runs.items():
        receipt = json.loads((source / "execution_receipt.json").read_text())
        check_hashes(receipt["frozen"]["sources"])
        check_hashes(receipt.get("aggregate_hashes", receipt.get("outputs", {})))
        check_hashes(receipt["task_receipts"])
        receipts[label] = receipt
        shutil.copy2(source / "execution_receipt.json", output / f"{label}_execution_receipt.json")
    for source, names in ((moments, ["correctness.csv", "benchmark.csv", "joint_law.csv", "tail.csv"]),
                          (bounded, ["directional.csv", "sharp_tree.csv", "fixed_q.csv"]),
                          (counting, ["counting.csv"])):
        for name in names:
            shutil.copy2(source / name, output / name)
    shutil.copy2(validation / "guard_checks.json", output / "resume_guard_checks.json")
    shutil.copy2(validation / "semivalue_mapping.json", output / "semivalue_mapping.json")
    c, b, law = (pd.read_csv(moments / name) for name in ("correctness.csv", "benchmark.csv", "joint_law.csv"))
    directions, trees, fixed = (pd.read_csv(bounded / name) for name in ("directional.csv", "sharp_tree.csv", "fixed_q.csv"))
    timing = b[b.status == "RUN"].groupby(["benchmark", "method"]).seconds.median().unstack()
    timing = timing.rename(columns={"quadrature": "quadrature_s", "coefficients": "coefficients_s", "pair_cells": "pair_cells_s"})
    timing = timing[["quadrature_s", "coefficients_s", "pair_cells_s", "mc_1024", "mc_16384"]].reset_index()
    for name in timing.columns[1:]:
        timing[name] = timing[name].map(lambda v: f"{v:.6f}" if np.isfinite(v) else "NOT RUN: cap")
    exact_table = trees[(trees.p == "1/2") & (trees.constant == .5) & (trees.target == 0)][["observed", "exact_width", "exact_variance", "leaves"]].copy()
    exact_table["observed"] = exact_table.observed.astype(int)
    exact_table["leaves"] = exact_table.leaves.astype(int)
    os.environ.setdefault("MPLCONFIGDIR", str(validation.resolve() / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(validation.resolve() / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for p, color in zip(["1/10", "1/2", "9/10"], ["#9B4B00", "#0067A5", "#4B751B"], strict=True):
        x = list(range(1, 6)); y = [float(scalar_width(m, Fraction(p))) for m in x]
        axes[0].plot(x, y, marker="o", color=color, label=f"p={p}")
    axes[0].set(xlabel="Observed binary features m", ylabel="Sharp within-model SHAP oscillation", ylim=(0, 1), xticks=range(1, 6))
    axes[0].legend(); axes[0].grid(alpha=.2); axes[0].set_title("Exact formula; LP checked at all markers")
    kernel = np.asarray(completion_kernel(2, Fraction(1, 2)), float)[:, :-1]
    vertices = np.asarray(list(product([-1, 1], repeat=3))) @ kernel.T
    hull = ConvexHull(vertices); polygon = vertices[hull.vertices]
    axes[1].fill(polygon[:, 0], polygon[:, 1], color="#0067A5", alpha=.15)
    axes[1].plot(*np.vstack([polygon, polygon[:1]]).T, color="#0067A5")
    axes[1].scatter(*vertices.T, color="#0067A5", s=20)
    axes[1].axhline(0, color="gray", lw=.6); axes[1].axvline(0, color="gray", lw=.6)
    axes[1].set(xlabel="Completion contrast of SHAP(A1)", ylabel="Completion contrast of SHAP(A2)", aspect="equal")
    axes[1].set_title("Exact contrast set: m=2, p=1/2")
    fig.suptitle("Bounded model, constant completed prediction, one hidden player", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"sharp_bounds.{suffix}", dpi=180)
    plt.close(fig)
    text = f"""# Novelty audit: sharp completion bounds, with prior-art collisions retained

## Decision

The stage produced a proved and executed **narrow candidate contribution (C3)**:
an explicit sharp bound on completion-induced interventional SHAP variation for
bounded models whose completed prediction is constant, including an O(m)-leaf
attaining tree and a fixed-completion-law variance envelope. **Paper-level
novelty remains PROVISIONAL / NOT ESTABLISHED.** The claim is a specialized
mathematical refinement, not a new explainer, learning algorithm, or established
human decision benefit. It survived the specific nearest-work comparisons below;
that is not an exhaustive absence-of-prior-art proof or human peer review.

The new user instruction to continue until novelty was found expanded the work
beyond F2. We executed research and rejected tempting broad claims instead of
renaming known results. C0 exact moments, C1/C2 arbitrary explanation laws and
C4 generic counting hardness are **NO-GO as standalone novelty claims**.
Their working code and negative findings are retained. Research branch:
`research/decision-value-novelty-20261008`; base `9312738bd57b500597aab43559e8dfe66151a1a2`.
Latest main was verified at `e7cac089d98500ac57771d419dff46804ced52c3` and is unchanged.

## C3: precise surviving claim

There are m observed binary players, one hidden player H, full-support product
reference P=Bernoulli(p)^m × q(H), and completion Q fixes all observed values to
one and retains q(H). The model is fixed, lies in [0,1] on the whole finite
reference support, and f(1,h)=c for every completion. All coordinates remain
SHAP players. Under these assumptions,

\\[
\\operatorname{{osc}}_Q(\\phi_i)\\le\\omega_m(p)
=(1-p)-(1-p)\\int_0^1t[p+(1-p)t]^{{m-1}}dt.
\\]

Here osc=max over two hidden states of the absolute attribution difference.
The exact maximum variance for a fixed finite q is beta(q)*omega^2, where
beta(q)=max_B q(B)(1-q(B)). A fair two-state q gives beta=1/4. The scalar bounds
are attained by one depth-(m+1) tree with **2m+2 leaves**, all in [0,1].

{table(exact_table)}

For an observed-feature direction v, the exact set of feasible two-completion
**contrasts** between two distinct hidden states is Kbar[-1,1]^(2^m-1).
With only one hidden state the contrast set is {{0}} and beta(q)=0.
Its support function is ||Kbar^T v||_1,
giving the sharp within-model directional oscillation and fixed-q variance
beta(q)||Kbar^T v||_1^2. This is not the feasible set of absolute SHAP vectors.
The full geometric width of the symmetric contrast set is **twice** this support
value; CSV fields `exact_width` mean within-model oscillation, avoiding that
factor-of-two ambiguity. Different directions can require different attaining
models; joint simultaneous attainment and linear-size general-direction trees
are not claimed.

Proof and precise scope: [BOUNDED_COMPLETION_THEOREM.md](../../../docs/BOUNDED_COMPLETION_THEOREM.md).
For a model output range [L,U], scale oscillation by U-L and variance by (U-L)^2.
A [0,1] range alone is not evidence of calibrated probabilities. One-hot encoding
H, dependent reference features, many hidden players, and a model-specific
certificate change the problem and are not covered.

![Sharp scalar bounds and exact contrast geometry](sharp_bounds.png)

Figure left: exact analytic values at the audited m=1,...,5; no extrapolation or
estimated confidence intervals. Right: the exact two-feature contrast polytope,
constructed from the eight endpoint combinations of its three free columns.

## REPRODUCED — actual coordinator runs

All runs used the existing environment on **CPU**; no model fitting or dataset
download. Runtimes below sum measured task work, excluding tests, literature,
setup, pauses, reporting and AI review. Repeated same-seed resume checks are not
additional independent evidence.

| Run | Executed work | Measured task seconds |
| --- | --- | --- |
| `{moments.name}` | 139 tasks: 100 correctness cases, 24 benchmark repeats, 12 joint-law fixtures, 3 tail checks | {receipts['moments']['task_seconds']:.3f} |
| `{bounded.name}` | 15 (m,p) groups / 45 (m,p,c) settings, 411 result rows | {receipts['bounded']['task_seconds']:.3f} |
| `{counting.name}` | 30 counting instances, 150 cardinality checks, 450 exact oracle queries | {receipts['counting']['task_seconds']:.3f} |

- **C0:** all 100 random finite-reference/completion cases agree with independent
  coalition/background/completion enumeration. Maximum absolute error across
  means, raw second moments and covariance: **{c[['quadrature_error','coefficient_error','pair_cell_error']].to_numpy().max():.3g}**.
  Continuous-Q threshold-cell, repeated-split, off-leaf SHAP and cross-tree
  cancellation fixtures also pass. SHAP is not constant on original model leaves;
  the compiler integrates hybrid-coalition factors, not raw leaf outputs alone.
- **C2:** all 12 arbitrary finite joint-law constructions have exactly constant
  prediction on their completion states; maximum definition-level SHAP error
  **{law.attribution_error.max():.3g}**, covariance error **{law.covariance_error.max():.3g}**.
- **C3:** every coefficient of 15 exact-rational operator-difference audits agrees;
  **{len(directions)} LP optima** agree with the sharp directional formula
  (max absolute error **{directions.lp_error.max():.3g}**). All **{len(trees)} bounded
  scalar trees** attain it (max width error **{trees.width_error.max():.3g}**, variance
  error **{trees.variance_error.max():.3g}**). **{len(fixed)} fixed-q checks** match in
  exact Fraction arithmetic. **{int(fixed.enumerated_endpoint_models.sum()):,} endpoint-valued
  models** were exhaustively checked across **{int((fixed.enumerated_endpoint_models>0).sum())}**
  small (m,p,q) settings. This finite audit supports the implementation; the proof
  supplies the all-m statement.
- **C4:** all 150 recovered cardinality counts equal brute-force integer counts,
  including zero and above-total targets. Correctness does not establish novelty.
- Every runner completed a real one-task pause, resumed with verified outputs,
  then verified a complete resume without recomputation. Separate sacrificial
  guard runs rejected config hash drift and corrupted completed outputs. Actual
  research artifacts were not corrupted. See `resume_guard_checks.json`.
- Full suite: **178 passed, 1 skipped (MPS unavailable), 3 existing SHAP plotting
  deprecation warnings**, 23.80 s (25.08 s wall). Research task receipts recorded no numerical
  warnings for C0/C3; C4 uses exact rational counting. Timing is local prototype
  evidence, not a target-GPU or production-performance claim.

## C0 matched comparator results — no algorithmic novelty claim

Median seconds over three repeats; two target attributions; exact same model,
P and Q. Boxes represent explicit trees with implicit zero leaves. MC times
include completion sampling and point-SHAP. All methods are local Python
implementations, not optimized reproductions of published software.

{table(timing)}

Benchmark IDs 0..5: four leaves with shared paths of depth 2,4,8,12,16,24.
IDs 6..7: 16 leaves, depths 4/6, disjoint hidden coordinates across leaves.
ID 0 has no random hidden coordinates and is a deterministic control. Pair-cell
enumeration uses only the thresholds in each leaf pair, not all global states.
ID 5 pair enumeration exceeds the frozen cap and is **NOT RUN**, not a measured
timeout. The 117 RUN method rows and 3 capped rows are all retained.

At depth 16, pair-cell enumeration is about
{float(b[(b.benchmark==4)&(b.method=='pair_cells')].seconds.median()/b[(b.benchmark==4)&(b.method=='quadrature')].seconds.median()):.1f}×
slower than quadrature here. MC-1024 is faster in these fixtures but approximate;
MC error does not decrease monotonically in every finite sample. Coefficient
integration is also polynomial, so avoiding completion enumeration is not unique
to C0. The max coefficient/quad discrepancy on this grid is
{b[b.method=='coefficients'].discrepancy_from_quadrature.max():.3g}; the grid does not
stress all possible cancellation/conditioning failures. No optimized generic
probabilistic-circuit baseline was run, so there is no claim of superiority over it.

## Prior-work adjudication — read from primary sources, not reproduced

Literature was checked on 2026-10-08 local time. Targeted searches included exact
tree-SHAP covariance, completion uncertainty, prediction-invariant attribution,
bounded removal-based robustness, Khintchine counting and inverse semivalues.
Sources were read; their reported experiments were **not rerun**. This is a
bounded audit, not a systematic-review completeness claim.

| Candidate | Closest verified primary work | Decision / surviving distinction |
| --- | --- | --- |
| C0 polynomial/quadrature TreeSHAP | [Quadrature-TreeSHAP, 2026](https://arxiv.org/abs/2605.04497), [LinearTreeShap, 2022](https://arxiv.org/abs/2209.08192), [TreeSHAP-IQ, 2024](https://arxiv.org/abs/2401.12069) | Quadrature/polynomial SHAP is prior art. |
| C0 exact completion moments | [Khosravi et al., 2019](https://arxiv.org/abs/1910.02182), [Vergari et al., 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html) | Compact SHAP factors can use existing compatible-circuit products/expectations; implementation/oracle contribution only. |
| General random-SHAP covariance | [Chau et al., 2023](https://arxiv.org/abs/2305.15167), [UbiQTree](https://arxiv.org/abs/2508.09639) | Random-function/model uncertainty differs from fixed-f input completion, but generic covariance novelty is false. |
| C1/C2 arbitrary finite SHAP laws at constant prediction | [Bilodeau et al., PNAS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10786278/), [Slack et al., 2020](https://arxiv.org/abs/1911.02508) | Exact finite-tree/full-support construction is useful, but the broad impossibility/manipulation phenomenon is known. NO-GO alone. |
| C3 sharp bounded completion refinement | [Lin, Covert & Lee, NeurIPS 2023](https://arxiv.org/html/2306.07462v2) | Their generic coalition-operator/input/model perturbation bounds do not supply this explicit constant-slice correction w_m(p), fixed-q sharp factor or O(m)-leaf scalar witness. Narrow candidate, not first SHAP robustness bound. |
| C4 directional counting hardness | [Gopalan, Nisan & Roughgarden](https://timroughgarden.org/papers/border.pdf), [Diakonikolas & Pavlou, AAAI 2019, Theorem 3](https://arxiv.org/html/1812.11712v1) | Generalized Khintchine hardness directly covers the derived zero-sum family via the mapping below. Valid specialized corollary; NO-GO as independent novelty. |

**Late C4 correction, preserved explicitly.** The executed reduction in
[DIRECTIONAL_COMPLETION_HARDNESS.md](../../../docs/DIRECTIONAL_COMPLETION_HARDNESS.md)
is valid, but its originality was weakened after execution by a direct mapping.
Set u(t)=p+(1-p)t and define normalized semivalue weights
s_k=2 integral_0^1 t u(t)^k(1-u(t))^(m-1-k) dt. These are positive rational weights;
sum_k binomial(m-1,k)s_k=1. The C4 coefficient satisfies
alpha_r=(1-p)(s_r+s_(r-1))/2. For zero-sum v, converting subset indicators to
Rademacher coordinates gives
D_m(v)=(1-p)Lambda(s) E_mu_s[|v·X|]/4. At p=1/2 the multiplier is Lambda/8.
Here Lambda=sum_r binomial(m,r)(s_r+s_(r-1)), taking out-of-range s terms as zero;
it is not generally 2. Seven additional exact rational checks for m=2,...,8
verify this mapping (`semivalue_mapping.json`); for m=2, Lambda=3.
The prior theorem already proves hardness of this generalized Khintchine
quantity, including restricted zero-sum directions. The explicit finite-difference
proof remains a reproducible derivation, not a new hardness technique. Ordinary
fixed-tree point SHAP under product P remains tractable; no contradiction exists.

Three AI agents generated initial perspectives and reviewed later coordinator
candidates. They are not human experts or independent empirical replications.
Two read-only reviewers found no blocking C3 algebra/code defect; both judged
paper-level originality provisional. Their raw transcripts stay local. Corrections
retained in this report: C0 originated with the coordinator; original-leaf SHAP
constancy is false; observed covariance need not have zero row sums when hidden
attributions balance it; and contrast support radius is not its full geometric width.

## REPORTED historical context; PROPOSED and NOT RUN

F0/F1/F2, Stable-Core, the 245-case rule baseline, archived architecture results,
and Round-7 method-pivot findings are **REPORTED only**, not rerun for novelty.
Taiwan/Polish inspected cohorts and all historical configs/results stayed fixed.
No existing tracked file was edited by this stage; all additions are new paths.

**NOT RUN:** human participants/expert task approval, financial data experiments,
new predictor fitting or test-based model selection, empirical decision value,
dependent-reference or multiple-hidden-player extensions, optimized circuit
software comparison, real-world completion calibration, approximate-hardness
claims, and external human novelty review. Exact endpoint enumeration above the
prespecified small-dimensional cap is also NOT RUN. No automatic merge to main.

**PROPOSED:** position C3 as a sharp, narrowly scoped mathematical proposition
supporting the decision-value problem, with C0/C2 as reproducible audit tools.
Before a paper claim, resolve whether the restricted norm/witness constitutes a
substantive advance beyond Lin et al.'s framework and semivalue corollaries.
Do not use additional viewed financial tests to manufacture that distinction.

## Reproduce and resume

Use new output directories with the existing locked environment:

```sh
rtk proxy .venv/bin/python scripts/run_completion_novelty_audit.py --output runs/decision_value_pilot/novelty_NEW --stop-after-tasks 1
rtk proxy .venv/bin/python scripts/run_completion_novelty_audit.py --output runs/decision_value_pilot/novelty_NEW --resume
rtk proxy .venv/bin/python scripts/run_bounded_completion_audit.py --output runs/decision_value_pilot/bounded_NEW --stop-after-tasks 1
rtk proxy .venv/bin/python scripts/run_bounded_completion_audit.py --output runs/decision_value_pilot/bounded_NEW --resume
rtk proxy .venv/bin/python scripts/run_directional_hardness_audit.py --output runs/decision_value_pilot/directional_NEW --stop-after-tasks 1
rtk proxy .venv/bin/python scripts/run_directional_hardness_audit.py --output runs/decision_value_pilot/directional_NEW --resume
rtk proxy .venv/bin/python -m pytest -q
```

Source/config/dependency and completed-output SHA256 checks prevent incompatible
resume. Actual creation times are in receipts; directory suffixes are unique run
identifiers. Raw per-task files, JSONL logs and tqdm transcripts stay ignored.
Aggregate allowlist and hashes are in `publication_receipt.json`; preservation,
tests and branch publication details are in `validation_receipt.json`.
"""
    (output / "RESULTS.md").write_text(text)
    paths = sorted(p for p in output.iterdir() if p.is_file())
    write_json(output / "publication_receipt.json", dict(
        source_runs={name: str(path) for name, path in source_runs.items()},
        reporter_sha256=digest(__file__), outputs={str(p): digest(p) for p in paths},
        scope="SYNTHETIC_MATHEMATICAL_AUDIT_NOT_HUMAN_OR_FINANCIAL_EVIDENCE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("moments", "bounded", "counting", "validation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report(**vars(args))
