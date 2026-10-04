# Prospective bounded audit, not a method-training pilot

Declared 2026-10-04 **before execution of the new finite-state checks**. Freeze
this file and METHOD_SPEC in git before running `audit_method_pivot.py`.
This is a local prospective record, not an independently registered study.

## Decision and primary question

No candidate passed the method gate. Therefore Phase 1 method training,
Phase 2 financial development and Phase 3 confirmation are **NOT AUTHORIZED BY
THE SCIENTIFIC GATE**. They are NOT RUN, not failed benchmarks. The work below
checks mathematical arguments without implementing A–D as predictors.

Question: can the advertised structural properties hold while prediction remains
uninformative or biased, and do the proposed formulas reduce to existing inference?
No AP improvement, confidence interval or publication novelty claim is inferred
from these checks. No post-check replacement generator or parameter tuning.

## Fixed finite worlds and comparators

Enumerate every state of three independent variables in {-1,+1}, uniform law P.
Use Bernoulli outcomes with probabilities from the following known logits:

- Additive: `f=-.8+.9*x1+.6*x2+.4*x3`.
- Interaction: same f plus `1.6*x1*x2`.

Evaluate all nested observation subsets for the exact tower check. The primary
partial-input comparison observes x1 and hides x2,x3. These are illustrative
financial-risk-shaped variables, **not simulated real customers or causal truth**.

Fixed comparators: exact P-conditional oracle, constant P-prevalence, mean-filled
point prediction, and exact integration under a misspecified q with
`q(x2=+1)=.9` (x1,x3 remain independent/uniform). No fitted model. Exact sums
replace random sampling; seeds and Monte Carlo confidence intervals are inapplicable.

Fixed checks:

1. Known conditional Bayes predictions obey the tower identity for every S subset T.
2. A constant predictor also obeys it; report expected Brier/log loss against the
   nonconstant conditional oracle so a vacuous coherence winner is visible.
3. Under the interaction generator, the same observed x1 permits different hidden
   responses; report the range and half-range lower bound on unavoidable worst-case
   approximation to the full-information model response.
4. Single-refinement squared agreement contains update variance; the expectation
   of the independent two-refinement product equals the conditional squared bias.
   Exhaustively sum all pairs; do not train a regularizer.
5. Exact inference under q has zero self-coherence but can differ from P. Report
   expected outcome loss and actual-law coherence as well as q-coherence.
6. For an additive logit, compare sigmoid(expected logit) with expected sigmoid.
   No claim that probability marginalization is exact merely because logits sum.
7. Add `2*x1` to a main effect and subtract it from its interaction allocation.
   Verify unchanged reconstructed logits but changed allocation. This tests
   nonidentifiability without centering, not all identified decomposition methods.
8. Mechanism shift only: keep P(X,Y) fixed; hide x2 with probabilities .9/.1
   for x2=+1/-1 in source, reversed in target. Compare conditional risks among
   hidden cases. This is a population oracle stress example, not a claim that
   all observable deployment distributions are indistinguishable.

Always show additive and interaction worlds, including differences near zero.
Stop and document any failed mathematical assertion; never change the generator
to obtain a desired effect. Tolerance 1e-12 for finite double-precision identities.

## Historical artifact audit

Read only the existing Region B, detector, paired difference, Stable-Core,
headroom elementary-bound CSVs. Record SHA256 and dimensions. Check declared
denominators and selected aggregate ratios, including the zero-query ceiling.
No hidden customer rows enter a predictor. If unavailable, mark UNAVAILABLE;
do not imply re-creation. Never open the old test/reserve for new selection.

## Budget and reproducibility

Audit executable budget: 60 seconds wall time, at most eight support states per
world; CPU only, no learning parameters, no SHAP calls, no sampler training.
Run the tiny finite calculation first; abort on exceeding the budget. Memory
expectation is below 256 MiB for the standard-library audit (estimate, not RSS
measurement). Test overhead is reported separately. Output to ignored
`outputs/method_pivot/`; public records contain only aggregate/synthetic receipts.

After implementation:

```bash
uv run --frozen python scripts/audit_method_pivot.py --historical
uv run --frozen pytest -q tests/test_method_pivot_audit.py
```

The runner hashes this protocol and its source and records its actual git commit,
elapsed time, artifact availability and identity errors. Tests cover invalid
probability laws, hidden-state invariance of conditional inputs, exact enumeration,
reduction, deterministic output and artifact provenance. Tests are engineering
checks; exhaustive toy identities are not a generalization benchmark.

## Gate for any later genuine pilot

This is a stop condition, **not a preregistration of a nonexistent method**.
A new revision must first supply an exact difference, then lock one primary
comparison/metric. Outcome AP/AUROC, Brier/log loss, classwise metrics at a
development-selected threshold, mechanism endpoint, informativeness and cost
must remain separate. LR/XGB25 and appropriate additive control stay; compare
the closest prior method, a simple objective-matched baseline and a reduction
ablation under equal information/split/calibration/tuning budgets.

Only then: one-seed sanity pilot, followed by at least three fixed seeds if it
passes. Split before every fitting step, independent calibration, customer-level
paired bootstrap (1,000 draws), no mask/completion pseudo-replication. Taiwan
development reuse is exploratory. Confirmation needs a frozen new licensed
cohort and target; none is selected here. No financial experiment is pre-emptively
promised while the algorithm and mechanism claim fail the first gate.
