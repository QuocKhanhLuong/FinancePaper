# C5 with retained reference moments: a sharper bound and its limits

2026-10-08. Branch `research/reference-moment-frontier-20261008`.
Base `8611ab8653b2288aac2ba7e047679a56515b63d1`.

## Result and research decision

**C5 survives as a sharp mean-only theorem.** Retaining its full reference-moment
vector yields a sharper bounded-moment certificate: it is strictly tighter in
159/231 configured moment profiles and never worse. All 105 configured C5
extremizing moment vectors recover the old sharp failure mass exactly.

The useful distinction is the information retained, not a claim that the old
theorem is false. The new comparator instantiates established first-moment
optimization. Its value as a SHAP-specific characterization is still a candidate
contribution; a new generic probability inequality or practical algorithmic
novelty is not established.

At a 5% failure budget, the new bound certifies 74/231 profiles versus 33/231
for C5. **Every one of those 74 has a reference moment exactly at 1.** Among
111 profiles with all moments strictly inside (-1,1), both certify zero at 5%.
Thus the coverage gain does not yet demonstrate robustness to estimated moments
or usefulness for real predictors. Profiles are configured synthetic conditions,
including duplicates, not independent people, datasets or deployment frequencies.

**Next research framing:** quantify the loss of rank certifiability when the
reference-moment vector is compressed to its weighted mean, then test whether
retaining uncertain moments still helps. Keep C3 as motivation, C5 as the
mean-only envelope, and the present result as an information-retention comparison.
Do not count these as independent major novelties.

## What was derived and implemented

Under C5's assumptions, let `b_k=E d_k(H)` be the cardinality-indexed reference
means. The normalized signed gap has the exact form

\[
G(H)=\sum_k\beta_k d_k(H)+c,\qquad
c=\sum_k(\alpha_k-\beta_k)b_k,
\]

where `d_k in [-1,1]`, `sum alpha=1`, and `sum beta=1/2`. Put
`X_k=(1-d_k)/2`, `mu_k=(1-b_k)/2`, and `tau=(1/2+c)/2`. For positive mean
gap, the sharp bound given the vector b is the largest z satisfying

\[
z\tau\le\sum_k\beta_k\min(z,\mu_k).
\]

It follows from `E[X_k 1_B]<=min(P(B),E X_k)` for the bad event B. No independence
between the X coordinates is imposed. Two hidden states attain the bound;
their contrasts preserve every b_k. Zero/unit-risk endpoints have one-state
attainers. Sorting the mu values and sweeping breakpoints computes the answer
with O(m log m) arithmetic/comparison operations after the coefficients and
moments are available; rational bit cost is not constant.

The [protocol and proof](../../../docs/REFERENCE_MOMENT_FRONTIER.md) specify
the complete argument, constructors and endpoint handling. The
[implementation](../../../src/financepaper/evaluation/reference_moments.py)
uses exact rational arithmetic. It supplies a specialized closed-form reduction
of a classical bounded-moment problem, not a new optimization paradigm.

The model remains bounded in [0,1], with iid Bernoulli(p) observed reference
variables, query all ones, ONE finite hidden SHAP player, and matching hidden
reference/completion laws. Predictions on the query slice are constant. This
does not establish conditional SHAP, multiple separate missing players, arbitrary
reference dependence, business correctness or a finite-sample confidence level.

### Concrete information-loss example, included before the audit

For m=3, p=1/2, and `b=(0,1)`, the normalized mean is 3/4:

| Available bound | Upper probability of a nonpositive pair gap |
|---|---:|
| Generic range | 25% |
| C5, retaining only the mean | 15.1388% |
| Scalar range-aware Markov | 10% |
| Full reference-moment bound | **0%** |

The exact endpoint `b_1=1` forces its bounded contrast to equal 1 almost surely.
The zero-risk result uses this strong information; it is not a measured zero
error rate, a claim of causal relevance, or a guarantee from an estimated mean.

## Actual execution: REPRODUCED / RUN

The [config](../../../configs/reference_moment_audit.json) fixes m in
{2,3,4,5,8,16,32}, p in {1/10,1/2,9/10}, eleven profiles per pair of m,p,
and five C5 attaining failure masses. The mathematical protocol and acceptance
threshold were fixed before outcomes. One numerical backend repair was needed
after a stopped first run, documented below; this is not an untouched numerical
confirmation.

| V2 check | Executed result |
|---|---|
| Configured reference-moment profiles | 231 |
| C5 extremizing-vector recovery cases | 105; exact equality in all |
| Independent generic moment LP solves | 336 |
| Exact attaining laws/contrasts | 336; moments and event mass verified rationally |
| Full SHAP definition checks, m<=5 | 132 profile cases |
| Full truth-table LP solves with fixed reference moments | 243 |
| Above-bound full-table probes | 111; all rejected by positive minimum bad gap |
| Smallest rejected probe gap | 0.0218354194, normalized units |
| Largest generic LP versus rational-bound discrepancy | 1.2073897437403502e-11, below 1e-9 threshold |
| New tests | 20 initial tests + 8 failure-driven numerical regression tests |
| Full repository suite | **250 passed, 1 skipped, 3 warnings in 23.71 s** |

The generic LP maximizes bad mass with individual event moments constrained by
the box support. It does not evaluate the sorted formula. The full-table LP
optimizes every bounded model-table entry while fixing the cardinality means
and query predictions. Its SHAP operator comes from independent exact
coalition/background enumeration, combined affinely across two positive hidden
reference laws. It does not use the C5 mixture to form its objective.

For the maximal proposed mass, the bad-gap minimum is nonpositive; at
`(1+z)/2` it is strictly positive when z<1. At z=0, a zero-mass bad state is
only a boundary LP diagnostic: the actual attaining witness has one positive-mass
hidden state. For m>5, full exponential tables were NOT RUN; compact witnesses
and moment LPs still ran. These complementary checks are not independent studies.

### Comparison on 231 profiles

The vector bound is tighter than C5 in 159 cases, equal to numerical display
precision in 72, and tighter than scalar range-aware Markov in 82. Strict
comparison counts use a 1e-12 reporting margin. Dominance checks in the runner
use exact rational bounds versus C5's conservative rational inverse.

Scalar Markov alone does **not** dominate C5: C5 is tighter in 6 profiles,
while Markov is tighter in 153. Their remaining 72 agree to display precision.
All 105 C5-attaining cases also attain scalar range-aware Markov when the
required reference information is supplied.

| Certificate | 1% budget | 5% budget | 10% budget |
|---|---:|---:|---:|
| Generic range | 24 | 33 | 55 |
| Scalar range-aware Markov | 24 | 35 | 67 |
| C5 | 24 | 33 | 59 |
| Full reference moments | 73 | 74 | 96 |

Counts use exact rational comparisons; C5 releases are checked against its
forward frontier to avoid inverse-bisection boundary rounding. The separate
attainer-recovery tasks are excluded from these denominators. There are 73
zero-risk and 30 unit-risk profiles. The strict-interior restriction above is
a descriptive limitation check, not a replacement benchmark or new selection
criterion. No new budget or model was selected after inspection.

## Numerical stop and repair — preserved, not hidden

The first run completed **272/336 tasks**, then stopped at m=16, p=9/10,
all b=0. The default HiGHS LP returned 0.9999999988978737 rather than 1:
error **1.1021262791288677e-9**, exceeding the fixed 1e-9 threshold. The
remaining 64 tasks were NOT RUN in that first run.

An exact primal point `z=1, s_k=1/2` satisfies every constraint with zero event
residual; the objective bound z<=1 proves optimality. The smallest beta is
about 4.17e-17. Disabling presolve and tightening dual-simplex tolerances did
not fix the discrepancy. Switching to IPM alone was also insufficient in the
subsequent regression test at a tighter optimality tolerance.

V2 applies an algebraically equivalent 1e6 scaling of the zero-RHS event row,
uses `highs-ipm` without presolve, and tightens internal solver tolerances.
It preserves the **same scientific 1e-9 threshold, model grid, mathematical
formula and full-table LP settings**. Tiny-coefficient handling is the likely
mechanism; this is not presented as an isolated upstream solver bug.

The [numerical amendment](../../../docs/REFERENCE_MOMENT_NUMERICS_V2.md) records
this failure-driven change. V2 reran every task in a new directory, including
the failing case. The first source snapshot, manifest, partial results and
failed validation log remain intact. Public evidence includes
[first-run failure and exact certificate](first_run_failure.json),
[first manifest](first_run_manifest.json), and [272 partial rows](first_run_partial.csv).
No acceptance tolerance was relaxed and the old output was not overwritten.

## Runtime, progress and artifact integrity

V2 used CPU, one numeric thread, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1,
pandas 2.3.3 and tqdm 4.70.1 on macOS 26.2 arm64. Actual interval:
**11:47:50.490–11:47:59.638 UTC**, including deliberate pause. Summed task
elapsed time was **2.068734 s**; the completing resume loop took 2.217116 s.
These are audit timings, not a latency comparison or moment-extraction benchmark.

V2 emitted no audit warnings. The suite skipped unavailable MPS hardware and
reported three existing SHAP/Matplotlib color-map pending deprecations.
The real pause completed one task; resume did 335 new / 1 reused; complete
resume did **0 new / 336 reused**. Scratch config drift and task corruption
were rejected. Detailed logs and per-task results stay in ignored run folders.

Public artifacts: [summary](summary.json), [profiles](profile.csv),
[C5 attainers](c5_attainer.csv), [exact coverage counts](coverage.csv),
[manifest](manifest.json), [execution receipt](execution_receipt.json),
[validation receipt](validation_receipt.json), [progress milestones](progress_milestones.json)
and [export hashes](artifact_export.json). No independent sub-agent rerun or
external human peer review occurred in this stage.

Fresh rerun, using new paths:

```bash
.venv/bin/python scripts/run_reference_moment_audit_v2.py --output runs/decision_value_pilot/reference_NEW --stop-after-tasks 1
.venv/bin/python scripts/run_reference_moment_audit_v2.py --output runs/decision_value_pilot/reference_NEW --resume
.venv/bin/python scripts/validate_reference_moment_audit_v2.py --run runs/decision_value_pilot/reference_NEW --output runs/decision_value_pilot/reference_validation_NEW
```

The original runner is retained to reproduce the numerical failure. Source and
config hashes prevent mixed-version resume. Directory timestamps are identifiers;
receipts contain actual times. All pre-existing tracked files remain unchanged.
Main was fetched for verification, not merged or modified.

## Prior evidence, novelty boundary and one next action

**REPORTED only:** the earlier C3/C5/C6 scientific audits and the
[compute gate](../../disclosure_compute_gate/20261008/RESULTS.md). This stage's
105 recovery tasks are a declared new targeted check of C5 extremizers, not a
replay of the entire previous experiment grid or an independent external replication.

Moment-based sharp probability bounds and optimization formulations have a long
prior literature. Bertsimas and Popescu (2005), sections 2 and 5, give the general
framework and tractable first-moment cases. This is a direct framework collision,
not evidence that their paper contains the exact SHAP cardinality formula.
[Primary paper](https://www.mit.edu/~dbertsim/papers/MomentProblems/Optimal-inequalities-in-probability-theory-A-convex-optimization-approach-SIAM15.pdf).
The present bounded search does not establish priority for the specialization.

**Availability limitation:** retaining b is free only in a computation that
already forms that vector. The current general `attribution_moments` API returns
mean, second moment and covariance, not cardinality-conditioned b. Extraction
cost on actual predictors was not measured. Therefore reject claims of universal
free extra information or established incremental deployment value.

**One next action — PROPOSED / NOT RUN:** test the moment-vector certificate
under prespecified nonzero moment uncertainty, with the same lower-confidence
information supplied to all comparators. This directly challenges whether the
observed endpoint-dependent certification gain survives imperfect information.

**NOT RUN / not established:** uncertain-moment evaluation or finite-sample
calibration; real-model extraction/cost benchmark; financial training/evaluation;
human judgments or utility; multiple separate missing players; full truth tables
above m=5; external peer review; confirmed publication novelty. The supported
result is a sharp information-retention characterization and an audited baseline,
with its endpoint sensitivity explicitly exposed.
