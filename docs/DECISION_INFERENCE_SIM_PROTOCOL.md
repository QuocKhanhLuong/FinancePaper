# F2 inference feasibility simulation — frozen specification

2026-10-07; continuation from `f5ae1de`. This is a **method feasibility and
assumption sensitivity study**, not empirical human evidence or sample-size
justification. No expert has accepted SESOI, time cost or task relevance. The
older protocol and results remain unchanged; this document specifies a new run.

## Scope and estimand

Validate only the proposed secondary two-way bootstrap candidate from
`DECISION_VALUE_PROTOCOL.md`. The planned logistic mixed-effects primary analysis
is **NOT RUN** here. No substitution of bootstrap power for GLMM power is allowed.
No new predictor, real-cohort model fitting, policy selection, participant
collection or invented expert responses occur. Bernoulli draws are explicitly
hypothetical mathematical variables; no simulated individual rating files are
saved, and they must never be merged with future human data.

The estimand is arm-4 minus arm-3 marginal accuracy, standardized equally across
four case strata and averaged over reviewer and case populations. Each reviewer
sees each case once in one of four arms. The 4-arm balanced rotation generalizes
the existing slot design: reviewer offsets contain each of 0..3 equally often,
randomly ordered; within every case stratum, each case phase 0..3 occurs equally
often, randomly ordered. Arm = 1 + (reviewer offset + case phase) mod 4. No
reviewer sees the same case in another arm. Arms 1/2 share the arm-3 simulation
law and are not part of the primary contrast in this feasibility check.

## Frozen scenario grid (illustrative, not fitted)

`configs/decision_inference_sim.json` fixes 36 scenarios: reviewer/case sizes
(16,16), (48,32), (96,64); marginal accuracy gains 0, .05 and .10; four profiles:

| Profile | Marginal arm-3 accuracy | Reviewer/case intercept SD | Reviewer/case arm-4 slope SD | Reviewer dropout / item missing |
| --- | --- | --- | --- | --- |
| iid | .60 | 0 / 0 | 0 / 0 | 0 / 0 |
| intercepts | .60 | .7 / .7 | 0 / 0 | 0 / 0 |
| slopes_missing | .60 | .7 / .7 | .5 / .5 | .10 / .10 |
| high_baseline | .75 | .7 / .7 | .5 / .5 | .10 / .10 |

All SDs are on the logit scale, not ICCs. Stratum offsets are [-.3,-.1,.1,.3].
Reviewers and cases have independent zero-mean Gaussian intercepts and slopes;
slopes apply only to arm 4. Simulated accuracy is Bernoulli with logistic mean.
Calibrate separate arm-3 and arm-4 intercepts by Gaussian quadrature/root solving
so their population means, averaged over strata, equal baseline and baseline+gain.
This prevents Jensen's inequality from changing the declared marginal effect
when slope variance is introduced. Compare 64/128-node quadrature for this grid.
At gain=0 with random slopes this is a **weak marginal null**, not an individual
sharp null. Reviewer dropout and item missingness are independent MCAR draws;
this does not validate MAR/MNAR handling or differential attrition.

No hypothetical profile is selected by the old Taiwan/Polish outcomes or by a
human pilot. There are no financial/business inputs to this simulation. The .05
and .10 gains are sensitivity scenarios, not an approved meaningful effect.

## Exact candidate analyses run on every replicate

Point estimate: for each stratum, observed accuracy in arm 4 minus arm 3; take
the mean of four contrasts. Missing outcomes are excluded from numerators and
denominators. If any arm/stratum has no observed outcome, analysis is invalid.

1. **Two-way percentile bootstrap:** independently resample the full reviewer
   index set and case indices within each of four strata, with replacement.
   All observations sharing a reviewer/case receive the same row/column count;
   each cell's weight is their product. Use the same weights for both arms.
   Recompute the standardized contrast on each bootstrap sample. Reviewer
   sequence and case-phase balance need not hold inside a bootstrap draw; this
   is part of the candidate being tested, not an exact randomization test.
   Use 499 bootstrap draws and 2.5/97.5 percentiles, without retuning the interval.
   Require >=99% finite bootstrap draws, otherwise mark invalid.
2. **Naive independent-ratings Wald comparator:** same point estimate, variance
   equal to sum over strata of [p4*(1-p4)/n4 + p3*(1-p3)/n3] divided by 16;
   normal 95% interval. It deliberately ignores reviewer/case dependence.
   Do not assume it must fail or that bootstrap must win before running.

Reject the zero-effect null exactly when the interval excludes zero. Report
coverage of the declared population marginal effect, interval width, estimate
bias, invalid rate and observed primary-rating counts. A failed replicate is
counted as non-rejection in the primary 1,000-replicate rate; also report the
conditional-on-valid rejection rate. No fitting convergence is involved for
these two methods; GLMM convergence remains NOT RUN.

Each scenario has 1,000 independent outer replicates; 95% Wilson intervals
quantify Monte Carlo uncertainty of rejection rates. Seeds derive from the
frozen base seed, scenario index and replicate index. Bootstrap RNG streams are
separate from data-generation streams. Retain all scenarios and failed draws.

## Predeclared diagnostic gates, not a deployment certificate

For each design/profile's null: Wilson upper bound of rejection <=.075 and
invalid fraction <=.01 is the screening gate against marked inflation. Rejection
below .025 is flagged as conservative; this can imply material loss of power.
The gate is a simulation diagnostic, not proof of exact .05 size. If a null gate
fails, mark matching nonzero-effect results as failing the calibration screen;
do not convert apparent power into a sample-size recommendation. No method,
cutoff, sample size or resampling scheme is tuned after these outcomes.

Power is conditional on the stated hypothetical law and on the actual bootstrap
method run. No final n is selected even if a row exceeds .80: task relevance,
expert-approved SESOI, acceptable delay, exact primary analysis and broader
missingness/heterogeneity checks are still unresolved.

## Execution, resume and evidence

Freeze config, source hashes and this protocol before scientific simulation.
Checkpoint every 50 replicates atomically; SHA256-protect chunks and reject
code/config/input drift on resume. Per-replicate **statistics** and detailed
logs stay local, not individual simulated outcomes. Progress JSONL and tqdm/ETA
use actual completed replicates. Run on CPU; record elapsed time and versions.
No new dependencies or modifications to historical configs/lockfile are needed.

One result report, aggregate scenario/gate CSVs, figure and publication receipt
may be committed to the research branch. No main merge. Full tests, bootstrap
versus literal resampling equivalence, marginal-calibration checks, RNG/resume
checks and a real partial-run resume are required before publication.

## Primary methodological sources

- Owen (2007), [The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111),
  DOI 10.1214/07-AOAS122: resampling rows and columns for crossed data; no claim
  that every finite-sample percentile interval for our contrast is exact.
- Owen and Eckles (2012), [Bootstrapping data arrays of arbitrary order](https://arxiv.org/abs/1106.2125),
  DOI 10.1214/12-AOAS547: product weights across factors and conditions for
  conservative variance. Our binary contrast, small design and percentile CI
  still require their own simulation checks; citation is not validation.
