# Stable-Core v2: measured results and falsification

Measured 2026-10-03. **No general coverage superiority or algorithmic novelty is
supported.** Variable-size stable reason sets are useful as a conservative
operational option, but they do not replace the strongest simple policies.

## Provenance, target and denominators

Protocol/source/input hashes preceded the run; all conditional models and release
policies were frozen before new assessment computation. XGBoost 25-view predictors,
preprocessing, semantic groups, masks and TreeSHAP references stayed unchanged.
Taiwan: 2,400 distinct assessment customers across three disjoint 800-customer
folds, four conditions, 9,600 customer-condition records. Polish: 1,182 assessment
statements, two conditions, 2,364 records; exact duplicate-feature clusters are
the bootstrap unit. These datasets were inspected previously: **exploratory
extension**, not independent confirmation. Freddie: **NOT RUN**, no official files.

Primary reason failure means restored observed-group contribution <=0.01 in raw
logit units. Ranked sensitivity additionally requires top-2 membership. Neither
is causal correctness, model correctness or expert validation. Current candidates
are **all** available groups with contribution >0.01; output size is variable.
These new reason-level definitions do not overwrite the historical whole-event.

Tables aggregate an equal mixture of declared environments, with reason-weighted
failure ratios. Customer coverage includes customers with no candidate reasons.
The 10% values below are calibration budgets, not guarantees. Tables for every
alpha .05/.10/.15, environment and conservative policy are in generated CSVs.

## Main target: meaningful positive reasons, 10% budget

| dataset | method | reason_risk | reason_coverage | customer_ge1 | customer_ge2 | mean_reasons |
|---|---|---|---|---|---|---|
| taiwan | release_all | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | whole_legacy2 | 1.61% | 62.84% | 71.99% | 71.99% | 1.4398 |
| taiwan | whole_mc1 | 1.15% | 40.42% | 92.61% | 0.00% | 0.9261 |
| taiwan | whole_mc_all | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | rank | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | variance | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | frequency | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | strength | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | lower_quantile | 2.56% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | stable_donor | 0.53% | 95.62% | 91.40% | 69.27% | 2.1907 |
| taiwan | stable_conditional | 0.54% | 94.70% | 91.14% | 68.75% | 2.1697 |
| taiwan | stable_both | 0.29% | 93.64% | 90.82% | 68.01% | 2.1454 |
| polish | release_all | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | whole_legacy2 | 1.97% | 52.25% | 78.47% | 78.47% | 1.5694 |
| polish | whole_mc1 | 1.21% | 31.35% | 94.16% | 0.00% | 0.9416 |
| polish | whole_mc_all | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | rank | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | variance | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | frequency | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | strength | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | lower_quantile | 3.90% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | stable_donor | 0.54% | 94.68% | 93.06% | 74.96% | 2.8435 |
| polish | stable_conditional | 0.41% | 93.52% | 92.94% | 74.58% | 2.8088 |
| polish | stable_both | 0.24% | 92.75% | 92.64% | 74.07% | 2.7855 |

Release-all already achieves 2.56%/3.90% aggregate reason risk with greater coverage
than Stable-Core on Taiwan/Polish. Thus the primary claim of greater coverage at
the same 10% or 15% budget **fails against this simple baseline**. The same endpoint
is substantially easier than preserving a top-k ranking. At 5%, Polish MCAR30
release-all and several empirically calibrated simple policies reach **5.58%**:
pooled success must not be mistaken for environment-specific risk control.

Stable-Core both retains 93.64%/92.75% of candidate reasons and reaches
90.82%/92.64% customer coverage with 0.286%/0.243% reason failure. It trades some
coverage and compute for much lower observed failure; this is a useful point,
not dominance at the declared budgets. No new sub-5% budget was tuned afterward.

95% paired customer-cluster bootstrap intervals (1,000 draws):

| dataset | reason_risk | reason_risk_lo | reason_risk_hi | customer_ge1 | customer_ge1_lo | customer_ge1_hi |
|---|---|---|---|---|---|---|
| taiwan | 0.29% | 0.21% | 0.38% | 90.82% | 89.80% | 91.77% |
| polish | 0.24% | 0.14% | 0.37% | 92.64% | 91.22% | 93.94% |

## Ranked sensitivity, 10% budget

| dataset | method | reason_risk | reason_coverage | customer_ge1 | customer_ge2 | mean_reasons |
|---|---|---|---|---|---|---|
| taiwan | release_all | 29.37% | 100.00% | 92.61% | 71.99% | 2.2910 |
| taiwan | whole_legacy2 | 3.71% | 62.84% | 71.99% | 71.99% | 1.4398 |
| taiwan | whole_mc1 | 1.33% | 40.42% | 92.61% | 0.00% | 0.9261 |
| taiwan | whole_mc_all | 2.64% | 34.85% | 50.19% | 29.65% | 0.7983 |
| taiwan | rank | 5.13% | 74.18% | 92.54% | 71.67% | 1.6995 |
| taiwan | variance | N/A | 0.00% | 0.00% | 0.00% | 0.0000 |
| taiwan | frequency | 4.96% | 74.02% | 92.47% | 71.44% | 1.6958 |
| taiwan | strength | 7.31% | 47.87% | 72.43% | 29.64% | 1.0968 |
| taiwan | lower_quantile | 7.03% | 46.34% | 71.03% | 27.92% | 1.0616 |
| taiwan | stable_donor | 1.82% | 70.02% | 91.39% | 68.59% | 1.6043 |
| taiwan | stable_conditional | 1.98% | 69.70% | 91.14% | 68.09% | 1.5969 |
| taiwan | stable_both | 1.24% | 68.46% | 90.76% | 66.02% | 1.5683 |
| polish | release_all | 43.73% | 100.00% | 94.16% | 78.47% | 3.0034 |
| polish | whole_legacy2 | 5.07% | 49.69% | 74.62% | 74.62% | 1.4924 |
| polish | whole_mc1 | 1.93% | 31.35% | 94.16% | 0.00% | 0.9416 |
| polish | whole_mc_all | 3.59% | 17.63% | 34.22% | 18.74% | 0.5296 |
| polish | rank | 7.19% | 60.10% | 94.16% | 78.05% | 1.8050 |
| polish | variance | N/A | 0.00% | 0.00% | 0.00% | 0.0000 |
| polish | frequency | 6.58% | 59.69% | 93.87% | 77.24% | 1.7927 |
| polish | strength | 4.91% | 5.73% | 13.32% | 3.17% | 0.1722 |
| polish | lower_quantile | 3.97% | 5.32% | 12.44% | 2.96% | 0.1599 |
| polish | stable_donor | 3.35% | 56.27% | 93.06% | 74.28% | 1.6899 |
| polish | stable_conditional | 3.71% | 56.11% | 92.94% | 73.69% | 1.6853 |
| polish | stable_both | 2.09% | 54.58% | 92.60% | 71.07% | 1.6392 |

Stable-Core both improves coverage relative to the exactly-two-reason whole policy,
but **rank-only and whole top-1 reach more customers**. Rank-only also releases
more reasons while satisfying the 10% empirical budget in each assessed environment.
It is not displaced by the more elaborate Stable-Core rule. Stable-Core has lower
observed reason risk than rank-only; their utilities differ, so neither claim
universal domination nor hide the strong baseline.

At the 10% operating point, every meaningful-target Stable-Core fit selects
`(tau_sign,tau_rank,q)=(0,0,.01)`: it reduces exactly to a positive lower-quantile
filter. Every ranked fit selects `(0,.5,.01)`. The extra sign-support condition is
inactive. This is direct empirical evidence against novelty of the compound rule,
not a reason to invent a more complex rule after assessment.

Crucial limitation: 28.15% of Taiwan and 42.52% of Polish current candidate reasons
already lie outside top-2 **before** verification. Ranked release-all failure is
therefore not a pure revision rate. It is a ranked-claim acceptance diagnostic.
Do not cite the 29.37%/43.73% ranked release-all failures as the original decoupling
phenomenon. Historical within-model top-2 revision results remain the reference.

## Eligibility and shorter-explanation controls

The common subset below requires >=2 current candidates, removing the eligibility
advantage from letting previously ineligible one-reason customers receive output:

| dataset | target | method | n | customer_coverage | reason_risk |
|---|---|---|---|---|---|
| polish | meaningful | rank | 1855 | 100.00% | 3.86% |
| polish | meaningful | stable_both | 1855 | 99.62% | 0.26% |
| polish | meaningful | whole_legacy2 | 1855 | 100.00% | 1.97% |
| polish | ranked | rank | 1855 | 100.00% | 7.44% |
| polish | ranked | stable_both | 1855 | 99.57% | 2.29% |
| polish | ranked | whole_legacy2 | 1855 | 95.09% | 5.07% |
| taiwan | meaningful | rank | 6911 | 100.00% | 2.50% |
| taiwan | meaningful | stable_both | 6911 | 99.74% | 0.28% |
| taiwan | meaningful | whole_legacy2 | 6911 | 100.00% | 1.61% |
| taiwan | ranked | rank | 6911 | 100.00% | 5.42% |
| taiwan | ranked | stable_both | 6911 | 99.65% | 1.35% |
| taiwan | ranked | whole_legacy2 | 6911 | 100.00% | 3.71% |

On the meaningful target, whole top-2 covers 100% of this common subset; Stable-Core
both covers 99.74%/99.62%. Hence its apparent aggregate customer gain over exactly
top-2 is largely candidate eligibility, not improved abstention. On ranked Polish
it retains a smaller real gain over whole top-2, but rank-only still covers 100%.

Paired differences in reason risk: Stable-Core both minus a control with **exactly
the same count on every same customer**; negative is better. Values below are
percentage-point differences, not relative percentage changes:

| dataset | target | baseline | difference_pp | lo_pp | hi_pp |
|---|---|---|---|---|---|
| taiwan | meaningful | stable_both_matched_strength | -0.4710 | -0.5958 | -0.3687 |
| taiwan | meaningful | stable_both_matched_random | -1.3304 | -1.5250 | -1.1300 |
| taiwan | ranked | stable_both_matched_strength | -0.7505 | -0.9505 | -0.5592 |
| taiwan | ranked | stable_both_matched_random | -21.3403 | -22.3305 | -20.2744 |
| polish | meaningful | stable_both_matched_strength | -0.6834 | -0.9173 | -0.4771 |
| polish | meaningful | stable_both_matched_random | -2.2931 | -2.6981 | -1.8843 |
| polish | ranked | stable_both_matched_strength | -2.4516 | -3.0623 | -1.8641 |
| polish | ranked | stable_both_matched_random | -32.3355 | -34.1329 | -30.4824 |

For meaningful reasons, the reductions versus matched strength are 0.47 pp on
Taiwan (95% CI 0.37–0.60 pp reduction) and 0.68 pp on Polish (0.48–0.92 pp).
Thus some **reason identity selection benefit survives matching set size**.
This does not establish superiority to optimally calibrated rank/frequency policies.
Size-matched controls are diagnostics, not lower-cost deployed alternatives: their
sizes are taken from the Stable-Core policy without consulting verification truth.

## Multiple completion families and risk-control limits

Donor-only -> both changes meaningful reason risk from 0.53% to 0.29% on Taiwan,
and 0.54% to 0.24% on Polish. Customer coverage decreases by about 0.57/0.42 pp.
Both-family agreement therefore does not collapse coverage in this study. The
family-specific policies were independently calibrated, so all disagreement/risk
tables must be read with their selected thresholds. Agreement is not worst-case
robustness: both completion families can share unsupported regions.

All **conservative finite-grid Hoeffding policies abstain completely**, at all
three budgets and both targets. The tested sample sizes and bound do not certify
useful coverage. This is a failure of the strong risk-control claim, not evidence
that empirical coverage is zero or that every possible risk bound would fail.
Do not replace the bound or threshold grid after seeing this result.

Stable-Core both releases 0–7 reasons on both datasets. Its meaningful-target
empty fractions are 9.18%/7.36%, including intrinsically candidate-empty records.
Customer any-released-reason failure is 0.65%/0.73%, distinct from micro reason
failure. Per-group tables, coverage>=2, count distributions and family disagreement
are all exported; no successful cases were selected for the headline.

## Unchanged predictive performance

These metrics describe the same frozen predictor on this explanation cohort;
Taiwan values are means over the three folds. They are not the larger historical
4,500-customer test or full mortgage performance. Every explanation policy shares
these predictions; none improves AP through a new risk model.

| dataset | condition | average_precision | roc_auc | brier | log_loss |
|---|---|---|---|---|---|
| polish | mcar10 | 0.6506 | 0.9090 | 0.0448 | 0.1682 |
| polish | mcar30 | 0.5458 | 0.8984 | 0.0523 | 0.1874 |
| taiwan | group_missing | 0.5019 | 0.7292 | 0.1485 | 0.4671 |
| taiwan | mar30 | 0.5104 | 0.7498 | 0.1465 | 0.4595 |
| taiwan | mcar10 | 0.5459 | 0.7636 | 0.1395 | 0.4433 |
| taiwan | mcar30 | 0.5084 | 0.7477 | 0.1449 | 0.4569 |

## Runtime

CPU end-to-end serving on the first 16 fixed MCAR30 customers, one warm-up,
five interleaved repetitions, including current TreeSHAP, completion generation,
completion TreeSHAP and policy scoring. Model loading/fitting excluded. Seconds
are per batch, not an individual request SLA:

| dataset | method | total_seconds_mean | total_seconds_std |
|---|---|---|---|
| polish | rank | 0.0785 | 0.0004 |
| polish | stable_both | 0.7896 | 0.0242 |
| polish | stable_conditional | 0.7133 | 0.0149 |
| polish | stable_donor | 0.0785 | 0.0004 |
| polish | strength | 0.0088 | 0.0001 |
| polish | whole_legacy2 | 0.0785 | 0.0003 |
| taiwan | rank | 0.0841 | 0.0007 |
| taiwan | stable_both | 0.2963 | 0.0012 |
| taiwan | stable_conditional | 0.2230 | 0.0032 |
| taiwan | stable_donor | 0.0848 | 0.0012 |
| taiwan | strength | 0.0094 | 0.0001 |
| taiwan | whole_legacy2 | 0.0850 | 0.0007 |

Both-family Stable-Core is approximately 3.5x/10.1x donor rank-only cost on
Taiwan/Polish. Donor Stable-Core and donor rank-only have essentially the same
attribution cost. Strength-only needs current SHAP but no completion attributions.
No cached-scoring versus uncached-inference comparison is used.

## Figures and reproducibility

Generated under `outputs/stable_core_study/analysis/`, each PNG plus PDF:

- A_reason_risk_coverage: declared operating points, no test-selected threshold.
- B_customer_risk_coverage: >=1-reason coverage versus verified risk.
- C_whole_vs_partial: whole/top-1/rank/partial comparison.
- D_reason_set_sizes: variable cardinality including zero.
- E_failure_by_group: all semantic groups, no group selection.
- F_completion_families: donor, conditional, both.

`metrics.csv` includes all risk budgets, conservative results and per-condition
95% CIs; `paired_differences.csv` uses the same customer resamples. Additional
outputs contain class-wise prediction metrics, common eligibility, pre-existing
rank exclusions, completion truth support and five timing repetitions. Raw/cached
row outputs, figures and models remain ignored; this report is an aggregate summary.

The software audit recomputed **2,009,952 customer-policy decisions** and
**861,408 matched-count comparisons** over 11,964 customer-condition records,
with zero observed/natural cell changes and zero invalid attributions. These are
checks, not independent samples. Test suite: **111 passed**, three pre-existing
SHAP/matplotlib deprecation warnings. Root-run software checks are not peer review.

Decision: retain partial reasons as an **optional operational policy**, reject a
new-algorithm/superior-coverage claim, and do not replace the whole-policy baseline.
The independent mortgage confirmation remains blocked by missing official files.
