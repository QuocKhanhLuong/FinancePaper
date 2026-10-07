# One-query verification headroom: development results

Measured 2026-10-04 (Asia/Ho_Chi_Minh). **NO-GO: stop the proposed
acquisition-method development at its first gate.** A query can recover some
coverage withheld by donor Stable-Core, but the strongest zero-query baselines
already cover every customer with an initial candidate at the declared budgets.
This result does not invalidate prediction–explanation decoupling; it invalidates
the proposed customer-coverage justification for this particular next method.

## What actually ran

The [prospective protocol](VERIFICATION_HEADROOM_PROTOCOL.md), source, masks,
model and unchanged calibrated thresholds were frozen before new query outcomes.
Polish: **500 distinct exact-feature clusters**, reused selector-development data,
MCAR10/MCAR30, 1,000 episodes. Frozen XGBoost, seven semantic groups, historical
raw-logit interventional TreeSHAP, donor K8. Every possible actual single-field
reveal plus STOP: **12,537 query actions, 13,537 states**. No predictor or release
threshold was refitted. Source complete truth is confined to evaluator code;
natural missing values were never restored. The donor pool is training-only.

Taiwan has **no new run**: all 21,000 train/development-pool records are in the
union of the three previously evaluated outer prediction folds. Excluding all
historical assessment IDs leaves zero eligible records. We did not open the old
test/reserve or label a previously evaluated fold as fresh development.
Freddie is **NOT RUN**, because official files have not been supplied.

Freeze timestamp: `2026-10-03T17:14:42.181990+00:00`.
Freeze SHA256: `05f29cb0beed4212efb27757c558e4f3f800e0d67d55fb32cbc4fa6f7a5cc6a4`.
Cohort/masks SHA256: `6fc0d47d9919cf30814a3ebc3389ae82a795f05d9ac2a098984a803efa3670fc`.
These are local reproduction artifacts, not an independent preregistration.

## Primary target and the decisive zero-query comparison

Keep the initial candidate reasons and initially observed attribution summands
fixed. A reason survives when its fully restored grouped contribution remains
above .01 raw logit. The primary risk is failed reasons / released reasons.
No requirement to display two reasons, no causal/correctness claim.

At alpha=10%, the table includes 95% customer-bootstrap intervals. Oracle rows
use true verification labels for action selection and are unattainable bounds.

| Condition | Method | Failed / released | Reason risk [95% CI] | Customer >=1 [95% CI] | Reason coverage | Query fraction |
| --- | --- | --- | --- | --- | --- | --- |
| mcar10 | zero_release_all | 35/1500 | 2.33% [1.61%, 3.16%] | 93.40% [91.20%, 95.40%] | 100.00% | 0.00% |
| mcar10 | zero_top1 | 2/467 | 0.43% [0.00%, 1.08%] | 93.40% [91.20%, 95.40%] | 31.13% | 0.00% |
| mcar10 | zero_rank | 35/1500 | 2.33% [1.61%, 3.16%] | 93.40% [91.20%, 95.40%] | 100.00% | 0.00% |
| mcar10 | zero_stable_donor | 7/1460 | 0.48% [0.14%, 0.85%] | 93.00% [90.80%, 95.00%] | 97.33% | 0.00% |
| mcar10 | oracle_rank_customers_b1 | 35/1500 | 2.33% [1.61%, 3.16%] | 93.40% [91.20%, 95.40%] | 100.00% | 0.00% |
| mcar10 | oracle_stable_donor_customers_b1 | 9/1473 | 0.61% [0.26%, 1.06%] | 93.20% [91.00%, 95.20%] | 98.20% | 2.60% |
| mcar30 | zero_release_all | 79/1428 | 5.53% [4.36%, 6.83%] | 95.00% [93.20%, 97.00%] | 100.00% | 0.00% |
| mcar30 | zero_top1 | 12/475 | 2.53% [1.27%, 4.19%] | 95.00% [93.20%, 97.00%] | 33.26% | 0.00% |
| mcar30 | zero_rank | 79/1428 | 5.53% [4.36%, 6.83%] | 95.00% [93.20%, 97.00%] | 100.00% | 0.00% |
| mcar30 | zero_stable_donor | 10/1314 | 0.76% [0.31%, 1.24%] | 92.20% [89.80%, 94.41%] | 92.02% | 0.00% |
| mcar30 | oracle_rank_customers_b1 | 79/1428 | 5.53% [4.36%, 6.83%] | 95.00% [93.20%, 97.00%] | 100.00% | 0.00% |
| mcar30 | oracle_stable_donor_customers_b1 | 30/1370 | 2.19% [1.44%, 2.95%] | 93.80% [91.80%, 95.80%] | 95.94% | 10.80% |

The elementary retained-claim ceiling is **93.4% for MCAR10 and 95.0% for
MCAR30**: the remaining customers have no initial candidate to preserve.
Top-1 reaches that ceiling without querying, with primary risk 0.43% and 2.53%.
Its upper bootstrap endpoints are 1.08% and 4.19%; these are empirical intervals,
not deployment certificates. Thus the maximum extra customer coverage over this
baseline is **zero at all three declared budgets (5/10/15%)** in this sample.
The same elementary customer bound applies to the ranked sensitivity here.

Top-1 releases fewer reasons, so it is not a reason-coverage dominator. At 10/15%
primary budgets, however, release-all also meets empirical risk and attains
**100% reason coverage**. At MCAR30/5%, release-all fails the budget (5.53%);
do not hide that exception or call pooled success environment-wise control.
Querying may trade computation against number/quality of reasons there, but
the predeclared +5-point customer-coverage hypothesis still fails.

## Separating query value from oracle knowledge

Both b=0 and b=1 oracles know verification labels and may abstain. They must use
the terminal mapping's entire output for their chosen state; no oracle deletion
of individual failing reasons is allowed. Binary optimization proved optimality
for all **96** finite-family solutions. These are exact within the enumerated
action/terminal family, not globally optimal deployable acquisition policies.

The table reports b=1 minus b=0 customer coverage, in **percentage points**.
Intervals resample customers while keeping selected oracle actions fixed; they
are optimistic and do not cover optimization uncertainty.

| Condition | Target | Alpha | Terminal rule | Query gain pp [conditional 95% CI] |
| --- | --- | --- | --- | --- |
| mcar10 | meaningful | 5.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | meaningful | 5.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar10 | meaningful | 10.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | meaningful | 10.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar10 | meaningful | 15.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | meaningful | 15.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar10 | ranked | 5.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | ranked | 5.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar10 | ranked | 10.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | ranked | 10.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar10 | ranked | 15.00% | rank | 0.00 [0.00, 0.00] |
| mcar10 | ranked | 15.00% | stable_donor | 0.20 [0.00, 0.60] |
| mcar30 | meaningful | 5.00% | rank | 1.00 [0.20, 2.00] |
| mcar30 | meaningful | 5.00% | stable_donor | 1.60 [0.60, 2.60] |
| mcar30 | meaningful | 10.00% | rank | 0.00 [0.00, 0.00] |
| mcar30 | meaningful | 10.00% | stable_donor | 1.60 [0.60, 2.60] |
| mcar30 | meaningful | 15.00% | rank | 0.00 [0.00, 0.00] |
| mcar30 | meaningful | 15.00% | stable_donor | 1.60 [0.60, 2.60] |
| mcar30 | ranked | 5.00% | rank | 1.60 [0.60, 2.80] |
| mcar30 | ranked | 5.00% | stable_donor | 2.40 [1.20, 3.80] |
| mcar30 | ranked | 10.00% | rank | 1.00 [0.20, 2.00] |
| mcar30 | ranked | 10.00% | stable_donor | 1.80 [0.80, 3.00] |
| mcar30 | ranked | 15.00% | rank | 0.20 [0.00, 0.80] |
| mcar30 | ranked | 15.00% | stable_donor | 1.80 [0.80, 3.00] |

For meaningful reasons, donor Stable-Core gains only **0.2 pp MCAR10** and
**1.6 pp MCAR30**, below the predeclared 5 pp requirement. Its best customer
coverage remains below the no-query top-1 ceiling. At 10/15% the rank terminal
chooses STOP throughout; there is no coverage or reason-count headroom over
release-all. Ranked-target gains cannot replace the failed primary endpoint.

## Matched-size controls and legitimate changes

For the reason-count-maximizing b=1 donor Stable-Core oracle, strength/random
controls release exactly the same number of reasons for each customer at the
same selected state. Their actions are inherited from the oracle, so these are
identity-selection diagnostics, not deployable acquisition competitors.

| Condition | Target | Selection | Failure [95% CI] | Released count |
| --- | --- | --- | --- | --- |
| mcar10 | meaningful | selected | 0.61% [0.26%, 1.06%] | 1473 |
| mcar10 | meaningful | selected_matched_strength | 0.81% [0.42%, 1.33%] | 1473 |
| mcar10 | meaningful | selected_matched_random | 1.63% [1.04%, 2.30%] | 1473 |
| mcar10 | ranked | selected | 3.65% [2.53%, 4.81%] | 877 |
| mcar10 | ranked | selected_matched_strength | 5.02% [3.74%, 6.45%] | 877 |
| mcar10 | ranked | selected_matched_random | 33.52% [30.38%, 36.63%] | 877 |
| mcar30 | meaningful | selected | 2.19% [1.44%, 2.95%] | 1370 |
| mcar30 | meaningful | selected_matched_strength | 2.85% [2.00%, 3.75%] | 1370 |
| mcar30 | meaningful | selected_matched_random | 4.53% [3.36%, 5.69%] | 1370 |
| mcar30 | ranked | selected | 8.72% [7.10%, 10.47%] | 895 |
| mcar30 | ranked | selected_matched_strength | 11.51% [9.69%, 13.38%] | 895 |
| mcar30 | ranked | selected_matched_random | 34.53% [31.65%, 37.49%] | 895 |

Some identity-selection benefit survives matched counts; it does not establish
coverage superiority or a novel acquisition algorithm. The optimizer maximizes
count subject to alpha, not minimum failure; querying can increase failure
while staying inside the empirical constraint. Do not call it uniformly safer.

Across candidate actions, 23/500 MCAR10 and 58/500 MCAR30 customers have an
initially nonpositive group become positive. These new claims are excluded from
primary gains. Counting them as preserved reasons would change the estimand.
Retractions of initially positive but ultimately failing claims are legitimate.
The query-order file retains these diagnostics and full selected-action arrays
locally. Initial own-SHAP importance/variance, realized prediction change and
rank-support change were compared with realized valid-candidate count. The
utility is mostly tied under this fixed universe (only 4/25 customers have
nonconstant utility versus own-importance at MCAR10/30); no ranking-superiority
claim is supportable. Expected entropy VOI, EDDI-style information gain and a
trained policy were **not implemented**, because the headroom gate failed.

## The phenomenon still appears in development

All 500 customers per condition are included below, including episodes with no
query available. Failure means at least one initially meaningful positive group
loses that property after full artificial restoration. This is not the historical
strict feature-top-3 event, nor the ranked target.

| Condition | Prediction shift cutoff | Prediction-stable N | Region B N | P(reason failure \| stable prediction) |
| --- | --- | --- | --- | --- |
| mcar10 | 0.01 | 375 | 18 | 4.80% |
| mcar10 | 0.02 | 416 | 22 | 5.29% |
| mcar10 | 0.05 | 451 | 27 | 5.99% |
| mcar30 | 0.01 | 279 | 32 | 11.47% |
| mcar30 | 0.02 | 353 | 48 | 13.60% |
| mcar30 | 0.05 | 415 | 55 | 13.25% |

This distinguishes two claims: some reasons fail despite a stable prediction,
yet a simple strongest-reason output can already provide broad useful coverage.
The former does not entail the need for the proposed acquisition algorithm.
These reused development results are not additional untouched external evidence.

## Compute, checks and artifacts

| Condition | States incl. STOP | Actual query options | Attributed rows incl. restoration | Completion seconds | State attribution seconds | Total seconds |
| --- | --- | --- | --- | --- | --- | --- |
| mcar10 | 3603 | 3103 | 32927 | 0.79 | 16.12 | 17.39 |
| mcar30 | 9934 | 9434 | 89906 | 2.27 | 46.36 | 49.51 |

Total approximately **66.90 seconds** on local CPU for evidence generation.
The 122,833 attributed rows include 1,000 full-restoration rows. This is one
end-to-end audit timing, **not** a repeated-latency benchmark; it does not estimate
the much more expensive proposed 8x8 hypothetical lookahead inference cost.
TreeSHAP dominates these timings. No GPU was required.

The independent checker verified source hashes, cohort disjointness, exhaustive
query coverage, natural-mask preservation, fixed attribution universe, all 96
oracle solutions/risk constraints, matched-set counts during reporting, and 24
independently recomputed attribution states. No mismatch was found. **121 tests
passed**, including brute-force comparisons of the optimizer with small exact
fixtures. The first unrestricted-thread full-suite run hung inside an existing
PyTorch/OpenMP sqrt call and was terminated; the complete suite passed with
OMP/MKL/VECLIB threads set to one. The financial runner already fixes these
variables, so no experimental definition/result was changed to address this
test-runtime issue.

```bash
uv run --frozen --extra temporal python scripts/run_verification_headroom.py freeze
uv run --frozen --extra temporal python scripts/run_verification_headroom.py evaluate
uv run --frozen --extra temporal python scripts/run_verification_headroom.py report
uv run --frozen --extra temporal python scripts/audit_verification_headroom.py
uv run --frozen --extra temporal python scripts/write_verification_headroom_report.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal pytest -q
```

Freeze/evaluate refuse overwriting completed studies. Local aggregate CSVs:
`analysis/metrics.csv`, `paired_query_gain.csv`, `elementary_bounds.csv`,
`solver_receipts.csv`; customer diagnostics remain in ignored `outputs/`.
`analysis/headroom.png` and `.pdf` compare primary alpha=10% coverage/failure.
Audit/analysis receipts include artifact hashes. No raw, processed rows, model
artifacts, or customer-level outputs are committed.

## Research and novelty decision

**NO-GO for one-query retained-reason acquisition as the next method contribution.**
Do not tighten alpha, change meaningful to ranked, add new reason candidates,
or drop top-1 to create apparent headroom. Do not implement the full nested
lookahead or a conditional-imputer acquisition grid after this failed gate.

The [nearest-work reassessment](NOVELTY_REASSESSMENT_2026.md) remains in force:
selective attribution, SHAP uncertainty, and explanation-guided acquisition
already exist; retargeting standard value of information is not automatically
new. This audit adds a useful negative result and a reason-coverage ceiling
diagnostic, **not enough evidence for a new algorithm paper**. There is still an
evaluation-paper story about verification targets, predictive/explanatory
decoupling, semantic grouping and simple policy baselines, with narrower claims
than a novel safe-explanation method. ESWA remains **promising but incomplete
for an evaluation paper; unsupported as a new-method claim**.

**One next action:** prepare an evaluation-paper manuscript outline with an
explicit claim–evidence table incorporating the negative Stable-Core and
acquisition results. Do not manufacture another method to satisfy a novelty
label. Large-scale confirmation remains unavailable until official data arrive.
