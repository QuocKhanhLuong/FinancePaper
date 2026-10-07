# Completion Monte Carlo: mechanism and failure audit

Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.

## What is actually measured

Correlation and fixed-missing-count detection are in `MC_correlations.csv` and
`fixed_missing_count_detection.csv`. MC is an operational any-reason revision
frequency; rank instability is the average fraction of reasons exiting. Their
near-equivalence is expected when most events involve one rank exit. Scalar
attribution variance can be large far from any decision boundary, while a small
near-tie shift can change top-k membership; target alignment explains the large
gain over variance. This is a descriptive mechanism, not causal identification.

The current score does not prove correct reasons. Rank-instability AP is
statistically indistinguishable from MC in Taiwan feature/group2 and Polish
feature-level comparisons. Polish group2 gives a small MC advantage; do not
generalize it into algorithmic novelty. Group-conditioned means do not identify
causal importance because multiple groups may be missing together.

## Support and validity

Donor IDs were independently checked against predictor-training pools. No target,
hidden query truth or assessment row enters donor selection. Observed values and
natural NaNs were checked at every draw. Taiwan categorical values come from
valid training donor codes; numeric completed values come from actual train rows,
not extrapolation. This ensures marginal validity, **not accounting-consistent
joint query/donor ratios or a calibrated conditional distribution**. Eight draws
typically use about7.0–7.2 distinct donors; diversity alone is not coverage.

At K8 MCAR30, numerical truth is inside the sampled min/max for81.02% of Taiwan
artificial numeric cells; true category appears for94.74% of categorical cells.
Polish numeric inclusion is75.09%. These are descriptive empirical supports,
not confidence intervals; cell counts are not independent-sample inference.
Truth-percentile values and all support checks are exported in a
**verification-only** table. Natural unknowns never contribute to this audit.

## Failure bins, not release decisions

High MC>=.5 with no verified feature revision: Taiwan MCAR30 n91; Polish n58.
Low MC<=.125 with revision: Taiwan n64; Polish n26. Definitions were fixed before
outcomes; they are not optimized classification thresholds. Near-tie fraction
(current third/fourth gap<=.01) is about56%/31% for Taiwan/Polish high-score false
positives, so **not all** false positives are ties. For low-score false negatives,
sampled truth support is lower (Taiwan about79%, Polish66%); consistent with missed
hidden regions, but not proof of the source of every error. No donor tuning follows.

## Deterministic representative cases inspected

The export contains first-five record IDs in each declared case bin/condition/fold,
280 cases total. These examples are the lowest IDs for each MCAR30 type, not
hand-picked successes. IDs are dataset row audit indices, not identities.

| Dataset / row | Type | Raw probability partial -> restored | MC8 | Inspection |
|---|---|---|---|---|
| Taiwan55, fold2 | High MC, no revision | .619932 -> .620146 | .75 | Third/fourth margin .00464; displayed scores barely move; all true hidden fields supported by sampled range/category. Broad hypothetical changes need not occur in the one realized verification. |
| Taiwan599, fold0 | Low MC, revision | .155584 -> .655144 | 0 | True hidden September PAY_0=2, all eight donor PAY_0=0. Missed critical category despite support for six of seven hidden fields and eight distinct donors. |
| Taiwan162, fold0 | Stable prediction, revised reasons | .366477 -> .349182 | .625 | BILL_AMT1 is an observed reason whose contribution/ranking changes when other bills are restored; delta p .01730. |
| Taiwan76, fold0 | Prediction shifts, stable reasons | .273560 -> .343218 | .125 | PAY_3, LIMIT_BAL, PAY_AMT6 stay positive/top3 despite delta p .06966. |
| Polish92 | High MC, no revision | .013761 -> .002898 | .75 | Margin .00610; true verification preserves Attr16/23/21 reasons despite many risky possible completions. |
| Polish442 | Low MC, revision | .237042 -> .289145 | .125 | True hidden net-profit/asset ratio −.26989 lies below sampled minimum −.22076; debt/equity-related reasons change rank. Seven distinct donors do not cover that region. |
| Polish2 | Stable prediction, revised reasons | .016104 -> .012474 | 1 | Delta p .00363; Attr10 rank changes near a .00528 boundary. Illustrates why semantic/tolerance checks matter. |
| Polish24 | Prediction shifts, stable reasons | .568735 -> .356223 | 0 | Attr41/39/35 remain the reason set despite delta p .21251; prediction and reason stability differ in both directions. |

Full before/restored scores, hidden fields, true values, sampled extrema and
donor diversity are in `representative_cases.csv`. Observed case associations
do not establish a causal explanation for the financial outcome.

## Audit receipt

Independent root recomputation checked66 cache batches,43,965 current rows,
703,440 completion rows,295,365 event rows and6,891,264 policy-release decisions.
No invalid current, restored or completion attributions were found; donor/source
provenance and artificial-only restoration passed. All results retain the original
frozen TreeSHAP estimand. Cross-explainer invariance is **not** established; prior
GRU IG/reference/permutation sensitivity remains a limitation in the historical
metric audit, not new confirmation here.
