# Monte Carlo K ablation

Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.

## Fixed method and Table B

K1/2/4 are prefixes of the original K8 stream; K16 appends an independent eight
draws. Frozen predictor, same customers/masks/TreeSHAP background/current reasons.
The historical K8 scores were checked exactly in all old Taiwan environments.
Default calibration is unchanged. New revision calibration uses the declared
four-environment mixture, so Brier/log-loss need not equal the old five-environment
report (e.g. Taiwan MCAR30 MC8 Brier .0882 here versus .0867 historically).

Taiwan MCAR30, feature top3: fold mean ± SD, not three independent studies.

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| prediction_only | 0.5728 ± 0.0262 | 0.2192 ± 0.0135 | 0.1464 ± 0.0049 | 0.4675 ± 0.0117 |
| learned_selector | 0.7963 ± 0.0299 | 0.4574 ± 0.0541 | 0.1209 ± 0.0062 | 0.3832 ± 0.0224 |
| prediction_variance | 0.5360 ± 0.0140 | 0.1996 ± 0.0072 | 0.1473 ± 0.0048 | 0.4712 ± 0.0106 |
| missing_fraction | 0.5330 ± 0.0137 | 0.1948 ± 0.0099 | 0.1469 ± 0.0046 | 0.4698 ± 0.0099 |
| attribution_variance | 0.5502 ± 0.0394 | 0.2043 ± 0.0227 | 0.1468 ± 0.0049 | 0.4690 ± 0.0117 |
| rank_instability | 0.8982 ± 0.0027 | 0.7171 ± 0.0417 | 0.0977 ± 0.0039 | 0.3163 ± 0.0179 |
| sign_instability | 0.5319 ± 0.0231 | 0.2247 ± 0.0275 | 0.1447 ± 0.0043 | 0.4653 ± 0.0099 |
| mc8 | 0.9000 ± 0.0001 | 0.7199 ± 0.0431 | 0.0882 ± 0.0016 | 0.2905 ± 0.0037 |

## Table C: all K, MCAR30

Taiwan feature top3:

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| mc1 | 0.7548 ± 0.0092 | 0.4472 ± 0.0266 | 0.1062 ± 0.0007 | 0.3631 ± 0.0052 |
| mc2 | 0.8287 ± 0.0127 | 0.5733 ± 0.0311 | 0.0922 ± 0.0023 | 0.3186 ± 0.0086 |
| mc4 | 0.8752 ± 0.0072 | 0.6567 ± 0.0307 | 0.0893 ± 0.0005 | 0.2990 ± 0.0050 |
| mc8 | 0.9000 ± 0.0001 | 0.7199 ± 0.0431 | 0.0882 ± 0.0016 | 0.2905 ± 0.0037 |
| mc16 | 0.9129 ± 0.0044 | 0.7578 ± 0.0452 | 0.0865 ± 0.0022 | 0.2895 ± 0.0063 |

Taiwan group top2:

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| mc1 | 0.7617 ± 0.0509 | 0.3927 ± 0.0552 | 0.0517 ± 0.0157 | 0.2003 ± 0.0561 |
| mc2 | 0.8345 ± 0.0393 | 0.5140 ± 0.0441 | 0.0460 ± 0.0140 | 0.1766 ± 0.0491 |
| mc4 | 0.8869 ± 0.0585 | 0.6147 ± 0.0598 | 0.0439 ± 0.0156 | 0.1631 ± 0.0557 |
| mc8 | 0.9207 ± 0.0437 | 0.6926 ± 0.0344 | 0.0428 ± 0.0155 | 0.1507 ± 0.0542 |
| mc16 | 0.9464 ± 0.0264 | 0.7633 ± 0.0488 | 0.0408 ± 0.0146 | 0.1400 ± 0.0490 |

Polish feature top3 (one fixed assessment, no invented seed SD):

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| mc1 | 0.8475 | 0.7361 | 0.1215 | 0.4111 |
| mc2 | 0.9085 | 0.8328 | 0.1005 | 0.3378 |
| mc4 | 0.9395 | 0.8987 | 0.0926 | 0.3015 |
| mc8 | 0.9514 | 0.9222 | 0.0915 | 0.2933 |
| mc16 | 0.9615 | 0.9366 | 0.0874 | 0.2765 |

Polish group top2:

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| mc1 | 0.8228 | 0.5846 | 0.0862 | 0.3091 |
| mc2 | 0.8854 | 0.7134 | 0.0704 | 0.2549 |
| mc4 | 0.9356 | 0.8129 | 0.0633 | 0.2183 |
| mc8 | 0.9513 | 0.8580 | 0.0622 | 0.2125 |
| mc16 | 0.9647 | 0.8917 | 0.0508 | 0.2129 |

All conditions, group1/2/3, raw/calibrated Brier and log loss are retained in
`B_detector_metrics.csv` / `C_K_ablation.csv`; the raw MC fraction is a completion
sensitivity estimate, not a calibrated posterior. At K1, sample prediction variance
is necessarily zero, so the prediction-variance comparator stays at its frozen K8
budget rather than pretending equal uncertainty information.

## Smallest useful K

The **predeclared** near-reference criterion was 95% of K16's AP gain above event
prevalence, averaged over Taiwan MCAR10/30/MAR30, within folds. Results:

| method | mean | min | max |
| --- | --- | --- | --- |
| mc1 | 0.4532 | 0.3554 | 0.5396 |
| mc16 | 1.0000 | 1.0000 | 1.0000 |
| mc2 | 0.6586 | 0.5264 | 0.7707 |
| mc4 | 0.8085 | 0.7584 | 0.8547 |
| mc8 | 0.9318 | 0.9105 | 0.9564 |

No cheaper K meets that criterion: K8 captures **93.18%**, K4 **80.85%**.
K16 is the smallest tested budget meeting its own reference criterion; that is
not evidence of convergence. K4 already detects useful signal, but “95% of the
benefit at a fraction of the cost” would be false here. Keep **K8 as the frozen
operational reference** and describe it as a budget compromise, not an optimum.
External performance does not select K. Increasing K also does not repair a
misspecified donor distribution, and Brier/log loss need not improve monotonically.

## End-to-end timing, five repetitions after one warm-up

Apple M4 Pro, 24 GiB unified memory, CPU single-thread inference. Persistent
model/background/explainer already loaded; includes transforms, current SHAP,
donors, completion predictions/SHAP, score and calibration/policy call. Excludes
disk loading. Repetitions measure timing variation on a fixed batch, not customer
population latency quantiles. Single-record results use one fixed diagnostic case.

| batch | method | latency_ms | relative_K1 | SHAP_calls | prediction_calls | completion_calls |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.733 ± 0.146 | 1.0000 | 2 | 2 | 1 |
| 1 | 2 | 2.378 ± 0.122 | 1.3719 | 3 | 3 | 1 |
| 1 | 4 | 3.888 ± 0.193 | 2.2427 | 5 | 5 | 1 |
| 1 | 8 | 6.665 ± 0.130 | 3.8450 | 9 | 9 | 1 |
| 1 | 16 | 12.345 ± 0.146 | 7.1220 | 17 | 17 | 1 |
| 1 | learned | 3.316 ± 0.301 | 1.9132 | 1 | 9 | 1 |
| 128 | 1 | 159.642 ± 0.700 | 1.0000 | 2 | 2 | 1 |
| 128 | 2 | 229.371 ± 0.734 | 1.4368 | 3 | 3 | 1 |
| 128 | 4 | 368.810 ± 1.495 | 2.3102 | 5 | 5 | 1 |
| 128 | 8 | 648.082 ± 2.913 | 4.0596 | 9 | 9 | 1 |
| 128 | 16 | 1205.498 ± 4.448 | 7.5512 | 17 | 17 | 1 |
| 128 | learned | 97.374 ± 0.596 | 0.6100 | 1 | 9 | 1 |

Timing breakdown (batch128, seconds; each stage mean ± SD):

| method | completion | transform | prediction | shap | scoring |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0208 ± 0.0004 | 0.0011 ± 0.0000 | 0.0004 ± 0.0000 | 0.1372 ± 0.0007 | 0.0001 ± 0.0000 |
| 2 | 0.0211 ± 0.0008 | 0.0018 ± 0.0001 | 0.0006 ± 0.0001 | 0.2057 ± 0.0009 | 0.0002 ± 0.0000 |
| 4 | 0.0210 ± 0.0002 | 0.0030 ± 0.0001 | 0.0011 ± 0.0001 | 0.3433 ± 0.0014 | 0.0004 ± 0.0000 |
| 8 | 0.0215 ± 0.0005 | 0.0052 ± 0.0002 | 0.0020 ± 0.0001 | 0.6187 ± 0.0026 | 0.0007 ± 0.0001 |
| 16 | 0.0217 ± 0.0008 | 0.0099 ± 0.0003 | 0.0036 ± 0.0001 | 1.1690 ± 0.0044 | 0.0013 ± 0.0001 |
| learned | 0.0211 ± 0.0004 | 0.0050 ± 0.0001 | 0.0017 ± 0.0001 | 0.0682 ± 0.0006 | 0.0014 ± 0.0001 |

Explicit batched SHAP calls are K+1 (current plus completions), completion sampler
calls one; K16 uses two eight-draw RNG blocks inside that call. TreeSHAP's own
additivity work is included in SHAP time. Most cost is SHAP, not the revision
head arithmetic. The generic selector still needs current SHAP and eight cheap
prediction draws. Its approximately .097 s/batch128 is a real deployment advantage
over MC4 .369 s and MC8 .648 s, with weaker detection. It is retained visibly.

Coverage at alpha .05/.10/.15 for **each** K is in `D_policy_intervals.csv` (pooled,
empirical and conservative; all customer-level CIs); no test-picked threshold or
oracle “coverage at risk” is reported as a deployable rule. Figure03 shows detection
and runtime; Figure04 uses whole-score ties for descriptive risk–coverage curves.

## Why the score works, and its limits

MC8 and rank instability are almost the same signal (Spearman .998 Taiwan and
.997 Polish feature MCAR30). Within-fold AP difference from rank instability on
Taiwan is .0028 [−.0010,.0082]; Polish feature difference is −.0002 [−.0024,.0019].
**No general superiority over rank instability is established.** Both need the
same completion explanations. MC adds substantial detection over scalar SHAP
variance/predictive uncertainty, not a novel uncertainty mechanism. See the
[failure audit](MONTE_CARLO_FAILURE_ANALYSIS.md).

Use `paired_fold_mean_differences.csv` for comparisons to fold-mean tables.
`paired_AP_differences.csv` is explicitly a pooled cross-fold score-ranking
diagnostic; differing fold calibrators can make its AP difference different.
Do not use that pooled difference to claim an algorithmic gain absent within folds.
