# External validation results — Polish corporate bankruptcy

Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.

## Provenance and independent assessment

[Official UCI source](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
[DOI10.24432/C5F600](https://doi.org/10.24432/C5F600), CC BY4.0, creator Sebastian
Tomczak. Download/license verified before use. `5year.arff` is 5,910 statements,
64 numerical ratios, 410 bankruptcy positives, 5,500 negatives, **one-year horizon**.
The UCI aggregate header is not this file's schema. ARFF SHA256:
`cb3f6f250ac46bd8d18e9a222f489fe8ee3e396fcec18959f5a0ef8e8169b2fc`.
Archive SHA256: `17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9`.

4,666 natural missing cells, 2,879 records with natural missingness, 3,031 complete
records, no infinite values. Attr37 is missing in2,548 statements. There are60
duplicate-feature rows; all exact duplicates were grouped before splitting.
Partitions (rows/positives): train2961/205; selector590/36; default-cal294/17;
revision-cal293/20; release-cal590/40; assessment1182/92. Assessment has627 originally
complete and555 naturally incomplete statements. No company ID means residual
identity or temporal overlap cannot be ruled out. No five-file pooling.

Donors are the1,544 complete **predictor-training** statements, never the target
or another partition. Only artificial MCAR10/30 cells have verifiable truth.
Natural unknowns remain NaN in partial, completion and restored raw records and
are handled by the frozen train-median transform. There is no MAR external test.

The [protocol](EXTERNAL_DATASET_PROTOCOL.md), all group assignments, K candidates,
model/selector recipes and calibration families were frozen before the external
assessment marker. No method or threshold was changed afterward. One external
split supports transfer of the phenomenon, not a general consumer-default claim.

## Table E: predictive controls, separate target

Fixed .5 classification threshold; raw and calibrated results are both retained.
The main question is explanation revision, not external classifier superiority.

| model | condition | probability | roc_auc | average_precision | recall | f1 | brier | log_loss | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | complete | prob_raw | 0.8872 | 0.5438 | 0.2065 | 0.3220 | 0.0525 | 0.2363 | 0.0382 |
| lr | complete | prob_calibrated | 0.8872 | 0.5438 | 0.2500 | 0.3740 | 0.0526 | 0.2407 | 0.0261 |
| lr | mcar10 | prob_raw | 0.8538 | 0.4698 | 0.1630 | 0.2564 | 0.0562 | 0.2377 | 0.0276 |
| lr | mcar10 | prob_calibrated | 0.8538 | 0.4698 | 0.2065 | 0.3140 | 0.0564 | 0.2423 | 0.0267 |
| lr | mcar30 | prob_raw | 0.7834 | 0.3155 | 0.1304 | 0.2124 | 0.0638 | 0.2790 | 0.0193 |
| lr | mcar30 | prob_calibrated | 0.7834 | 0.3155 | 0.1304 | 0.2124 | 0.0644 | 0.2920 | 0.0291 |
| xgb | complete | prob_raw | 0.9206 | 0.6806 | 0.4022 | 0.5401 | 0.0416 | 0.1506 | 0.0164 |
| xgb | complete | prob_calibrated | 0.9206 | 0.6806 | 0.3152 | 0.4640 | 0.0433 | 0.1603 | 0.0279 |
| xgb | mcar10 | prob_raw | 0.9090 | 0.6506 | 0.3913 | 0.5333 | 0.0436 | 0.1590 | 0.0161 |
| xgb | mcar10 | prob_calibrated | 0.9090 | 0.6506 | 0.3478 | 0.5000 | 0.0448 | 0.1682 | 0.0259 |
| xgb | mcar30 | prob_raw | 0.8984 | 0.5458 | 0.2283 | 0.3415 | 0.0499 | 0.1754 | 0.0186 |
| xgb | mcar30 | prob_calibrated | 0.8984 | 0.5458 | 0.1957 | 0.3051 | 0.0523 | 0.1874 | 0.0345 |

Post-hoc default calibration **worsens** XGB Brier/log loss in all three conditions:
MCAR30 Brier .0499 raw -> .0523 calibrated. Calibration used only294 statements
with17 positives and was not refit after this result. No calibration-improvement
claim is made. Classwise precision/recall/F1/support are in E_external_prediction.csv.

## Transfer of decoupling, raw |delta p| <= .02

| variant | condition | eligible | A | B | C | D | revision_among_stable | overall_revision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature3 | mcar10 | 1182 | 828 | 141 | 153 | 60 | 0.1455 | 0.1701 |
| feature3 | mcar30 | 1180 | 522 | 273 | 214 | 171 | 0.3434 | 0.3763 |
| group1 | mcar10 | 1118 | 869 | 37 | 192 | 20 | 0.0408 | 0.0510 |
| group1 | mcar30 | 1108 | 641 | 83 | 333 | 51 | 0.1146 | 0.1209 |
| group2 | mcar10 | 929 | 674 | 47 | 180 | 28 | 0.0652 | 0.0807 |
| group2 | mcar30 | 926 | 467 | 85 | 283 | 91 | 0.1540 | 0.1901 |
| group3 | mcar10 | 712 | 480 | 37 | 157 | 38 | 0.0716 | 0.1053 |
| group3 | mcar30 | 678 | 274 | 61 | 248 | 95 | 0.1821 | 0.2301 |

All three cutoffs and rank gaps are in the [group validation report](DOMAIN_REASON_VALIDATION_RESULTS.md).
At the widest .02 logit rank gap, grouped top2 MCAR30 still has60/552 stable cases
with revision (10.87%). The coarse financial groups attenuate, but do not erase,
the external phenomenon. Group2 MCAR30 RegionB at the frozen event is15.40%
[12.30,18.41] among stable eligible statements.

## Detectors, MCAR30

Feature-top3 target:

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| prediction_only | 0.5662 | 0.4562 | 0.2358 | 0.6667 |
| learned_selector | 0.7719 | 0.6380 | 0.1924 | 0.5623 |
| prediction_variance | 0.5061 | 0.3900 | 0.2524 | 0.7086 |
| missing_fraction | 0.5665 | 0.4173 | 0.2336 | 0.6596 |
| attribution_variance | 0.5370 | 0.4058 | 0.2423 | 0.6811 |
| rank_instability | 0.9511 | 0.9224 | 0.1129 | 0.3663 |
| sign_instability | 0.5692 | 0.4628 | 0.2282 | 0.6627 |
| mc8 | 0.9514 | 0.9222 | 0.0915 | 0.2933 |

Group-top2 target:

| Detector | roc_auc | average_precision | brier | log_loss |
| --- | --- | --- | --- | --- |
| prediction_only | 0.6059 | 0.2622 | 0.1506 | 0.4758 |
| learned_selector | 0.5947 | 0.2559 | 0.1516 | 0.4796 |
| prediction_variance | 0.5231 | 0.1990 | 0.1564 | 0.4952 |
| missing_fraction | 0.5423 | 0.2225 | 0.1537 | 0.4856 |
| attribution_variance | 0.5609 | 0.2168 | 0.1544 | 0.4858 |
| rank_instability | 0.9240 | 0.8219 | 0.0666 | 0.4189 |
| sign_instability | 0.6148 | 0.3682 | 0.1368 | 0.4459 |
| mc8 | 0.9513 | 0.8580 | 0.0622 | 0.2125 |

Paired AP/AUROC differences (MC8 minus prediction-only), 1000 duplicate-cluster draws:

| variant | metric | difference | low | high |
| --- | --- | --- | --- | --- |
| feature3 | AP | 0.4660 | 0.4201 | 0.5081 |
| feature3 | AUROC | 0.3852 | 0.3518 | 0.4198 |
| group1 | AP | 0.6140 | 0.5369 | 0.6785 |
| group1 | AUROC | 0.4318 | 0.3715 | 0.4891 |
| group2 | AP | 0.5957 | 0.5294 | 0.6518 |
| group2 | AUROC | 0.3454 | 0.2929 | 0.3939 |
| group3 | AP | 0.5348 | 0.4599 | 0.5993 |
| group3 | AUROC | 0.3823 | 0.3262 | 0.4390 |

Rank instability ties MC on the feature target; this is an important simple
completion-explanation comparator, not a negative result to hide. The learned
selector remains weaker. Grouped selector results test a **fixed feature-target
selector recalibrated** to grouped labels, not a newly trained group-optimal selector.

## Natural-missingness sensitivity

| variant | condition | has_natural_missing | n | eligible | revision | stable_n | B | B_given_stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature3 | mcar30 | False | 627 | 625 | 0.3936 | 444 | 161 | 0.3626 |
| feature3 | mcar30 | True | 555 | 555 | 0.3568 | 351 | 112 | 0.3191 |
| group1 | mcar30 | False | 627 | 574 | 0.1202 | 393 | 49 | 0.1247 |
| group1 | mcar30 | True | 555 | 534 | 0.1217 | 331 | 34 | 0.1027 |
| group2 | mcar30 | False | 627 | 487 | 0.1930 | 311 | 43 | 0.1383 |
| group2 | mcar30 | True | 555 | 439 | 0.1868 | 241 | 42 | 0.1743 |
| group3 | mcar30 | False | 627 | 342 | 0.2544 | 186 | 37 | 0.1989 |
| group3 | mcar30 | True | 555 | 336 | 0.2054 | 149 | 24 | 0.1611 |

Natural unknowns do not eliminate RegionB: grouped top2 is43/311 (13.83%) in
originally complete and42/241 (17.43%) in naturally incomplete stable statements.
This is observational stratification, not a causal missingness effect. Restoration
is partial: it returns to the original natural-missing record, not all-information
truth. Complete-case donors may be a selected population, and shared financial
ratio constraints can be broken by mixing query and donor subsets. Neither is
repaired after assessment.

## External success/failure gate

All four bounded transfer criteria are met: RegionB persists; prediction-only
detection is weaker; MC discriminates revision; useful selected coverage remains
at10/15% declared budgets. This is **qualified transfer**, not universal risk control:
MCAR only, one split/target, dictionary-grounded groups without human validation,
calibration deterioration for default, no useful robust5% coverage, and no
independent-company calibration guarantee. Taiwanese and Polish prediction metrics
are never pooled. Figure06 presents the two targets separately.
