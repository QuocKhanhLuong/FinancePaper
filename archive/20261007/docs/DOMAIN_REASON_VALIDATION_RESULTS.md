# Domain-grouped reason validation results

Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.

## Verdict

**Keep the phenomenon, narrow its magnitude and operational interpretation.**
Grouping removes much feature-level turnover, but Region B remains under every
reported group size. These are model-reason changes, not human-validated errors.
Groups/tolerances were specified in [DOMAIN_REASON_GROUPS.md](DOMAIN_REASON_GROUPS.md).
Signed observed-member aggregation is primary; whole-group sums are a separate
sensitivity including imputed contributions. No metric replaces the old event.

## Table A: XGBoost, raw probability shift <= .02

Rates below are fractions. Eligibility differs by definition; denominators must
not be suppressed. Taiwan includes MCAR10/30/MAR30; Polish only MCAR10/30.

| dataset | variant | condition | eligible | stable_n | B | revision_among_stable | overall_revision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | feature3 | mar30 | 1970 | 1040 | 72 | 0.0692 | 0.1345 |
| taiwan | feature3 | mcar10 | 2214 | 1657 | 101 | 0.0610 | 0.0921 |
| taiwan | feature3 | mcar30 | 2009 | 898 | 116 | 0.1292 | 0.1792 |
| taiwan | group1 | mar30 | 2229 | 1138 | 25 | 0.0220 | 0.0435 |
| taiwan | group1 | mcar10 | 2252 | 1694 | 25 | 0.0148 | 0.0213 |
| taiwan | group1 | mcar30 | 2210 | 1011 | 39 | 0.0386 | 0.0434 |
| taiwan | group2 | mar30 | 1710 | 814 | 35 | 0.0430 | 0.0854 |
| taiwan | group2 | mcar10 | 1838 | 1365 | 30 | 0.0220 | 0.0310 |
| taiwan | group2 | mcar30 | 1638 | 727 | 42 | 0.0578 | 0.0818 |
| taiwan | group3 | mar30 | 965 | 416 | 21 | 0.0505 | 0.0829 |
| taiwan | group3 | mcar10 | 1153 | 844 | 28 | 0.0332 | 0.0451 |
| taiwan | group3 | mcar30 | 881 | 349 | 26 | 0.0745 | 0.0976 |
| polish | feature3 | mcar10 | 1182 | 969 | 141 | 0.1455 | 0.1701 |
| polish | feature3 | mcar30 | 1180 | 795 | 273 | 0.3434 | 0.3763 |
| polish | group1 | mcar10 | 1118 | 906 | 37 | 0.0408 | 0.0510 |
| polish | group1 | mcar30 | 1108 | 724 | 83 | 0.1146 | 0.1209 |
| polish | group2 | mcar10 | 929 | 721 | 47 | 0.0652 | 0.0807 |
| polish | group2 | mcar30 | 926 | 552 | 85 | 0.1540 | 0.1901 |
| polish | group3 | mcar10 | 712 | 517 | 37 | 0.0716 | 0.1053 |
| polish | group3 | mcar30 | 678 | 335 | 61 | 0.1821 | 0.2301 |

## Prediction cutoffs and confidence intervals

All .01/.02/.05 cutoffs are reported, not selected. Primary group top-2 illustration:

| dataset | condition | cutoff | stable_n | B | B_given_stable | low | high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | mar30 | 0.0100 | 552 | 18 | 0.0326 | 0.0177 | 0.0482 |
| taiwan | mar30 | 0.0200 | 814 | 35 | 0.0430 | 0.0295 | 0.0569 |
| taiwan | mar30 | 0.0500 | 1244 | 73 | 0.0587 | 0.0459 | 0.0716 |
| taiwan | mcar10 | 0.0100 | 1120 | 18 | 0.0161 | 0.0090 | 0.0240 |
| taiwan | mcar10 | 0.0200 | 1365 | 30 | 0.0220 | 0.0146 | 0.0302 |
| taiwan | mcar10 | 0.0500 | 1670 | 42 | 0.0251 | 0.0179 | 0.0323 |
| taiwan | mcar30 | 0.0100 | 451 | 19 | 0.0421 | 0.0256 | 0.0616 |
| taiwan | mcar30 | 0.0200 | 727 | 42 | 0.0578 | 0.0414 | 0.0751 |
| taiwan | mcar30 | 0.0500 | 1200 | 84 | 0.0700 | 0.0556 | 0.0844 |
| polish | mcar10 | 0.0100 | 629 | 42 | 0.0668 | 0.0461 | 0.0879 |
| polish | mcar10 | 0.0200 | 721 | 47 | 0.0652 | 0.0461 | 0.0839 |
| polish | mcar10 | 0.0500 | 824 | 60 | 0.0728 | 0.0554 | 0.0911 |
| polish | mcar30 | 0.0100 | 410 | 57 | 0.1390 | 0.1051 | 0.1722 |
| polish | mcar30 | 0.0200 | 552 | 85 | 0.1540 | 0.1230 | 0.1841 |
| polish | mcar30 | 0.0500 | 724 | 129 | 0.1782 | 0.1498 | 0.2062 |

Full A/B/C/D counts and percentages for **all** variants/cutoffs/rank gaps are in
`analysis/A_phenomenon_sensitivity.csv`; `region_B_intervals.csv` adds clustered
95% intervals and the calibrated-probability sensitivity. Stable here is an
oracle restoration diagnostic, **not a deployable confidence or correctness label**.

## Rank-gap sensitivity, MCAR30, cutoff .02

The sign boundary remains 1e-6; only rank gap changes. Units are raw-logit
attributions. No normalization was introduced to make tolerances look favorable.

| dataset | variant | tolerance | overall_revision | B | revision_among_stable |
| --- | --- | --- | --- | --- | --- |
| taiwan | feature3 | event | 0.1792 | 116 | 0.1292 |
| taiwan | feature3 | event_gap_0 | 0.1792 | 116 | 0.1292 |
| taiwan | feature3 | event_gap_0.005 | 0.1220 | 73 | 0.0813 |
| taiwan | feature3 | event_gap_0.01 | 0.0856 | 49 | 0.0546 |
| taiwan | feature3 | event_gap_0.02 | 0.0528 | 25 | 0.0278 |
| taiwan | group1 | event | 0.0434 | 39 | 0.0386 |
| taiwan | group1 | event_gap_0 | 0.0434 | 39 | 0.0386 |
| taiwan | group1 | event_gap_0.005 | 0.0376 | 31 | 0.0307 |
| taiwan | group1 | event_gap_0.01 | 0.0312 | 24 | 0.0237 |
| taiwan | group1 | event_gap_0.02 | 0.0253 | 18 | 0.0178 |
| taiwan | group2 | event | 0.0818 | 42 | 0.0578 |
| taiwan | group2 | event_gap_0 | 0.0818 | 42 | 0.0578 |
| taiwan | group2 | event_gap_0.005 | 0.0635 | 33 | 0.0454 |
| taiwan | group2 | event_gap_0.01 | 0.0531 | 31 | 0.0426 |
| taiwan | group2 | event_gap_0.02 | 0.0403 | 21 | 0.0289 |
| taiwan | group3 | event | 0.0976 | 26 | 0.0745 |
| taiwan | group3 | event_gap_0 | 0.0976 | 26 | 0.0745 |
| taiwan | group3 | event_gap_0.005 | 0.0783 | 17 | 0.0487 |
| taiwan | group3 | event_gap_0.01 | 0.0613 | 13 | 0.0372 |
| taiwan | group3 | event_gap_0.02 | 0.0477 | 9 | 0.0258 |
| polish | feature3 | event | 0.3763 | 273 | 0.3434 |
| polish | feature3 | event_gap_0 | 0.3763 | 273 | 0.3434 |
| polish | feature3 | event_gap_0.005 | 0.3297 | 228 | 0.2868 |
| polish | feature3 | event_gap_0.01 | 0.2771 | 179 | 0.2252 |
| polish | feature3 | event_gap_0.02 | 0.2161 | 127 | 0.1597 |
| polish | group1 | event | 0.1209 | 83 | 0.1146 |
| polish | group1 | event_gap_0 | 0.1209 | 83 | 0.1146 |
| polish | group1 | event_gap_0.005 | 0.1119 | 75 | 0.1036 |
| polish | group1 | event_gap_0.01 | 0.1056 | 70 | 0.0967 |
| polish | group1 | event_gap_0.02 | 0.0875 | 58 | 0.0801 |
| polish | group2 | event | 0.1901 | 85 | 0.1540 |
| polish | group2 | event_gap_0 | 0.1901 | 85 | 0.1540 |
| polish | group2 | event_gap_0.005 | 0.1793 | 79 | 0.1431 |
| polish | group2 | event_gap_0.01 | 0.1706 | 73 | 0.1322 |
| polish | group2 | event_gap_0.02 | 0.1501 | 60 | 0.1087 |
| polish | group3 | event | 0.2301 | 61 | 0.1821 |
| polish | group3 | event_gap_0 | 0.2301 | 61 | 0.1821 |
| polish | group3 | event_gap_0.005 | 0.2109 | 52 | 0.1552 |
| polish | group3 | event_gap_0.01 | 0.1962 | 45 | 0.1343 |
| polish | group3 | event_gap_0.02 | 0.1755 | 37 | 0.1104 |

Taiwan grouped Region B becomes small under the widest .02 gap: 1.78%, 2.89%,
2.58% among stable eligible customers for group1/2/3, versus 8.01%, 10.87%, 11.04%
on Polish. Thus the original feature-level magnitude was substantially driven
by fine rankings. It did not collapse to zero, but “frequent serious errors” is
not established. The unvalidated financial/human importance of a small logit gap
must not be inferred from these percentages.

Whole-group top2 gives higher MCAR30 revision (Taiwan20.04%, Polish53.13%) because
it adds imputed/restored contributions. This is a different estimand, **not** the
preferred result. The observed subset is fixed at both endpoints.

## Common eligibility and analysis controls

| dataset | variant | condition | common_eligible | feature_revision | group_revision | group_only_events | feature_only_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | group1 | mcar30 | 1979 | 0.1789 | 0.0384 | 47 | 325 |
| taiwan | group2 | mcar30 | 1594 | 0.1763 | 0.0803 | 82 | 235 |
| taiwan | group3 | mcar30 | 881 | 0.1793 | 0.0976 | 53 | 125 |
| polish | group1 | mcar30 | 1108 | 0.3718 | 0.1209 | 60 | 338 |
| polish | group2 | mcar30 | 926 | 0.3629 | 0.1901 | 76 | 236 |
| polish | group3 | mcar30 | 678 | 0.3746 | 0.2301 | 70 | 168 |

Saved analysis controls, no refitting (absolute GRU/tree comparisons retain different explainer estimands):

| dataset | model | variant | eligible | stable_n | B | overall_revision |
| --- | --- | --- | --- | --- | --- | --- |
| taiwan | additive | feature3 | 1872 | 843 | 0 | 0.0000 |
| taiwan | additive | group1 | 2224 | 1016 | 0 | 0.0000 |
| taiwan | additive | group2 | 1643 | 723 | 0 | 0.0000 |
| taiwan | additive | group3 | 858 | 351 | 0 | 0.0000 |
| taiwan | lr | feature3 | 2254 | 975 | 0 | 0.0000 |
| taiwan | lr | group1 | 2333 | 1008 | 0 | 0.0000 |
| taiwan | lr | group2 | 1975 | 818 | 0 | 0.0000 |
| taiwan | lr | group3 | 1365 | 526 | 0 | 0.0000 |
| taiwan | mask_delta | feature3 | 2231 | 1173 | 178 | 0.1649 |
| taiwan | mask_delta | group1 | 2327 | 1217 | 58 | 0.0460 |
| taiwan | mask_delta | group2 | 2007 | 1034 | 80 | 0.0917 |
| taiwan | mask_delta | group3 | 1331 | 655 | 98 | 0.1608 |
| taiwan | vanilla | feature3 | 2223 | 1322 | 85 | 0.0873 |
| taiwan | vanilla | group1 | 2298 | 1376 | 20 | 0.0187 |
| taiwan | vanilla | group2 | 1882 | 1090 | 44 | 0.0526 |
| taiwan | vanilla | group3 | 1142 | 634 | 42 | 0.0937 |
| polish | lr | feature3 | 1148 | 528 | 0 | 0.0000 |
| polish | lr | group1 | 1149 | 529 | 0 | 0.0000 |
| polish | lr | group2 | 957 | 397 | 0 | 0.0000 |
| polish | lr | group3 | 626 | 200 | 0 | 0.0000 |

Additive controls' zero observed-reason revision is structural and does not
establish superior explanations. Group3 eligibility is only 36.7% on Taiwan
MCAR30; lower revision or selected coverage must be interpreted with that loss.
No timing groups, demographics or Polish ratios were regrouped after outcomes.

Figures: `analysis/01_decoupling.{png,pdf}`, `02_feature_group_revision`,
`06_external_regions`. A=both stable; B=prediction stable/reasons revised;
C=prediction shifts/reasons stable; D=both change.
