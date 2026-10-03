# Release policy validation

Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.

## Decision and assumptions

**Robust-environment calibration is the most defensible declared stress policy,
not a distribution-free or mechanism-specific deployment guarantee.** One common
threshold is fit against every simulated calibration environment and is used at
inference without a mechanism label. Pooled calibration is the reference; observable
strata use total missing fraction <=.15, (.15,.30], >.30 and never the hidden
MCAR/MAR mechanism. Small strata fail closed. Policies were fixed before each
assessment; alpha5/10/15% all remain visible.

Separate revision Platt pool and release pool. Empirical risk constraints and
simultaneous finite-grid binomial-upper-bound rules are reported separately.
The grid is unchanged 0,.01,...,1; ties are not outcome-sorted. Overall means
equal environment mixture, excluding complete cases, unlike the historical
five-condition pooled report. Historical thresholds were not overwritten.

Intervals are 1000 paired customer bootstrap draws, stratified by Taiwan outer
fold; Polish resamples exact-feature duplicate clusters. These are conditional
on trained models and fitted thresholds, not uncertainty over repeated studies.
Multiple masks of one customer are never independently resampled. Zero releases
have **undefined** risk. Bounds across environments use Bonferroni correction;
no independence between environments is required, but independent customers within
an environment are an assumption.

Polish release calibration has 590 statements/585 exact-feature clusters and no
company IDs. The frozen binomial rule counts statements. Therefore even its
“conservative” label is a procedure description, **not corporate-level certification**.
See [implementation/assumption audit](VALIDATION_IMPLEMENTATION_NOTES.md).

## Table D: main MC8 feature-top3 policies, all risk targets

Conservative-rule results; coverage denominator includes all sampled customers.


### taiwan

| variant | alpha | family | condition | Coverage [95% CI] | Revision [95% CI] |
| --- | --- | --- | --- | --- | --- |
| feature3 | 0.0500 | pooled | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | mar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | group_missing | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | mar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | group_missing | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | mar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | group_missing | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.1000 | pooled | overall | 67.48% [66.31, 68.56] | 4.01% [3.51, 4.53] |
| feature3 | 0.1000 | pooled | mcar10 | 80.71% [79.04, 82.25] | 2.48% [1.80, 3.24] |
| feature3 | 0.1000 | pooled | mcar30 | 62.29% [60.42, 64.17] | 5.15% [4.07, 6.27] |
| feature3 | 0.1000 | pooled | mar30 | 66.12% [64.33, 67.92] | 3.53% [2.66, 4.51] |
| feature3 | 0.1000 | pooled | group_missing | 60.79% [58.87, 62.71] | 5.41% [4.30, 6.62] |
| feature3 | 0.1000 | robust | overall | 42.92% [41.96, 43.85] | 2.48% [2.00, 2.98] |
| feature3 | 0.1000 | robust | mcar10 | 52.38% [51.04, 53.75] | 1.51% [0.88, 2.23] |
| feature3 | 0.1000 | robust | mcar30 | 39.88% [38.33, 41.62] | 3.45% [2.31, 4.68] |
| feature3 | 0.1000 | robust | mar30 | 42.54% [40.96, 44.04] | 2.15% [1.31, 3.05] |
| feature3 | 0.1000 | robust | group_missing | 36.88% [35.12, 38.62] | 3.16% [2.02, 4.28] |
| feature3 | 0.1000 | stratified | overall | 16.09% [15.50, 16.75] | 1.94% [1.24, 2.68] |
| feature3 | 0.1000 | stratified | mcar10 | 26.00% [25.12, 26.92] | 2.24% [1.15, 3.49] |
| feature3 | 0.1000 | stratified | mcar30 | 8.54% [7.50, 9.58] | 1.95% [0.46, 3.98] |
| feature3 | 0.1000 | stratified | mar30 | 12.62% [11.54, 13.71] | 1.32% [0.31, 2.74] |
| feature3 | 0.1000 | stratified | group_missing | 17.21% [16.04, 18.46] | 1.94% [0.73, 3.24] |
| feature3 | 0.1500 | pooled | overall | 77.14% [75.99, 78.15] | 8.21% [7.54, 8.93] |
| feature3 | 0.1500 | pooled | mcar10 | 87.00% [85.58, 88.29] | 4.98% [4.03, 5.93] |
| feature3 | 0.1500 | pooled | mcar30 | 74.58% [72.79, 76.29] | 10.78% [9.33, 12.14] |
| feature3 | 0.1500 | pooled | mar30 | 74.62% [72.96, 76.29] | 7.65% [6.48, 8.93] |
| feature3 | 0.1500 | pooled | group_missing | 72.33% [70.50, 74.04] | 10.02% [8.56, 11.50] |
| feature3 | 0.1500 | robust | overall | 71.07% [69.88, 72.16] | 5.29% [4.70, 5.85] |
| feature3 | 0.1500 | robust | mcar10 | 82.96% [81.38, 84.38] | 3.37% [2.60, 4.19] |
| feature3 | 0.1500 | robust | mcar30 | 66.50% [64.75, 68.46] | 6.64% [5.42, 7.79] |
| feature3 | 0.1500 | robust | mar30 | 69.62% [67.88, 71.42] | 4.85% [3.88, 5.91] |
| feature3 | 0.1500 | robust | group_missing | 65.21% [63.12, 67.04] | 6.84% [5.60, 8.12] |
| feature3 | 0.1500 | stratified | overall | 62.57% [61.39, 63.61] | 4.46% [3.92, 5.01] |
| feature3 | 0.1500 | stratified | mcar10 | 83.38% [81.79, 84.92] | 3.95% [3.12, 4.89] |
| feature3 | 0.1500 | stratified | mcar30 | 50.58% [48.58, 52.67] | 5.27% [4.03, 6.46] |
| feature3 | 0.1500 | stratified | mar30 | 57.08% [55.16, 59.00] | 3.80% [2.87, 4.83] |
| feature3 | 0.1500 | stratified | group_missing | 59.25% [57.33, 61.17] | 5.13% [4.04, 6.31] |

### polish

| variant | alpha | family | condition | Coverage [95% CI] | Revision [95% CI] |
| --- | --- | --- | --- | --- | --- |
| feature3 | 0.0500 | pooled | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | pooled | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | robust | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | overall | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | mcar10 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.0500 | stratified | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.1000 | pooled | overall | 65.40% [63.29, 67.46] | 3.17% [2.32, 4.12] |
| feature3 | 0.1000 | pooled | mcar10 | 78.00% [75.66, 80.24] | 2.49% [1.59, 3.53] |
| feature3 | 0.1000 | pooled | mcar30 | 52.79% [49.96, 55.76] | 4.17% [2.70, 5.79] |
| feature3 | 0.1000 | robust | overall | 58.88% [56.65, 61.06] | 1.72% [1.08, 2.53] |
| feature3 | 0.1000 | robust | mcar10 | 73.10% [70.49, 75.66] | 1.27% [0.58, 2.12] |
| feature3 | 0.1000 | robust | mcar30 | 44.67% [42.00, 47.59] | 2.46% [1.31, 3.76] |
| feature3 | 0.1000 | stratified | overall | 31.81% [30.41, 33.35] | 0.80% [0.26, 1.54] |
| feature3 | 0.1000 | stratified | mcar10 | 63.62% [60.81, 66.70] | 0.80% [0.26, 1.54] |
| feature3 | 0.1000 | stratified | mcar30 | 0.00% [0.00, 0.00] | undefined |
| feature3 | 0.1500 | pooled | overall | 73.35% [71.35, 75.23] | 6.98% [5.81, 8.19] |
| feature3 | 0.1500 | pooled | mcar10 | 82.91% [80.71, 84.93] | 4.59% [3.42, 5.83] |
| feature3 | 0.1500 | pooled | mcar30 | 63.79% [61.21, 66.47] | 10.08% [7.88, 12.15] |
| feature3 | 0.1500 | robust | overall | 69.25% [67.23, 71.24] | 4.34% [3.39, 5.40] |
| feature3 | 0.1500 | robust | mcar10 | 80.80% [78.53, 82.93] | 3.56% [2.47, 4.77] |
| feature3 | 0.1500 | robust | mcar30 | 57.70% [55.04, 60.59] | 5.43% [3.84, 7.12] |
| feature3 | 0.1500 | stratified | overall | 64.21% [62.18, 66.29] | 3.10% [2.28, 4.05] |
| feature3 | 0.1500 | stratified | mcar10 | 80.03% [77.79, 82.27] | 3.28% [2.22, 4.54] |
| feature3 | 0.1500 | stratified | mcar30 | 48.39% [45.55, 51.40] | 2.80% [1.56, 4.17] |

## Grouped top2 at 10% target


### taiwan

| variant | alpha | family | condition | Coverage [95% CI] | Revision [95% CI] |
| --- | --- | --- | --- | --- | --- |
| group2 | 0.1000 | pooled | overall | 68.19% [66.68, 69.77] | 3.90% [3.37, 4.42] |
| group2 | 0.1000 | pooled | mcar10 | 74.21% [72.50, 76.00] | 1.57% [0.98, 2.12] |
| group2 | 0.1000 | pooled | mcar30 | 64.12% [62.29, 66.17] | 5.07% [3.99, 6.20] |
| group2 | 0.1000 | pooled | mar30 | 67.08% [65.17, 68.92] | 5.47% [4.35, 6.54] |
| group2 | 0.1000 | pooled | group_missing | 67.33% [65.38, 69.21] | 3.77% [2.86, 4.71] |
| group2 | 0.1000 | robust | overall | 65.69% [64.13, 67.23] | 2.55% [2.13, 2.97] |
| group2 | 0.1000 | robust | mcar10 | 72.79% [71.04, 74.58] | 1.09% [0.62, 1.58] |
| group2 | 0.1000 | robust | mcar30 | 61.46% [59.62, 63.38] | 3.25% [2.32, 4.20] |
| group2 | 0.1000 | robust | mar30 | 63.96% [62.08, 65.75] | 3.45% [2.63, 4.31] |
| group2 | 0.1000 | robust | group_missing | 64.54% [62.50, 66.46] | 2.65% [1.90, 3.46] |
| group2 | 0.1000 | stratified | overall | 48.78% [47.51, 50.09] | 1.69% [1.32, 2.05] |
| group2 | 0.1000 | stratified | mcar10 | 53.62% [52.04, 55.25] | 0.85% [0.39, 1.37] |
| group2 | 0.1000 | stratified | mcar30 | 38.88% [37.00, 40.79] | 1.61% [0.79, 2.43] |
| group2 | 0.1000 | stratified | mar30 | 38.75% [36.96, 40.50] | 1.29% [0.64, 2.03] |
| group2 | 0.1000 | stratified | group_missing | 63.88% [61.88, 65.75] | 2.67% [1.92, 3.50] |

### polish

| variant | alpha | family | condition | Coverage [95% CI] | Revision [95% CI] |
| --- | --- | --- | --- | --- | --- |
| group2 | 0.1000 | pooled | overall | 71.74% [69.27, 74.13] | 6.37% [5.28, 7.53] |
| group2 | 0.1000 | pooled | mcar10 | 75.47% [72.95, 78.00] | 5.04% [3.64, 6.52] |
| group2 | 0.1000 | pooled | mcar30 | 68.02% [65.17, 70.73] | 7.84% [5.97, 9.67] |
| group2 | 0.1000 | robust | overall | 64.21% [61.90, 66.81] | 1.71% [1.12, 2.30] |
| group2 | 0.1000 | robust | mcar10 | 70.22% [67.46, 72.87] | 1.33% [0.60, 2.13] |
| group2 | 0.1000 | robust | mcar30 | 58.21% [55.34, 61.17] | 2.18% [1.17, 3.29] |
| group2 | 0.1000 | stratified | overall | 51.18% [49.11, 53.30] | 4.05% [2.91, 5.13] |
| group2 | 0.1000 | stratified | mcar10 | 74.62% [72.08, 77.14] | 5.22% [3.73, 6.70] |
| group2 | 0.1000 | stratified | mcar30 | 27.75% [25.21, 30.20] | 0.91% [0.00, 2.01] |

Grouped top1/top3, all grouped alpha targets, all K and detector controls are in
`analysis/D_policy_intervals.csv`; this includes every empirical/conservative
result, not just successful thresholds. `paired_policy_differences.csv` gives
paired robust-minus-pooled/stratified differences. Plot05 shows alpha10% with
undefined/no-release conditions labelled explicitly.

## What failed

The empirical pooled rule misses environment targets: feature MCAR30 alpha10%
gives Taiwan10.78% and Polish10.08%; Polish grouped top2 reaches12.43%. These are
observations, not proof of a guarantee violation since no such guarantee exists.
Even empirical robust Polish feature MCAR30 is10.08%, slightly above10%.
At alpha5%, conservative robust feature policies release **nothing** on either
dataset; Polish grouped top2 also releases nothing. This operating point is not
practical with the present calibration sample sizes and coarse MC scores.
Observable-stratified conservative feature policy releases nothing on Polish
MCAR30 even at10%; conditioning fragments the data and can lose most coverage.
Robust is more cautious, not uniformly Pareto-superior to pooled calibration.

## Necessary baseline: release all currently eligible reasons

No MC computation or learned threshold. MCAR30 only below; full environments in
`release_all_eligible.csv`. These are observed descriptive rates, not calibrated rules.

| dataset | variant | coverage [CI] | revision [CI] |
| --- | --- | --- | --- |
| taiwan | feature3 | 83.71% [82.17, 85.17] | 17.92% [16.25, 19.54] |
| taiwan | group1 | 92.08% [91.00, 93.08] | 4.34% [3.51, 5.25] |
| taiwan | group2 | 68.25% [66.50, 70.13] | 8.18% [6.87, 9.50] |
| taiwan | group3 | 36.71% [34.88, 38.58] | 9.76% [7.87, 11.62] |
| polish | feature3 | 99.83% [99.58, 100.00] | 37.63% [34.74, 40.24] |
| polish | group1 | 93.74% [92.37, 95.00] | 12.09% [10.17, 13.77] |
| polish | group2 | 78.34% [75.95, 80.66] | 19.01% [16.43, 21.63] |
| polish | group3 | 57.36% [54.49, 60.32] | 23.01% [20.00, 26.30] |

Taiwan group1/group2 already have low unconditional revision. At a10% target,
it would be misleading to credit MC for that baseline semantic aggregation effect.
Polish grouped top2 is more decisive: unselected78.34% coverage/19.01% revision,
versus robust conservative58.21% coverage/2.18% revision (95% CI1.17–3.29%) under
MCAR30. This demonstrates a useful tradeoff; it does not certify unobserved shifts
or establish an optimal policy. The current method is useful at10/15% in the
declared environments; a universal5% operational claim is rejected.
