# Prediction stability and explanation revision — 2026-10-03

The new development evidence reproduces **stable prediction + revised reasons**.
It does not establish that a stable prediction is correct, well calibrated for
that customer, or epistemically certain. Restoration shift is an oracle diagnostic
available only after verification. Entropy and completion variance are inference-
time proxies; they answer different questions.

## Four regions, with an explicit operational interpretation

| | No strict reason revision | Strict reason revision |
|---|---|---|
| Absolute raw probability shift ≤ declared cutoff | A: stable prediction, stable offered reasons | **B: stable prediction, revised offered reasons** |
| Shift > cutoff | C: shifted prediction, stable offered reasons | D: both shift |

Use this measurable table instead of equating the row labels with “reliable” and
“uncertain.” The 37/148 historical pilot observation remains exploratory and is
not the expected proportion in a new cohort.

## Development replication

384 randomly selected diagnostic customers/fold, three separately fitted folds.
Rows below count eligible, attribution-valid **customer-fold cases**; some original
customers can occur in another fold's diagnostic partition. Bootstrap uses original
customer clusters rather than counting those repetitions as independent people.

| Model | Condition | B/stable at .01 | B/stable at .02 | B/stable at .05 |
|---|---|---:|---:|---:|
| XGB25 | MCAR10 | 31/632 (4.9%) | 47/781 (6.0%) | 65/970 (6.7%) |
| XGB25 | MCAR30 | 31/281 (11.0%) | 53/440 (12.0%) | 119/741 (16.1%) |
| XGB25 | MAR30 | 41/327 (12.5%) | 56/473 (11.8%) | 95/715 (13.3%) |
| Vanilla GRU | MCAR10 | 20/814 (2.5%) | 24/956 (2.5%) | 31/1045 (3.0%) |
| Vanilla GRU | MCAR30 | 16/445 (3.6%) | 39/654 (6.0%) | 63/870 (7.2%) |
| Vanilla GRU | MAR30 | 22/544 (4.0%) | 25/677 (3.7%) | 46/839 (5.5%) |
| Mask/delta GRU | MCAR10 | 39/727 (5.4%) | 52/902 (5.8%) | 66/1056 (6.2%) |
| Mask/delta GRU | MCAR30 | 45/318 (14.2%) | 79/544 (14.5%) | 129/880 (14.7%) |
| Mask/delta GRU | MAR30 | 47/430 (10.9%) | 67/606 (11.1%) | 104/839 (12.4%) |

For example, the complete MCAR30 A/B/C/D counts at .02 are XGB
387/53/425/106; vanilla 615/39/370/52; mask-delta 465/79/448/86. These account for
all eligible cases, not only illustrative failures. All cutoffs/conditions and
zero-revision additive controls are in `analysis/diagnostic_case_b.csv`.

The separate [metric audit](REVISION_METRIC_AUDIT.md) shows sensitivity to rank
tolerance and grouping. Thus the supported phenomenon is revision of a declared
reason list, not proof that an explanation was false or a broad causal distinction.

## Does predictive uncertainty detect revision?

Experiment B freezes XGB25, fits generic selectors on 2,000 held-out revision-
training customers/fold and evaluates on all 800 diagnostic customers/fold.
Prediction inputs: current calibrated probability, entropy, eight-donor completion
probability variance, missing fraction. The paired attribution recipe adds only
current observed attribution summaries. No restored values enter either recipe.

| Condition | Prediction-only AUROC / AP | + Attribution AUROC / AP | + Attribution and mask groups AUROC / AP |
|---|---:|---:|---:|
| MCAR10 | .6631 / .1464 | .8173 / .2984 | .8323 / .3235 |
| MCAR30 | .5741 / .2125 | .7527 / .3872 | .7731 / .3976 |
| MAR30 | .5871 / .2115 | .7881 / .3759 | .7957 / .3901 |

Values are arithmetic means of fold metrics for the same fixed boosted classifier.
On pooled predictions with original-customer bootstrap (1,000 draws), adding
attribution increases AP by .1661 [95% interval .1195, .2085] at MCAR30 and
.1652 [.1189, .2115] at MAR30. These are development, pointwise intervals, not
confirmatory multiplicity-adjusted tests. Mean fold differences need not equal
pooled differences because AP is nonlinear.

This falsifies “these tested predictive signals almost perfectly detect revision”
in the diagnostic setting. It does **not** establish statistical independence of
all predictive and explanatory uncertainty, nor rule out a stronger uncertainty
estimator. Completion variance is sensitive to the approximate donor distribution;
we do not label it calibrated epistemic uncertainty. A generic selector already
works, so no new neural-head contribution follows.

## Bounded outer assessment

The later [results](REVISION_AWARE_RESULTS.md) report the frozen strategy on disjoint
outer customer folds, including all three cutoffs, critical-case detection,
fixed missing-count controls, and release-policy failures. The strategy shares
the **same XGB probabilities and explanations**; its four-region counts therefore
equal XGB's before selective release. It predicts risk of revision rather than
changing the underlying phenomenon. Raw per-case outcomes and the six required
plot families are regenerated with `scripts/report_revision_study.py`.

Calibration/selection decisions never use the historical test. Nevertheless,
outer folds come from a historically explored benchmark and their customer pools
overlap other folds' development roles. This entire research narrative needs
external confirmation. A claim that Taiwan's stable/revised frequency is universally
25%, or that “small shift means a safe prediction,” is not supported.
