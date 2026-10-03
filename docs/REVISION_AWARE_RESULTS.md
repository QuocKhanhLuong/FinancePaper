# Revision-aware study — measured 2026-10-03

**The phenomenon survives; the learned selector loses to a simple completion
Monte Carlo revision estimator. Do not promote a new neural head or claim a
methodological win.** The selected generic selector detects revision substantially
better than predictive uncertainty, but the strongest explanation-based baseline
is better. This is the intended falsification, not a reason to tune on these results.

The [pre-assessment GO decision](NEXT_MODEL_DECISION.md) authorized one bounded
strategy test. **Post-assessment decision: NO-GO for a new architecture or claimed
selector superiority; continue only as a narrowed evaluation/release-policy study.**
Retain frozen XGB + Monte Carlo as the strong reference for future independent
validation, not a newly selected SOTA architecture. No second architecture was
developed or tuned after assessment. Earlier reports are unchanged.

## Cohort and interpretation

Three folds over 21,000 original train/development/probability-calibration customers;
7,000 outer predictions and 800 label-independent explanation customers per fold.
Outer customers are disjoint across folds. The historical 4,500 test and reserved
4,500 risk-calibration customers are excluded. Each fold trains its predictor on
8,000 customers, with distinct inner early-stopping, default calibration, revision
training, diagnostic, revision calibration and release calibration partitions.
All models/calibrators/policies froze before any outer assessment.

These are **internal exploratory results on a historically studied benchmark**.
Global research decisions used pooled diagnostic evidence, whose customer pools
overlap other folds' training/development roles. This is not independent confirmation
of the entire adaptive research process. Fold SD combines customer split and
training/mask restart variation; three folds are not three independent studies.

## Credit prediction: no need to replace XGBoost

Mean ± sample SD across folds. AP/AUROC higher, Brier lower. Default probabilities
are calibrated on a separate pool. The revision strategy leaves XGB probabilities
and explanations exactly unchanged.

| Model | AP clean | AP MCAR30 | AP MAR30 | Brier MCAR30 | Brier MAR30 |
|---|---:|---:|---:|---:|---:|
| LR25 | .5359 ± .0181 | .5036 ± .0152 | .4952 ± .0242 | .1437 ± .0022 | .1457 ± .0029 |
| Additive tree control | .5368 ± .0193 | .5061 ± .0175 | .4894 ± .0226 | .1435 ± .0019 | .1465 ± .0028 |
| Vanilla GRU + masking | .5326 ± .0122 | .5042 ± .0102 | .4872 ± .0168 | .1429 ± .0012 | .1463 ± .0029 |
| Mask/delta GRU + masking | .5327 ± .0132 | .5051 ± .0061 | .4916 ± .0184 | .1429 ± .0018 | .1452 ± .0034 |
| XGB25 / same predictor + selector | .5465 ± .0222 | .5117 ± .0195 | .5035 ± .0243 | .1417 ± .0025 | .1441 ± .0031 |

XGB clean/MCAR30/MAR30 AUROC is .7739/.7534/.7494. At MAR30, mask/delta minus
XGB AP is −.01192 [customer-paired 95% bootstrap interval −.01865, −.00508],
Brier +.00110 [.00036, .00187]. At MCAR30, AP difference is −.00658
[−.01336, −.00014]. These are conditional-on-fitted-model, pointwise intervals,
not a claim of universal superiority. Training sizes differ from the historical
15,000-customer training experiments, so do not interpret this table as temporal
performance drift. Raw/calibrated metrics, classwise precision/recall/F1/support,
log loss, ECE and calibration-bin counts remain in each fold's prediction JSON.
The 50/50 blend remains a historical tradeoff comparator, not a new outer candidate.

## Revision and the critical stable-prediction case

Strict k3/.01 event, eligible valid pairs only; rates across TreeSHAP and IG are
descriptive, not equivalent explanation-correctness rankings.

| Model | MCAR30 revision | MAR30 revision | MCAR30 revised among stable at .02 | MAR30 revised among stable at .02 |
|---|---:|---:|---:|---:|
| XGB25, unchanged by selector | 360/2009 = 17.92% | 265/1970 = 13.45% | **116/898 = 12.92%** | 72/1040 = 6.92% |
| Vanilla GRU | 194/2223 = 8.73% | 148/2182 = 6.78% | 85/1322 = 6.43% | 62/1448 = 4.28% |
| Mask/delta GRU | 368/2231 = 16.49% | 263/2179 = 12.07% | **178/1173 = 15.17%** | 131/1281 = 10.23% |
| LR / additive | 0 | 0 | 0 | 0 |

At MCAR30, XGB stable/revised frequencies for .01/.02/.05 are 11.05%/12.92%/15.35%;
mask/delta 14.37%/15.17%/16.24%. The phenomenon reproduces, but **the historical
25% is not a universal rate**. `analysis/outer_case_b.csv` contains A/B/C/D counts
for every cutoff/model/condition. Stability means raw probability stability,
not correctness or low inference uncertainty.

All tree/linear attribution pairs pass completeness. For vanilla/mask-delta at
MCAR30, pair-valid counts are 2,383/2,387 of 2,400; preverification eligibility
2,235/2,240 falls to 2,223/2,231 after requiring restored validity. Those exclusions
are visible and never an inference-time release filter. Additive zero revision is
structural; it cannot establish good explanations. The metric's sensitivity and
shared-explainer controls are in [the audit](REVISION_METRIC_AUDIT.md).

## Revision-risk detection: the decisive comparison

Same frozen XGB, customers, masks and event. Mean ± SD across folds; calibrated
revision probabilities. All generic classifiers use fixed compact recipes.

| Detector | AUROC MCAR30 | AP MCAR30 | Brier MCAR30 | AUROC MAR30 | AP MAR30 |
|---|---:|---:|---:|---:|---:|
| Entropy | .5075 | .1910 | .1528 | .5552 | .1653 |
| Missing fraction | .5330 | .1948 | .1475 | .6140 | .1799 |
| Mask-only learned control | .5693 | .2082 | .1472 | .6386 | .2062 |
| Prediction-only learned control | .5728 ± .0262 | .2192 ± .0135 | .1468 ± .0057 | .6377 ± .0221 | .2059 ± .0592 |
| Selected generic revision selector | .7963 ± .0299 | .4574 ± .0541 | .1226 ± .0074 | .8095 ± .0315 | .4170 ± .1399 |
| **8-donor Monte Carlo revision** | **.9000 ± .0001** | **.7199 ± .0431** | **.0867 ± .0006** | **.9123 ± .0057** | **.6520 ± .0409** |

Completion probability variance, max probability, temporal missing fraction,
attribution variance, sign/rank instability and both logistic/boosted generic
selectors are also reported in `outer_revision_metrics.csv`. Rank instability
nearly matches MC revision; variance alone is substantially weaker. Completion
Monte Carlo uses **training donors only**, not the true hidden evaluation values.

Paired bootstrap on pooled customer predictions, selected minus reference:

| Condition | Reference | AP difference [95% CI] | AUROC difference [95% CI] |
|---|---|---:|---:|
| MCAR30 | Prediction-only | +.2338 [.1906, .2790] | +.2351 [.1989, .2696] |
| MAR30 | Prediction-only | +.1815 [.1248, .2349] | +.1631 [.1284, .2002] |
| MCAR30 | Monte Carlo revision | **−.2847 [−.3384, −.2285]** | −.0990 [−.1226, −.0738] |
| MAR30 | Monte Carlo revision | **−.2595 [−.3252, −.1862]** | −.0957 [−.1236, −.0660] |

The 1,000-draw intervals resample original customers, retaining all their copied
masks; they do not refit models and do not measure full training-process uncertainty.
Pooled AP differences differ from differences of fold-mean AP. No dozens-of-tests
significance claim or arbitrary metric scalar is used.

At exactly seven missing cells under MCAR30, selected AUROC/AP is .8036/.4605,
prediction-only .5709/.2663, mask-only .4200/.1694, MC .8827/.7053 (353 cases,
69 events). Fixed counts 5/6/8 tell the same qualitative story. The head is not
merely a missing-rate proxy. Field-specific mask indicators provide the matched-
count/different-group control, but artificial missingness is still not a natural
default signal.

Within the critical |delta p|≤.02 XGB stratum, selected AUROC/AP is .7956/.3628;
prediction-only .5082/.1262; MC .9327/.7175 (898 cases, 116 events). Explanation
information helps precisely where probability stability is misleading. This
oracle-defined subgroup is evaluation only, never an inference feature.

## Release risk: pooled calibration does not certify every mechanism

Policies fixed on 600 release-calibration customers/fold, one condition per person.
Coverage denominator is **all** 2,400 explanation-sampled customers; withheld
ineligible cases count against coverage. Conservative = simultaneous one-sided
binomial upper bounds over the fixed 101 thresholds; empirical = observed
calibration revision≤10%. Neither threshold is optimized on outer outcomes.

| Policy | Detector | Pooled coverage / risk | MCAR30 coverage / risk | MAR30 coverage / risk |
|---|---|---:|---:|---:|
| Empirical | Selected | 85.62% / 9.88% | 79.71% / **15.84%** | 78.79% / **11.63%** |
| Empirical | Monte Carlo | 86.21% / 9.47% | 81.00% / **15.33%** | 80.08% / **11.60%** |
| Conservative | Prediction-only | 46.67% / 4.82% | 15.96% / **12.27%** | 25.79% / 7.59% |
| Conservative | Selected | 71.58% / 5.59% | 58.42% / 9.06% | 61.54% / 6.91% |
| Conservative | Monte Carlo | **81.00% / 5.61%** | **72.75% / 9.11%** | **74.00% / 7.09%** |

The empirical policy fails the 10% target under each 30% mechanism. Even the
conservative prediction-only policy fails under MCAR30; its guarantee assumptions
concern the pooled mixture, not this shifted subpopulation. Selected MCAR30 risk
95% customer-bootstrap interval is [7.62%, 10.51%]; MC is [7.79%, 10.50%]. Therefore
even point estimates below 10% do **not** establish a mechanism-specific ≤10%
population guarantee. Pooled selected coverage interval is [69.79%, 73.29%], MC
[79.46%, 82.63%]. No unknown revision labels were released by XGB policies.
Binomial bounds additionally rely on iid calibration assumptions; stratified
benchmark construction and future deployment shift must not be waved away.

Revision calibration is not an automatic improvement. Selected MCAR30 Brier
raw/calibrated .1207/.1226 and log loss .3830/.3863 slightly worsen. For MC,
Brier .0793/.0867 worsens but log loss .5716/.2958 improves by avoiding extreme
finite-draw 0/1 probabilities. Report both; do not present calibration as a cure.

## Pareto interpretation and compute

The selected strategy adds detection information over prediction-only selection
without changing default AP/calibration. It does **not** establish improvement
over the strongest explanation baseline: MC is better on detection/calibration
and retains more coverage at similar observed risk. The conservative selected
risk is fractionally lower, so do not call MC a strict all-metric dominator either.
The joint prediction/explanation frontier is not reduced to a scalar score.

The learned score has a real computational tradeoff: on one fixed inner 128-case
batch, five warm CPU timing repetitions give .1053 ± .0012 seconds for current
SHAP + donors/predictions + learned score, versus .7271 ± .0186 seconds including
eight completion SHAP evaluations. No score was changed using these timings.
This ~6.9× batch latency difference is a measurement on this machine, not novel
amortization or evidence that the cheaper selector meets the desired risk budget.

Measured stage times: diagnostics 612.24s, selector baselines 176.13s,
calibration/freeze 76.91s, outer assessment 374.40s; total **20.66 minutes**,
excluding browsing, software tests, reporting/bootstrap and timing probes.
Neural controls trained on MPS (vanilla 21,953 parameters, mask/delta 23,265);
best epochs 17/3/16 and 17/2/22 respectively, at most 25 epochs. Trees/selectors
ran on CPU with one numerical thread. Peak hardware memory was not measured.

## Artifacts and checks

`outputs/revision_study/` holds phase freezes, split IDs, masks, fitted models,
attribution arrays, per-record predictions, current-only features, and separate
verification outcomes. Normal diagnostic JSONL excludes all verification-only
fields; null uncertainty/head fields for other predictors mean not implemented,
not measured zero uncertainty. Detailed GRU diagnostics remain available in the
existing debug forward path.

`analysis/` contains seven PNG/PDF pairs: discrimination versus missingness,
calibration versus missingness, revision versus missingness, prediction versus
explanation shift, selective prediction, selective explanation, calibration curves.
Curves are descriptive; line segments across whole-score ties are visual guides,
not attainable policies or thresholds selected on assessment data. Paired intervals,
critical-case detection, fixed-count controls, calibration bins and Pareto accounting
are CSV/JSON files regenerated by the scripts below.

Root-run recomputation verified 54 frozen source hashes, 12 frozen artifacts,
283 result artifacts, exclusion of 9,000 historical/reserved customers, 21,000
unique outer customers, 525,000 prediction rows and 88,800 reason rows. This is
software verification, not an independent peer review. Added tests cover target
equivalence, donor preservation, forbidden inference fields, split isolation,
whole ties, unknown outcomes, finite-sample abstention, and common serving schema.
Final full suite: **77 passed**; three upstream SHAP plotting deprecation warnings.
Two postprocessing-only defects (monthly NumPy indexing and nullable event dtype)
were corrected before successful export/reporting. Neither changed fitted models,
attributions, labels, policies or recorded prediction/revision metrics.
Publication removes one empty EOF line in `revision_study.py`; the exact measured
bytes are archived locally under `source_at_measurement/`. The auditor verifies
the old hash and proves this is only trailing whitespace, rather than silently
rewriting the measurement freeze.

```bash
# After the four study stages in README:
uv run --frozen --extra temporal python scripts/report_revision_study.py
uv run --frozen --extra temporal python scripts/summarize_revision_study.py
uv run --frozen --extra temporal python scripts/serialize_revision_diagnostics.py
uv run --frozen --extra temporal python scripts/benchmark_revision_inference.py
uv run --frozen --extra temporal python scripts/audit_revision_study.py
```

No dataset/model/output cache is committed. No external experiment, head superiority,
independent population guarantee, or ESWA acceptance is claimed. The next scientific
action is a preregistered external known-cell restoration/release evaluation and
domain-grounded revision tolerance study. If the strict reason-list phenomenon
disappears under meaningful grouped/tolerant reasons, or MC cannot retain useful
coverage at a fixed independently calibrated risk budget, abandon or substantially
narrow the proposed operational contribution.
