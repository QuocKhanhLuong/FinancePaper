# Decisive validation protocol, 2026-10-03

Freeze before new results. No architecture, loss, donor-distance or predictor
search. Historical results and their source files remain intact. AI-assisted
implementation/analysis; software audit is not independent peer review.

Taiwan reuses all three `04bf054` frozen predictor bundles and the same 800 outer
explanation customers per fold, plus the existing disjoint 600 revision-probability
and 600 release-calibration customers. The earlier test/reserve remain excluded.
This is an internal stress validation on previously inspected customers, not a
new Taiwan confirmation cohort. Controls use saved LR/additive/GRU attributions;
no retraining. XGBoost 25-view is the sole main predictor.

Keep complete/MCAR10/MCAR20/MCAR30/MAR30 masks exactly. Add one declared group-missing
environment: uniformly hide one of the three six-field history groups, using
partition mask seed + 20000, independent of target/values. It is a 6/23 field
stress scenario, not MAR30. Generate on the entire partition before subsetting.
Grouping and all sensitivity definitions are in DOMAIN_REASON_GROUPS.md.

## Completion budget and evidence

Unchanged training-only donor reference (max 2048), nearest 32 by observed-field
IQR distance clipped at 5/categorical mismatch, joint sampling with replacement.
K=1/2/4 are prefixes of the **same original K=8 draw stream**; K=16 appends an
independent eight-draw stream with seed + 100000. This preserves historical K8
draws despite the old row-major RNG consuming K draws per customer. No hidden
truth/target enters completion, current diagnostics, scores or release inference.

Cache partial/full/completion attributions in ignored outputs. Export verification
separately. Completion audit uses truth only afterward: numeric empirical percentile
and min/max inclusion, categorical support, donor uniqueness/entropy, observed-cell
invariance, reference IDs, natural-missing invariance. No imputer improvement after
external evaluation. Failures are analysis, never a search trigger.

Primary MC comparison K8 remains fixed. Report all K; a descriptive Taiwan-only
smallest-useful-K criterion is >=95% of K16 AP **above event prevalence**, averaged
over MCAR10/30/MAR30, with fold ranges/paired uncertainty. It is not a new external
K selection. If no smaller K meets it, say so. K16 is not a convergence oracle.

## Calibration policies

Default calibration remains frozen. For each reason definition and K, independently
fit the existing monotone positive-slope Platt revision calibrator on the designated
revision-calibration pool. One condition per customer, chosen by RNG before scores.
New release comparisons use an equally assigned mixture MCAR10/MCAR30/MAR30/group
(Polish: MCAR10/MCAR30); complete/MCAR20 are diagnostic only. The old five-condition
pooled policy and its thresholds are preserved and reported separately as history.

All policies use the existing calibrated-score grid 0,.01,...,1 and whole ties.
Targets .05/.10/.15 are all evaluated. Unknown verification outcomes count as failures.
Eligibility uses current evidence only. No releases -> undefined risk, not zero.

* A pooled: maximum coverage on one assigned environment per calibration customer.
* B observable-stratified: missing fraction <=.15, (.15,.30], >.30; current total
  missingness only, including natural missingness externally. Independent thresholds
  in each stratum, minimum 50 assigned calibration customers; otherwise withhold.
* C robust-environment: one shared threshold satisfying the risk constraint in
  **every** declared calibration environment; maximize average calibration coverage.
  Each environment includes one mask per customer; reuse across environments is
  handled as paired clusters, never independent replicates. The mechanism label is
  calibration-only; inference uses only the one common threshold.

Report empirical constraints and conservative simultaneous Clopper–Pearson upper
bounds separately. Conservative family correction is .05/(101 * number of tested
strata/environments); alpha risk budget is distinct from confidence delta=.05.
No distribution-free or arbitrary-shift claim. Fixed-policy customer-bootstrap CIs
do not include training/calibration-estimation uncertainty. At least 1000 paired
customer draws, including all a customer's masks and preserving outer folds for
Taiwan. Primary descriptive overall population is equal environment mixture.

## Diagnostics, runtime, reporting

Detection AP/AUROC, raw and calibrated Brier/log loss; all reason definitions,
all K, condition-specific results and eligibility. Paired differences vs predictive
entropy, prediction-variance and existing prediction-only/generic learned selectors.
Generic scores are controls, not new architectures; Taiwan models stay frozen.
For grouped labels recalibrate the same fixed raw selector, do not refit it.

Report A/B/C/D counts and percentages and P(B|prediction-stable), at all three
cutoffs; common eligibility; normalized signed-attribution L1 shift separately.
MC versus variance/rank/sign/missing-fraction correlations and fixed-condition /
fixed-missingness strata. Deterministically export first five record IDs of each
failure/contrast type per condition, not just successful examples. High MC >=.5,
low MC <=.125 are descriptive case bins, not release thresholds. Inspect support
and near-ties without changing method.

Timing on fixed fold0 diagnostic customers, batch128 and single-record serving:
one warm-up and five timed repetitions; report mean/SD; exclude disk/model loading,
include preprocessing, current explanation, completion, prediction and scoring.
Separate SHAP, prediction, donor generation, transform and scoring; report call
counts. Generic learned selector still needs current SHAP plus K8 predictions.
Use CPU XGBoost/SHAP; no GPU needed. Record device, packages, hardware and source,
artifact/data hashes. At least six PNG/PDF figures and machine-readable tables.

External preparation/training/calibration ends in a hashed freeze **before** any
external assessment predictions/attributions. Protocol, groups, K, method and
families cannot change afterward. Code bugs halt the stage and get a written
incident report before further action; no silent repairs/reruns.
