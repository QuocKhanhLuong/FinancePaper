# Stable-Core v2: prospective operational comparison on inspected benchmarks

Declared 2026-10-03 before computing the new completion-family/policy results.
This user-requested extension supersedes the **two-candidate cap** in the initial
Freddie feasibility specification; it does not change historical events/results.
Freddie has not been acquired or evaluated. Freeze this v2 specification before
future Freddie assessment; no method adaptation after mortgage outcomes.

## Question and evidence status

Can partial reason sets increase explanation coverage at the same **verified
reason-failure** budget, beyond merely stating fewer reasons? Keep the fitted
XGBoost 25-view predictors, preprocessing, backgrounds, semantic groups, masks
and customer partitions from decisive validation. No risk-model retraining.
Taiwan and Polish assessment outcomes were inspected historically: this is an
exploratory paired extension, not new independent confirmation. Polish is corporate
bankruptcy; its results are not pooled with Taiwan credit-card default.

Taiwan: three saved folds, 600 revision-calibration customers for current-score
grid construction, 600 separate release-calibration customers, 800 assessment
customers per fold. Conditions MCAR10, MCAR30, MAR30, group-missing. Polish: same
frozen training and partitions; 590 release-calibration and 1,182 assessment rows;
MCAR10/30. Exact duplicate Polish feature rows are one bootstrap/calibration
cluster. Repeated conditions/reasons/completions are never independent units.

## Two targets, explicitly distinguished

Use raw-logit interventional TreeSHAP and the original pre-verification availability
mask to sum original fields into groups at every endpoint. The predictor and
background stay the same. Natural missing values remain unknown.

Candidate set: **all available groups with current contribution >0.01**. No top-k
cap and no requirement to emit a fixed number. Fixed schema order resolves ties.

1. **Primary meaningful-positive target:** a candidate fails if its restored
   contribution is <=0.01. It asks whether the positive contribution remains
   material on the declared model scale. It does not assert an adverse-action
   reason, causality, predictive correctness or human usefulness.
2. **Ranked sensitivity:** failure if restored contribution <=0.01 OR at least two
   other originally available groups exceed it by >1e-6. This retains a top-2
   importance requirement but still permits any number of emitted reasons with
   ties. Report separately; do not silently equate this with the historical event
   whose sign threshold was 1e-6 and candidate set was exactly top-2.

Historical grouped revision remains unchanged and is a separate reference.
If only rank swaps fail and meaningful-positive failures are negligible, narrow
the paper claim. A released empty set has undefined conditional failure risk.

## Evidence and fixed strategy families

K=8. For every candidate: positive-sign support (>1e-6), meaningful support (>0.01),
top-2 support (rank tolerance 1e-6), joint meaningful/top-2 support, attribution
variance and quantiles 0.1/0.5/0.9. These are completion frequencies/quantiles, not
posterior probabilities or confidence intervals for causal truth.

Completion A: exactly the saved historical K8 training-neighbor donor draws.
Completion B: training-only conditional forest already specified in the initial
spec (16 trees/field, depth6, min_leaf40, max50k training rows, two fixed sweeps).
Same predictor/attribution reference, same erased inputs; no labels/verification
values in either family. Train B on the original predictor training partition only.
Do not change its hyperparameters in response to results. Sample validity and
observed/natural-cell invariance are mandatory gates.

Stable-Core rule: sign support >=s AND top-2 support >=r AND lower attribution
quantile >q. Grid s,r in {0,.5,.625,.75,.875,1}; q in {.01,.02,.05}. Thus 108 pairs
of support/magnitude settings (108 triples). Calibrate A-only, B-only and both
separately. Both uses minima of support and lower quantile across A/B and therefore
must pass the same thresholds in each family. It is completion-model agreement,
not worst-case robustness. Also report disagreement and hidden-truth support.

Baselines use the SAME reason-failure target/calibration customers:

- Release all candidates (uncontrolled reference).
- Whole-explanation MC: top-1, top-2 and all candidates; release the entire chosen
  set if estimated any-reason failure is below its calibrated threshold. The main
  historical-score comparator uses the unchanged sign/rank MC top-2 score; its
  threshold is recalibrated to the common reason-level target for this experiment.
- Reason rank support only; SHAP variance only; reason-level empirical verified-
  target frequency only; current strength only; lower quantile only.
- Prediction entropy only, applying one release decision to all candidates.
- Stable-Core A, B, A-and-B.

Frequency/whole MC grids use {0,1/8,...,1}; strength/variance/quantile/entropy grids
use 21 quantiles of **current-only revision-calibration evidence**, plus endpoints.
That pool does not fit final release thresholds or use restoration labels. All
current-score grids freeze before release calibration, so conservative finite-grid
checks do not choose candidate thresholds from their own labelled samples.

## Calibration, objective and fairness

For each target and alpha .05/.10/.15, maximize **candidate-reason coverage** on
release calibration subject to each declared environment's empirical micro risk
<=alpha. Tie-break on customer coverage>=1, then deterministic grid order. All
policies share this rule. No threshold chosen on assessment. A separately reported
conservative sensitivity uses independent clusters, maximum G reasons/record,
`D_cluster = mean_rows((failed-alpha*released)/G)` within environment, and
`mean(D)+sqrt(log(number_grid_points * environments/.05)/(2*n_clusters)) <=0`.
This targets ratio of expected cluster-mean counts under stationarity/independence;
it is not a finite-sample guarantee under shift. Primary empirical ratios remain
row/reason weighted; report cluster-weighted conservative calibration explicitly.
Empty environment releases cannot certify a policy; if no candidate passes, abstain.

Report whole-explanation any-failure AND reason-level micro risk. The two budgets
are different: never use improvement under the easier reason target as evidence of
better any-reason risk control. Show calibration and assessment separately and
count each assessment environment above its nominal budget, without retuning.

For every Stable-Core policy, construct paired current-strength and deterministic
random controls with **exactly the same number of reasons on each same customer**.
They receive no verification labels. Report paired reason-failure differences.
These size-matched controls diagnose reason identity benefit; they are not cheaper
deployable policies because their sizes are inherited from Stable-Core. Standalone
strength/top-1/top-2 baselines supply practical comparisons.

## Evaluation, runtime, plots and gates

Primary metrics: reason risk, candidate-reason coverage, customers with >=1/>=2
reasons, mean/count distribution, candidate-survivor recall, any-failure risk,
per-group failure, and invalid-attribution counts. Report per environment and equal
environment mixture, dataset separately. Bootstrap 1,000 times over original
customer IDs (Polish duplicate clusters), preserving Taiwan folds and all repeated
conditions. Use paired draws for difference CIs; no completion/reason pseudoreplication.
Fixed-policy CIs omit training/calibration selection uncertainty.

Plot A risk/reason coverage; B risk/customer coverage; C whole/partial release;
D reason-count distribution; E per-group failure; F completion-family comparison.
Policy families' full calibrated tradeoff paths are descriptive assessment curves,
not a source of new operating thresholds. Log cold fit time, completion and SHAP
time, and five repetitions of end-to-end serving on fixed first16 MCAR30 customers
per dataset (fold0 Taiwan). Do not compare cached scoring time against uncached MC.

Success needs useful coverage at attained reason risk, benefit over the strongest
whole/top-1 baseline, and reason identity benefit at matched size. Failure of rank
or strength controls invalidates an algorithmic advantage claim. Both-family empty
sets, false stability, conditional-model dependence, or failed Freddie transfer are
valid negative findings. No new model search is a fallback.

Serving returns risk raw/calibrated, candidate groups, released groups, withheld
groups, frequency/quantile diagnostics, declared target/budget/calibration scope,
and status NONE/PARTIAL/ALL_CANDIDATES. A status describes set size, not a guarantee.
Restored attributions and failure labels exist only in separate verification output.
