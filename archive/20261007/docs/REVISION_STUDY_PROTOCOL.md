# Prospective development study, 2026-10-03

Status: frozen before the new diagnostic runs. Historical reference commit:
`897a76d`. This is a bounded research continuation, not a final paper experiment.
The previous 4,500 test customers are excluded from every new operation except
checking exclusion IDs. The original 4,500 risk-calibration customers stay reserved.
Earlier reports and metric implementations remain unchanged.

## Boundaries

Use the union of the original train/development/probability-calibration partitions
(21,000 customers). A newly declared stratified three-fold development protocol
uses seed 20261003. Each outer training pool has 14,000 customers, partitioned
before preprocessing into predictor train 8,000, early stopping 1,000, default
probability calibration 1,000, revision training 2,000, diagnostic development
800, revision probability calibration 600, release-risk calibration 600.
All copies of a customer stay together. Restart seeds are 101, 102, 103, one per
outer fold; variation consequently mixes split and training/mask effects.
This is internal cross-validation on a historically explored benchmark, **not
three independent confirmation datasets or a newly untouched test set**.

Do not inspect any outer assessment until all folds' diagnostics, decision,
models, calibration and policies are frozen. No candidate grid or outer-driven
retry. A later independent dataset/time cohort is still required.

## Diagnostic phase (before conditional strategy implementation)

Refit LR, 25-view XGBoost (depth3/200 trees), additive depth1/600-tree control,
augmented vanilla GRU and augmented mask/delta GRU. Preserve old preprocessing,
25-view unit total weight per customer, BCE, 25 epochs/early stopping, and 64
training reference customers. Directed GRU ablations are mask-only, delta-only,
static-only and temporal-only. These are controls, not a predictor selection grid.
Use identical monthly mapping and masks across all models. Main conditions:
complete, nested MCAR10/20/30 and anchor MAR30.

On 384 label-independent diagnostic customers per fold, audit k=2/3/5,
positive magnitude .001/.01/.05, rank tolerance 0/1 and rank tie gap
1e-6/.005/.01 raw-logit units. Keep the sign boundary at 1e-6 separately.
The frozen operational target stays k3/.01/rank0/epsilon1e-6. Report eligibility,
pair-validity, events, top-k overlap, sign agreement, Spearman correlation and
observed normalized L1 shift (sum absolute change divided by the sum of endpoint
absolute attributions). Do not optimize these choices.

Semantic sensitivity sums **only originally observed** monthly contributions
into repayment, bill and payment groups, with five static fields separate.
It explains an observed subset of a history, not hidden facts. Correlated-month
swaps inside one group disappear by construction; grouping is a diagnostic,
not proof that financial fields are interchangeable.

For recurrent models compare fixed-mean IG with average IG over four empirical
training references. On a fixed 48-customer subset, also use a common original-
field permutation Shapley game (availability fixed, same training references and
permutations before/after) for tree and recurrent predictors. Finite-permutation
noise and differing games must remain explicit.

Measure fixed-pair logit second differences; same-imputed-value predictions
and explanations under all-observed masks/ordinary monthly deltas; literal
zero observed-mask inputs and zero deltas as deliberately inconsistent stress
inputs; and value-only versus full restoration. These distinguish functional
sensitivity, not causal mechanism or natural informative missingness.

Experiment B fits generic revision classifiers on held-out revision-training
customers, using current probability/entropy/missing fraction/completion
prediction variance, then adding current attribution statistics. Restoration
outcomes are labels only. Report fixed-rate comparisons and paired customer
bootstrap intervals. Cases B use all three fixed |delta p| cutoffs .01/.02/.05;
probability stability is not correctness or an inference-time confidence label.

## Decision gate and one conditional strategy

Continue only if robust revision is non-trivial (at least 5% in a nonlinear
model under more than one missing condition), stable/revised cases are present
(at least 20 pooled at .02), predictive signals are not near-perfect revision
detectors (AUROC/AP .95 would prompt stopping), and the literature does not
establish the same verification target as already solved. These are feasibility
criteria, not hypothesis-test cutoffs or final novelty evidence.

If GO, pursue **frozen 25-view XGBoost plus a separate calibrated generic
revision selector**. No joint neural head, interaction suppression or new loss.
The simple selector is deliberately the strongest architectural falsifier:
if it works, contribution is evaluation/target/policy, not a novel network.
Fixed regularized logistic and small histogram-gradient-boosted selectors are
controls; choose their fixed feature-set recipe using inner development only.
Primary proposed strategy uses current prediction, mask-group and attribution
statistics; completion-based baselines share the same completion budget.
No raw hidden values, restored prediction or restored explanation enter selector
inference. This definition is frozen before results and may be rejected, not
replaced with another architecture after assessment.

Conditional completion sensitivity uses a training-only nearest-neighbour donor
distribution over 32 neighbours, 8 draws, preserving joint hidden-field values
from a donor. This is a cheap approximate conditional imputer, not MICE, a
posterior, or an imputation novelty claim. Keep currently observed cells exact;
never use y in neighbour selection. Missing fraction, entropy, max probability,
completion probability variance, attribution variance, rank/sign instability,
and Monte-Carlo revision are explicit comparators. The single-imputation reason
set is fixed; completion draws estimate its possible revision, not its selection.

Calibrate default and revision probabilities on distinct designated pools.
Learn release thresholds on release calibration only. Report empirical 10%-risk
thresholds and conservative simultaneous binomial upper-bound thresholds
separately, with one condition per original customer for the pooled policy.
Evaluate condition-specific distribution shift separately. Ties are included
whole; zero release means undefined conditional risk, never zero risk.

Outer evaluation: all 7,000 predictions per fold, fixed label-independent 800
explanation customers per fold. Report paired original-customer bootstrap
(1,000 draws), fold mean/SD, raw/calibrated metrics, classwise metrics and
all eligibility/invalid counts. Use Pareto dimensions individually, never a
weighted scalar. Six required plots plus calibration curves are generated.
External data remain a recommendation until provenance/license are verified.

AI-assisted source verification, code and analysis are disclosed. Root-run
checks are software audits, not independent peer review.
