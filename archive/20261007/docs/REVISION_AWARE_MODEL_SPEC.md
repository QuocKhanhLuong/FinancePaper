# Frozen predictor + calibrated revision selector

Status: selected after development gate, before outer assessment. One strategy;
all learned components are standard. See [decision](NEXT_MODEL_DECISION.md).

## Tensors and boundary

Existing canonical 23 original fields, six monthly blocks and five static fields
remain unchanged. Predictor is the existing 200-tree depth3 XGBoost with 25
train-only masking views, unit total customer weight, existing median/mode,
scaling/one-hot and mask/delta metadata. Training uses 8,000 customers/fold.
Default calibration is independent positive-slope Platt, class threshold selected
on early-stopping development. Its prediction is **identical** with/without selector.

`CurrentEvidence` accepts only `[B]` current probability, `[B,23]` hidden mask,
`[B,23]` current original-field TreeSHAP, `[B,8]` completion probabilities, optional
`[B,8,23]` completion attributions. No full-information values, targets, probabilities
or explanations are accepted. Current-only features use explicit column allowlists.
Offline `collect` writes `current.csv` and `verification.csv` separately; their
keys must match before fitting supervision. Verification labels never enter
the serving object's feature extraction.

Predicted quantity is `P(E=1 | current information, current eligibility)` for the
frozen strict k3/.01 event. No label exists when three observed reasons cannot
be released; those records are withheld, not mislabeled as reliable negatives.
TreeSHAP validity must pass. Unknown restored validity is evaluation-only.

## Features, fitting and controls

Prediction recipe: calibrated probability, entropy, completion probability
variance, missing fraction. Attribution recipe adds count of observed positive
contributions, observed absolute/positive/negative sums, third-reason magnitude,
third/fourth rank margin. Full recipe additionally adds static/temporal missing
fractions, original-field masks and chronological monthly missing fractions.
Hidden-field attributions are excluded from these summary features.

Fixed selector controls: standardized logistic C=1 or histogram gradient boosting
120 iterations, 7 leaves, minimum leaf40, learning rate .05, L2=5. No search over
these parameters. Fit on 2,000 held-out revision-training customers (all conditions
remain together); choose a full/attribution recipe within each fold using inner
diagnostic AP. Prediction-only and mask-only controls are retained separately.
No encoder gradients or joint revision loss; no class weights by default.

Completion sampler uses only predictor-training customers: 2,048 deterministic
reference rows, observed-field IQR-scaled clipped numeric distance/categorical
mismatch, 32 nearest donors, 8 draws of joint hidden values. Observed cells stay
exact. This is an approximation/sensitivity distribution, not a posterior or
MC-dropout estimate. Never include default labels or hidden query truth in distance.
Full completion SHAP supplies variance, rank/sign instability and MC revision
baselines; the current single-imputation reason set remains fixed. The proposed
selector uses completion prediction variance, not completion SHAP. Budget and
runtime of the more expensive baselines are reported.

## Calibration and release

Fit separate revision Platt calibration on 600 customers, one randomly assigned
condition per customer from complete/MCAR10/20/30/MAR30 (fixed RNG, before scores).
Release-risk calibration uses another 600 customers and independent assignment.
This avoids treating five masks of one person as independent Bernoulli trials.
Baseline scores are also calibrated; degeneracy uses training/calibration prevalence
only and is reported. Revision calibration never changes default predictions.

Use the fixed threshold grid `{0,.01,...,1}` on calibrated revision probability.
An explanation is released iff currently eligible and probability≤threshold.
Choose maximum calibration coverage satisfying empirical revision≤.10; also
report a conservative policy requiring a one-sided Clopper–Pearson upper bound
≤.10 simultaneously across the finite threshold family (Bonferroni .05/101).
No feasible threshold means withhold all; conditional risk is undefined, not zero.
Ties remain whole. Bounds assume iid customers from the declared pooled condition
mixture; they do not certify MAR separately or survive arbitrary distribution shift.
The empirical policy has no finite-sample risk guarantee.

All folds' selector artifacts/calibrators/thresholds are hashed and frozen before
outer batches are constructed. Assess 7,000 outer predictions and a random fixed
800-customer explanation subset/fold. Do not refit after seeing assessment.

## Outputs and cost

Common inference dictionary: prediction `{logit,prob_raw,prob_calibrated}`;
missingness `{fraction_total,fraction_temporal,fraction_static,features_missing,
months_missing}`; uncertainty `{predictive_entropy,prediction_variance,
ensemble_or_mc_std}`; explanation `{top_features,top_scores,signs,rank}`;
reliability `{revision_score,revision_probability,release_decision}`.
The uncertainty std is specifically completion sensitivity, not a calibrated
interval. Verification-only `{prob_full,prob_shift,revision_event,num_revised_reasons}`
belongs to a separate evaluator; serving output has no such field. No qualitative
confidence labels are hardcoded. Debug tensors optional; no embeddings are invented
for tree models. Normal serving needs current SHAP and 8 inexpensive prediction
draws, not 8 completion explanations.

Trees/selectors use CPU. Existing GRU/IG controls use MPS when available, then CUDA,
then CPU; this strategy requires no CUDA, mixed precision or new dependency.
Source hashes, dataset hash, partition IDs, device, runtime, predictor epochs and
parameter counts live in manifests/fit summaries. Hardware peak-memory measurement
is not implemented and must not be inferred from tensor dimensions.
