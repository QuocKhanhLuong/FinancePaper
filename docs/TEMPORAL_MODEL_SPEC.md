# Temporal pilot specification

Frozen 2026-10-03 after [literature decision](TEMPORAL_MODEL_RESEARCH.md), before
implementation/evaluation. This additive branch preserves the existing static
pilot and its event. Stage F and the final five-seed study are not part of this run.

## Data, splits and tensors

Use the existing checksum-pinned Taiwan loader and seed-42 five-way stratified
split: train/development/probability calibration/reason-risk calibration/test =
50/10/10/15/15%. All masks/copies stay in their original-record partition. Preserve
the reason-risk calibration split unused until a release selector is implemented.

[Official mapping](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients):

| Month | status | bill | previous payment |
|---|---|---|---|
| April | PAY_6 | BILL_AMT6 | PAY_AMT6 |
| May | PAY_5 | BILL_AMT5 | PAY_AMT5 |
| June | PAY_4 | BILL_AMT4 | PAY_AMT4 |
| July | PAY_3 | BILL_AMT3 | PAY_AMT3 |
| August | PAY_2 | BILL_AMT2 | PAY_AMT2 |
| September | PAY_0 | BILL_AMT1 | PAY_AMT1 |

Static order is LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE. No static repetition in
time; no derived financial ratios in this pilot. PAY codes are categorical,
including unusual codes; do not invent meanings or merge categories.

Fit one shared preprocessor on complete training records. Numerical imputation
median and mean/standard-deviation scaling are per original field. Categorical
fill is each original field's training mode (smallest code breaks ties). Use one
training-only union vocabulary for all six PAY fields, separate static category
vocabularies, and all-zero encoding for unseen categories. Persist mappings,
statistics and training record IDs. No target enters preprocessing.

Raw temporal tensor is `[B,6,3]`. Encoded temporal values are
`[B,6,C_PAY+2]`, status one-hot then scaled bill/payment. Static values are
`[B,2+C_SEX+C_EDUCATION+C_MARRIAGE]`. Observed masks have shape `[B,6,3]`
and `[B,5]`, `1=observed`. The existing dataframe hidden mask uses the opposite
convention, `True=hidden`; conversion must be explicit.

Monthly delta `[B,6,3]`: `delta[:,0]=0`; for `t>0`,
`delta[t]=1+(1-observed[t-1])*delta[t-1]`. Retain elapsed months in diagnostics;
divide by five on model input. Fully observed history has `[0,1,1,1,1,1]`,
not all zeros. Sequence-origin zero means unknown pre-April gap is not inferred.

`TemporalPreprocessor.transform(X, hidden_mask)` returns `TemporalBatch` with
numpy float32 `temporal`, `static`, `temporal_observed`, `static_observed`,
`delta`; `take(indices)`, `to_torch(device)`, `flatten(include_metadata=True)`
provide aligned views. Metadata-enabled flatten returns `(matrix, origins)`;
values-only flatten returns a matrix. `transform` fills the masked fields itself; never send
unmasked hidden truth into normal prediction. For diagnostic value-only
restoration, combine complete encoded values with the original context explicitly.
Encoded-origin metadata maps each value dimension to one of the original 23 fields.

## Missingness

Evaluation conditions: complete, nested MCAR10/MCAR30 using the same uniform
matrix per split, and MAR30. MCAR may hide any of the 23 fields. MAR leaves AGE
and LIMIT_BAL observed, fits anchor means/scales and intercept on train only,
then draws independently per nonanchor field with
`sigmoid(intercept + .75*z_age - .75*z_limit)`.
Choose intercept by bisection so expected **all-23-field** training missing rate
is .30 (thus nonanchor expectation `.30*23/21`). Never condition on y or hidden
target-field values. Report actual rate on every split; MAR30 and MCAR30 differ
in mechanism and eligible fields, not only a severity label.

Training augmentation: per original record per epoch sample rate uniformly from
`[0,.1,.2,.3]`, independent Bernoulli fields. Same number of optimizer steps as
complete training. Dedicated deterministic RNG streams for shuffling, masking,
dropout/model initialization, and evaluation. No artificial copy crosses a split.

## Architecture and loss

Primary `MissingAwareGRU`: ordinary one-layer GRU (explicit six-step GRUCell loop),
hidden 64, input `[encoded values, observed mask, delta/5]`. Static MLP output32
from `[encoded static values, static observed mask]`. Concatenate final recurrent
state + static embedding; fusion Linear64, ReLU, dropout .1; scalar risk-logit
head. No attention, learned decay, reconstruction or revision head. `TemporalGRU`
is the same value/static architecture without mask/delta inputs. Configurable
`use_mask`, `use_delta`, `use_static`, `use_temporal` support future ablations.

Forward interface accepts a tensor dictionary matching `TemporalBatch` fields,
and `debug=False`. Returns a nested dictionary:

```python
{
  'risk': {'logit': tensor, 'prob_raw': tensor, 'prob_calibrated': None},
  'missingness': {'fraction': tensor, 'temporal_fraction': tensor,
                 'static_fraction': tensor, 'per_month': tensor},
  'temporal': None,  # debug: hidden_states, summary_embedding
  'embeddings': None,  # debug: static, fused
  'explanation': None,  # populated by attribution/evaluation, not fake labels
  'uncertainty': None,  # no ensemble in this first branch
  'reliability': {'revision_risk': None, 'release': None}
}
```

Normal inference does not serialize hidden tensors. Debug NPZ is opt-in, with
record IDs and original-field/month names. Explanation output includes feature,
score, sign, rank, monthly totals, missing field names and eligibility. Eligibility
means enough positive observed reasons; it is **not** a calibrated release decision.

Default loss BCEWithLogits. Weighted BCE `pos_weight=n_negative/n_positive`
uses train labels only. Focal `-alpha_t*(1-p_t)**gamma*log(p_t)`, gamma2,
positive alpha .75 (negative .25), independently configured; no additional weights
or resampling. Compute stably with log-sigmoid/BCE. Log exact losses and ratios.

## Stages, budget and selection

A: reproduce existing CPU static pilot separately, byte-compare scientific files.
It uses its original preprocessing/SHAP definition and is labelled `legacy`.

B–D one-seed exploratory runs: shared flat LR and XGBoost complete; both with
one fixed augmented copy per training record; vanilla complete/augmented GRU;
mask-delta complete/augmented GRU; augmented mask-delta weighted BCE/focal.
Flat shared features include exactly the values, masks and deltas available to
the primary GRU, flattened without repeating static features. Static trees see
month identity through column identity. The fixed augmented LR/XGBoost use the same
first-epoch masks as neural augmentation, and the same number of rows; it is a
conservative bounded control, not equal exposure to every neural epoch's masks.
Complete-only XGBoost with masking indicators constant in training cannot learn
their effect. Distinguish that from its augmented control.

Neural AdamW, lr .001, weight decay .0001, batch256, max25 epochs, patience5,
gradient-norm clip5. Checkpoint by mean development AP over complete/MCAR10/
MCAR30; ties break by lower unweighted mean log loss. MAR is held out as mechanism
stress, not used to select checkpoint. No hyperparameter search in the first run.
Flat LR C1; XGBoost depth3, lr.05, 200 estimators, min_child_weight5, n_jobs1.
These fixed shared controls are separate from tuned legacy baselines.

Fit a positive-slope Platt calibrator per frozen model on equal pooled
complete/MCAR10/MCAR30 probability-calibration records. MAR tests calibration
transfer. Choose raw and calibrated F1 thresholds separately on pooled development
records at those same three conditions, deterministic lower tie. Any later
intervals resample original records, not their correlated condition copies.
Report calibration metrics separately for every condition. Test is opened
only after model checkpoints, calibrators, thresholds and attribution rules freeze.

E: predeclared explanation models are vanilla augmented BCE, mask-delta augmented
BCE, and augmented shared XGBoost; no selection using test scores. Prediction
uses all 4500 test records. Expensive attribution uses a label-independent fixed
subset of 400 test and 256 development IDs selected by a separate seed and shared
across models. Report that smaller denominator prominently. Full-test attribution
is a configurable later run. Same train-background64 IDs as the original pilot.

## Attribution and restoration

IG on raw logits, eval mode/dropout off, frozen train64 encoded-mean value
baseline. Integrate gradients with Gauss–Legendre quadrature (initial32 nodes,
retry64/128 if completeness fails). Hold each endpoint's masks/delta fixed on its
path. Sum signed one-hot contributions into the corresponding financial field.
Original-feature `[N,23]` output feeds existing `extract_reasons` and
`revision_event`, k3 and rank tolerance0. Log baseline/input logits and residual;
tolerance is `abs(residual)<=.002+.001*abs(logit-baseline_logit)`; exclude failed
attribution pairs from reason conclusions and report the failures, never silently
zero them. Record integration nodes actually used.

TreeSHAP reference for shared XGBoost uses the same64 complete training IDs and
raw margin; metadata attributions are retained separately, excluded from released
financial reasons. Additivity applies to all encoded values **and** metadata.
The old XGBoost pilot remains the pure value-SHAP reference. IG is conditional on
metadata and TreeSHAP is joint; this limits cross-model interpretation.

Two restoration diagnostics with the same predictor:
1. value-only: true hidden financial values, original mask/delta;
2. full: true values and fully observed mask/recomputed deltas.

The second is primary verification. Always pass the **original hidden mask** to
the event so newly restored fields cannot become observed-reason competitors.
Also log same-imputed-values/full-observation-context probability change to
isolate direct metadata influence; this intentionally inconsistent input is only
a sensitivity diagnostic, not a real customer counterfactual or causal effect.

Development-only reason magnitude grid `[.001,.01,.025,.05]`: largest threshold
retaining90% of eligibility at .001, pooled MCAR10/MCAR30. No high/medium/low
reliability labels. Report event/coverage and risk-coverage diagnostics sorted by
missing fraction (ties handled together); the simple revision score is exactly
the originally hidden fraction, larger meaning higher risk. Report its AUROC/AP
when both event classes exist; restored values never enter this score. Any
coverage at empirical risk<=10% is descriptive, not a calibrated guarantee.
For matched coverage, first intersect eligible, attribution-valid IDs across
the three models for each condition. The denominator remains all400 selected
test IDs. Clip target coverages .25/.50/1.0 to the common eligible fraction,
select the same number per model using descending pre-restoration kth reason
strength (record-ID tie breaking), and report common/selected counts. Missing
reason sets cannot be filled with ineligible rows. This is an evaluation
diagnostic, not a deployed selector. These subset estimates never establish the
five-seed improvement gates in the research document.

## Reporting, reproducibility and hardware

Report all requested discrimination/calibration metrics and classwise support,
probability shift/full-vs-value-only restoration, top-k overlap/rank/sign agreement,
revision rate conditional on eligibility, and stable-prediction/revised-reason
counts at raw |delta p|<=.02. Keep AP primary. Ten fixed-width calibration bins
include counts and empty bins; ECE is sample-weighted, descriptive/bin-dependent.
No certainty/insufficient-information label from an arbitrary probability cutoff.

Persist resolved config, data/source/dependency hashes, splits, masks/seeds,
preprocessor/background IDs, frozen checkpoint selection, calibration parameters,
epoch histories, parameter counts, device/version, wall time, predictions,
summary/calibration/reason CSVs and attribution NPZs under an ignored output folder.
Refuse overwrite of a nonempty output. Debug tensors/models/raw data stay ignored.
Record CPU/MPS reproducibility limits; seeds do not guarantee byte identity across
hardware/PyTorch versions. Repeat CPU deterministic smoke, not every full training.

Device priority: MPS then CUDA then CPU; explicit device override for tests.
Float32 only; no AMP/CUDA-only dependencies. Use standard Linear, GRUCell,
sigmoid/tanh, autograd and losses. Device-info prints PyTorch version, selected
device, MPS availability and parameter count. Unsupported MPS operations must
be identified; no silent fallback claimed as MPS execution. A CPU override is
the documented fallback. [PyTorch MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html).

Implementation-time runtime finding: PyTorch2.14.1 on this Mac passes isolated
GRUCell MPS forward/backward, but the combined CPU numerical-library test process
segfaulted under default thread pools. Explicit `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` before imports resolved the combined
suite. The CLI supplies these defaults; test commands set them explicitly. This
is a CPU thread-pool compatibility workaround, not an unsupported MPS operation
or a silent device fallback. It can reduce CPU throughput; record it in runtime
metadata and keep normal float32 MPS execution.

Required tests: exact chronology, train-only scaling/vocabularies/unseen values,
mask/delta runs, shared flatten parity, deterministic MCAR/MAR/augmentation,
vanilla context invariance, absent branch ablations, finite losses/train ratio,
CPU deterministic training smoke, MPS forward/backward/IG when available,
IG completeness/aggregation, unchanged original revision semantics, held-out
calibration boundaries and clean=restored negative control. Five seeds, full
structural ablations, reconstruction and a calibrated revision head remain planned.
