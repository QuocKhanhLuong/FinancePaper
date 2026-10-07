# First reproducible pilot

This implements the requested Phase 1 sanity pipeline. The methodological
precedence remains EXPERIMENT_PROTOCOL → MODEL_PIPELINE → RESEARCH_CONCEPT →
README. The coding task explicitly narrows the protocol's broader pilot matrix
to **Taiwan, one seed, complete + MCAR 30%, single median/mode imputation**.
MAR, conditional/multiple imputation, native-missing robustness, selectors,
probability/risk calibration, five seeds and South German are deferred. No
paper-level risk–coverage or accepted-risk guarantee is reported.

## Data and fitting boundaries

1. Load complete records from the pinned official XLS (or explicitly supplied,
   fingerprinted CSV). ID becomes an audit index, never a predictor. Target is
   separated immediately. Preserve all categorical codes without ad hoc recoding.
2. Stratify original records into train/development/probability calibration/risk
   calibration/test with 50/10/10/15/15% proportions. The last two calibration
   partitions remain reserved and unused. Each original record has one mask per
   evaluated condition and never appears in a different split.
3. Train on complete training records. Both models use numeric train medians,
   categorical train modes, and a train-fitted one-hot encoder; LR also scales
   numeric fields. PAY repayment-status codes are categorical. Unknown categories
   are encoded as all-zero dummy blocks without changing the fitted vocabulary.
4. LR selects C by three-fold stratified CV **inside train**, refitting the entire
   preprocessing pipeline in each fold. Selection minimizes mean log loss.
   XGBoost searches depths 2/3/4, uses complete development records for log-loss
   early stopping, and retains exactly the best trees with `save_best=True`.
   This ensures SHAP and probability prediction use the identical tree ensemble.
   No class weighting is enabled. Fitting uses one thread.
5. Draw a fixed background of 64 training IDs, shared by both model families.
   Its model-specific encoded representation is frozen for all explanations.
6. Independently mask original development and test cells with Bernoulli
   probability 0.30, before encoding. The generator uses only shape and axes.
   Actual cell missingness is reported; no conditioning forces exactly 30% or
   enough observed features. Seeds for splitting/model fitting, development
   masks, test masks and background are recorded separately.

## Attributions and development choices

Both models explain **default-class raw log-odds**, not probability-scale SHAP.
LR uses exact independent-background linear SHAP:
`phi_j = beta_j * (z_j - mean(background_j))`. XGBoost uses exact interventional
TreeSHAP, with explicit `model_output="raw"` and a fixed background. This is an
attribution convention, not a causal interpretation. Encoded contributions are
summed **with their signs** back to the original 23 fields before ranking.

For each model, on MCAR development records only:

- Choose the probability threshold maximizing F1 over 0.01, 0.02, …, 0.99;
  the smaller threshold wins an exact tie. Use this frozen threshold for all
  subsequent prediction metrics in both test conditions.
- Choose the largest minimum attribution in `[0.001, 0.01, 0.025, 0.05]` retaining
  at least 90% of development explanation coverage at the smallest magnitude.
  This is a declared pilot heuristic for meaningful positive reasons. It uses
  no restored attributions, revision labels or test data, and is **not** risk
  calibration. If no development record is eligible, choose the largest value
  and report zero eligibility honestly.

The grids, 90% retention rule, `k=3`, numerical epsilon `1e-6`, rank tolerance
zero and stable-prediction diagnostic `|p_before - p_restored| <= 0.02` are
configured before running. `frozen_selection.json` is written before test masks,
predictions or explanations are computed. The 0.02 definition means a shift of
at most two percentage points; it is a diagnostic convention, not a learned or
validated reliability threshold.

## Reason eligibility and the revision event

Let O be the original observed-feature set, and let R contain the three highest
**positive signed** original-field contributions in O exceeding the selected
minimum magnitude. Exact ties break by canonical schema order. If fewer than
three qualify, withhold: store `eligible=false`, an empty reason list and a
**null** revision event. Do not pad the list or count withholding as success.

Restore the true hidden values only in evaluation, then apply the **same frozen
preprocessor, predictor and SHAP background**. For an eligible explanation:

```text
E = 1 if any j in R satisfies either:
    phi_restored[j] <= epsilon
    count(l in O where phi_restored[l] > phi_restored[j] + epsilon)
        >= k + rank_tolerance
E = 0 otherwise
```

The first rule treats zero and contributions below numerical resolution as
non-increasing. The second requires a meaningful exit from the top-k observed
fields; near-ties do not cause failures. Newly restored hidden fields never
enter O, even if their contributions are large. Internal rank swaps among the
same accepted reasons are diagnostics, not revision events. The threshold for
initial eligibility must exceed epsilon, so unchanged attributions cannot fail
the sign test. Ranks use signed contributions consistently with positive
risk-increasing reasons, not absolute contribution magnitude.

An initially withheld case has no event even if it could receive reasons after
restoration. The restored reason list in the CSV uses the same original O and
minimum magnitude. The event tests the original R under the stated sign/rank
rule; it does not require every surviving reason to exceed the initial magnitude
again. These are explicit pilot operationalizations of the protocol's event.

## Outputs and denominators

- Coverage = eligible / all test records. Here all eligible lists are treated as
  released under a simple eligibility-only baseline; no risk selector exists.
- Reason-revision rate = revised / eligible. If eligible = 0, report null.
- Report prediction metrics on **all** test records, and separately on eligible
  records, before and after restoration: ROC-AUC, AP, recall, F1, Brier and log
  loss. The probabilities are uncalibrated. Undefined AUC/AP are null.
- Report stable-prediction eligible counts, stable-prediction revised counts,
  and revision rate conditional on stable prediction **and** eligibility.
- Diagnostics use only originally observed fields: mean absolute attribution
  distance, sign agreement, signed-rank Spearman correlation and top-k overlap.
  Overlap divides by `min(k, number observed)` and includes nonpositive ranked
  fields; it is distinct from positive-reason eligibility. Empty/constant
  undefined diagnostics are null. Means exclude undefined cases.

The complete-data condition uses an all-observed mask and identical before/after
records. It must have zero revision among eligible explanations. LR must have
unchanged observed-field attributions after restoration under this fixed,
independent background. TreeSHAP may show observed-field changes. Neither a
positive result nor stable SHAP proves causality, explanation correctness,
fairness, useful risk selection or generalization beyond this one pilot.

## Reproduction and tests

`uv.lock` records exact dependencies; `.python-version` defaults to Python 3.11.
Use the same lock, input checksum and configuration for reruns. Compare
`summary.json`, `records.csv`, `frozen_selection.json`, the mask/split CSVs, and
arrays in each NPZ. Small floating-point differences may occur across operating
systems or CPU/library builds; the manifest records that environment. Binary
model serialization and plots are convenience artifacts, not portability claims.
The manifest also records hashes of source files, so an uncommitted working tree
does not masquerade as the recorded Git commit alone. Keep local result folders
when auditing a run; changes to code require a new run directory.

The offline suite checks malformed data, checksums, source/target separation,
stratified disjoint splits, value-independent MCAR, exact restoration, train-only
statistics/vocabularies, signed aggregation and additivity, LR invariance,
TreeSHAP interactions, early-stopping model identity, sign/rank/tie events and
withholding. A synthetic end-to-end run is repeated and its numerical artifacts
compared. It spies on preprocessing fits and changes calibration/test predictors
to verify that frozen selection is unaffected. Synthetic results are never
presented as Taiwan findings.

API references: [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
and [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
