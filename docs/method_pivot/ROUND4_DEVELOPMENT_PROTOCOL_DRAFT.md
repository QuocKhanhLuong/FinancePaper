# Round 4 development protocol: reviewed and frozen for a bounded audit

Date: **2026-10-05**. Starting branch `research/method-pivot`, audit base `9f047cd`.
This coordinator-reviewed protocol supersedes the unaccepted worker draft.
It is **exploratory development reuse**, not independent confirmation or a new
method. No historical model, mask, attribution, event or result is changed.

## Question and primary comparison

Can richer statistics of the **same K=8 completion predictions**, with the same
missing mask, approach completion-explanation evidence for detecting grouped
reason revision? Primary comparison: AP(MC8) minus AP(prediction-distribution
boosted detector), on **MCAR30**, fold 0. Secondary: the incremental AP from adding
explanation evidence to the identical detector. This is a predictive comparison,
not a conditional-independence test or information-theoretic proof.

The previous entropy/variance controls were incomplete coverage of possible
prediction-distribution methods. This pilot is still not learned DMV reproduction.
See [MVU source audit](ROUND4_MVU_SOURCE_AUDIT.md).

## Exact data and role boundary

UCI Taiwan local file SHA-256:
`30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
The historical seed-42 train/development/probability-calibration union has
21,000 records. The 4,500 historical test and 4,500 risk-calibration records are
excluded. This audit does not establish that the latter have never been viewed.
All Taiwan development records appeared in prior research; no clean-holdout claim.

Use only fold 0 caches under
`outputs/decisive_validation/taiwan/fold_0/`:

- Existing `revision_calibration`: 600 customers. Sort IDs, permute once with
  NumPy default_rng(20261005). First 400 = detector fit; remaining 200 = score
  calibration. Every condition for an ID follows its assigned role.
- Existing `release_calibration`: 600 customers = evaluation only for this audit.
  It was calibration in an earlier study, so this new role does not make it
  uninspected confirmation data. No release threshold is selected here.
- Do not load historical `outer`, `revision_train` labels or test/reserve rows.
- Check exact IDs against `outputs/revision_study/fold_0/partition_ids.json`;
  fit/cal/evaluation must be disjoint from predictor training and each other.

Partition file SHA-256:
`0fdc5f6abe8409118dec4214d9fd58c7b767b3a0e06fc014fc15e82c869a6454`.
Predictor artifact SHA-256:
`16369aa6a254bc7d22fbd55acd8ba7c1987955817ad6180768113e59dc3ba144`.
Caches include completion probabilities and original-feature attributions;
**no regeneration or new SHAP calls is needed**. Hash all current/verification
files and validate each existing complete.json receipt before evaluation.
The original 8,000-row XGB25 training, preprocessing, background and positive-slope
Platt calibration remain frozen inside the cached outputs. No predictor reload
or training is necessary for this pilot.

The old three-fold scheme isolates roles **within each fold**, but reuses people
across folds. Counts summed over folds are episodes, not unique customers.

## Frozen target and scenarios

Conditions: MCAR10, MCAR30, MAR30, group_missing, exactly their existing caches.
Primary target: `group2` from `reliability/validation.py`.
`aggregate_groups(..., observed_only=True)` sums signed original-feature SHAP
contributions over **initially observed members** of each Taiwan group, applying
the same original mask before and after restoration and to every completion.
Eight groups: repayment, balance, payment, LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE.

These cached attributions are **raw logit units** (FlatAdapter.attribute), not
L1-normalized quantities. Candidate groups have contribution > .01. Display two
positive eligible groups using stable sorting. A displayed reason revises when
its restored contribution <= 1e-6 or at least two eligible observed groups exceed
it by more than 1e-6. The competitor ranks are not restricted to the originally
displayed groups. Use revision_arrays(k=2), unchanged.
Eligibility uses the current candidate count, current validity, all 16 cached
completion validity flags (historical rule); restored validity only determines
whether the offline target is measurable. Report exclusions, never silently skip.
The truth and restored outputs remain confined to labels/offline diagnostics.

## Fixed controls and information budget

Use the first eight stored completions for **every** score. Probability controls
receive calibrated current probability, natural-log entropy, completion mean,
variance/std, .1/.5/.9 quantiles, positive-vote fraction, modal hard confidence,
soft confidence, current-action agreement, all eight **sorted** probabilities,
23 missing indicators and missing fraction. Sorted probabilities preserve the
whole empirical univariate distribution; no completion explanations are supplied.
The threshold tie convention is >= .5; hard-vote and current-action agreement
are distinct (`prediction_distribution_features`).

Evaluate these fixed scores, without choosing a winner to define a new method:

1. Current entropy.
2. Completion prediction variance.
3. 1 - hard confidence.
4. 1 - current-action agreement.
5. Missing fraction.
6. Prediction-distribution boosted detector on the above current-only features.
7. MC8 grouped revision frequency.
8. Grouped rank-instability fraction.
9. Same boosted detector plus four explanation statistics: MC8, rank instability,
   sign instability and mean observed-group attribution variance.

Both learned controls use the historical generic classifier capacity:
HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
min_samples_leaf=40, learning_rate=.05, l2_regularization=5,
early_stopping=False, random_state=101). No hyperparameter search.
This is **baseline adaptation**, not a proposed revision head.
Train on eligible fit episodes from all four conditions. Fit separate positive-slope
Platt score maps on the calibration role only; use logit(clipped score) for bounded
frequency/probability scores, identity for entropy/variance. Report raw ranking AP
and AUROC, calibrated Brier/log loss. No thresholds or risk guarantees are claimed.

## Statistics and stopping rule

Primary endpoint: paired AP difference MC8 minus prediction-distribution detector
on MCAR30 eligible evaluation customers. Secondary paired endpoint: augmented
minus prediction-only detector on that same cohort. Bootstrap original customer
IDs with 1,000 draws, seed 20261005; all scores use identical draw weights.
Reject single-class bootstrap draws and report the retained count. Report events,
prevalence, AP/AUROC, Brier/log loss by condition and the two differences with 95%
percentile intervals. No formal multiple-comparison significance claim.

Pilot seed: 101 (fixed predictor restart), split/bootstrap seed 20261005.
A positive mechanism-necessity gate requires **both** paired lower bounds > 0 and
primary AP gain >= .05. Otherwise stop expansion; call the result negative or
inconclusive as supported. This gate would justify a three-restart baseline audit,
not a novel-method claim. Any future novel construction still needs a separate
formulation/prior-art gate. No predictor training or architecture search.

Stop on file/hash drift, customer-role overlap, invalid completion/mask contract,
nonfinite score, fewer than 20 eligible evaluation events/non-events on the primary
condition, or single-class fit/calibration. Do not switch target/cohort to recover
a result. Total pilot budget <= 600 seconds CPU; interrupt before expanding if
exceeded. Preflight feature extraction on a bounded slice before full computation.
No measured runtime is asserted before the run.

## Audit evidence and correction boundary

Worker receipts under `outputs/method_pivot/round4/protocol_audit/` locate existing
schemas and hashes. Their initial source-only PASSED labels and timing guesses
are **not accepted as execution evidence**. The runner must independently assert
row IDs, donor isolation, artificial/natural separation and input allowlists.
The worker loaded ten fitting CSV rows for schema types; no performance aggregate
or evaluation outcome was used to choose this protocol. The frozen final cache
selection above is by role and availability, not favorable results.

This protocol must be committed and its configuration/input manifest frozen
before opening evaluation labels. All raw arrays, row outputs and fitted detector
artifacts remain ignored. Aggregate reports and independent code can be committed.
