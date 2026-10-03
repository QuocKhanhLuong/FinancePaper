# Freddie large-scale validation protocol

Declared 2026-10-03 before acquiring new raw files or seeing mortgage outcomes.
This additive branch leaves Taiwan/Polish models, masks, events and results intact.
Current execution status is [NOT RUN on real Freddie data](LARGE_SCALE_VALIDATION_RESULTS.md).

## Data, access and reproducibility

Use the single selected [Release 47 cohort](LARGE_DATASET_DECISION.md). The authorized
researcher manually downloads official annual samples after registration and
accepting the applicable Clarity/SFLLD terms. No automated download or mirror.
Store ZIPs under a private local raw directory and write an acquisition receipt
using `configs/freddie_acquisition.example.json`. Receipt fields establish a
researcher attestation, not cryptographic proof of provider authorship. The script
refuses missing receipts, wrong releases, unexpected members and layout drift.

```bash
uv run --frozen python scripts/prepare_freddie.py \
  --raw-dir /absolute/path/to/official/downloads \
  --receipt /absolute/path/to/acquisition.json --stage pilot

uv run --frozen python scripts/prepare_freddie.py \
  --raw-dir /absolute/path/to/official/downloads \
  --receipt /absolute/path/to/acquisition.json --stage development

# Only after fitting predictors/completion models/calibration policies:
uv run --frozen python scripts/prepare_freddie.py \
  --raw-dir /absolute/path/to/official/downloads \
  --receipt /absolute/path/to/acquisition.json --stage confirmation \
  --freeze-manifest outputs/freddie/analysis_freeze.json
```

Pilot prepares 2000–2001 only; development prepares all 12 non-test vintages;
confirmation prepares only 2020–2022 and checks entity/date separation against the
development metadata. The same acquisition receipt covers this release. The
confirmation freeze JSON must contain `release: 47`,
`development_intake_sha256` (the completed development intake manifest hash), and
`artifacts`, mapping exactly `predictor_lr`, `predictor_xgb`, `preprocessor`,
`shap_background`, `donor_completion`, `conditional_completion`, `release_policies`
and `protocol` to objects containing repository-relative `path` and actual `sha256`.
These must refer to fitted, reviewed artifacts, not placeholder files. This file
will be produced by the future experiment integration; the current code validates
its completeness/hashes before opening test ZIPs. A hash is not proof that an
artifact producer obeyed the protocol. Never claim confirmation because a template
passes JSON parsing. Failed intakes remain marked INCOMPLETE and are not overwritten.

Only local files are read. All generated rows/IDs/models/caches stay under ignored
`data/processed/` or `outputs/`. Each intake records ZIP/member hashes, sizes,
row counts, receipt hash, date and protocol/config/code hashes. SHA256 is provenance
metadata, not a license grant. No raw hashes are invented before download. Fail on
unknown columns/field counts; never use the old 32-column origination parser for
the new 31-column origination / 35-column performance layout described by the guide.
Real official-file conformance remains an intake gate despite synthetic tests.

The preparer streams one annual sample at a time, retains only the six history and
twelve outcome months, and writes feature/natural-mask files separately from target
files. It retains censored labels explicitly. It never prints row contents. The
source cohort is all 50k official loans/year, not an outcome-stratified sample.
No substitution of the illustrative Release 47 example files for annual samples.

## Predictor and feature freeze

See [target](FREDDIE_TARGET_DEFINITION.md) and [allowlist](FREDDIE_LEAKAGE_AUDIT.md).
All train-only learned transformations operate before masking augmentation.
Median numerical / mode categorical baseline, one-hot categorical encoding with
unknown categories ignored and encoded columns summed back to original fields.
No demographic proxies or high-cardinality identifiers are added after outcomes.

Exactly LR and XGBoost: LR L2 C=1/max_iter=2000, train-standardized numeric features;
XGBoost CPU hist, logistic objective, depth3, learning_rate .05, max600 rounds,
min_child_weight20, reg_lambda10, row/column subsampling1, early stopping30 on
development log loss. No hyperparameter search. Train XGBoost with 25 deterministic
MCAR views at rates0/.1/.2/.3; weight copies1/25 so they are not 25 independent loans.
LR uses the same masking distribution as a linear control. Use a streaming or
disk-backed matrix if expanded data exceed local RAM; no silent change of views.
Seeds101/102/103, fixed data split, shared evaluation masks. These are training/
mask restarts, not independent population replications. Platt calibration on2014;
show raw and calibrated metrics, even if calibration worsens them.

## Semantic reason groups (fixed before outcomes)

| Group | Fields |
|---|---|
| Credit-score history | Origination classic FICO |
| Income-relative debt burden | Original DTI |
| Collateral leverage and insurance | Original LTV, CLTV, MI% |
| Contract size and terms | Original UPB, rate, term |
| Property use and structure | Units, occupancy, property type |
| Origination context | First-time-homebuyer, channel, purpose, one/multiple borrowers |
| Repayment behavior | Six monthly delinquency buckets |
| Outstanding balance history | Six monthly actual UPBs |
| Contract-rate history | Six monthly interest rates |

These dictionary-based groups are not expert-approved adverse-action reasons.
Use raw-logit interventional TreeSHAP with a fixed64-row training background, same
reference before/after verification. LR uses coefficient × displacement on the
same transformed background. Sum signed encoded contributions to original fields,
then originally observed members to groups at both endpoints. Feature top3 remains
a diagnostic; grouped top2 is primary; group1/3 are sensitivity. Current minimum
contribution .01, sign epsilon1e-6, rank epsilon1e-6, with .005/.01/.02 rank-gap
sensitivity. These are raw-logit units, not normalized scores. No comparison claims
explanation correctness. Partial stable sets are governed by the separate spec.

## Missingness and completion

MCAR10, MCAR30 and one group-missing environment: uniformly choose one of repayment,
balance or rate histories and hide its originally observed cells. Report realized
fractions. No labels affect masks; natural and artificial masks are disjoint.
Restore only artificial cells. Current inference does not receive truth.

Compare K8 training-neighbor donors (historical max2048, nearest32) to a fixed
conditional-forest completion model; details in the stable-core spec. Do not select
one using assessment performance. Report each and the predeclared intersection
separately. No third imputer or new predictor. Audit support, category validity,
unique donors, current-value invariance and runtime; verification truth is audit only.

## Calibration and assessment

Use2017 for independent release calibration, with a fixed label-independent hash
sample of10k eligible customers per seed; evaluate each environment on the same
customers. Default calibration2014 is disjoint. Policies: empirical pooled and
conservative robust-environment thresholds at alpha .05/.10/.15. For whole releases,
retain historical calibrated risk scoring only if fit on a separate role; this new
branch instead calibrates thresholds directly on raw MC/rank/uncertainty scores,
without claiming they are probabilities. Prediction entropy/variance, rank
instability, MC and stable-core all receive identical calibration customers.

No restore-derived predictor statistic is allowed. Freeze model, preprocessor,
background, completion models and policies with hashes before assessment predictions.
Evaluate prediction on **every labelled assessment loan**. Attributions are on a
fixed10k-ID sample per assessment vintage (max30k loans), chosen before target
inspection. Report the explanation-subset denominator distinctly; do not imply
SHAP was computed on750k loans. Hash selection uses loan ID+fixed salt; retain all
seeds/masks of a customer in one cluster. No optional stopping or outcome sampling.

Report AUC/AP, class-wise precision/recall/F1/support, Brier/log loss/ECE and curves;
threshold .5 plus one development-F1 threshold frozen before assessment. Report
Region B at |delta raw p| .01/.02/.05, reason revision/eligibility, detector AUC/AP,
whole risk–coverage and stable-core reason precision/recall/customer coverage.
No arbitrary combined scalar. Conditional-imputer dependence is a main result,
including intersection empty sets and cases where rank instability matches MC.

Paired1000-draw bootstrap over unique loans, retaining masks/seeds/reasons, stratified
by assessment vintage; show each vintage separately. Fixed-policy intervals do not
cover training/calibration selection uncertainty or household dependence. Repeat
latency five times on a fixed batch128 and single loan, partition completion,
prediction, SHAP and scoring. Log runtime/RSS, CPU/memory, packages, epochs/trees,
and hashes. Overnight feasibility is a target, not a measured claim yet.

## Publication and stop rules

Only reviewed non-reconstructive aggregate results/figures and original code enter
Git; no loan-level examples or weights. Suppress small cells (<20 loans) and any
aggregate combination that reveals records; a count threshold is not a formal
privacy guarantee. All preprocessing/result artifacts remain local by default.

Stop on provenance mismatch, ambiguous labels/leakage, inadequate cohort/event
support, negligible grouped Region B, mostly empty stable sets, no improvement
over cheaper rank evidence, or large incompatible completion-family conclusions.
Record failures without changing cohort, endpoint, K, groups or calibration family.
The predictor/assessment runner is intentionally not claimed executed or validated
on unavailable files; the local intake and policy primitives are independently
testable while access is pending.
