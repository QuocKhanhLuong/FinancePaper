# External financial-risk validation: Polish companies

Protocol declared 2026-10-03 before external model outcomes. This validates an
explanation-reliability phenomenon across a different financial-risk target;
it does **not** replicate Taiwan consumer next-month default.

Official source: [UCI Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
[dataset DOI](https://doi.org/10.24432/C5F600), creator Sebastian Tomczak (2016).
License reverified 2026-10-03: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Download only the [official archive](https://archive.ics.uci.edu/static/public/365/polish%2Bcompanies%2Bbankruptcy%2Bdata.zip),
extract only `5year.arff`, record archive/member SHA256. Do not commit raw data.

The selected cohort has 5,910 statements, 64 numeric ratios and binary bankruptcy
within **one year**: 410 positive and 5,500 negative. The website's aggregate
10,503/65 header describes another/combined representation, not this ARFF schema.
Do not add year as a feature or combine the five files: horizons differ and no
usable company identifier is supplied. Same-firm overlap not detectable beyond
identical-feature rows remains a limitation. Not a time-ordered external cohort.
Natural missingness exists; exact counts and duplicate checks are recorded by
the preparation command, independently of model outcomes.

## Split and model

Predeclare StratifiedGroupKFold(20, shuffle=True, random_state=20261004), grouping
identical 64-feature vectors including their missing pattern. Fold numbers 0–9
predictor training, 10–11 revision-selector training, 12 default calibration,
13 revision-probability calibration, 14–15 release calibration, 16–19 assessment.
Approximate proportions 50/10/5/5/10/20%; retain all records and report exact counts.
All copies remain together, including conflicting target duplicates. No metadata
or label enters model features/donors. All fitted transforms/background/donors use
predictor training only. Assessment IDs are frozen before fitting; no new split seed.

Fixed LR (C1, unweighted) and XGBoost (200 depth3 trees, learning rate .05,
min_child_weight5, lambda1, hist, one CPU thread). Same 25-view augmentation with
per-customer total weight one, per-row rates from 0/.1/.2/.3. Artificial masks only
touch observed cells. Train-median fill, train mean/std scaling, append observed
mask indicators. All-natural-missing training column, if any: fixed zero fill,
unit scale and explicit audit flag. No winsorization, ratio engineering, class
weights, loss tuning, hyperparameter search or test-selected threshold. Report
class metrics at .5; discrimination/calibration are primary.

Raw-logit interventional TreeSHAP with 64 fixed training background rows; LR exact
linear attribution with that same reference. Mask attribution logged separately,
excluded from observed financial reasons. Completeness check remains
.002+.001*abs(logit-reference). Default positive-slope Platt fit on independent
default calibration, balanced complete/MCAR10/MCAR30 views. No predictor selection.

## Verification and completion

Keep `natural_missing_mask` and `artificial_verification_mask` separate.
Nested MCAR10/30 masks: uniform draw per cell AND originally observed. Partial
inputs hide those cells; verification restores **only** those cells. Natural
unknowns stay NaN at both endpoints and never supply ground-truth labels for
imputation support. Primary eligible reasons exclude both sets at both endpoints.
No external MAR: two MCAR conditions are the bounded confirmation protocol.

Use the same nearest-32 joint donor procedure (max2048 reference rows, numeric
IQR distance floor1/clipping5), restricted to **complete predictor-training rows**.
This avoids inventing true values in donor records. If fewer than32 complete
training rows exist, stop and document infeasibility; no adaptive fallback.
Only artificial cells are completed; naturally missing cells remain unknown and
are median-imputed by the frozen preprocessing at every endpoint. Distance uses
currently observed query cells only. Audit complete-case donor selection bias.
K8 reference, all K1/2/4/8/16 as frozen nested streams. No changes after assessment.

## Detectors, policies and external success

Use raw entropy/max-probability, completion prediction variance, missing fraction,
attribution variance, rank/sign instability and MC revision. Stronger standard
controls: fixed HGB (120 iterations, 7 leaves, minleaf40, lr.05, L2=5) fitted on
revision-training only: prediction/entropy/prediction variance/missing fraction;
generic adds current attribution statistics and missing-field indicators. No
feature/parameter selection. Recalibrate independently for each grouped target
without refitting the generic selector, which is trained on feature top3 only.

Policies, rank-gap sensitivity, group definitions, cutoffs and alpha targets are
exactly the predeclared validation protocol. External groups are fixed in
DOMAIN_REASON_GROUPS.md. External results never choose K, groups, tolerances or
calibration family. Report Taiwan and Polish separately, including partial/failure
transfer. Success needs non-negligible RegionB, explanation-specific evidence
stronger than prediction signals, useful MC discrimination, and nonzero useful
coverage at declared risk budgets. No final claim from raw feature-level rates
alone. Bootstrap at original statement/duplicate cluster level (1000 draws),
never masks/completions as independent observations.
