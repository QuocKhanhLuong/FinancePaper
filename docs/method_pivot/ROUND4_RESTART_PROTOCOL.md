# Round 4 gated continuation: two remaining development restarts

Date: 2026-10-05. This extension is committed **before opening new evaluation
labels for folds 1 and 2 in this audit**, after the predeclared fold-0 gate passed.
These caches and customers are historically inspected development evidence.
This is not independent confirmation, official DMV reproduction or a new method.

## Why continuation is allowed

The [original protocol](ROUND4_DEVELOPMENT_PROTOCOL_DRAFT.md), committed at
`1d49cee`, required both paired AP-difference lower bounds > 0 and MC8 minus
prediction-distribution AP >= .05. The executed fold-0 run at `110b5d2` met it:
MC8 minus prediction distribution = .67805, 95% CI [.49759,.81539]; adding four
explanation features = .51509, CI [.35725,.74410]. There were 25 events among
426 eligible MCAR30 evaluation customers. This enables a three-restart baseline
audit only; it does not establish method novelty.

## Frozen extension

Run existing folds **1 and 2**, with predictor restarts **102 and 103** from
`configs/revision_study.yaml`. Change only paths/hashes and the detector's matching
random_state to 102/103. Preserve all other original choices: 400 fitting and
200 score-calibration customers from revision_calibration, 600 evaluation
customers from release_calibration; sort/permutation seed 20261005; same K8,
group2 target, 44/48 feature columns, fixed detector capacity, nine controls,
four conditions and 1,000 paired customer bootstrap draws. No new model fitting
beyond those same two detector controls; no SHAP regeneration. Budget 600 CPU
seconds per restart, freeze before run, no failure-driven target/threshold changes.

All original stopping rules apply, including >=20 eligible evaluation events and
non-events for MCAR30. A fold failing this count gate is **not run to performance**;
report the insufficient support and stop expansion instead of lowering the gate.
Report all attempted folds. Do not replace an unfavorable fold.

Configs:

- `configs/prediction_distribution_audit_fold1.yaml`
- `configs/prediction_distribution_audit_fold2.yaml`

Each contains exact partition/predictor hashes. The runner separately hashes all
current/verification cache files against historical receipts and freezes current
features before opening labels. Record actual protocol/code commits in manifests.
The original `protocol_commit` config field references the base preregistration;
the extension's exact file digest and pre-run Git commit appear in freeze.json.

## Reporting rule

Per-fold AP, AUROC, calibrated Brier/log loss and both paired differences remain
primary. Report mean and sample SD across three restarts only when all complete.
These are descriptive restart summaries: folds reuse people and are not three
independent populations. Never pool rows as independent customers. No pooled
bootstrap or across-fold confidence interval is claimed.

Evidence of recurring incremental explanation value requires the predeclared
fold-0 gate to hold in each completed restart. If not, describe heterogeneity or
insufficient evidence; do not average away a failed gate. Even if it holds, the
strong rank-instability baseline still blocks a new-method claim. This audit
can motivate a future mechanism, not certify its novelty or performance.
