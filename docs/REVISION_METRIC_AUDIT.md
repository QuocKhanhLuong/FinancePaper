# Revision metric audit — 2026-10-03

Decision: **keep the frozen operational event; require sensitivity reporting**.
It is a brittle top-list editing target, not explanation correctness. New results
come from diagnostic partitions in [the declared study](REVISION_STUDY_PROTOCOL.md),
384 customers × 3 folds, with fold-specific predictors/references. These are
development evidence, not final confirmation. Prior reports remain unchanged.

## Definition and estimand

For a fixed trained predictor/explainer, select exactly three originally observed
fields with largest positive attribution > .01 raw-logit units. Otherwise withhold.
After true hidden values are restored, revise if any released reason becomes
nonpositive (epsilon 1e-6) or has at least three **originally observed** fields
with attribution larger by >1e-6. Newly revealed fields cannot displace these
reasons in this target. Reordering within the same top-three set is not an event.
This tests whether the originally offered reasons survive verification. It does
not test whether the full-information top-three includes newly verified facts.
Predictor weights and reference distributions do not change across endpoints.

The new audit separates sign epsilon from rank-tie tolerance. Increasing the old
shared epsilon would also call small positive reasons sign failures. Primary
behavior is regression-tested against the old implementation; no target relabeling
was used to improve a model. A continuous diagnostic is observed-field normalized
L1 shift, `sum |phi_partial-phi_full| / sum (|phi_partial|+|phi_full|)`;
zero denominator gives zero. No arbitrary weighted severity score is introduced.

## Sensitivity: event percentage among eligible, valid pairs

| Variant | XGB MCAR30 / MAR30 | Vanilla MCAR30 / MAR30 | Mask-delta MCAR30 / MAR30 |
|---|---:|---:|---:|
| Frozen k3/.01 | 16.37 / 16.00 | 8.46 / 6.09 | 15.31 / 14.19 |
| k2 | 15.61 / 11.88 | 6.18 / 5.22 | 10.88 / 10.73 |
| k5 | 22.98 / 17.01 | 10.43 / 9.68 | 20.28 / 17.07 |
| Magnitude .001 | 18.26 / 17.22 | 9.03 / 6.95 | 15.79 / 14.30 |
| Magnitude .05 | 13.08 / 14.98 | 8.21 / 4.91 | 13.20 / 12.48 |
| Rank tie gap .005 | 12.05 / 9.88 | 4.93 / 3.04 | 9.00 / 7.71 |
| Rank tie gap .01 | 8.55 / 6.66 | 3.81 / 2.19 | 6.40 / 5.81 |
| Permit one rank displacement | 4.84 / 3.33 | 1.77 / 1.90 | 3.99 / 2.76 |
| Semantic groups | 8.68 / 9.81 | 7.32 / 6.36 | 15.31 / 12.77 |

All 1,152 customers/model/condition are in the attribution sample; frozen eligible
denominators MCAR30/MAR30 are XGB 971/931, vanilla 1,076/1,051, mask-delta 1,078/1,050.
Changing k/magnitude/grouping changes eligibility. The machine-readable audit
also records intersection-with-primary denominators/rates; these marginal rows
are **not paired effect sizes**. Grouping sums only originally observed repayment,
bill and payment fields, alongside five static fields; it is a different level of
financial description, not a claim that correlated months are interchangeable.

The strict top-k event is sensitive to small rank changes: one extra allowed rank
removes most events. Nevertheless, finite tie gaps and semantic grouping retain
nontrivial revision. The phenomenon does not disappear, but “material revision”
needs domain/user validation before real credit use. .01 is a declared engineering
threshold on raw logits, not a regulatory or human-meaningfulness threshold.

## Attribution-method control

48 fixed diagnostic customers/fold; same training reference rows and 32 original-
field permutations across endpoints/models, availability held fixed. MCAR30
revision is XGB 24/125 = 19.20%, vanilla 15/138 = 10.87%, mask-delta 24/139 = 17.27%,
additive 0/125. Four-reference expected IG gives vanilla 14/129 = 10.85% and
mask-delta 18/124 = 14.52%. This is expected IG, not stochastic GradientSHAP.
The small subset and finite permutations preclude a precise cross-model ranking.
Completeness verifies summation, **not Monte Carlo convergence or correctness**.

Primary TreeSHAP is interventional over encoded inputs including metadata,
then signed-summed to original financial fields; metadata is omitted as a reason.
IG varies financial values while holding masks/deltas fixed. These are different
games. IG's mean/four-reference paths can pass through fractional categorical
encodings; interventional SHAP can break financial correlations. Neither method
is causal. The common-field game checks a shared estimand on a small subset;
within-model partial/restored comparisons remain the primary interpretation.

## Degenerate stability and guardrails

LR and additive trees give zero revision as expected: with fixed-reference
additive logits, an observed field's contribution does not depend on other hidden
fields. This does not make their explanations correct or their predictions best.
Always report discrimination, calibration, eligibility, attribution validity and
revision separately. A constant explanation/predictor cannot win by revision alone.
Observed rank correlation, sign agreement and overlap remain separate diagnostics.
Unknown labels from invalid restored attributions are reported, never treated as
non-events or supplied to inference-time eligibility.

Artifacts: `outputs/revision_study/fold_*/metric_audit.csv`, `*_*.npz`,
`explainer_sensitivity.csv`; `audit_manifest.json` records source/data/artifact
hashes and 612.24 seconds runtime. No historical test customer was used.
