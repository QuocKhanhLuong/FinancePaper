# Temporal feasibility pilot — measured 2026-10-03

**Recommendation: retain the temporal models as experimental comparators and
keep explanation reliability as the paper's primary contribution.** This pilot
does not establish an overall missing-aware GRU improvement over XGBoost. It
does show the relevant failure case: better missing-input prediction can coexist
with less stable reasons.

The [research decision](TEMPORAL_MODEL_RESEARCH.md) and
[specification](TEMPORAL_MODEL_SPEC.md) were written before implementation and
test evaluation. This is one seed, one dataset and simulated missingness. It is
not the five-seed paper experiment, a temporal generalization test, a calibrated
reason-release system or evidence of causal/correct explanations.

## Execution and provenance

- Official checksum-pinned Taiwan workbook: 30,000 rows, 23 financial fields.
- Original five-way split retained: 15,000 train; 3,000 development; 3,000
  probability calibration; 4,500 reserved reason-risk calibration; 4,500 test.
- Seed 42; identical masks, train-only shared preprocessing and initialization
  seeds across relevant comparisons. Same 64 training background IDs as the
  original static pilot. The temporal branch has its own evaluation mask stream;
  legacy versus new-branch results are not a matched-mask comparison.
- Ten fits: complete/augmented flat LR and XGBoost; complete/augmented vanilla
  and mask/delta GRU; weighted-BCE and focal variants of augmented mask/delta GRU.
- PyTorch 2.14.1, float32, MPS; no MPS fallback environment variable enabled.
  CPU threads explicitly limited for the documented native-library interaction.
- Vanilla GRU: 21,953 parameters; primary mask/delta GRU: 23,265. Six neural fits
  took about 67 seconds combined (7.9–13.4 seconds each). Whole run, including
  development/test attribution and output generation: 154.09 seconds.
- Full-test predictions: 180,000 rows (10 models × 4 conditions × 4,500 records).
  Reasons: 4,800 rows (3 models × 4 conditions × 400 shared test IDs).
- Actual test cell-missing rates: MCAR10=9.9266%, MCAR30=29.9063%, MAR30=30.3024%.
  MAR preserves AGE/LIMIT_BAL; its train-fitted intercept targets 30% over all 23
  fields. No labels enter mask generation.

Local artifact directory: `outputs/temporal_pilot/` (ignored, regenerate rather
than commit). `manifest.json` records data/source/dependency/artifact hashes,
device, thread settings and time; `resolved_config.yaml`, split/mask files,
background IDs, checkpoints, preprocessing metadata, histories and frozen
selection are retained. A separate `outputs/temporal_stage_a/` reproduced the
legacy pilot's summary, records, selection, splits and masks byte-for-byte.

## Prediction: all 4,500 test records

AP is Average Precision. Brier columns use the independent pooled Platt
calibrator; raw and calibrated AUC/AP coincide because its slope is positive.
All classwise precision/recall/F1/support, ROC-AUC, log loss, ECE and calibration
bins are in `prediction_metrics.json` and `calibration_test.csv`.

| Model | AP complete | AP MCAR10 | AP MCAR30 | AP MAR30 | Brier MCAR30 | Brier MAR30 |
|---|---:|---:|---:|---:|---:|---:|
| LR complete | .5337 | .5153 | .4790 | .4716 | .1473 | .1498 |
| LR augmented | .5334 | .5217 | .5021 | .4940 | .1436 | .1468 |
| XGBoost complete | .5600 | .5450 | .5158 | .4935 | .1425 | .1472 |
| XGBoost augmented | .5524 | .5415 | .5232 | .5038 | .1400 | .1445 |
| Vanilla GRU complete | .5508 | .5380 | .5080 | .4917 | .1430 | .1472 |
| Vanilla GRU augmented BCE | .5518 | .5404 | .5154 | .5016 | .1415 | .1454 |
| Mask/delta GRU complete | .5523 | .5393 | .5088 | .5001 | .1425 | .1464 |
| **Mask/delta GRU augmented BCE** | .5540 | .5444 | .5200 | .5168 | .1403 | .1425 |
| Mask/delta GRU augmented weighted BCE | .5482 | .5392 | .5206 | .5212 | .1396 | .1415 |
| Mask/delta GRU augmented focal | .5499 | .5401 | .5209 | .5183 | .1395 | .1419 |

Mask augmentation helps both recurrent and flat models. Primary GRU improves
over augmented vanilla on MAR30 AP by .0152 and MCAR30 AP by .0046. Against augmented
XGBoost it improves MAR30 by .0130 but loses MCAR30 by .0032. Complete-data XGBoost
has the highest complete-data AP. Thus the specified improvement across both
missingness conditions is not supported by this screen.

These comparisons also change static-mask access when moving from vanilla to
the primary GRU. They establish the effect of the full missingness-metadata
package, not the isolated effect of temporal masks, deltas or recurrence.
Flat augmented models receive one fixed masked view per row; recurrent models
receive new masks per epoch. A stronger exposure-matched tree control and the
planned branch/order ablations are necessary before attributing gains to the
architecture.

### Loss/calibration tradeoff under MCAR30

| Primary architecture's loss | Raw Brier | Calibrated Brier | Raw log loss | Calibrated log loss | Raw ECE |
|---|---:|---:|---:|---:|---:|
| BCE | .1404 | .1403 | .4455 | .4453 | .0120 |
| Weighted BCE | .1917 | .1396 | .5753 | .4429 | .2224 |
| Focal | .2075 | .1395 | .6066 | .4428 | .2444 |

Weighted/focal scores are severely miscalibrated before correction in this run.
Calibration substantially repairs them, with small improvements in calibrated
Brier/AP over BCE. This does not justify publishing their raw outputs as
default probabilities. Keep BCE as the default; the loss variants remain
ablations. Their reason-revision evaluation was not run, so their predictive
scores cannot establish superior explanation reliability.

## Reasons: the fixed 400-record attribution subset

IG explains recurrent raw logits conditional on each endpoint's mask/delta;
TreeSHAP jointly explains financial values and metadata for shared XGBoost.
Only originally observed financial fields can be released or become competitors.
Cross-model IG/SHAP differences are descriptive, not proof of explanation
correctness or a common attribution estimand.

| Model | MCAR30 revised / evaluable eligible | MCAR30 revision | MAR30 revised / evaluable eligible | MAR30 revision |
|---|---:|---:|---:|---:|
| Vanilla GRU augmented BCE | 52 / 331 | 15.71% | 42 / 321 | 13.08% |
| Mask/delta GRU augmented BCE | 97 / 346 | 28.03% | 56 / 337 | 16.62% |
| XGBoost augmented | 92 / 354 | 25.99% | 83 / 356 | 23.31% |

At 50% coverage of the same 400 IDs, restricted first to common eligibility and
selected by pre-verification reason strength, MCAR30 revision is 14.0% for
vanilla, 22.5% for mask/delta, 22.0% for XGBoost (200 selected each). MAR30 is 11.5%,
16.0%, 19.0%. Thus the primary model's worse revision than vanilla survives this
coverage control. These are subset estimates without multi-seed confidence
intervals, not deployment guarantees.

For primary GRU under MCAR30, **37 of 148** evaluable eligible explanations with
raw probability shift ≤ .02 nevertheless revise: 25.0%. All complete-information
negative controls have zero revision among evaluable eligible explanations.

Numerical checks are explicit: after up to 128 IG nodes, 3–5 of 400 primary pairs
per neural model/condition fail completeness and are excluded from event
conclusions. MCAR30 valid-pair counts are 395 vanilla, 397 mask/delta, 400 XGBoost.
Before-verification eligibility and evaluation eligibility are separate columns;
failed restoration attribution is not disguised as a successful release.
Completeness is necessary but not sufficient for per-feature numerical accuracy;
baseline and quadrature sensitivity remain limitations for a final paper.

### Availability context matters

The primary model's MCAR30 full-restoration revision rate is 28.03%, versus 13.95%
for value-only restoration among eligible cases with a valid secondary endpoint.
Those denominators differ slightly and the rates are not an additive causal
decomposition. At full-test level, changing only the availability context while
holding imputed values fixed changes primary probabilities by .02827 on average,
versus .01341 for XGBoost and zero for vanilla. This intentionally inconsistent
input is a sensitivity diagnostic, not a real financial counterfactual.

Full-restoration absolute probability shifts are .03840 primary, .03889 vanilla
and .04009 XGBoost under MCAR30. A slightly smaller prediction shift does not
imply stable reasons. The conditional IG reference logit also changes with
availability; both references and completeness residuals are logged.

The simple missing-fraction revision score has MCAR30 event AUROC .6013 / AP .3586
for primary GRU. Risk-coverage curves and empirical 10% crossings are exported,
but small selected sets and test-derived crossings do not certify a 10% risk
guarantee. No learned selector or calibrated release decision has been fitted.

## Verification and exact scope

The combined suite passed 59 tests, including the original static pilot, official
chronology, train-only statistics, hidden-value exclusion, MAR anchors, mask/delta
alignment, loss separation, MPS forward/backward/IG, conditional attribution,
aggregation, matched coverage and a synthetic end-to-end held-out-data mutation
test. That mutation preserved neural weights and every scientific selection.
CPU training was tested synthetically; full Taiwan CPU runtime was not measured.
An independent audit verified all 43 source and 63 artifact hashes and recomputed
reasons, eligibility and restoration revision events from all 12 attribution
archives with zero mismatches.

Run from the repository root:

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal financepaper download
uv run --frozen --extra temporal python scripts/run_temporal_pilot.py --device-info
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
uv run --frozen --extra temporal python scripts/run_temporal_pilot.py --config configs/temporal_pilot.yaml --output outputs/temporal-repeat
```

The script refuses nonempty output directories. MPS results need not be
byte-identical across devices/library versions. Inspect `manifest.json` before
comparing runs. Training configurations and every unsuccessful result remain
visible; no test-driven replacement model was trained.

Implemented: stages A–E for this bounded specification; optional debug tensors,
monthly/original-field attributions, calibration, diagnostics and record-level
artifacts. Not run/implemented: five-seed confirmatory study, structural/order
ablation experiments, full GRU-D decay, reconstruction, MC uncertainty ensembles,
conditional multiple imputation, learned revision head or calibrated release.
The proposed four-way reliability categories therefore remain unavailable.

## Answers to the ten research questions

1. **Use an RNN?** As an experimental comparator, yes. This run does not justify
   replacing XGBoost or making recurrence the paper's central method.
2. **Which architecture?** Retain vanilla as control and the selected small
   GRU-Simple value/mask/delta model as the candidate. Full GRU-D decay is not yet
   justified by six categorical/money snapshots; it remains a future ablation.
3. **Enough temporal structure?** Enough to test a six-month ordered model,
   already supported by official chronology and prior Taiwan LSTM work. Actual
   incremental value of order remains unproved until branch/order controls run.
4. **Default loss?** BCE plus held-out calibration. Weighted/focal are useful
   calibrated comparisons, not raw-probability defaults or new methods.
5. **Potential contribution?** The verification-defined reason event and a
   validated future release-risk estimator, not this recurrent architecture.
6. **Known components?** Credit GRU/LSTM, temporal/static branches, mask/delta,
   decay, augmentation, reconstruction, class weights/focal, calibration,
   SHAP/IG and generic explanation uncertainty/selection. The recent Taiwan
   temporal/uncertainty preprint also rules out a broad combination claim.
7. **Falsifier?** The frozen five-seed comparison requires useful AP gains on
   both MCAR30/MAR30 without calibration harm, plus improved revision at matched
   coverage and successful order/branch/reference controls. The current screen
   already fails the across-condition AP improvement and vanilla-relative
   revision improvement; no universal inferiority claim follows.
8. **One Apple Silicon Mac?** Yes for the implemented stages, measured directly:
   about 2.6 minutes here. Later grids/heads/imputation require separate budgets;
   their runtime has not been measured.
9. **Log what?** Raw/calibrated risk, masks/month counts/deltas, model/data/seed
   identities, optional hidden/static/fused tensors, original/month attributions,
   baseline logits/residuals, reasons/sign/rank/eligibility, both restorations,
   metadata sensitivity and revision labels. Unimplemented uncertainty/release
   outputs stay null, never invented high/medium/low labels.
10. **Paper focus?** Explanation reliability. Better prediction and worse
    reason stability coexist here. Defer Stage F and any central model claim
    until the controls and independent selector evaluation support them.
