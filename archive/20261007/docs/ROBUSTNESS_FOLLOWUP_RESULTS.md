# Robustness follow-up — measured 2026-10-03

**No all-metric winner was found.** The 50/50 additive/tree blend reduces reason revision, but the stronger multi-view XGBoost control exposes prediction/calibration costs. Retain this as an exploratory tradeoff result; do not promote the blend or the recurrent model as universally superior.

The [frozen protocol](ROBUSTNESS_FOLLOWUP_PROTOCOL.md) and development-only search precede the new test evaluation. Both selected recipes were frozen for all five restarts before test batches were constructed. This result does not change the research question.

## Scope and selection

- Five training/masking restart seeds: 42, 43, 44, 45, 46. The seed-42 customer partitions, evaluation masks and training background64 stay fixed.
- Taiwan test: 4,500 customers per model/condition; explanations: the same fixed 400 customers. No customer copy crosses a partition; the 4,500 reason-risk calibration customers remain unused.
- Thirteen candidate recipes require four additional fitted tree components in the development screen. The selected interaction component is the regularized depth3/600-tree, 25-view model; its additive partner is the depth1/600-tree, 25-view model.
- Development AP selected interaction weight 0.75. The joint prediction/revision gate selected weight 0.50 and passed its predeclared screening tolerances. These weights did not change on later seeds or test results.
- All 11 evaluated predictors are shown below. The unblended 25-view tree and the exact interaction component are explicit controls. Each copied customer has total sample weight one.
- Mean ± sample SD below measures training/masking variability on a fixed cohort. It is not a confidence interval, five independent test datasets, or the performance of a five-model prediction ensemble.
- The test cohort was already inspected in the earlier pilot. This follow-up remains exploratory; it is not an independent confirmation.

Complete LR/XGBoost predictions are deterministic here. Their matched-coverage
revision statistics can still vary because the common eligible set depends on
the other models' restarts. Revision SD therefore includes this evaluation-set
variation; it is not solely variation in that predictor's own weights.

## Prediction under missing information

AP is Average Precision (higher better); calibrated Brier is lower better. Positive-slope Platt calibration uses only the reserved probability-calibration split.

| Model | AP complete | AP MCAR10 | AP MCAR30 | AP MAR30 | Brier MCAR30 | Brier MAR30 |
|---|---:|---:|---:|---:|---:|---:|
| LR complete | 0.5337 ± 0.0000 | 0.5153 ± 0.0000 | 0.4790 ± 0.0000 | 0.4716 ± 0.0000 | 0.1473 ± 0.0000 | 0.1498 ± 0.0000 |
| LR one-view | 0.5343 ± 0.0019 | 0.5226 ± 0.0024 | 0.4982 ± 0.0048 | 0.4901 ± 0.0043 | 0.1443 ± 0.0007 | 0.1467 ± 0.0006 |
| XGBoost complete | 0.5600 ± 0.0000 | 0.5450 ± 0.0000 | 0.5158 ± 0.0000 | 0.4935 ± 0.0000 | 0.1425 ± 0.0000 | 0.1472 ± 0.0000 |
| XGBoost one-view | 0.5545 ± 0.0028 | 0.5442 ± 0.0020 | 0.5194 ± 0.0049 | 0.5064 ± 0.0043 | 0.1401 ± 0.0006 | 0.1436 ± 0.0008 |
| Vanilla GRU + masking | 0.5494 ± 0.0036 | 0.5380 ± 0.0025 | 0.5151 ± 0.0017 | 0.5011 ± 0.0021 | 0.1413 ± 0.0002 | 0.1453 ± 0.0003 |
| Mask/delta GRU + masking | 0.5527 ± 0.0032 | 0.5431 ± 0.0025 | 0.5215 ± 0.0023 | 0.5172 ± 0.0046 | 0.1400 ± 0.0005 | 0.1420 ± 0.0009 |
| XGBoost 25-view | 0.5560 ± 0.0010 | 0.5448 ± 0.0013 | 0.5237 ± 0.0016 | 0.5138 ± 0.0008 | 0.1396 ± 0.0001 | 0.1427 ± 0.0001 |
| Regularized depth3 25-view | 0.5529 ± 0.0011 | 0.5462 ± 0.0012 | 0.5227 ± 0.0004 | 0.5108 ± 0.0010 | 0.1395 ± 0.0001 | 0.1424 ± 0.0001 |
| Additive depth1 25-view | 0.5385 ± 0.0002 | 0.5348 ± 0.0003 | 0.5174 ± 0.0002 | 0.5025 ± 0.0002 | 0.1421 ± 0.0000 | 0.1455 ± 0.0000 |
| Blend 75% interaction (predictive) | 0.5524 ± 0.0010 | 0.5467 ± 0.0010 | 0.5237 ± 0.0006 | 0.5111 ± 0.0006 | 0.1397 ± 0.0001 | 0.1427 ± 0.0001 |
| Blend 50% interaction (joint) | 0.5492 ± 0.0007 | 0.5436 ± 0.0006 | 0.5235 ± 0.0004 | 0.5089 ± 0.0004 | 0.1401 ± 0.0001 | 0.1433 ± 0.0001 |

The stronger exposure control matters. Averaged AP for one-view versus 25-view XGBoost rises from 0.5194 to 0.5237 under MCAR30 and from 0.5064 to 0.5138 under MAR30. The 50/50 blend has MCAR30 AP 0.5235 but MAR30 AP 0.5089; mask/delta GRU remains better on MAR30 AP at 0.5172. The blend therefore does not justify replacing the best predictive controls.

## Explanation revision at matched coverage

Each model selects 200 of the 400 fixed explanation customers from the per-seed/per-condition common eligible set, using only current third-reason strength. Selected customer identities may differ by model. All runs attained 50% coverage. The magnitude threshold is fixed at .01 for this experiment; the previous pilot used model-specific thresholds and a different common-model set, so its revision table is not directly interchangeable with this one.

| Model | MCAR10 revision | MCAR30 revision | MAR30 revision |
|---|---:|---:|---:|
| LR complete | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| LR one-view | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| XGBoost complete | 5.9 ± 0.2% | 14.9 ± 0.7% | 14.4 ± 1.1% |
| XGBoost one-view | 9.5 ± 1.9% | 20.9 ± 3.3% | 19.6 ± 2.2% |
| Vanilla GRU + masking | 5.8 ± 1.2% | 12.3 ± 2.6% | 11.0 ± 2.2% |
| Mask/delta GRU + masking | 10.7 ± 2.4% | 18.4 ± 3.7% | 15.2 ± 3.4% |
| XGBoost 25-view | 6.9 ± 1.1% | 14.0 ± 0.6% | 15.2 ± 0.8% |
| Regularized depth3 25-view | 12.1 ± 2.4% | 22.1 ± 2.5% | 22.5 ± 2.6% |
| Additive depth1 25-view | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% |
| Blend 75% interaction (predictive) | 8.1 ± 0.9% | 17.3 ± 2.3% | 18.7 ± 3.1% |
| Blend 50% interaction (joint) | 5.1 ± 1.5% | 10.4 ± 2.5% | 13.4 ± 1.2% |

The 50/50 blend reduces MCAR30 revision from 14.0% to 10.4% relative to the 25-view XGBoost control (3.6 percentage points), and from 22.1% to 10.4% relative to its exact unblended depth3 component. Under MAR30 it reduces the latter from 22.5% to 13.4%, but vanilla GRU is more stable at 11.0%. These are descriptive fixed-cohort estimates; no significance claim follows.

LR and the depth1 additive model have zero revision under this fixed-reference interventional attribution protocol. Observed-field contributions in an additive logit cannot depend on restored other fields. This expected structural invariance is not evidence of correct/causal explanations or of an accurate predictor. Tree SHAP and recurrent conditional IG also retain different attribution estimands.

The common eligible sets contain 227–344 of the 400 customers across conditions
and restarts. IG completeness excludes 0–5 pairs per vanilla-GRU group and 0–12
per mask/delta-GRU group; all tree/linear/blended groups pass. Matched coverage is
an evaluation conditional on valid attribution pairs, not an inference-time
release rule or proof that omitted explanations would have been reliable.

## Calibration and minority detection costs

All values here use held-out calibrated probabilities and development-selected F1 thresholds. Lower recall cannot be hidden by a better AP or ROC-AUC.

| Condition | Model | ROC-AUC | Recall | F1 | Brier | Log loss | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| mcar30 | XGBoost 25-view | 0.7632 | 0.5091 | 0.5161 | 0.1396 | 0.4426 | 0.0183 |
| mcar30 | Mask/delta GRU + masking | 0.7580 | 0.5290 | 0.5180 | 0.1400 | 0.4444 | 0.0111 |
| mcar30 | Blend 75% interaction (predictive) | 0.7616 | 0.5104 | 0.5168 | 0.1397 | 0.4431 | 0.0167 |
| mcar30 | Blend 50% interaction (joint) | 0.7610 | 0.5045 | 0.5165 | 0.1401 | 0.4442 | 0.0223 |
| mar30 | XGBoost 25-view | 0.7646 | 0.4716 | 0.4993 | 0.1427 | 0.4484 | 0.0246 |
| mar30 | Mask/delta GRU + masking | 0.7583 | 0.4959 | 0.4996 | 0.1420 | 0.4480 | 0.0158 |
| mar30 | Blend 75% interaction (predictive) | 0.7625 | 0.4770 | 0.4964 | 0.1427 | 0.4486 | 0.0230 |
| mar30 | Blend 50% interaction (joint) | 0.7622 | 0.4637 | 0.4941 | 0.1433 | 0.4501 | 0.0252 |

Complete classwise precision/recall/F1/support, raw and calibrated metrics, and calibration-bin counts are retained in `evaluation/prediction_metrics.json` and `.csv`. The deployment class threshold was not adjusted using test recall.

## Explicit dominance check

Each row checks 70 declared condition/metric cells, including classwise metrics and explanation coverage/rank/sign/overlap diagnostics. These cells are correlated and partly redundant; win counts are an accounting device, not independent statistical tests or a model-selection score. Ties use numerical tolerance 1e-8. Undefined values would fail the gate; none occurred.

| Candidate | Baseline | Wins | Ties | Losses | Weak Pareto dominance |
|---|---|---:|---:|---:|---|
| Blend 50% interaction (joint) | LR one-view | 47 | 0 | 23 | No |
| Blend 50% interaction (joint) | LR complete | 50 | 1 | 19 | No |
| Blend 50% interaction (joint) | Mask/delta GRU + masking | 39 | 0 | 31 | No |
| Blend 50% interaction (joint) | Regularized depth3 25-view | 27 | 4 | 39 | No |
| Blend 50% interaction (joint) | Vanilla GRU + masking | 45 | 0 | 25 | No |
| Blend 50% interaction (joint) | XGBoost one-view | 39 | 0 | 31 | No |
| Blend 50% interaction (joint) | XGBoost complete | 52 | 0 | 18 | No |
| Blend 50% interaction (joint) | XGBoost 25-view | 25 | 0 | 45 | No |
| Blend 75% interaction (predictive) | LR one-view | 48 | 0 | 22 | No |
| Blend 75% interaction (predictive) | LR complete | 49 | 0 | 21 | No |
| Blend 75% interaction (predictive) | Mask/delta GRU + masking | 41 | 0 | 29 | No |
| Blend 75% interaction (predictive) | Regularized depth3 25-view | 40 | 0 | 30 | No |
| Blend 75% interaction (predictive) | Vanilla GRU + masking | 44 | 0 | 26 | No |
| Blend 75% interaction (predictive) | XGBoost one-view | 50 | 0 | 20 | No |
| Blend 75% interaction (predictive) | XGBoost complete | 44 | 0 | 26 | No |
| Blend 75% interaction (predictive) | XGBoost 25-view | 24 | 0 | 46 | No |

Neither selected candidate weakly dominates any of the eight baselines across all declared cells; neither strictly wins every cell. The 50/50 choice loses 45 cells against 25-view XGBoost. This is a negative result for an all-metric claim, even though its reason-revision improvement is useful.

## Runtime, checks and artifacts

- Complete corrected run: 377.73 seconds fitting/search and 326.42 seconds evaluation, 11.74 minutes total. PyTorch neural training/IG used MPS; XGBoost used one CPU thread. This is observed local runtime, not a universal performance guarantee.
- 66 tests passed, including synthetic train-only/held-out mutation checks, blend-logit attribution additivity, additive restoration invariance, unit per-customer training weight, missing-model coverage failure, and failure to label missing metrics or calibration harm as dominance.
- A separate root-run recomputation verified 46 source hashes and 346 frozen artifacts; all six primary prediction metrics from 990,000 CSV prediction rows; reasons, eligibility and 6,746 revision events from 88,000 explanation rows / 220 attribution archives; shared masks, matched coverage and complete negative controls. No mismatch was found.
- The corrected run is `outputs/robustness_followup_controlled/`. `fit_manifest.json` predates test; `manifest.json` completes the run. The additional `evaluation/audit_receipt.json` records the post-run recomputation and is intentionally outside the immutable run manifest.
- The earlier interrupted fit-only attempt at `outputs/robustness_followup/` is retained. It opened no test batches and supplied no reported test results.
- Requested Orca run `run_e85605ccca99`, task `task_0984a86463b0`: two Codex readiness timeouts and one stopped Claude shell-parse failure. All three temporary worker terminals were released. No worker produced a deliverable; this is direct root execution with software/artifact verification, **not an independent worker review**.

```bash
uv sync --frozen --extra temporal
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
uv run --frozen --extra temporal python scripts/run_robustness_followup.py --output outputs/followup-repeat --stage fit
uv run --frozen --extra temporal python scripts/run_robustness_followup.py --output outputs/followup-repeat --stage evaluate
uv run --frozen --extra temporal python scripts/audit_robustness_followup.py --output outputs/followup-repeat
```

## Research decision

Keep the explanation-reliability framing and the stronger 25-view XGBoost control. Retain the 50/50 blend as a useful tradeoff comparator. Do not claim a universally winning model, a novel architecture, or a calibrated reason-release guarantee. The current five restarts do not resolve split/generalization uncertainty; further confirmation needs a separately declared evaluation and ultimately independent customer/time data. No test-driven enlargement of this search was performed.
