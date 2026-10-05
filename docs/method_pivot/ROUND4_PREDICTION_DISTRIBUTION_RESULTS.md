# Round 4: richer prediction controls — executed development results

Date: **2026-10-05**. **MEASURED EXPLORATORY BASELINE AUDIT**, not new-method evidence or independent confirmation.

The protocol was committed at `1d49cee`; code/tests at `110b5d2`. Fold 0 passed the registered continuation gate. The extension to folds 1/2 was committed at `43bd1b2` before their evaluation labels were opened in this audit. No predictor, mask, attribution, revision target or historical report changed.

## Primary MCAR30 results

All controls use the same K=8 cached completions and initially observed-member grouped top-2 revision target. AP/AUROC below rank **reason revision**, not next-month default. Brier/log loss measure separately calibrated revision-event probabilities, not default calibration. Higher AP/AUROC is better; lower Brier/log loss is better. The four-condition fit pools repeated masks of the same fitting customers; roles are assigned by customer before pooling.

| Detector | AP mean ± SD | AUROC mean ± SD | Brier mean ± SD | Log loss mean ± SD |
|---|---:|---:|---:|---:|
| entropy | 0.0918 ± 0.0241 | 0.4854 ± 0.0808 | 0.0784 ± 0.0279 | 0.2907 ± 0.0782 |
| prediction_variance | 0.1083 ± 0.0386 | 0.5748 ± 0.0978 | 0.0786 ± 0.0278 | 0.2921 ± 0.0775 |
| hard_uncertainty | 0.0940 ± 0.0362 | 0.4843 ± 0.0453 | 0.0786 ± 0.0280 | 0.2919 ± 0.0788 |
| action_disagreement | 0.0937 ± 0.0306 | 0.4836 ± 0.0474 | 0.0786 ± 0.0280 | 0.2915 ± 0.0787 |
| missing_fraction | 0.1161 ± 0.0322 | 0.6003 ± 0.0330 | 0.0779 ± 0.0277 | 0.2873 ± 0.0779 |
| prediction_distribution | 0.0991 ± 0.0402 | 0.5290 ± 0.0300 | 0.0787 ± 0.0284 | 0.2916 ± 0.0794 |
| prediction_plus_explanation | 0.5253 ± 0.0974 | 0.8924 ± 0.0610 | 0.0548 ± 0.0180 | 0.1874 ± 0.0686 |
| mc8 | 0.6139 ± 0.1117 | 0.9017 ± 0.0578 | 0.0481 ± 0.0207 | 0.1697 ± 0.0761 |
| rank_instability | 0.6012 ± 0.1256 | 0.8830 ± 0.0735 | 0.0529 ± 0.0220 | 0.1881 ± 0.0845 |

These are descriptive mean/sample SD across three restarts, **not** a confidence interval over independent populations. The 1,800 evaluation-role appearances contain **1,769 unique customers**, with overlaps across folds. No row-pooled inference was used.

| Restart | Eligible MCAR30 | Events | MC8 − prediction-distribution AP [95% paired CI] | Added-explanation AP gain [95% paired CI] | Gate |
|---|---:|---:|---|---|---|
| 101 | 426 | 25 | 0.6780 [0.4976, 0.8154] | 0.5151 [0.3573, 0.7441] | PASS |
| 102 | 420 | 32 | 0.4151 [0.2246, 0.5696] | 0.3117 [0.1605, 0.4723] | PASS |
| 103 | 408 | 51 | 0.4513 [0.3124, 0.5885] | 0.4517 [0.3176, 0.5946] | PASS |

Both registered comparisons have positive lower bounds in every restart. All 1,000 paired customer bootstrap draws per restart retained both classes. Primary revision prevalence varies from 5.87% to 12.50%; AP values must be interpreted relative to that prevalence. No claim of a formal conditional-independence test is made.

## What this changes — and what it does not

The prediction-only learned control has 44 features: current probability/entropy, completion moments/quantiles/votes, **all eight sorted completion probabilities**, 23 missing indicators and missing fraction. The matched augmented control adds four grouped explanation statistics and changes no hyperparameter. Incremental explanation value survives this stronger control in this small-data fitting regime.

This is not evidence that *all* predictive-uncertainty learners fail. Fit support is limited (400 customers and correlated masking episodes); another conditional distribution estimator or larger fitting sample may do better. The official learned DMV implementation was **not run**; these are independent finite-completion controls. We cannot claim to beat DMV.

**Negative finding retained:** rank instability remains a strong baseline: mean AP .6012 versus MC8 .6139; no predeclared superiority test between them was run. Both consume the same completion explanations, so ranking is not presumed to avoid SHAP cost. The learned augmented detector averages below direct MC8. No new algorithm, method necessity beyond these controls, or method-level novelty has been demonstrated.

## All environments and exclusions

Every condition below was predeclared. No condition was selected after seeing performance. Eligible means a current explanation has two positive observed groups and valid cached attribution checks; restored invalidity further excludes unverifiable labels. Ineligible customers are not negative revision examples.

| Fold | Condition | Candidates | Current ineligible | Restored invalid among eligible | Evaluated | Events |
|---|---|---:|---:|---:|---:|---:|
| 0 | mcar10 | 600 | 143 | 0 | 457 | 15 |
| 0 | mcar30 | 600 | 174 | 0 | 426 | 25 |
| 0 | mar30 | 600 | 159 | 0 | 441 | 27 |
| 0 | group_missing | 600 | 173 | 0 | 427 | 19 |
| 1 | mcar10 | 600 | 128 | 0 | 472 | 12 |
| 1 | mcar30 | 600 | 180 | 0 | 420 | 32 |
| 1 | mar30 | 600 | 141 | 0 | 459 | 29 |
| 1 | group_missing | 600 | 153 | 0 | 447 | 34 |
| 2 | mcar10 | 600 | 152 | 0 | 448 | 41 |
| 2 | mcar30 | 600 | 192 | 0 | 408 | 51 |
| 2 | mar30 | 600 | 176 | 0 | 424 | 61 |
| 2 | group_missing | 600 | 170 | 0 | 430 | 56 |

Full per-restart metric table (raw ranking metrics, held-out-calibrated probability metrics):

| Fold | Condition | Detector | AP | AUROC | Brier | Log loss |
|---|---|---|---:|---:|---:|---:|
| 0 | group_missing | action_disagreement | 0.0445 | 0.3689 | 0.0432 | 0.1890 |
| 0 | group_missing | entropy | 0.0514 | 0.4430 | 0.0425 | 0.1822 |
| 0 | group_missing | hard_uncertainty | 0.0445 | 0.3689 | 0.0427 | 0.1840 |
| 0 | group_missing | mc8 | 0.6516 | 0.9074 | 0.0268 | 0.1014 |
| 0 | group_missing | missing_fraction | 0.0445 | 0.5000 | 0.0425 | 0.1820 |
| 0 | group_missing | prediction_distribution | 0.0583 | 0.5947 | 0.0425 | 0.1812 |
| 0 | group_missing | prediction_plus_explanation | 0.5897 | 0.9128 | 0.0271 | 0.1029 |
| 0 | group_missing | prediction_variance | 0.0409 | 0.4710 | 0.0434 | 0.1892 |
| 0 | group_missing | rank_instability | 0.6516 | 0.9074 | 0.0282 | 0.1053 |
| 0 | mar30 | action_disagreement | 0.0806 | 0.4864 | 0.0579 | 0.2352 |
| 0 | mar30 | entropy | 0.0964 | 0.5403 | 0.0577 | 0.2330 |
| 0 | mar30 | hard_uncertainty | 0.0601 | 0.4775 | 0.0578 | 0.2336 |
| 0 | mar30 | mc8 | 0.6861 | 0.9070 | 0.0306 | 0.1161 |
| 0 | mar30 | missing_fraction | 0.0722 | 0.5745 | 0.0575 | 0.2297 |
| 0 | mar30 | prediction_distribution | 0.0738 | 0.5441 | 0.0577 | 0.2328 |
| 0 | mar30 | prediction_plus_explanation | 0.6600 | 0.8892 | 0.0362 | 0.1351 |
| 0 | mar30 | prediction_variance | 0.0626 | 0.5148 | 0.0581 | 0.2362 |
| 0 | mar30 | rank_instability | 0.6810 | 0.9068 | 0.0356 | 0.1291 |
| 0 | mcar10 | action_disagreement | 0.0328 | 0.4706 | 0.0320 | 0.1467 |
| 0 | mcar10 | entropy | 0.0328 | 0.4493 | 0.0319 | 0.1464 |
| 0 | mcar10 | hard_uncertainty | 0.0328 | 0.4706 | 0.0319 | 0.1463 |
| 0 | mcar10 | mc8 | 0.5540 | 0.9204 | 0.0194 | 0.0744 |
| 0 | mcar10 | missing_fraction | 0.0516 | 0.6865 | 0.0315 | 0.1399 |
| 0 | mcar10 | prediction_distribution | 0.0665 | 0.6259 | 0.0317 | 0.1445 |
| 0 | mcar10 | prediction_plus_explanation | 0.4990 | 0.9614 | 0.0209 | 0.0756 |
| 0 | mcar10 | prediction_variance | 0.0475 | 0.6692 | 0.0319 | 0.1460 |
| 0 | mcar10 | rank_instability | 0.5540 | 0.9204 | 0.0200 | 0.0736 |
| 0 | mcar30 | action_disagreement | 0.0747 | 0.5266 | 0.0554 | 0.2252 |
| 0 | mcar30 | entropy | 0.0712 | 0.5159 | 0.0554 | 0.2255 |
| 0 | mcar30 | hard_uncertainty | 0.0657 | 0.5227 | 0.0554 | 0.2252 |
| 0 | mcar30 | mc8 | 0.7358 | 0.9635 | 0.0274 | 0.0896 |
| 0 | mcar30 | missing_fraction | 0.0869 | 0.6291 | 0.0552 | 0.2231 |
| 0 | mcar30 | prediction_distribution | 0.0578 | 0.4948 | 0.0555 | 0.2263 |
| 0 | mcar30 | prediction_plus_explanation | 0.5728 | 0.9575 | 0.0370 | 0.1160 |
| 0 | mcar30 | prediction_variance | 0.0901 | 0.6833 | 0.0556 | 0.2265 |
| 0 | mcar30 | rank_instability | 0.7427 | 0.9644 | 0.0299 | 0.0959 |
| 1 | group_missing | action_disagreement | 0.0736 | 0.4758 | 0.0704 | 0.2699 |
| 1 | group_missing | entropy | 0.0612 | 0.3961 | 0.0711 | 0.2762 |
| 1 | group_missing | hard_uncertainty | 0.0739 | 0.4762 | 0.0704 | 0.2702 |
| 1 | group_missing | mc8 | 0.6860 | 0.9010 | 0.0432 | 0.1549 |
| 1 | group_missing | missing_fraction | 0.0761 | 0.5000 | 0.0703 | 0.2693 |
| 1 | group_missing | prediction_distribution | 0.0947 | 0.5274 | 0.0703 | 0.2696 |
| 1 | group_missing | prediction_plus_explanation | 0.6424 | 0.9228 | 0.0432 | 0.1534 |
| 1 | group_missing | prediction_variance | 0.1068 | 0.4588 | 0.0704 | 0.2707 |
| 1 | group_missing | rank_instability | 0.6308 | 0.8682 | 0.0487 | 0.1784 |
| 1 | mar30 | action_disagreement | 0.0808 | 0.5050 | 0.0592 | 0.2357 |
| 1 | mar30 | entropy | 0.0642 | 0.4928 | 0.0594 | 0.2371 |
| 1 | mar30 | hard_uncertainty | 0.0644 | 0.4872 | 0.0593 | 0.2363 |
| 1 | mar30 | mc8 | 0.6169 | 0.9166 | 0.0372 | 0.1324 |
| 1 | mar30 | missing_fraction | 0.1234 | 0.6668 | 0.0582 | 0.2268 |
| 1 | mar30 | prediction_distribution | 0.1009 | 0.5893 | 0.0588 | 0.2325 |
| 1 | mar30 | prediction_plus_explanation | 0.4237 | 0.9188 | 0.0433 | 0.1467 |
| 1 | mar30 | prediction_variance | 0.0842 | 0.6129 | 0.0593 | 0.2364 |
| 1 | mar30 | rank_instability | 0.6028 | 0.8994 | 0.0426 | 0.1495 |
| 1 | mcar10 | action_disagreement | 0.0254 | 0.4728 | 0.0266 | 0.1367 |
| 1 | mcar10 | entropy | 0.0204 | 0.3585 | 0.0270 | 0.1398 |
| 1 | mcar10 | hard_uncertainty | 0.0254 | 0.4739 | 0.0265 | 0.1364 |
| 1 | mcar10 | mc8 | 0.6304 | 0.9056 | 0.0143 | 0.0624 |
| 1 | mcar10 | missing_fraction | 0.0350 | 0.6182 | 0.0249 | 0.1184 |
| 1 | mcar10 | prediction_distribution | 0.0363 | 0.5822 | 0.0258 | 0.1290 |
| 1 | mcar10 | prediction_plus_explanation | 0.6383 | 0.9176 | 0.0144 | 0.0617 |
| 1 | mcar10 | prediction_variance | 0.0328 | 0.5728 | 0.0264 | 0.1351 |
| 1 | mcar10 | rank_instability | 0.5303 | 0.8197 | 0.0199 | 0.0843 |
| 1 | mcar30 | action_disagreement | 0.0773 | 0.4327 | 0.0706 | 0.2708 |
| 1 | mcar30 | entropy | 0.0859 | 0.5466 | 0.0704 | 0.2692 |
| 1 | mcar30 | hard_uncertainty | 0.0816 | 0.4344 | 0.0707 | 0.2716 |
| 1 | mcar30 | mc8 | 0.5165 | 0.8490 | 0.0481 | 0.1782 |
| 1 | mcar30 | missing_fraction | 0.1109 | 0.6076 | 0.0698 | 0.2648 |
| 1 | mcar30 | prediction_distribution | 0.1015 | 0.5506 | 0.0702 | 0.2685 |
| 1 | mcar30 | prediction_plus_explanation | 0.4132 | 0.8367 | 0.0543 | 0.1932 |
| 1 | mcar30 | prediction_variance | 0.0821 | 0.5479 | 0.0708 | 0.2723 |
| 1 | mcar30 | rank_instability | 0.5031 | 0.8216 | 0.0552 | 0.2065 |
| 2 | group_missing | action_disagreement | 0.1294 | 0.4953 | 0.1137 | 0.3887 |
| 2 | group_missing | entropy | 0.0920 | 0.3183 | 0.1135 | 0.3881 |
| 2 | group_missing | hard_uncertainty | 0.1306 | 0.4982 | 0.1137 | 0.3891 |
| 2 | group_missing | mc8 | 0.6760 | 0.9203 | 0.0637 | 0.2118 |
| 2 | group_missing | missing_fraction | 0.1302 | 0.5000 | 0.1133 | 0.3871 |
| 2 | group_missing | prediction_distribution | 0.2506 | 0.6300 | 0.1095 | 0.3729 |
| 2 | group_missing | prediction_plus_explanation | 0.6484 | 0.8865 | 0.0639 | 0.2310 |
| 2 | group_missing | prediction_variance | 0.1351 | 0.4528 | 0.1153 | 0.3943 |
| 2 | group_missing | rank_instability | 0.6507 | 0.9078 | 0.0692 | 0.2284 |
| 2 | mar30 | action_disagreement | 0.1435 | 0.4992 | 0.1242 | 0.4166 |
| 2 | mar30 | entropy | 0.1230 | 0.4461 | 0.1241 | 0.4161 |
| 2 | mar30 | hard_uncertainty | 0.1407 | 0.5004 | 0.1242 | 0.4169 |
| 2 | mar30 | mc8 | 0.6394 | 0.8785 | 0.0745 | 0.2648 |
| 2 | mar30 | missing_fraction | 0.2162 | 0.6143 | 0.1205 | 0.4013 |
| 2 | mar30 | prediction_distribution | 0.1914 | 0.5687 | 0.1230 | 0.4125 |
| 2 | mar30 | prediction_plus_explanation | 0.5659 | 0.8806 | 0.0856 | 0.2872 |
| 2 | mar30 | prediction_variance | 0.1759 | 0.5666 | 0.1246 | 0.4180 |
| 2 | mar30 | rank_instability | 0.6259 | 0.8710 | 0.0790 | 0.2789 |
| 2 | mcar10 | action_disagreement | 0.0934 | 0.5043 | 0.0835 | 0.3081 |
| 2 | mcar10 | entropy | 0.0738 | 0.4000 | 0.0836 | 0.3087 |
| 2 | mcar10 | hard_uncertainty | 0.0984 | 0.5071 | 0.0834 | 0.3076 |
| 2 | mcar10 | mc8 | 0.7532 | 0.9240 | 0.0377 | 0.1373 |
| 2 | mcar10 | missing_fraction | 0.1086 | 0.5930 | 0.0829 | 0.3005 |
| 2 | mcar10 | prediction_distribution | 0.0991 | 0.5302 | 0.0842 | 0.3112 |
| 2 | mcar10 | prediction_plus_explanation | 0.7349 | 0.9475 | 0.0373 | 0.1387 |
| 2 | mcar10 | prediction_variance | 0.1128 | 0.5719 | 0.0833 | 0.3065 |
| 2 | mcar10 | rank_instability | 0.7377 | 0.9120 | 0.0426 | 0.1572 |
| 2 | mcar30 | action_disagreement | 0.1289 | 0.4914 | 0.1097 | 0.3784 |
| 2 | mcar30 | entropy | 0.1183 | 0.3939 | 0.1095 | 0.3774 |
| 2 | mcar30 | hard_uncertainty | 0.1347 | 0.4960 | 0.1098 | 0.3788 |
| 2 | mcar30 | mc8 | 0.5894 | 0.8926 | 0.0688 | 0.2412 |
| 2 | mcar30 | missing_fraction | 0.1506 | 0.5643 | 0.1088 | 0.3739 |
| 2 | mcar30 | prediction_distribution | 0.1381 | 0.5416 | 0.1104 | 0.3800 |
| 2 | mcar30 | prediction_plus_explanation | 0.5898 | 0.8830 | 0.0730 | 0.2529 |
| 2 | mcar30 | prediction_variance | 0.1526 | 0.4933 | 0.1094 | 0.3776 |
| 2 | mcar30 | rank_instability | 0.5577 | 0.8630 | 0.0738 | 0.2618 |

## Runtime, test and artifact provenance

CPU, one thread, Apple M4 Pro / 24 GiB verified in this session. Cached-audit runtimes were 0.783 s, 0.763 s, 0.764 s. These exclude historical predictor training, completion generation and SHAP. They are **not** end-to-end MC latency measurements. Bounded preflight used 16 current-only rows before full evaluation. No new SHAP call or financial predictor training occurred.

Final combined suite: **181 passed, 0 failed, 0 skipped**, 11.96 s; three upstream SHAP/Matplotlib deprecation warnings. New audit tests cover customer isolation, hidden-truth exclusion, source drift, current-only APIs, 44→48 feature reduction, exact rank-event values and completion-prefix behavior. Engineering passes are not scientific validation.

| Local ignored artifact | SHA-256 |
|---|---|
| `outputs/method_pivot/round4/distribution_audit_01/freeze.json` | `8b4abb5028c3b8f4f5691c9237dbcab27ed99067657467d3a58f9379f9ac3f7a` |
| `outputs/method_pivot/round4/distribution_audit_01/results.json` | `85bab3db573a4512001901a4674acb045f906b141dff90b460cbe4c41878a1d1` |
| `outputs/method_pivot/round4/distribution_audit_01/metrics.csv` | `fd38b9036e7e34be85b1fa5fe2df1e08ca7f6f5e82fc9a38975a044138c3756e` |
| `outputs/method_pivot/round4/distribution_audit_01/cohorts.csv` | `d90d8eabafeb35aee45dcfe2071d2e4e7ca9b29504f0417404cf7fd1756eda07` |
| `outputs/method_pivot/round4/distribution_audit_fold1_01/freeze.json` | `5174a69a636ff06e25679bcfe927731236b1c8de758eb62d4499a7b14500c3f6` |
| `outputs/method_pivot/round4/distribution_audit_fold1_01/results.json` | `740b286dff545d18b2ad1cc232a82f6b3abc67a74c69cd9e7f2b722bfeb474fb` |
| `outputs/method_pivot/round4/distribution_audit_fold1_01/metrics.csv` | `39ff2cb045132a6611dba17b01e6e0ce45b95869c932f19719f4940eaafbf844` |
| `outputs/method_pivot/round4/distribution_audit_fold1_01/cohorts.csv` | `e6987bd2f38260dcc13c75080c5770f4a985d326b51e075b6382049da4a52a41` |
| `outputs/method_pivot/round4/distribution_audit_fold2_01/freeze.json` | `79e931c7b814a031dd2f1e792278f4736ce4c73a83b6b7d5b413fbff290ada62` |
| `outputs/method_pivot/round4/distribution_audit_fold2_01/results.json` | `900135fdd2486290eed8f180285f92e6c8fbde5c17439b3ea897b23a588199d3` |
| `outputs/method_pivot/round4/distribution_audit_fold2_01/metrics.csv` | `72e4a191083428e6190394ca58a4ac9a9924986235ddf4dc12df87eca1476659` |
| `outputs/method_pivot/round4/distribution_audit_fold2_01/cohorts.csv` | `26366025b3302d2170583c82a2e32350e16dee4b79d727425dab7f78005afcfb` |
| `outputs/method_pivot/round4/tests_final.log` | `257cb6436d4dc791764796ef68f5db501787ecacac3a499e085d78f810598e31` |

The input manifests hash every used current/verification array against its existing completion receipt, exact partition/predictor artifacts and relevant source/config files. They record clean implementation commits. Row scores, customer IDs, detectors, cached inputs and logs remain ignored. The public document reports aggregates only.

## Reproduction

```bash
.venv/bin/python scripts/run_prediction_distribution_audit.py \
  --stage freeze --output outputs/method_pivot/round4/reproduction_fold0
.venv/bin/python scripts/run_prediction_distribution_audit.py \
  --stage run --output outputs/method_pivot/round4/reproduction_fold0

# Repeat both stages with --config configs/prediction_distribution_audit_fold1.yaml
# and fold2.yaml, each using its own new output directory.
```

The runner refuses missing assets, receipt/source drift, role overlap, insufficient primary events, and an already-started output directory. Historical caches are required; they are not committed. No silent regeneration or synthetic fallback.

**VERIFIED:** current receipts, isolation/mask/donor checks, three executed baseline audits, paired intervals and tests. **REPORTED:** historical training and earlier research conclusions; predictors were not retrained here. **NOT RUN:** official DMV, either proposed candidate, new external confirmation, TabM, Freddie. **ASSUMED:** declared completion model, fixed explainer and the within-fold development sampling conditions.
