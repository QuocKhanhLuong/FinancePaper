# FinancePaper

Research repository for:

> **When Should Credit-Risk Models Withhold Reasons? Verification-Aware Selective Explanations under Missing Information**

Working title; the project is now an **evaluation + selective explanation policy
study**, not a new credit-classifier architecture paper.

The [2026-10-04 conformal-envelope novelty audit](docs/CONFORMAL_EXPLANATION_NOVELTY_AUDIT.md)
is **NO-GO for a new-method claim on the current specification**. Existing
multilabel conformal methods already construct inner/outer sets, and prior work
already calibrates explanation outputs. The exact future-verification financial
target remains a possible evaluation contribution; the audit does not claim it
has been fully solved elsewhere. No envelope experiment or TabM training was
started after this gate. Historical negative results below remain unchanged.

The [2026-10-03 novelty reassessment](docs/NOVELTY_REASSESSMENT_2026.md) incorporates
explanation-guided acquisition and the September 2026 EDFA preprint. It supports
only a bounded [budgeted reason-verification feasibility study](docs/BUDGETED_REASON_VERIFICATION_PLAN.md),
not a new-algorithm claim. The first development headroom gate is now **measured
and NO-GO**; the full expected-value acquisition policy remains NOT RUN. All
historical results and frozen confirmation protocols below remain in force.

## One-query verification: completed headroom gate, NO-GO

The [prospective protocol](docs/VERIFICATION_HEADROOM_PROTOCOL.md) and
[measured report](docs/VERIFICATION_HEADROOM_RESULTS.md) cover 500 Polish
development clusters, MCAR10/30, and all 12,537 possible single-field reveals.
At the primary meaningful-reason target, zero-query strongest top-1 already
reaches the retained-candidate customer ceiling: 93.4%/95.0% coverage with
0.43%/2.53% observed reason failure. A hindsight oracle improves donor Stable-Core
by only 0.2/1.6 coverage points, below the predeclared 5-point gate and still below
top-1. At 10/15% budgets, release-all also attains full reason coverage. This does
not justify developing a new acquisition method or changing the target to make
one win. Prediction–explanation decoupling remains measurable.

Taiwan has no new run: all its development-pool records have appeared in a
historical outer prediction fold. The old test/reserve remains excluded.
Freddie is NOT RUN. This is explicitly development reuse, not new confirmation.
Reproduction commands, 1,000-draw cluster intervals, optimizer checks and the
121-test receipt are documented in the report. No new classifier was trained.

## Stable-Core: implemented and tested, no coverage-superiority claim

The [v2 study](docs/STABLE_CORE_STUDY_PROTOCOL.md) evaluates **variable-size reason
sets**, two completion families, B1–B6 baselines, and controls matched on each
customer's reason count. It preserves all previous models/masks/results and uses
the inspected Taiwan/Polish cohorts as an explicitly exploratory extension.

At a 10% empirical calibration budget, donor+conditional Stable-Core releases
meaningful-positive reasons for 90.82%/92.64% of Taiwan/Polish customer-condition
cases, with reason failure 0.286%/0.243%. However, release-all already has aggregate
failure 2.56%/3.90% and **higher coverage**; rank-only is also a strong control on
the ranked target. The main superior-coverage hypothesis is not supported.
Matched-size strength controls have higher failure, so some reason identity
selection benefit exists. All conservative finite-grid policies release nothing.

Read the [results and six figure descriptions](docs/STABLE_CORE_RESULTS.md),
[final ten-question decision](docs/STABLE_CORE_DECISION.md), and
[updated nearest-work review](docs/STABLE_CORE_RESEARCH_UPDATE.md).
Stable-Core remains an optional operational strategy, not a replacement or a
novel uncertainty algorithm. Freddie confirmation is still NOT RUN.

Requires the historical local artifacts reproduced by the stages below. All new
rows/caches/figures/models stay in ignored `outputs/stable_core_study/`.

```bash
uv run --frozen --extra temporal python scripts/run_stable_core_study.py freeze
uv run --frozen --extra temporal python scripts/run_stable_core_study.py calibrate
uv run --frozen --extra temporal python scripts/run_stable_core_study.py evaluate
uv run --frozen --extra temporal python scripts/report_stable_core_study.py
uv run --frozen --extra temporal python scripts/audit_stable_core_study.py
uv run --frozen --extra temporal python scripts/supplement_stable_core_study.py
# Run timing without concurrent tests/training/report generation.
uv run --frozen --extra temporal python scripts/benchmark_stable_core_study.py
uv run --frozen --extra temporal python scripts/write_stable_core_reports.py
```

The initial `stable_core.py` primitive is retained as design history; v2 serving
and calibration use `reliability/reason_sets.py`. This study does not silently
reinterpret the historical top-k reason-revision event.

## Large-scale validation: provenance accepted, real experiment not run

The [provenance audit](docs/DATASET_PROVENANCE_AUDIT.md) selects exactly one new
dataset: **Freddie Mac SFLLD, Standard annual samples, Release 47 (July 2026),
ACCEPT WITH RESTRICTIONS**. The researcher has not supplied official files. No
mortgage data were downloaded, no mortgage model was trained, and no large-scale
result is claimed.

The [frozen decision](docs/LARGE_DATASET_DECISION.md) declares 750,000 source loans
across 15 vintages, with separate temporal train/development/calibration/test roles.
Eligible counts remain unknown. The endpoint is observed **90+ DPD in the next
12 reporting months before termination**, not generic default.

- [Target and censoring](docs/FREDDIE_TARGET_DEFINITION.md), [leakage audit](docs/FREDDIE_LEAKAGE_AUDIT.md), and [data protocol](docs/FREDDIE_DATA_PROTOCOL.md).
- [Citation and access package](docs/DATASET_CITATION.md), [repository data policy](data/README.md).
- [Stable-core specification and prior work](docs/STABLE_CORE_EXPLANATION_SPEC.md).
- [Execution status and remaining gates](docs/LARGE_SCALE_VALIDATION_RESULTS.md).

The offline preparer requires user-acquired official ZIPs and a completed local
receipt based on `configs/freddie_acquisition.example.json`. It performs no login,
download or terms acceptance. Start with `--stage pilot`, then `development`;
`confirmation` refuses assessment access without hashes of fitted artifacts and
policies. See the protocol for commands. Intake, conditional-completion and
stable-core primitives have synthetic tests; the mortgage training/evaluation
runner remains to be integrated after official intake. Historical results below
are unchanged.

## Decisive semantic, external and release-policy validation

The [final decision](docs/FINAL_PAPER_DECISION.md) is **continue with a narrowed
evaluation paper; promising but incomplete for ESWA**. New grouped and external
results support prediction/explanation decoupling, but not a new uncertainty
algorithm or a deployment risk guarantee. No new neural head or predictor search.

- [Frozen semantic groups](docs/DOMAIN_REASON_GROUPS.md) and [grouped results](docs/DOMAIN_REASON_VALIDATION_RESULTS.md).
- [Monte Carlo K/runtime ablation](docs/MONTE_CARLO_K_ABLATION.md) and [failure analysis](docs/MONTE_CARLO_FAILURE_ANALYSIS.md).
- [Release-policy validation](docs/RELEASE_POLICY_VALIDATION.md).
- [External protocol](docs/EXTERNAL_DATASET_PROTOCOL.md) and [Polish bankruptcy results](docs/EXTERNAL_VALIDATION_RESULTS.md).
- [Pre-evaluation protocol](docs/DECISIVE_VALIDATION_PROTOCOL.md) and [implementation/assumption notes](docs/VALIDATION_IMPLEMENTATION_NOTES.md).

At MCAR30 and raw probability shift <=.02, **group top2** reasons revise in 42/727
stable eligible Taiwan cases (5.78%) and 85/552 Polish cases (15.40%). MC8 revision
detection AP is .6926/.8580 versus .1447/.2622 for prediction-only detectors.
These target-specific findings are not pooled credit-default prediction metrics.
The strong rank-instability baseline is often indistinguishable from MC.

The reference remains **XGBoost 25-view + completion MC K8**. K4 captures only 81%
of K16's AP gain above prevalence on Taiwan; K8 captures 93%, below the predeclared
95% criterion. At 10% declared risk target, grouped-top2 robust calibration retains
61.46%/58.21% coverage with observed MCAR30 revision 3.25%/2.18%. Robust feature-level
5% policies release nothing. Report failures and the release-all baseline.

Run from the repository root. This phase requires the existing frozen
`outputs/revision_study/fold_*/{predictors,release_model}.joblib`, partition files
and old attribution/current caches. They are intentionally not committed. On a
fresh checkout, first reproduce the four historical revision-study stages below
using `--output outputs/revision_study` (audit -> baselines -> freeze -> assess),
or recover the verified local artifacts from `04bf054`. `financepaper download`
fetches the checksum-pinned Taiwan workbook. No historical artifact is overwritten
by the validation stage. Polish download is official UCI, checksum pinned and
automatic in `external-fit`; only `5year.arff` is used.

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal financepaper download
uv run --frozen --extra temporal python scripts/run_decisive_validation.py freeze
uv run --frozen --extra temporal python scripts/run_decisive_validation.py taiwan
uv run --frozen --extra temporal python scripts/run_decisive_validation.py external-fit
# External method/predictors/policies are now hashed and immutable.
uv run --frozen --extra temporal python scripts/run_decisive_validation.py external-evaluate
# Time without concurrent training/report processes.
uv run --frozen --extra temporal python scripts/benchmark_decisive_validation.py
uv run --frozen --extra temporal python scripts/report_decisive_validation.py
uv run --frozen --extra temporal python scripts/supplement_decisive_validation.py
uv run --frozen --extra temporal python scripts/serialize_decisive_validation.py
uv run --frozen --extra temporal python scripts/audit_decisive_validation.py
uv run --frozen --extra temporal python scripts/write_decisive_reports.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
```

The default validation directory is `outputs/decisive_validation`; a completed
external evaluation refuses replacement. Current-only NPZ/JSONL and verification
truth/labels are separate. Six PNG/PDF figure pairs and tables A–E, all k/tolerance/
alpha sensitivities, 1000-draw customer-cluster intervals, 280 deterministic cases,
runtime repetitions and provenance/audit receipts are generated under `analysis/`.
Every mask/completion of a customer stays in its bootstrap cluster. Polish exact
duplicates stay in one split and one evaluation cluster, but unknown company
identities and row-level calibration still prevent entity-level guarantees.
New validation is CPU-only and ran on Apple M4 Pro, 24 GiB; no CUDA/rented GPU.
This is a bounded frozen protocol: YAML records the declared constants, while
the implementation fixes their corresponding grids. It is not a general
configuration-driven experiment search; changing a definition requires a new
prospective protocol before evaluation.

## Historical verification-supervised revision study

The previous branch froze the credit predictor and learned when its currently
observed reasons may need revision. It uses **XGBoost 25-view plus a generic
calibrated selector**, not a new recurrent architecture. Start with the
[SOTA comparison](docs/SOTA_COMPARISON_2026.md),
[metric audit](docs/REVISION_METRIC_AUDIT.md),
[decision](docs/NEXT_MODEL_DECISION.md) and
[specification](docs/REVISION_AWARE_MODEL_SPEC.md).
The [results](docs/REVISION_AWARE_RESULTS.md) and
[prediction/explanation analysis](docs/PREDICTION_EXPLANATION_DECOUPLING.md)
separate measured results from remaining validation.
The measured selector beats prediction-only uncertainty but loses to completion
Monte Carlo revision estimation. The post-assessment recommendation is to stop
new-head superiority claims and retain the evaluation/release-policy question.

This study excludes the old 4,500 inspected test records **and** the untouched
4,500 risk-calibration reserve. Three internal folds use the original 21,000
training/development/calibration pool; all policies freeze before outer assessment.
It remains an exploratory benchmark study, not independent confirmation of the
whole adaptive research process. See the [frozen protocol](docs/REVISION_STUDY_PROTOCOL.md).

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/run_revision_study.py --stage audit --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/run_revision_study.py --stage baselines --output outputs/revision-repeat
# Continue only after reviewing the diagnostic gate and recording the decision.
uv run --frozen --extra temporal python scripts/run_revision_study.py --stage freeze --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/run_revision_study.py --stage assess --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/report_revision_study.py --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/summarize_revision_study.py --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/serialize_revision_diagnostics.py --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/benchmark_revision_inference.py --output outputs/revision-repeat
uv run --frozen --extra temporal python scripts/audit_revision_study.py --output outputs/revision-repeat
```

The three-stage manifests record phase-specific source hashes. Assessment rejects
changes to frozen sources, models or policies. Generated tables and seven PNG/PDF
plot pairs live in `outputs/revision-repeat/analysis/`. Common current diagnostic
JSONL and verification-only JSONL are separate files; serving code cannot accept
restored values. A 10% empirical calibration threshold is not a guarantee; the
conservative finite-family binomial policy is reported separately. All artifacts
remain ignored by Git. CPU fallback: append `--device cpu` to study stages.

## Reproducible first pilot

The first pilot implements Taiwan, one seed, complete and MCAR 30% test records,
train-fitted median/mode imputation, Logistic Regression, XGBoost, and restoration
of the true hidden values. It tests whether predictions can stay relatively
stable while reasons from the **same fixed model** change. It does not yet
estimate or calibrate a reason-release policy.

From the repository root, with [uv](https://docs.astral.sh/uv/) and Python 3.11+:

```bash
uv sync --frozen
uv run --frozen financepaper download
uv run --frozen pytest -q
uv run --frozen financepaper pilot --config configs/pilot.yaml
```

On macOS, XGBoost also requires the OpenMP runtime: `brew install libomp`.

The download is explicit and checksum-verified. Tests use synthetic local data
and require no network. The default run uses all 30,000 official records and
evaluates all 4,500 test records, without sampling a favorable subset.

To repeat the run without overwriting results:

```bash
uv run --frozen financepaper pilot --config configs/pilot.yaml --output outputs/pilot-repeat
cmp outputs/pilot/summary.json outputs/pilot-repeat/summary.json
```

Raw data, output directories, fitted models, plots and caches are ignored by Git.
Commit the source, configuration and `uv.lock`; regenerate experiment artifacts.
The CLI refuses to write into a nonempty output directory. A completed run has
`manifest.json`; an interrupted run may leave partial artifacts without that
completion marker.

Main artifacts in `outputs/pilot/`:

| File | Purpose |
| --- | --- |
| `summary.json` | Prediction metrics, coverage, revision denominators and stable-prediction/revised-reason counts |
| `records.csv` | All test cases, model/condition, reasons, predictions and nullable revision event |
| `frozen_selection.json` | Train/development model selection, development thresholds and feature mapping, saved before test evaluation |
| `split_assignments.csv`, `background_ids.csv`, `*_masks.csv` | Audit record assignments, training reference rows and exact hidden cells |
| `*_attributions.npz` | Signed original-feature attributions before/restored, row IDs and feature order |
| `stable_prediction_revised_examples.json` | Representative qualifying cases, without assuming the phenomenon must occur |
| `prediction_vs_explanation.png`, `calibration.png`, `calibration.csv` | Diagnostic scatterplot and uncalibrated prediction reliability curve |
| `*_pipeline.joblib`, `*_background.npy` | Fitted preprocessing/model and fixed encoded SHAP reference |
| `manifest.json`, `config.yaml` | Dataset/artifact/code hashes, versions, seeds and resolved configuration |

See [pilot definitions and limitations](docs/PILOT_IMPLEMENTATION.md) for the
precise event, development selection, denominators and reproducibility checks,
and [data provenance](data/README.md) for the pinned UCI source. The broader
research plan below describes later phases, not features already implemented.

## Experimental temporal branch

The [literature decision](docs/TEMPORAL_MODEL_RESEARCH.md) authorizes a small
experimental comparison, not a new GRU architecture claim. The primary candidate
is GRU-Simple with financial values, observed masks and elapsed monthly deltas,
plus a separate static branch. Vanilla GRU and the original LR/XGBoost remain
baselines. The [specification](docs/TEMPORAL_MODEL_SPEC.md) fixes the official
April-to-September mapping, split boundaries, losses and attribution protocol.

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/run_temporal_pilot.py --device-info
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run --frozen --extra temporal pytest -q
uv run --frozen --extra temporal python scripts/run_temporal_pilot.py --config configs/temporal_pilot.yaml
# Explicit fallback or CPU verification:
uv run --frozen --extra temporal python scripts/run_temporal_pilot.py --config configs/temporal_pilot.yaml --device cpu --output outputs/temporal-cpu
```

PyTorch is optional. Automatic device choice is MPS, then CUDA, then CPU;
training uses float32. The script sets the three CPU thread limits above before
loading numerical libraries (unless already explicitly configured). On the
tested macOS/PyTorch 2.14.1 environment, the combined XGBoost/SHAP/PyTorch test
process crashed with default thread pools and passed with these limits. They
also prevent CPU oversubscription; MPS computation still runs on the GPU.
The bounded one-seed run compares complete/masked training
and BCE/weighted BCE/focal loss. It reports raw and independently calibrated
prediction metrics for all test records under complete, MCAR10/30 and MAR30.
Integrated Gradients explanations use a predeclared shared subset of 400 test
records; expensive full-test attribution is a separate configurable run.

The model exposes missingness diagnostics and optional hidden states/embeddings.
Reason eligibility is not a calibrated release decision. Reconstruction,
uncertainty ensembles and a trained revision-risk head remain deferred; their
outputs are explicitly unavailable. The paper remains about explanation
reliability unless controlled evidence supports a stronger temporal-model claim.

The [measured temporal pilot report](docs/TEMPORAL_PILOT_RESULTS.md) records the
one-seed MPS run, negative results and limitations. The primary GRU improved MAR
prediction but did not beat augmented XGBoost on MCAR AP, and its reasons revised
more often than vanilla GRU at matched coverage. This supports retaining the
explanation-reliability framing.

## Bounded robustness follow-up

The [follow-up protocol](docs/ROBUSTNESS_FOLLOWUP_PROTOCOL.md) tests stronger
multi-view XGBoost controls and additive/tree logit blends. It freezes a small
development-only candidate search, then compares five training restarts on the
same customer split. The previous test cohort has already been inspected;
these results remain exploratory. No model is promised to win every metric.

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/run_robustness_followup.py
# Or separate all fitting/selection from the test phase:
uv run --frozen --extra temporal python scripts/run_robustness_followup.py --stage fit --output outputs/followup-repeat
uv run --frozen --extra temporal python scripts/run_robustness_followup.py --stage evaluate --output outputs/followup-repeat
uv run --frozen --extra temporal python scripts/audit_robustness_followup.py --output outputs/followup-repeat
```

The fit phase writes all checkpoints, development choices, calibration and
thresholds before test evaluation. The evaluation phase verifies frozen source,
data and artifact hashes. Reuse of a nonempty run/evaluation directory is refused.
Outputs include the full development leaderboard, per-restart prediction and
reason metrics, matched coverage, calibration bins, per-record predictions and
attributions, and explicit per-metric win/tie/loss tables. Data and artifacts
remain ignored by Git. See the protocol for the recorded Orca launch failures;
this follow-up has no independent worker review.

The [measured follow-up report](docs/ROBUSTNESS_FOLLOWUP_RESULTS.md) records the
five-restart result: the 50/50 additive/tree blend reduces MCAR30 reason revision
at matched coverage, but loses calibration and some prediction metrics. Neither
selected candidate dominates all declared metrics. The stronger 25-view XGBoost
control remains essential.

## Core idea

Credit-risk models may still produce a stable risk score when a customer's financial record is incomplete, while the **reasons shown to the user can change substantially after the missing information becomes available**.

This project studies:

1. How incomplete information affects model explanations, not only predictions.
2. Whether we can estimate the risk that a currently displayed explanation will need to be revised after the true missing values are revealed.
3. When the system should **show an explanation** and when it should **withhold the explanation / request manual review**.

The proposed target is called **reason-revision risk**.

## Main research question

> **When a credit record is incomplete, can we identify which explanations are reliable enough to publish before the missing information is verified?**

## Main datasets

### Primary: UCI Default of Credit Card Clients

- ~30,000 records.
- Target: default payment.
- Used for the main default-prediction experiments.
- Original data are complete, so values can be artificially hidden and later restored for verification.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### External validation: Polish Companies Bankruptcy, fifth-year cohort

- 5,910 statements; 64 financial ratios; 410 bankruptcy events.
- Target: corporate bankruptcy within one year, not consumer next-month default.
- Natural missingness remains unknown; only artificially hidden observed cells
  are restored. The measured external assessment is 1,182 statements.
- [Official UCI source](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data), CC BY 4.0.

### Earlier proposed robustness dataset: South German Credit

- 1,000 records.
- Target: good / bad credit.
- Not used in the new decisive validation; retained as an earlier secondary option.
- The target is **not identical to default payment**, so results are reported separately.

Source: https://archive.ics.uci.edu/dataset/522/south+german+credit

## Historical broader model plan

- Logistic Regression — additive / interpretable baseline.
- XGBoost — main nonlinear predictor.
- Median/mode imputation — simple baseline.
- Conditional / MICE-style imputation — stronger missing-data baseline.
- Multiple imputation — represent uncertainty over missing values.
- Native-missing XGBoost — baseline that does not require explicit imputation.
- SHAP — local feature attribution.

The list above is the original broader plan, not a claim that every item was
implemented. The current frozen validation uses XGBoost 25-view and train-only
joint nearest-neighbour donor completions; MICE and native-missing comparisons
are not new experiments in this phase. Deep learning is **not required** for the
current explanation-release contribution.

## High-level pipeline

```text
Complete credit record
        |
Artificially hide selected fields
        |
Observed record + missing mask
        |
Training-only plausible donor completions
        |
Several plausible completed records
        |
Logistic Regression / XGBoost
        |
SHAP explanations
        |
Candidate reasons from observed features only
        |
Estimate reason-revision risk
        |
Risk-calibrated release threshold
        |
+----------------------+-----------------------+
|                                              |
Low revision risk                         High revision risk
Show explanation                         Withhold explanation
                                         / manual review
        |
Reveal original hidden values
        |
Measure whether the released reasons
actually need to be revised
```

## Repository docs

- [Research concept and novelty](docs/RESEARCH_CONCEPT.md)
- [Model pipeline](docs/MODEL_PIPELINE.md)
- [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Related work and gap](docs/RELATED_WORK_GAP.md)

## Important interpretation rule

This project does **not** claim that SHAP identifies the true causal reasons for credit default.

The study only measures whether the explanation produced by the **same predictive model** remains consistent after previously hidden information is restored.

## Current target venue

**Expert Systems with Applications (ESWA)** is being considered as a target venue, but venue fit depends on whether experiments show a meaningful problem and a contribution beyond standard multiple imputation + SHAP uncertainty.
