# FinancePaper

Research repository for:

> **When Should Credit-Risk Models Withhold Reasons? Verification-Aware Selective Explanations under Missing Information**

## Verification-supervised revision study

The next branch freezes the credit predictor and learns when its currently
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

### External validation: South German Credit

- 1,000 records.
- Target: good / bad credit.
- Used only as a secondary credit-risk validation set.
- The target is **not identical to default payment**, so results are reported separately.

Source: https://archive.ics.uci.edu/dataset/522/south+german+credit

## Minimal model stack

- Logistic Regression — additive / interpretable baseline.
- XGBoost — main nonlinear predictor.
- Median/mode imputation — simple baseline.
- Conditional / MICE-style imputation — stronger missing-data baseline.
- Multiple imputation — represent uncertainty over missing values.
- Native-missing XGBoost — baseline that does not require explicit imputation.
- SHAP — local feature attribution.

Deep learning is **not required** for the first paper version.

## High-level pipeline

```text
Complete credit record
        |
Artificially hide selected fields
        |
Observed record + missing mask
        |
Conditional multiple imputation
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
