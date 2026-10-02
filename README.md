# FinancePaper

Research repository for:

> **When Should Credit-Risk Models Withhold Reasons? Verification-Aware Selective Explanations under Missing Information**

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
