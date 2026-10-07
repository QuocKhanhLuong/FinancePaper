# Research Concept

## Working title

**When Should Credit-Risk Models Withhold Reasons? Verification-Aware Selective Explanations under Missing Information**

Alternative descriptive title:

**Reliable Explainable Credit-Risk Prediction under Incomplete Financial Information**

## 1. Practical problem

Credit models are increasingly expected to provide both:

- a risk prediction; and
- an explanation of why the prediction is high or low.

However, a credit record may be incomplete when a decision is made.

A common pipeline fills missing values and then produces a prediction and a SHAP explanation.

The problem is that:

- the prediction may remain similar after the true information becomes available;
- but the explanation may change substantially.

Example:

```text
Incomplete record:
Risk = 78%
Main reasons = Payment history, Credit utilization, Debt

After hidden information is restored:
Risk = 76%
Main reasons = Loan amount, Age, Credit utilization
```

The score barely changes, but the explanation does.

## 2. Main research question

> **When a credit record is incomplete, can we determine whether the current explanation is reliable enough to publish before the missing information is verified?**

## 3. Sub-questions

### RQ1

How strongly does incomplete financial information affect local explanations compared with predictive performance?

### RQ2

Can explanations for **already observed variables** still change when other missing variables are restored?

### RQ3

Can we estimate which explanations are likely to require revision after the missing information is verified?

### RQ4

Can a calibrated release policy expose more useful explanations at the same accepted reason-revision risk than strong baselines such as missing-rate heuristics, predictive uncertainty and SHAP variance?

## 4. Testable hypotheses

### H1 — Prediction robustness is not explanation robustness

Prediction performance / probability may change only slightly while the identity, rank or sign of important explanation features changes substantially.

### H2 — Missingness can affect explanations of observed variables

For nonlinear models with feature interactions, the attribution assigned to an observed feature can change when other variables are completed.

### H3 — Revision-targeted uncertainty is more useful than generic uncertainty

A score designed to estimate whether the currently displayed reasons will need revision can provide better risk–coverage trade-offs than generic prediction variance or SHAP variance.

If these hypotheses are not supported, the proposed methodological contribution should be dropped or reduced to an empirical study.

## 5. Proposed contribution

### Contribution 1 — Verification-based evaluation target

Define **reason-revision risk**:

> the risk that reasons published from an incomplete record must be materially changed after the hidden true values are restored.

This is not the same as SHAP variance.

### Contribution 2 — Mechanism analysis

Study when missing variables alter explanations for variables that were already observed, including an additive Logistic Regression negative control and a nonlinear XGBoost model.

### Contribution 3 — Selective explanation policy

Estimate revision risk from plausible completions of the missing information and calibrate a threshold for deciding whether to:

- release the explanation; or
- withhold it / request manual review.

## 6. What is NOT claimed as novelty

The following are already known research directions and are not sufficient contributions by themselves:

- missing-data imputation;
- multiple imputation;
- SHAP;
- SHAP uncertainty;
- explanation stability;
- selective explanation / abstention;
- probability calibration.

The paper must demonstrate value specifically for **verification-aware reason revision under incomplete credit information**.

## 7. Success criterion

The key comparison is:

> At the same maximum accepted reason-revision risk, can the proposed method safely release explanations for more cases than the strongest baseline?

Example evaluation:

```text
Allowed revision risk: <= 10%

SHAP-variance selector:
Coverage = 55%

Learned baseline selector:
Coverage = 63%

Proposed revision-targeted selector:
Coverage = 72%
```

These numbers are illustrative only. No result is assumed in advance.

## 8. Interpretation limitation

Stable SHAP explanations do not imply:

- causal correctness;
- fairness;
- economic correctness;
- correct underwriting decisions.

The reference used here is only:

> explanation from the same fixed model after the originally hidden values are restored.
