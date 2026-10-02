# Experimental Protocol

## 1. Scope

Primary claim:

> We evaluate whether explanations produced from incomplete credit records remain valid for the same fixed model after the originally hidden information is restored.

The study does **not** claim causal explanations.

---

## 2. Datasets

### Main — UCI Default of Credit Card Clients

Role:

- main credit-default dataset;
- primary statistical conclusions.

Target:

- default payment / non-default payment.

Why suitable:

- approximately 30,000 records;
- no original missing values;
- hidden values can be restored exactly after simulated missingness.

Source:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Validation — South German Credit

Role:

- secondary credit-risk validation.

Target:

- good / bad credit.

Important:

- this target is not identical to default payment;
- results are reported separately;
- do not merge performance metrics across the two datasets.

Source:
https://archive.ics.uci.edu/dataset/522/south+german+credit

---

## 3. Data split before any learned preprocessing

### Taiwan

For each random seed:

- Train: 50%
- Development: 10%
- Probability calibration: 10%
- Reason-risk calibration: 15%
- Test: 15%

Use stratification by target.

Run 5 seeds.

All trainable transformations are fit only on the appropriate training portion.

### South German

Because the dataset is small, treat it as secondary validation.

Use repeated stratified splits / folds and report high uncertainty honestly.

Do not use South German to tune decisions already fixed using Taiwan.

---

## 4. Leakage rules

Before preprocessing:

1. split records;
2. remove target from all imputation inputs;
3. learn encoders only from training data;
4. learn scalers only from training data;
5. learn imputers only from training data;
6. tune predictor only inside training / development;
7. calibrate probabilities only on probability-calibration data;
8. calibrate explanation release threshold only on risk-calibration data;
9. open test results only after all definitions are frozen.

Artificial copies / masks of one original record must remain inside the same outer split.

---

## 5. Missingness mechanisms

Let M be the mask.

### MCAR

Missingness is generated independently of feature values.

Use target cell-missing rates:

- 10%
- 20%
- 30%

### MAR

Missingness of one feature depends on other features that remain observed.

Example:

```text
P(M_j = 1 | x_observed)
    = sigmoid(a + b1*z1 + b2*z2)
```

Choose anchor variables in advance and keep them observed in that scenario.

Do not use the target label to generate missingness in the main study.

### MNAR — stress test only

Missingness probability depends on the value that is itself hidden.

Example:

```text
P(M_j = 1 | x_j)
    = sigmoid(a + b*x_j)
```

The simulation generator can see x_j, but the imputer cannot.

Interpretation rule:

> simulated MNAR does not prove that the real credit dataset has an MNAR missingness mechanism.

### Group missingness

Add a realistic structural stress test.

Taiwan groups:

- repayment-status history;
- bill-amount history;
- previous-payment history;
- demographic fields.

Hide an entire group for a subset of records and report the resulting actual cell-missing rate.

---

## 6. Models

### Logistic Regression

- L2 regularization.
- Tune regularization strength.
- Main linear baseline.
- Use as a sanity / negative control for interaction-driven explanation changes.

### XGBoost

- Main nonlinear predictor.
- Compact hyperparameter search.
- Early stopping.

### Direct missing-value tree baseline

Include one tree configuration that accepts missing values without the explicit multiple-imputation pipeline.

Purpose:

- test whether the observed explanation problem is only an artifact of explicit imputation.

---

## 7. Imputation conditions

Required comparison:

### A. Simple single imputation

- median for numerical;
- most frequent for categorical.

### B. Conditional single imputation

One sample from the stronger conditional imputer.

### C. Conditional multiple imputation

K plausible completions.

Primary:

- K = 20.

Ablation:

- 10
- 20
- 40

### D. Native missing handling

No explicit imputation for the chosen direct-missing tree baseline.

---

## 8. Prediction metrics

Report:

- ROC-AUC;
- Average Precision;
- Recall;
- F1 at a development-selected threshold;
- Brier score;
- log loss;
- calibration curve.

Prediction performance is reported:

1. on all test records;
2. on only records whose explanations are released.

The second result must never replace the first.

---

## 9. Explanation metrics

Main verification metric:

### Reason-revision rate

Among explanations that were released:

```text
revision risk
    = number requiring material revision after true values are restored
      / number released
```

Also report:

- coverage = fraction of cases receiving an explanation;
- top-k overlap;
- rank correlation / rank stability;
- sign agreement;
- attribution distance;
- prediction shift before vs after restoration.

These are diagnostics.

The main operating result is the **risk–coverage trade-off**.

---

## 10. Fair selector comparison

Compare:

- missing fraction;
- predictive variance;
- SHAP variance;
- rank stability;
- sign stability;
- learned verification-error selector;
- proposed revision-targeted score.

For every selector:

- same model;
- same incomplete records;
- same reason-generation rule;
- same risk-calibration split;
- same target risk level.

Primary target:

- reason-revision risk <= 10%.

Primary comparison:

> maximum explanation coverage while satisfying the fixed revision-risk limit.

---

## 11. Confidence intervals and seeds

Use:

- 5 random seeds;
- paired bootstrap by original record;
- 95% confidence intervals.

If multiple masks are generated for one record, resample at the **record level**, not at the mask/imputation level.

Do not treat K imputations as K independent customers.

---

## 12. Required ablations

### A1 — number of imputations

- K = 10 / 20 / 40.

Question:

- does extra computation materially improve revision detection?

### A2 — imputer

Compare at least:

- simple imputation;
- one conditional imputer;
- multiple conditional imputation.

Question:

- is the result specific to one imputation model?

### A3 — predictor family

- Logistic Regression;
- XGBoost.

Question:

- does interaction/nonlinearity drive explanation revision?

### A4 — selector

Replace proposed revision score with:

- SHAP variance;
- learned selector.

Question:

- is the contribution more than standard explanation uncertainty?

### A5 — explanation background

At least:

- 64 train examples;
- 128 train examples.

Question:

- is the finding an artifact of SHAP reference choice?

---

## 13. Failure cases to analyze

Manually inspect representative examples of:

1. stable prediction, unstable explanation;
2. unstable prediction, stable explanation;
3. low imputation variance but wrong / revised explanation;
4. correlated variables swapping attribution rank;
5. insufficient meaningful observed reasons;
6. selector confidently releasing an explanation that later fails;
7. distribution / missingness shift breaking risk calibration.

---

## 14. Minimum experiment matrix

### Pilot

Dataset:
- Taiwan only.

Models:
- Logistic Regression;
- XGBoost.

Missingness:
- complete;
- MCAR 30%;
- MAR 30%.

Imputation:
- simple;
- conditional multiple.

Seeds:
- 1.

Goal:
- confirm the revision phenomenon and validate code.

### Main paper experiment

Datasets:
- Taiwan;
- South German validation.

Models:
- LR;
- XGBoost.

Missingness:
- MCAR 10%;
- MCAR 30%;
- MAR 30%.

Seeds:
- 5.

Selectors:
- all required baselines + proposed.

### Stress tests

Main dataset only:

- MNAR 30%;
- grouped missingness;
- different imputer;
- different K;
- calibration under shifted missingness.

---

## 15. Continue / stop criteria

Continue toward a method paper if:

- observed-feature explanations show a non-trivial revision problem;
- the phenomenon exists across more than one missingness setting;
- the proposed selector improves the risk–coverage trade-off against the strongest baseline;
- the gain is not explained only by greater computation.

Stop claiming a new method if:

- SHAP variance or the learned selector performs equivalently or better;
- almost all explanations must be withheld;
- the phenomenon disappears once bugs / ties / correlated-feature grouping are controlled.

In that case, reposition the work as:

> an empirical study / evaluation framework for explanation reliability under incomplete credit information.

---

## 16. What is required vs optional

### Required to support the paper claim

- leakage-safe split;
- LR and XGBoost;
- simple + conditional multiple imputation;
- MCAR + MAR;
- restoration-based reason-revision target;
- SHAP-variance baseline;
- strong learned selector;
- independent risk calibration;
- 5 seeds + confidence intervals;
- failure analysis.

### Optional extensions

- deep tabular models;
- TabPFN / TabM;
- user study;
- dashboard;
- additional modern dataset;
- causal explanations;
- real-bank naturally missing data.

Do not add optional components until the core hypothesis survives the pilot.
