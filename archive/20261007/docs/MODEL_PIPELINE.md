# Model Pipeline

## Recommended paper-grade pipeline

The first implementation should remain deliberately simple:

- **Predictor 1:** Logistic Regression.
- **Predictor 2:** XGBoost.
- **Main imputer:** conditional multiple imputation.
- **Simple imputer baseline:** median / most-frequent.
- **Explanation:** SHAP.
- **Main research object:** reason-revision risk.
- **No deep learning in the first version.**

The novelty should come from the research question and evaluation target, not from adding a large neural architecture.

---

## 1. End-to-end flow

```text
Original complete record x*
        |
        |  simulate missing information
        v
Incomplete record x_obs + missing mask m
        |
        +-------------------------------+
        |                               |
        | simple baseline               | conditional multiple imputation
        |                               |
        v                               v
single completed record        K plausible completed records
                                        |
                              +---------+---------+
                              |                   |
                              v                   v
                    prediction f(x^k)      SHAP phi(x^k)
                              |                   |
                              +---------+---------+
                                        |
                           aggregate over K samples
                                        |
                              candidate reason set R
                           using observed fields only
                                        |
                           Monte-Carlo revision score
                                  u_revision(x)
                                        |
                            risk-calibrated threshold
                                        |
                  +---------------------+---------------------+
                  |                                           |
                  v                                           v
          release explanation                     withhold explanation
                                                   / manual review
                  |
                  v
       reveal original hidden values x*
                  |
                  v
       recompute explanation phi*
                  |
                  v
        did released reasons need revision?
                  |
                  v
          reason-revision event E
```

---

## 2. Data representation

Let the original complete record be:

```text
x* = [x1, x2, ..., xd]
```

Create a missing mask:

```text
m_j = 1 if feature j is hidden
m_j = 0 if feature j is observed
```

The model receives:

```text
x_obs + m
```

The hidden values are retained only by the evaluation code and are **never available to the predictor, imputer or selector at inference time**.

Missingness is simulated **before preprocessing / encoding** so the research object remains an original financial field rather than an encoded dummy column.

---

## 3. Preprocessing

### Taiwan Default of Credit Card Clients

Drop:

- ID.

Suggested grouping:

**Continuous / monetary**
- LIMIT_BAL
- AGE
- BILL_AMT1 ... BILL_AMT6
- PAY_AMT1 ... PAY_AMT6

**Categorical / coded**
- SEX
- EDUCATION
- MARRIAGE
- PAY_0, PAY_2 ... PAY_6

For the main comparable pipeline:

- numeric -> median / conditional numerical imputation -> StandardScaler for LR;
- categorical -> conditional categorical imputation -> OneHotEncoder;
- XGBoost can use the same encoded representation for a clean apples-to-apples comparison.

Repayment-status codes should not silently be interpreted as equal-distance continuous variables in Logistic Regression.

### South German Credit

Only these are truly quantitative in the supplied codebook:

- duration
- amount
- age

The remaining coded variables should be treated as categorical or ordered categorical according to the code table, not as arbitrary continuous integers.

For the first implementation, one-hot encoding all non-quantitative variables is the safest common representation.

---

## 4. Predictive models

### Model A — Logistic Regression

Purpose:

- transparent baseline;
- strong classical credit-scoring reference;
- negative control for interaction-driven explanation revision.

Recommended:

- L2 regularization;
- tune C inside training CV;
- class weighting tested as an ablation, not enabled silently.

For a purely additive linear logit with a fixed SHAP background, changing a different hidden variable should not change the attribution of an already observed feature. This gives a useful sanity check.

### Model B — XGBoost

Purpose:

- main nonlinear predictor;
- captures interactions that may cause explanations of observed features to change when other information is restored.

Tune only a compact search space:

- max_depth: 2–6
- learning_rate: 0.02–0.15
- n_estimators: early stopping / upper bound
- min_child_weight
- subsample
- colsample_bytree
- reg_lambda

Do not perform a huge hyperparameter search. The research question is explanation reliability, not leaderboard optimization.

### Native-missing robustness baseline

Add one tree-boosting configuration that directly accepts missing values.

This baseline answers:

> Is the problem caused only by explicit imputation, or does explanation revision remain when the predictor handles missing values internally?

Keep this as a robustness baseline rather than the main method.

---

## 5. Imputation

### Baseline 1 — Single simple imputation

- numeric: train-set median;
- categorical: train-set most-frequent category.

Purpose:

- cheap baseline;
- exposes how much benefit comes from conditional / multiple imputation.

### Baseline 2 — Single conditional imputation

Generate one conditional completion of the missing fields.

Purpose:

- separates **quality of the imputer** from **benefit of multiple samples**.

### Main — Multiple conditional imputation

Learn:

```text
q_theta(X_missing | X_observed)
```

using **training predictors only**.

Do not give the target label to the imputer in the main experiment.

At inference, sample:

```text
x^(1), ..., x^(K)
```

with K = 20 initially.

Recommended implementation:

- MICE / miceforest-style conditional models;
- numerical and categorical variables use suitable conditional models;
- fit on train only;
- apply frozen imputation models to development/calibration/test.

Ablate:

- K = 10
- K = 20
- K = 40

If K = 10 performs equivalently, prefer it because explanation computation is expensive.

---

## 6. Prediction under missing information

For each completion:

```text
p_k = f(x^(k))
```

Aggregate:

```text
p_bar = mean(p_1, ..., p_K)
```

Prediction uncertainty baseline:

```text
u_pred = variance(p_1, ..., p_K)
```

Probability calibration must be fit on a separate probability-calibration split.

Recommended first choice:

- Platt / logistic calibration.

Report:

- ROC-AUC
- Average Precision
- Brier score
- log loss
- calibration curve

---

## 7. Explanation generation

For every completion:

```text
phi^(k) = SHAP(f, x^(k))
```

Use:

- LinearSHAP / exact linear attribution for Logistic Regression;
- TreeSHAP for XGBoost.

Use a fixed background sampled from training data only.

Recommended initial background:

- 64 train records.

Sensitivity:

- 128 records.

When one-hot encoding is used, aggregate SHAP values back to the **original feature** before selecting reasons.

Example:

```text
EDUCATION_university
EDUCATION_high_school
EDUCATION_other
        |
        v
single original feature: EDUCATION
```

---

## 8. Candidate reasons

Only explanations about **currently observed original features** are eligible to be shown.

Reason:

A value generated by the imputer should not be presented to a reviewer as if it were an observed fact about the customer.

For each observed feature j:

```text
phi_bar_j = mean_k(phi_j^(k))
```

Initial setting:

- choose top 3 positive risk-increasing features;
- require a minimum attribution magnitude;
- if fewer than 3 meaningful observed reasons exist, withhold rather than filling the list with weak reasons.

The exact threshold is selected on development data and frozen before test.

---

## 9. Proposed revision-targeted score

Let R be the candidate reason set.

Split the K imputations into two groups:

- Group A: creates the candidate reasons.
- Group B: estimates how often those reasons fail under alternative plausible completions.

For every completion k in Group B, check whether a reason:

1. changes sign from risk-increasing to non-increasing; or
2. leaves the top-k important observed features by more than a predefined rank tolerance.

Define:

```text
u_revision(x)
    = (# plausible completions in which R would need revision)
      / (# Group-B completions)
```

Interpretation:

> Under the imputation model, how often would the reasons we are about to publish fail to remain materially similar?

This is a **Monte-Carlo operational score**, not a guaranteed posterior probability.

---

## 10. Verification target

Because the benchmark starts from complete data, restore the actual hidden values:

```text
x_obs + true hidden values = x*
```

Compute:

```text
phi* = SHAP(f, x*)
```

Define the binary verification event:

```text
E = 1  -> released reason set requires material revision
E = 0  -> released reason set remains acceptable
```

The exact rule should use:

- sign preservation;
- top-k membership among the features that were already observed;
- a small attribution tolerance to avoid counting numerical ties as failures.

This event is the main evaluation target.

---

## 11. Release threshold

The final policy is:

```text
if u_revision(x) <= threshold:
    release explanation
else:
    withhold explanation / manual review
```

Do not choose the threshold on test data.

Use the independent risk-calibration split.

Primary operating point:

```text
accepted reason-revision risk <= 10%
```

Among thresholds satisfying the risk constraint, choose the one with the highest coverage.

Report the full **risk–coverage curve**, not only one threshold.

---

## 12. Selector baselines

The proposed revision score must be compared against:

1. missing-rate heuristic;
2. prediction variance across imputations;
3. SHAP attribution variance;
4. top-k SHAP rank stability;
5. sign stability;
6. a learned selector trained on development data to predict the verification event E.

All selectors:

- see the same incomplete cases;
- use the same predictor;
- use the same candidate reason definition;
- receive the same calibration budget;
- are evaluated on the same test masks.

---

## 13. Recommended MVP

Implement in this order:

### Phase 1 — sanity pipeline

```text
Taiwan
-> one train/test seed
-> MCAR 30%
-> median imputation
-> LR + XGBoost
-> SHAP
-> restore true values
-> measure reason-revision rate
```

Goal: determine whether the phenomenon exists.

### Phase 2 — strong imputation

```text
MCAR + MAR
-> conditional multiple imputation K=20
-> u_revision
-> risk–coverage
-> SHAP variance baseline
```

Goal: determine whether a revision-targeted score adds value.

### Phase 3 — paper experiment

- five seeds;
- multiple missing rates;
- strong learned selector;
- independent probability and risk calibration;
- South German external validation;
- MNAR and grouped-missingness stress tests;
- ablations.

Do not start with Phase 3.
