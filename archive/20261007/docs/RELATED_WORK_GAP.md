# Related Work and Research Gap

## Papers that directly constrain the novelty claim

### 1. Explainability of Machine Learning Models under Missing Data

- Studies how missing-data handling changes model explanations.
- Shows that an imputation method that improves prediction does not necessarily preserve explanation quality.
- Directly rules out a novelty claim based only on "missing data affects SHAP".

Preprint:
https://arxiv.org/abs/2407.00411

Applied Soft Computing DOI:
https://doi.org/10.1016/j.asoc.2026.115105

### 2. Imputation Uncertainty in Interpretable Machine Learning Methods

- Explicitly studies uncertainty induced by imputation in interpretable ML.
- Uses multiple imputation / pooling ideas to quantify uncertainty in interpretations.
- Directly rules out a novelty claim based only on "multiple imputation + SHAP uncertainty".

https://arxiv.org/abs/2512.17689

### 3. Selective Explanations — NeurIPS 2024

- Studies when explanations should be accepted or replaced by a more expensive explanation procedure.
- Means that "only show explanations when confidence is high" is not new by itself.

https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html

### 4. Calibrated Explanations — Expert Systems with Applications 2024

- Adds uncertainty information and calibration to local explanations.
- Means that explanation calibration itself is not novel.

DOI:
https://doi.org/10.1016/j.eswa.2024.123154

### 5. Interpretable machine learning for imbalanced credit scoring datasets — EJOR 2024

- Studies explanation behavior/stability in credit scoring under imbalance.
- Means that "credit explanation stability is underexplored" is too broad a claim.

DOI:
https://doi.org/10.1016/j.ejor.2023.06.036

### 6. Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring — EJOR 2025

- Studies SHAP/LIME stability in a cost-sensitive credit-scoring setting.
- Further prevents a generic "stability in credit scoring" novelty claim.

DOI:
https://doi.org/10.1016/j.ejor.2025.05.039

## Gap we are testing

The candidate gap is narrower:

> Existing work measures explanation sensitivity, uncertainty or stability, but does not directly make the **post-verification revision event** the operational target: whether the reasons we are about to show from an incomplete record will still be defensible after the hidden true values become known.

The proposed study therefore evaluates:

1. explanation before verification;
2. restoration of the actual held-out value;
3. explanation after verification;
4. whether the originally released reasons materially change;
5. whether this event could have been predicted before verification.

## Important caution

This is a **candidate research gap**, not a proven novelty claim.

The idea should only be retained as a methodological contribution if:

- the revision phenomenon is non-trivial;
- simple baselines do not already solve it;
- the proposed score provides a reproducible risk–coverage benefit;
- results generalize beyond one model and one missingness scenario.
