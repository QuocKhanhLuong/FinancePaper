# Round 4: Missing Value Uncertainty source audit

Audit date: **2026-10-05**. FinancePaper starting commit:
`9f047cdad5d743f3d3794d3a33a47bda69ac9e74`, branch `research/method-pivot`.
This is a baseline and formulation audit, not a reproduction or a new method.

## Verified identity and access

Burnett, David J.; Ganguli, Surojit; Kaplan, Lance M.; Upadhyay, Devesh;
and Inouye, David I. (2026). *Missing Value Uncertainty: Could Collecting
Missing Values Change the Prediction?* Transactions on Machine Learning Research,
August 2026. [Official accepted-paper listing](https://jmlr.org/tmlr/papers/),
[paper](https://openreview.net/pdf?id=BRWTS5e03Z),
[official code release](https://github.com/inouye-lab/MissingValueUncertainty/releases/tag/tmlr).
The old ICLR submission is not evidence of ICLR acceptance.

The `tmlr` release was inspected at exact commit
`c26359cd743d4c605370c2c9916e2d3bca34903a`. Source checkout is local and ignored:
`outputs/method_pivot/round4/mvu_sources/official`.
Direct OpenReview PDF retrieval encountered a browser-verification challenge;
indexed primary-PDF passages were readable. We inspected those passages and the
listed implementation files, **not the entire final accepted PDF**.

No root or nested license/copying file was found outside Git internals at that
pin. The current official GitHub repository API returned `license: null`.
**Third-party code reuse permission: UNRESOLVED.** This is an evidence boundary,
not a legal conclusion that all research about the algorithm is prohibited.
No official dependency installation, training, benchmark, or vendoring was done.
A separately written implementation of published mathematical definitions must
be labeled an adaptation, not an official reproduction.

## Target and reduction that matter to FinancePaper

For a fixed complete-input predictor f and observed information W=(S,x_S), the
relevant distribution is the law of P=f(X) conditional on W. Its hard-vote
confidence is the probability of its modal decision, whereas soft confidence
uses the mean class-probability vector. DMV learns a conditional distribution of
complete-input probabilities from masked training examples; the inspected
implementation uses a Dirichlet family. The paper also calibrates confidence
against decision consistency after revelation. These are prediction targets,
not reason-revision targets. This paragraph summarizes the primary paper;
the operational details below come from direct code inspection.

For binary classification, independently spelling out those statistics gives:

```text
P_k = f(x_S, completion_k)
r = mean_k 1[P_k >= 0.5]
c_hard = max(r, 1-r)
c_soft = max(mean_k P_k, 1-mean_k P_k)
```

The modal decision in `c_hard` need not equal the decision of FinancePaper's
current median/mode-imputed record. Therefore also report the separately named
**current-action agreement**:

```text
c_current = mean_k 1[1[P_k >= 0.5] == 1[p_current >= 0.5]]
```

This last definition is a baseline adaptation, not an attributed MVU theorem.
The finite K frequencies above are Monte Carlo controls, **not learned DMV**.
Probability 0.5 needs an explicit tie convention; the displayed convention is
positive-at-equality and must be tested, since generic multiclass argmax ordering
can choose the opposite class at an exact tie.

Hard confidence 1 does not imply unchanged probability: completions predicting
0.51 and 0.99 have the same class. Nor does it imply unchanged reasons. An
explanation can vary along a level set of f. This distinction motivates a fair
comparison; it does not establish an empirical win over MVU or a new algorithm.

## Source-level contracts at the pinned release

All paths below are relative to the official repository at the pin above.

| Source inspected | Observed behavior | Consequence for our baseline |
|---|---|---|
| `README.md`, `requirements.txt` | Python 3.9.18 is the documented environment; Torch ~2.1.0 and image-oriented dependencies; release offers image classifier checkpoints | Do not silently install its old environment over FinancePaper or claim a tabular benchmark was supplied |
| `learn_dirichlet_network.py` | Supports an optional frozen teacher loaded through `NeuralNetworkRegressor.load`; masked/student branch gets gradients; script also evaluates its dataset test partition | The supplied CLI is not a direct XGB25 adapter and must not be pointed at the repeatedly inspected historical test as a development shortcut |
| `mvu/model/loss.py` | Combines clean classification, masked classification, and Dirichlet NLL on clean probabilities; logit form maps concentration through exp | With a frozen teacher, the clean classification term is constant with respect to student parameters; any tabular adaptation must disclose all nonconstant terms |
| `mvu/explanation/decision.py` | Chooses the modal minimum-loss action over sampled class-probability vectors and returns its agreement frequency | It is not automatically agreement with our current imputed action |
| `mvu/explanation/dirichlet.py` | Scales predicted concentration and samples probability vectors before computing best actions; wrapper expects a neural regressor/mask convention | Direct distribution estimation avoids feature-space completion, but this implementation can still sample in probability space to obtain decision confidence |
| `mvu/explanation/calibration.py` | MVCE bins hard confidence and compares it with consistency against a complete-input decision; outcome accuracy is a separate calculation | MVCE is not outcome calibration, SHAP calibration, or verified-reason risk calibration |
| `mvu/util.py` | Device helper chooses CUDA if requested/available, otherwise CPU | CPU exists; native MPS selection was not found in this helper. No measured Mac runtime claim |

The optional frozen teacher is important: replacing the predictor with a new DMV
classifier would change both the financial task function and its verified
explanations. A fair explanation-detection experiment should keep XGB25 frozen
and explicitly distinguish a distribution surrogate from the risk predictor.

## What a strong control must receive

1. Same incomplete records, masks, permitted fit customers and frozen predictor.
2. Same completions for Monte Carlo prediction and explanation statistics.
3. Prediction-only controls: current probability, entropy, completion mean,
   variance, quantiles, hard confidence and current-action agreement.
4. The same detector-training/calibration budget when supervised revision
   detection is used; distinguish an unsupervised statistic from a fitted detector.
5. A learned DMV adaptation, if later run, trained only on permitted fit data,
   with full probabilities as supervision only. Its calibration against decision
   consistency must remain distinct from calibration for revision detection.
6. Explicit cost accounting: rank instability and reason Monte Carlo both use
   completion explanations. Hard-vote prediction controls do not require SHAP.

No theorem makes prediction-only summaries sufficient for an explanation event.
Conversely, the historical weak entropy selector does not show that a well-fitted
conditional prediction-distribution baseline is weak. Both must be tested on the
same development protocol before expanding claims.

## Decision

**GO for a bounded, independently implemented prediction-distribution baseline
audit; NO-GO for claiming a new method from this audit alone.** The smallest
first comparison can reuse same-K completion predictions without copying the
unlicensed code. It must not be called DMV reproduction. Learned DMV requires
an explicitly specified adaptation, separate training and fair capacity/budget.

**VERIFIED:** accepted-paper identity; source pin; source contracts above;
license API result; finite-statistic distinctions.
**REPORTED:** paper performance is not reproduced here and no numbers are imported.
**NOT RUN:** official MVU/DMV, a tabular DMV adaptation, new financial comparison.
**ASSUMED:** the chosen completion law represents plausible missing values;
its misspecification is not solved by taking hard votes.
