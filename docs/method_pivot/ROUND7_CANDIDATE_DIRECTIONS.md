# Round 7 — broaden the search, then audit exact reductions

Primary-source search/review: **2026-10-07**. Repository read:
`a9de607946d94c087ec2ebe09f76e2e9252e62f4`, `research/method-pivot`.
Root generated ideas before reading two independent Orca AGY reports. Those
reports are proposals, not trustworthy empirical/theoretical evidence until
checked. [Corrections and orchestration receipt](ROUND7_ORCHESTRATION_REVIEW_RECEIPT.md).

The four directions below differ in what changes: learning objective,
measurement allocation, inference ambiguity, or training observation structure.
They are not four classifier architectures. The original A–D and rounds 1–6
remain historical; this document does not replace them.

## Closest-work ledger

F = relevant full-text sections inspected by coordinator; A = official
abstract/metadata only. A is insufficient to claim a theorem or reproduce an
algorithm. No listed implementation was executed in this round. No code was
copied; code licenses not inspected mean **not cleared for reuse**, not forbidden.

| Paper | Status/year | Exact problem | Method | Assumptions | Closest overlap | What is still missing | Proposed difference | Official source / code status |
|---|---|---|---|---|---|---|---|---|
| Gögl, Xing, Yau, *Martingale-Consistent Self-Supervised Learning* | Preprint, May 2026; F §§3–4, C.2 | Coarse/refined prediction coherence | Conditional-mean penalty, independent refinement draws | Correct conditional draws for unbiased actual-law objective; learned imputer approximates this | Direction 1's entire coherence objective | Verification correction with misspecified imputer is not established by that objective alone | Anchor with observed audits; must beat AIPW/conditional moments | [arXiv v1](https://arxiv.org/html/2605.11846v1); no code reused |
| Robins, Rotnitzky, Zhao, *Estimation of Regression Coefficients When Some Regressors Are not Always Observed* | JASA 1994; A | Missing regressors | Augmented weighted estimating equations | MAR, positivity, regularity | Audit pseudo-outcome correction | Not a ready-made joint neural coherence implementation | Fixed-teacher proposal reduces to this statistical machinery | [Publisher](https://doi.org/10.1080/01621459.1994.10476818) |
| Zhao, Candès, *Imputation-Powered Inference* | Preprint 2025; F §§2–5 | Infer parameters with incomplete records and arbitrary imputer | Correct imputation using complete records | Primary MCAR setting; broader target needs extra assumptions | Correct task functionals rather than raw missing values | Not arbitrary MNAR recovery or automatically joint-training guarantee | No distinction from a generic correction has been derived | [Primary text](https://arxiv.org/html/2509.13778v1); [author code](https://github.com/sarmxzh/imputation-powered-inference), license not audited |
| Kluger et al., *Prediction-Powered Inference with Imputed Covariates and Nonuniform Sampling* | Preprint 2025; A | Inference using imputed covariates and nonuniform complete sample | Predict-then-debias/bootstrap | Sampling design and inferential regularity; not arbitrary unidentified missingness | Audits need not be uniformly sampled | Abstract does not establish arbitrary jointly trained encoders | Merely adding nonuniform audits is insufficient | [Primary record](https://arxiv.org/abs/2501.18577); [author code](https://github.com/DanKluger/Predict-Then-Debias_Bootstrap), license not audited |
| Muandet, Jitkrittum, Kübler, *Kernel Conditional Moment Test via Maximum Moment Restriction* | UAI 2020; F method | Conditional moment restrictions | RKHS embedding / maximum moment criterion | Suitable kernels, moments and sampling assumptions | Direction 1's residual conditional mean | Finite-budget computational/statistical advantage remains unshown | Ordinary critic duality alone adds no mechanism | [Proceedings](https://proceedings.mlr.press/v124/muandet20a.html) |
| Zrnic, Candès, *Active Statistical Inference* | ICML 2024; A + official abstract | ML-assisted measurement allocation | Bias-corrected inference with targeted sampling | Budget, positive observation probabilities | Direction 2's audit allocation | Different financial operational costs need a defined estimand | Credit application alone is not a new allocation algorithm | [Proceedings](https://proceedings.mlr.press/v235/zrnic24a.html); [author code](https://github.com/tijana-zrnic/active-inference), license not audited |
| Li, Zrnic, Candès, *Robust Sampling for Active Statistical Inference* | NeurIPS 2025; F §§2–3, theorem 1 | Bad uncertainty estimates can harm active sampling | Budget-preserving interpolation and robust optimization | Asymptotic theorem needs consistent path selection; ambiguity-set assumptions matter | Beyond uniform-versus-Neyman already studied | Does not automatically cover every end-to-end prediction objective | Must outperform this control, not just uniform audits | [Official paper](https://papers.nips.cc/paper_files/paper/2025/file/6389470564214983604d1ac81631c2c5-Paper-Conference.pdf); code not reused |
| Kukliansky, Shamir, *Attribute Efficient Linear Regression with Distribution-Dependent Sampling* | ICML 2015; F algorithm/variance discussion | Limited-attribute regression | Distribution-dependent measurement sampling | Bounded regression setting; moments known/estimated | Directions 2/4: adaptive observation efficiency | Group costs and nonlinear loss need their own derivation | Variance-aware sampling itself is already known | [Proceedings](https://proceedings.mlr.press/v37/kukliansky15.html) |
| Dorn, Guo, *Sharp Sensitivity Analysis for IPW via Quantile Balancing* | JASA 2023; A + author abstract | Unobserved-confounding sensitivity | Sharp scalar bounds | Tan sensitivity model, its actual propensity constraints | Direction 3: scalar worst-case contrasts | Attribution application differs from causal estimand | Claimed joint advantage fails independently of paper's details | [Correct DOI](https://doi.org/10.1080/01621459.2022.2069572), [author manuscript](https://jacobdorn.info/files/QuantileBalancing.pdf) |
| Yadlowsky et al., *Bounds on the Conditional and Average Treatment Effect with Unobserved Confounding Factors* | Annals of Statistics 2022; A | Conditional treatment-effect bounds | Loss-based bound estimation | Bounded confounding / identification assumptions | Customer-conditional bounds not a new general idea | Financial completion law is a different object | Must specify ambiguity mapping; cannot import guarantee | [Author manuscript/metadata](https://pmc.ncbi.nlm.nih.gov/articles/PMC10694186/), DOI 10.1214/22-AOS2195 |
| Laberge et al., *Partial Order in Chaos: Consensus on Feature Attributions in the Rashomon Set* | JMLR 2023; F intro/method | Attribution disagreement across good models | Consensus statements / partial orders | Specified model set and explanation convention | Direction 3: statements surviving ambiguity | Completion ambiguity differs from model ambiguity | Cartesian product of two ambiguity sets is not enough | [Official paper](https://jmlr.org/papers/v24/23-0149.html) |
| Cesa-Bianchi, Shalev-Shwartz, Shamir, *Efficient Learning with Partially Observed Attributes* | JMLR 2011; A | Learning with measurement budgets | Attribute-limited learning | Stated training/test budget model | Direction 4's problem family | Raw-feature interaction measurement costs require explicit comparison | Cannot claim invention of budgeted feature observation | [Official paper](https://www.jmlr.org/papers/v12/cesa-bianchi11a.html) |
| Hazan, Koren, *Linear Regression with Limited Observation* | ICML 2012; F §3/algorithm 1 | Few observed coordinates per example | Unbiased randomized gradient estimates | Bounded norm/appropriate loss; independent sampling | Direction 4: stochastic loss/gradient moments | Naive lifting may spend redundant raw queries | Must beat symbolic simplification + unbiased sampling | [Official paper](https://icml.cc/2012/papers/433.pdf) |
| Bullins, Hazan, Koren, *The Limits of Learning with Missing Data* | NeurIPS 2016; F intro/theorem 1 | Observation limits by loss | Lower bounds and algorithms | Particular limited-observation model/classes | Direction 4: loss determines identification requirements | Not a universal lower bound for every financial missingness problem | Need a sharper distinct result, not re-label lower bounds | [Official paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/955a1584af63a546588caae4d23840b3-Paper.pdf) |

Additional checks: [Sudak and Tschiatschek's VAE posterior-consistency work](https://arxiv.org/abs/2310.16648)
(official abstract inspected) already addresses inconsistent missing-pattern
posteriors. [Zhu and Ying, MSML 2020](https://proceedings.mlr.press/v107/zhu20a.html)
(official abstract) studies the double-sampling obstacle for squared conditional
residuals in RL. This does not solve our financial problem directly, but means
discovering that obstacle is not itself a new theory contribution. These are
related-work boundaries, not reproduced benchmarks.

## 1. Verification-anchored learning, including joint training

Another necessary control is [Lopez-Paz et al., *Unifying distillation and
privileged information*, ICLR 2016](https://bottou.org/papers/lopez-paz-2016)
(author abstract/metadata verified): using richer training information to teach
a deployment student is already an established formulation. If Y is available,
direct masked supervised ERM must also receive the same training labels. A
teacher's extra verified-data budget is not free.

**Strongest rejection.** With the teacher frozen, the proposed correction and
student score are exactly AIPW pseudo-outcome regression. With both predictors
trained, naive squared corrected targets optimize a variance-penalized objective.
A conditional-moment critic fixes the population objective through known duality.

**Defensible difference sought.** A finite-audit, finite-compute algorithm for
jointly learned nested predictors could still differ through statistical or
optimization efficiency. It must show that advantage against the same-audit
conditional-moment/AIPW controls. Neither independent proposal supplies that
step. Thus **selected for deeper falsification, not accepted as a method**.
Mac/toy feasibility is excellent; a new financial run is premature.

## 2. Learn which records to verify under costs

**Strongest rejection.** The worker's sigma/sqrt(cost) allocation solves a
classical variance-allocation problem. It is not a bound on arbitrary classifier
excess risk. Strong modern controls already address unreliable uncertainty
scores; comparison only with uniform acquisition is inadequate.

**Defensible difference sought.** Coupled group-verification costs or changing
learned targets could require a different design objective. No corresponding
estimator or proven need is supplied. The two-stratum gain is real arithmetic
for a known rule, not methodological novelty. **NO-GO for this proposal**.
It also differs from the historical one-field per-customer release task; that
old negative result is not the reason for rejection here.

## 3. Joint sensitivity certificate for reasons and decisions

**Strongest rejection.** For the proposed top-1 expected-attribution criterion,
infimum over ambiguity and finite minimum over competitors commute. Solving
against a shared density does not improve the exact pairwise criterion. The
worker's apparent gain changes ANY-competitor failure into ALL-competitor failure.

**Defensible difference sought.** A genuinely different coupled decision loss
could require joint inference. It has not been formulated here, and could not
silently replace the historical verified-reason target. Density-ratio ambiguity
also needs support and sensitivity assumptions; it is not automatically an
odds-ratio MNAR model. **NO-GO for the supplied mechanism**.

## 4. Learn interactions without complete training records

**Strongest rejection.** For binary signed coordinates, monomials satisfy
chi_A * chi_B = chi_(A symmetric-difference B). Expanding squared loss first
can reduce measured raw-coordinate requirements relative to naive feature
lifting. But generic algebraic simplification plus Horvitz–Thompson moment
estimation obtains exactly the same estimator. That is the matched baseline.

**Defensible difference sought.** Unknown interaction support, dependence-aware
measurement design or a strictly sharper budget/sample-complexity result could
matter. None is derived. A star of pairwise interactions is only an illustrative
special case, not evidence that continuous financial models need fewer queries.
**NO-GO for the proposed identity-only algorithm**; no implementation added.

## Discarded diagnostic, not a fifth method

The worker's information-theoretic coverage certificate lacks needed target
entropy assumptions and fails on a constant reason. Its XOR attribution example
is also incorrect. Valid impossibility theory can be valuable, but an invalid
inequality cannot be repaired by citing Fano or attribution-impossibility work.
See the [executed checks](ROUND7_FALSIFICATION_RESULTS.md).

**Selection outcome:** no construction currently passes technical difference,
necessity, evidence and nontriviality together. This is an exact-reduction /
counterexample decision, not a demand that every ingredient be unprecedented.
Search absence is not proof of novelty; a broad prior-art label is not proof
that all future combinations are equivalent.
