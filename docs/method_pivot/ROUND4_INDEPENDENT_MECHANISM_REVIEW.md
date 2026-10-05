# Round 4: independent proposals after coordinator review

Review date: **2026-10-05**. Starting commit `9f047cd`, branch
`research/method-pivot`. Two AGY proposals were reviewed against primary sources.
The first worker draft was **not accepted**: it overstated several reductions,
misstated metadata, and included unmeasured runtime estimates as facts. This
corrected version supersedes that uncommitted draft, preserved locally under
`outputs/method_pivot/round4/review/`. No candidate method was implemented.

## The precise gap

For frozen predictor f, observed state W=(S,x_S), completion law q and a fixed
attribution map phi, distinguish the conditional laws of f(X) and phi(X).
Attributions may vary even on a constant-prediction fiber. The exact synthetic
counterexample is in [the executed sanity report](ROUND4_BASELINE_SANITY_RESULTS.md).
It is not a new theorem, financial prevalence estimate, or proof every trained
prediction-only detector must be weak.

For a fixed-reference **efficient full attribution vector**, sum(phi)=f_margin-b,
so the prediction is reconstructible from phi after the known link. Therefore
sigma(f(X)) is contained in sigma(phi(X)); the rejected draft's assertion of
neither inclusion was wrong under this convention. Observed-only subsets and
ranks generally lose this reconstruction property. At a regular point of a
smooth scalar function, a level set has local dimension d-1; this is not true
without the nonzero-gradient/regularity qualification and does not describe a
tree step function's differential geometry. Attribution variation along such a
fiber is possible, not inevitable.

## Candidate 1 — learn the conditional attribution distribution directly

Input W; training target Phi=phi(X) from complete permitted training rows and a
frozen predictor/explainer. A conditional density model minimizes

```text
E_(X,S) [-log p_theta(phi(X) | X_S,S)].
```

At inference, sample attribution vectors from this density rather than completing
financial fields and repeatedly calling SHAP. A signed vector needs an appropriate
support; normalizing positive shares into a Dirichlet changes the target and
requires handling the no-positive case. No such replacement is frozen here.

| Closest work | Status / official source | Exact overlap | Difference and assumptions |
|---|---|---|---|
| Burnett et al., Missing Value Uncertainty | TMLR, August 2026; [official listing](https://jmlr.org/tmlr/papers/), [paper](https://openreview.net/pdf?id=BRWTS5e03Z) | Direct conditional distribution learning of complete-input model probabilities from partial input; Dirichlet NLL implementation | Prediction probability is not a signed attribution vector; complete training supervision and a suitable observation law are required. Official code reuse license remains unresolved in our source audit |
| Jethani, Sudarshan, Covert, Lee & Ranganath, FastSHAP | ICLR **2022**; [official paper](https://openreview.net/pdf?id=Zq2G_VTV53T), [author preprint](https://arxiv.org/abs/2107.07436) | Amortizes Shapley estimation using a learned explainer and Shapley regression | Estimates attribution values; it does not by itself learn a future-verification conditional attribution distribution. No official experiment was reproduced |
| Yamaguchi & Nishida, Explanation Bottleneck Models | [author manuscript v2](https://arxiv.org/html/2409.17663v2), December 2024, sections 2.2–2.3 inspected | Distills a teacher's explanation distribution into a text decoder while learning the task | Text generation, not missing financial fields or verified SHAP vectors. Its mode-focused sequence-distillation approximation is not a theorem about all NLL training |

**Strongest rejection.** The proposal currently only substitutes another frozen
response into ordinary conditional density estimation. Amortization and
teacher-generated explanation supervision are established. It contains no
specified representation, objective correction, or inference operation that
improves over matched conditional density baselines. This is a reduction of the
*present algorithm*, not a claim that a new target can never motivate a method.

**Correction to the draft.** Expected NLL on stochastic targets satisfies
cross-entropy = H(q)+KL(q||p_theta). With sufficient family, data and optimization,
it can recover q, including entropy. Mode-only distillation may discard variation;
it does not follow that every NLL estimator does so. One observed complete row
per partial input is not intrinsically invalid supervised conditional-density
training; it does require statistical sharing and appropriate assumptions.

**Defensible difference still needed.** An actual algorithm that exploits signed
attribution structure or verification sampling to improve accuracy/cost over a
matched generic density estimator, with a nontrivial reduction and measured
benefit. None is supplied yet. **NO-GO for this construction as a new method**.
A two-support synthetic oracle is feasible, but candidate training/timing is
**NOT RUN**. Under a changed hidden-value law, a confidently wrong learned density
would falsify robustness unless an explicit adaptation assumption/mechanism exists.

## Candidate 2 — search for attribution changes while prediction stays stable

For a declared feasible completion region C(W), current prediction p0 and fixed
prediction tolerance delta, search

```text
max_z D(reasons(phi(x_S,z)), reasons_current)
subject to z in C(W), |f(x_S,z)-p0| <= delta.
```

This can find witnesses for the historical phenomenon without estimating their
probability. A certificate would concern the specified region **and prediction
slice**, not all plausible future observations. Empty feasible regions require
an explicit infeasibility status, never a reliability claim.

| Closest work | Status / official source | Exact overlap | Difference and assumptions |
|---|---|---|---|
| Devos, Meert & Davis, Versatile Verification of Tree Ensembles | ICML 2021; [proceedings](https://proceedings.mlr.press/v139/devos21a.html), official PDF section 3 inspected | Constrained tree-ensemble optimization with anytime bounds | A top-k attribution discrepancy must first be represented by compiled attribution functions/rank inequalities; that encoding can be expensive. Generic representability is not evidence of an equally efficient off-the-shelf query |
| Dombrowski et al., Explanations can be manipulated and geometry is to blame | NeurIPS 2019; [official paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/bb836c01cdc9120a9c984c525e4b1a4a-Paper.pdf) | Explanation variation while predictions remain nearly unchanged, analyzed through geometry | Perturbations of differentiable models differ from verification of missing coordinates in a tree model |
| Slack, Hilgard, Jia, Singh & Lakkaraju, Fooling LIME and SHAP | AIES 2020; [DOI](https://doi.org/10.1145/3375627.3375830), [author preprint](https://arxiv.org/abs/1911.02508) | Shows post-hoc explanations can be misleading while in-distribution predictions are retained | Constructs an adversarial model wrapper exploiting explainer queries; it is **not** the same fixed-model, input-level constrained optimization problem |

**Strongest rejection.** With no new search/pruning/representation step, this is
a verification task for known solver families. Pairwise attribution differences
can be represented using the finite step functions already studied in round 2;
rank events require additional constraints. This establishes a required baseline,
not a polynomial-time reduction or equal-cost claim.

**Defensible difference still needed.** A demonstrated structural reduction in
attribution-event search complexity or an admissible bound that makes previously
intractable queries practical against the same-region generic verifier. None has
been derived. **NO-GO for the current solver-free proposal**.

**Falsifier and limits.** A broad completion box may include implausible points and
make useful certificates rare; this is a risk to test, not a measured fact that
coverage is always zero. The synthetic example has rank difference h-3/4 and a
flip boundary at 3/4, derivable by hand. No solver/runtime was benchmarked.

## Decision and evidence boundary

Neither proposal currently passes the method gate. This says nothing conclusive
about all conditional learning or all verification algorithms. Components being
old alone is not a rejection: the issue is that these **specific** proposals add
no demonstrated necessary mechanism beyond changing the target/query.

The next factual gate is the same-completion prediction-distribution comparison.
A conditional prediction law can correlate with revision and a learned detector
can infer its baseline frequency; the synthetic fiber example does not establish
that MVU/DMV must perform poorly empirically. Historical claims remain restricted
to controls actually run.

**VERIFIED:** coordinator checked the mathematical corrections, primary metadata,
XBM objective/distillation sections, Veritas formulation, and the executed
synthetic control. **REPORTED:** historical financial and round-two results.
**NOT RUN:** either candidate, official DMV, new confirmation. **ASSUMED:** fixed
attribution convention, completion support/law, adequate training coverage.
No claim of causal validity, complete missing-value recovery or sufficient
method novelty is supported. Continued targeted research remains possible;
this bounded review is not an impossibility proof or a reason to invent novelty.
