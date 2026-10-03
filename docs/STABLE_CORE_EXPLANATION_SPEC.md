# Verification-calibrated stable reason sets — feasibility specification

**Version note:** the user-requested [Stable-Core v2 study](STABLE_CORE_STUDY_PROTOCOL.md)
supersedes the initial two-candidate cap and adds a magnitude-based target, all
candidate reason sets and matched-size controls, before Freddie data access.
The original specification below is retained as design history. Historical
Taiwan/Polish revision definitions and results remain unchanged.

**GO for bounded testing, not a proven improvement or novelty claim.** Frozen
2026-10-03 before Freddie access. No new risk predictor, neural head or architecture.
Mortgage efficacy, calibration and completion-family comparisons remain NOT RUN.

## Prior work checked through 2026-10-03

Primary paper/proceedings pages were searched for robust/missing-data explanations,
distributional/interval SHAP, partial/selective attribution, stable explanations,
set-valued explanations and verification calibration. This is a targeted nearest-
work audit, not proof of exhaustive absence. DOI failures are not substituted with
invented publication metadata; preprints are labelled as such.

| Work / primary source | Established idea | Difference from our test |
|---|---|---|
| [Vo et al., Explainability of ML Models under Missing Data](https://arxiv.org/abs/2407.00411) | Missing-value treatment affects predictive and explanatory behavior | Does not make our frozen observed-group post-verification release event automatic novelty |
| [Golchian & Wright, 2025 preprint](https://arxiv.org/abs/2512.17689) | Multiple-imputation uncertainty for SHAP/PFI/PDP interval coverage | Excludes claiming imputation intervals as new; we evaluate released reasons against held-out true values |
| [Laberge et al., JMLR 2023](https://jmlr.org/papers/v24/23-0149.html) | Consensus attribution statements/partial orders over a Rashomon set | Very close conceptual precedent for retaining only stable statements; their uncertainty is across models, ours across completions of one frozen model |
| [Cifuentes et al., ECAI 2024, DOI 10.3233/FAIA240586](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240586) | SHAP scores under uncertainty about the population distribution | Distributional attribution uncertainty is known; sampled two-family agreement is not their formal uncertainty region |
| [Selective Explanations, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html) | Detect inaccurate amortized explanations and allocate explanation computation | Selective explanation is known; our target is revision of originally observed reasons after verification |
| [Löfström et al., Guarded Explanations, COPA 2026](https://proceedings.mlr.press/v329/lofstrom26a.html) | Filter unsupported perturbation-based rule conditions using empirical guard scores | Closest operational warning: a plausible-looking score is not a finite-sample risk guarantee; guard checks support, not future verification survival |
| [Johansson et al., COPA 2025](https://proceedings.mlr.press/v266/johansson25a.html) | SHAP explanations of conformal prediction sets | Set-valued predictions are not the same object as a subset of releasable reasons |
| [Goldwasser et al., Statistical Significance of Feature Importance Rankings](https://raw.githubusercontent.com/mlresearch/v286/main/assets/goldwasser25a/goldwasser25a.pdf) | Statistical verification of attribution rankings/top-k sets | Sampling error of rankings differs from change caused by unknown financial fields; both require explicit selection handling |
| [Xiang et al., RoSHAP, 2026 preprint](https://arxiv.org/abs/2605.15154) | Distributional stability/activity/strength criterion under data/model fitting variation | No generic stable-attribution or robust-ranking novelty; ours freezes predictor and evaluates actual hidden-value restoration |

Candidate contribution only: **verification-calibrated stable reason sets for
incomplete financial information**, evaluated against all-or-nothing release under
temporal holdout and completion-family disagreement. No directly identical procedure
was established by this search; this is not a first-ever claim. If rank instability
and a simple per-reason threshold perform equivalently, frame as evaluation/policy.

## Inputs and frozen estimand

For each loan i, `phi_current[N,G]`, `phi_completion[N,K,G]`, `group_hidden[N,G]`.
These already sum raw-logit original-field TreeSHAP over **originally observed**
members at every endpoint. Current and completion explanations share predictor
and64-row training reference. G follows FREDDIE_DATA_PROTOCOL; K=8.

Let C_i be **up to two** currently available groups with contribution >.01,
ordered by contribution then fixed schema order. Allow one candidate when only
one is eligible; whole-explanation baseline still requires exactly two. Report
both common-eligible and all-loan coverage to avoid an eligibility trick.

For each candidate g and completion b define:

```
positive_igb = phi_igb > 1e-6
top_igb = count(available h: phi_ihb > phi_igb + 1e-6) < 2
survives_igb = positive_igb AND top_igb
s_ig = mean_b positive_igb
r_ig = mean_b top_igb
j_ig = mean_b survives_igb
```

Log quantiles .1/.5/.9 of attribution, not a posterior interval. K8 frequencies
have increments1/8; never display invented98% confidence. The verified per-reason
label uses the same positive/top2 rule on the restoration endpoint, restricted to
the same original availability. Newly revealed groups do not become competitors.
It indicates model-attribution survival, not financial truth or causal correctness.

## Release and independent calibration

Release `S_i = {g in C_i: s_ig >= tau_sign AND r_ig >= tau_rank}`. Fixed threshold
grid for each coordinate:0,.5,.625,.75,.875,1 (36 pairs). The joint-survival fraction
is a separate simple score baseline; two marginal frequencies alone do not guarantee
their intersection probability. Calibrate actual verified failure, not a product of
those frequencies. Never tune the grid on assessment.

For each of alpha .05/.10/.15, select on2017 calibration only. Primary robust policy
must satisfy the risk criterion in each MCAR10/MCAR30/group environment. Prefer
maximum fraction of customers with >=1 reason, then number of released reasons,
then stricter lexicographic thresholds. Empirical pooled calibration is descriptive.
No satisfactory pair => empty sets, with undefined conditional precision/risk.

**Reason-level micro false-stable risk:** `sum_i failures_i / sum_i |S_i|`.
Reason outputs from one customer are dependent; do not apply a binomial interval
to independent reason rows. For a conservative sensitivity define, for each loan,
`D_i = (failures_i - alpha * |S_i|)/2`, in `[-alpha,1-alpha]`.
For n independent calibration loans in one environment, use
`mean(D) + sqrt(log(36 * E / .05)/(2*n)) <= 0` for every environment E.
This finite-grid Hoeffding union-bound test targets a population ratio of expected
counts under its independence/stationarity assumptions, not each customer's error.
Require nonzero release and unique customer IDs within each environment. Simulated
environments may share loans; union bound does not require independence between
environments. It does not protect against temporal population shift or unknown
same-borrower loans. Bootstrap over loans supplies descriptive assessment CIs.

Whole-release MC/rank policies target any-reason revision; stable-core targets
per-reason failure. They are **not interchangeable risks**. Also evaluate both with
the same loan-level `any released reason fails` endpoint and report reason counts.
No claim of dominating whole-release without that matched-risk comparison.

## Required metrics and comparisons

- Reason precision = surviving released reasons / released reasons; false-stable
  rate is its complement. Undefined when no reason released.
- Candidate-survivor recall = surviving released / all surviving current candidates.
  Report denominator explicitly; do not imply recall of all possible restored reasons.
- Mean released reasons per loan, fractions with0/1/2, customer coverage>=1,
  and customer-level any-failure risk conditional on at least one release.
- Feature/group revision, Region B and whole-explanation risk–coverage retained.
- Compare release-all eligible; prediction entropy/variance; completion rank/sign
  instability; MC K8; simple per-reason joint survival; stable-core thresholds.
- Same masks, background, calibration loans, timing and loan-cluster1000-draw CIs.

## Two completion families, separately evaluated

**A: historical neighbor/donor K8.** Training-only complete feature reference,
max2048 records, nearest32 by observed numerical IQR distance (floor1, clipped5)
and categorical mismatch; joint donor row sample with replacement. If <32 complete
training rows remain, fail this comparator explicitly, not silently change it.

**B: conditional forest K8.** Training features only; fixed max50k label-independent
training-row sample. For each field, fit16 ExtraTrees (regression for numerical,
classification for categorical), depth6/min_leaf40, one thread. Regressors use other
fields with training median fills and missing indicators; categorical fields are
one-hot encoded in the conditioning design. Field targets use only genuinely
observed training values. At inference, initialize missing fields by training
median/mode, then do two fixed-order conditional sweeps. Choose a tree uniformly
and sample an observed training target from its reached leaf. This yields a sampled
conditional imputation with valid observed univariate support, not a Bayesian
posterior or guaranteed joint financial distribution. No target y is accepted.
Natural missing fields may be latent internally during sweeps but are reset to NaN
in returned endpoints; only artificial cells are exposed as completions. Original
observed values never change. No assessment-based fit/refit or imputer selection.

**Intersection sensitivity:** take the coordinate-wise minimum s/r across A/B,
then independently calibrate the same36-pair policy on2017. Two families are not16
independent draws, nor a proven ambiguity set. Compare release disagreement,
verified failures conditional on disagreement, hidden-truth support and cost before
claiming completion-model robustness. If both share bias, agreement can still fail.

Implementing a conditional imputer is within the newly authorized phase; it does
not retroactively change the historical Taiwan/Polish K8 result. The completion
comparison and model training remain an unrun protocol until official intake passes.

## API and privacy boundary

Serving APIs accept current attributions, completion attributions, original mask,
and frozen thresholds only. Verification arrays/labels enter separate evaluation
and calibration calls. Tests change hidden truth and future loan rows without
affecting current features or release; assert shape/finite/unit/ID contracts.
No per-loan explanation, truth, ID, completion value, model weight or background row
is published. Export aggregate tables locally for manual governance review.

Stop rather than retune if stable sets are mostly empty, rank baseline suffices
with less work, or conclusions depend strongly on imputer family. A stronger model
is not a fallback. Domain meaning still needs expert validation.
