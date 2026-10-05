# Second-round mechanism search — 2026-10-05

Read base: `da7a0f1b3cad815b6c57c0f2a2c6287819be7fda`, branch
`research/method-pivot`. This continues the user's request after the first-round
NO-GO. It does not overwrite that decision or reinterpret unrun methods as failed
experiments. No new external dataset is acquired. The previously inspected Taiwan
and Polish cohorts cannot supply independent confirmation.

## Scope of this search

The first-round A–D mechanisms remain in [CANDIDATE_DIRECTIONS.md](CANDIDATE_DIRECTIONS.md).
This round asks whether a concrete estimator or inference algorithm can address
their identified limitation. Discovery searches and relevant primary method
sections were inspected on 2026-10-05. This is a bounded novelty audit, not proof
of absence of prior art. Two independent research reviewers were asked to try
to reduce the proposals to known methods. Their judgments are not peer review.

| Construction | Closest reduction | Decision at formulation stage |
|---|---|---|
| Verify some records, tilt a completion law while preserving calibrated prediction moments | Calibration weighting / imputation-powered inference | NO-GO as a new algorithm without an additional identified mechanism |
| Correct release-policy risk using inverse-propensity verification residuals | Augmented inverse-probability estimation + finite-policy risk control | NO-GO as a new algorithm |
| Bound explanation changes invisible to prediction moments | Linear partial identification over a probability simplex | Useful diagnostic; NO-GO as a method by itself |
| Learn the conditional joint distribution of predictions and attributions | Conditional response distillation with an efficiency constraint | NO-GO as currently specified; fidelity alone does not ensure realizability |
| Compile a shallow tree ensemble's attribution function, then integrate its conditional joint moments | Path decomposition, grid attribution functions, tractable moment inference | Narrow computational candidate undergoing independent reduction/cost checks |

## Nearest primary sources inspected

F denotes inspection of relevant full-text passages; A denotes abstract or
official metadata only. No external code was installed; code licenses not
checked here are NR, not assumed permissive.

| Work | Status and inspection | Exact overlap and remaining distinction | Official source |
|---|---|---|---|
| Gögl, Xing & Yau, Martingale-Consistent Self-Supervised Learning | May 2026 preprint v1, F §§3.3–4 | Conditional updating objective and independent-refinement estimator already exist; wrong refinement laws remain a limitation | [arXiv](https://arxiv.org/html/2605.11846v1) |
| Shin, Gail & Pfeiffer, risk estimation with missing covariates | Biostatistics 23(3), 2022; reviewer method inspection | Calibration of phase-two weights using auxiliary risk information precedes a risk-preserving audit tilt | [Publisher](https://academic.oup.com/biostatistics/article/23/3/875/6146183) |
| Kluger et al., prediction-powered inference with imputed covariates | 2025 preprint, A | Imputation and nonuniform sampling corrections are already inference tools | [arXiv](https://arxiv.org/abs/2501.18577) |
| Zhao & Candès, Imputation-Powered Inference | September 2025 preprint v1, F setup | Complete-case correction of black-box imputation for downstream estimands; not an identification theorem for each customer's hidden truth | [arXiv](https://arxiv.org/html/2509.13778v1) |
| Duan & Pelger, Imputation-Powered Inference for Missing Covariates | NBER working paper 34535, December 2025, A | Bias-corrected inference under heterogeneous observation patterns; distinct paper from Zhao–Candès | [Official PDF](https://www.nber.org/papers/w34535.pdf) |
| Li, Zrnic & Candès, Robust Sampling for Active Statistical Inference | NeurIPS 2025, reviewer proceedings verification | Robust audit-sampling design already protects against inaccurate uncertainty-guided sampling | [Proceedings PDF](https://papers.nips.cc/paper_files/paper/2025/file/6389470564214983604d1ac81631c2c5-Paper-Conference.pdf) |
| Donnelly et al., Rashomon Importance Distribution | NeurIPS 2023, A | Importance uncertainty across similarly predictive models; our completion-law object freezes the model, but that change alone is insufficient | [Proceedings](https://papers.nips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html) |
| Laberge & Pequignot, Understanding Interventional TreeSHAP | September 2022 preprint v1, F | Dummy reduction and leaf-path games supply the algebraic starting point; they are not our invention | [Full text](https://arxiv.org/html/2209.15123v1) |
| Filom et al., On marginal feature attributions of tree-based models | arXiv v4 May 2024, F §§3.2, 3.4, Appendix G | Attribution is constant on a refined split grid; lookup compilation for oblivious trees is established. A non-grid, path-depth-bounded joint-moment compiler would need a distinct cost advantage | [Full text](https://arxiv.org/html/2302.08434v4) |
| Khosravi et al., On Tractable Computation of Expected Predictions | NeurIPS 2019, F §4 | Exact moments for compatible circuits are known; arbitrary incompatible structures need not be tractable | [Proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/fccc64972a9468a11f125cadb090e89e-Paper.pdf) |
| Ancona et al., DASP | ICML 2019, A | Approximate Shapley computation by moment propagation; different randomness from a fixed explainer evaluated at uncertain inputs | [Proceedings](https://proceedings.mlr.press/v97/ancona19a.html) |
| Bley et al., Explaining predictive uncertainty by exposing second-order effects | Pattern Recognition 160, 2025, A | Covariance of attributions explains ensemble uncertainty; covariance as an explanation object is already known | [DOI](https://doi.org/10.1016/j.patcog.2024.111171) |
| UbiQTree | Patterns, online February 2026, F Algorithms 1–5 | Model/tree resampling and SHAP uncertainty decomposition; not exact integration of a fixed model over hidden input values | [DOI](https://doi.org/10.1016/j.patter.2025.101454) |
| Jia & Pei, Shapley Value on Uncertain Data | January 2026 preprint; institutional record lists TKDE DOI, A | Moments of random data-owner Shapley contributions. Data valuation differs from local feature attribution; stochastic Shapley moments are not new | [arXiv](https://arxiv.org/abs/2601.14543), [institutional publication record](https://scholars.duke.edu/publication/1916971) |
| Nadel & Wettenstein, WOODELF | AAAI 2026, F §§4–9 | Compiles decision-path patterns and background statistics for fast exact point attributions. Directly excludes claiming precompilation as our novelty | [Proceedings/DOI](https://doi.org/10.1609/aaai.v40i29.39630), [author code, MIT](https://github.com/ron-wettenstein/woodelf) |
| Wettenstein, Nadel & Boker, WOODELF-HD | April 2026 preprint v1, F §§1–4 | Repeated-feature merging and depth-dependent precomputation are already optimized. Must be a strong numerical comparator, not omitted in favor of stock SHAP | [Full text](https://arxiv.org/html/2604.10569v1) |
| Gorji, Amrollahi & Krause, FourierSHAP | NeurIPS 2025, F abstract and representation section | Sparse exact tree representation and amortized attribution already exist. Changing basis alone does not justify a new method | [Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/1b331c20064e37e204a5bcd12481bfac-Abstract-Conference.html) |
| Wan et al., Conditional Evidence Reconstruction and Decomposition | April 2026 preprint, A | Incomplete-modality representations and logit evidence decomposition already coexist outside credit; a financial application does not establish new evidence-decomposition machinery | [arXiv](https://arxiv.org/abs/2604.17030) |

## Strongest rejections of the nonselected constructions

**Audit correction.** A weighted verification residual is an AIPW/PPI construction.
Constraining a density tilt to preserve risk moments does not identify missing
outcome-relevant directions; it can preserve a biased risk estimate. No new
estimator follows just from naming the moments “explanation evidence.”

**Prediction-invisible directions.** With finite completion weights q, moments
Aq=b and a reason functional c'q, identification in an interior feasible set is
equivalent to c belonging to the row span of A augmented by normalization.
At boundaries, active support constraints matter. Minimizing/maximizing c'q is
standard linear partial identification. Keeping the *entire* prediction law
fixed permits redistribution only inside prediction fibers, not arbitrary
changes of distinct prediction values.

**Response-space learning.** Learning P((prediction, attribution)|observed input)
avoids reconstructing every hidden feature but is conditional density estimation.
An attribution-sum constraint does not ensure that a generated vector corresponds
to any feasible completion of the frozen predictor. No revision-head experiment
is rerun under this name.

## Candidate requiring a real computational test

For an arbitrary, not necessarily oblivious, numeric decision tree, write a leaf
contribution as v times a product of path-interval indicators. Collapse repeated
splits of the same feature into one interval. Dummy reduction limits each local
game to the distinct features of that path. Compile its attribution as a sum of
weighted rectangle indicators, rather than enumerate the global split grid.
After substituting observed inputs, merge identical residual rectangles and
integrate their intersections against a declared completion law.

The possible contribution is the **compiled representation and query algorithm
for joint attribution moments**, if it is correct and useful at matched numerical
accuracy/cost. Neither linearity, rectangle integration, covariance, SHAP nor
Monte Carlo uncertainty is a novel component. There is no new financial predictor
or release threshold in this candidate.

An important rejected shortcut: attribution is **not generally constant within an
original prediction leaf**. Off-path features affect coalition evaluations.
Original-leaf masses therefore do not suffice for exact attribution moments.
Filom et al.'s refined grid, not the original tree partition, has the relevant
constancy property. An independent review initially suggested that shortcut;
the counterexample is being included in the mathematical correctness gate.

Scope limits are decisive: exact moments under q do not make q correct; first and
second moments do not identify top-k or sign-event probabilities; raw-logit
moments do not equal probability moments after a sigmoid. The method must expose
these limits, not turn a numerical certificate into a reliability guarantee.

The cost can be quadratic in the number of residual rectangles. A result that is
exact but slower than direct enumeration or budget-matched TreeSHAP without a
useful accuracy advantage fails the practical mechanism gate. Independent review
and a prospective bounded numerical pilot must precede financial development.
