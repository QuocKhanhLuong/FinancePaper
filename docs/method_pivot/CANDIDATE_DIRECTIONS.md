# Four method directions: nearest-work audit

Searched **2026-10-04**. Targeted primary-source review, not a systematic review
or proof that no future technical contribution exists. The decision concerns the
specific constructions below. Old components are allowed; a combination needs
a consequential mechanism that survives reduction to the closest existing method.

Evidence notation: **F** = relevant full-text method/assumption sections inspected;
**A** = official abstract/metadata only. NR means not verified, not absent.
No external implementation was installed or benchmarked. Preprints are not
silently promoted to peer-reviewed papers. Source-specific claims below concern
their methods, not reproduction of their reported performance.

## A — Faithful observed/inferred evidence decomposition

Candidate: a predictor exposes additive terms whose inputs are all observed,
and conditional expectations of terms needing hidden fields. A fixed centering
law supplies a reproducible convention; the displayed parts reconstruct the
declared score. This is distinct from attaching uncertainty to SHAP afterward.

| Paper | Status/year | Exact problem | Method | Assumptions | Closest overlap | What is still missing | Proposed difference | Official source | Code/license if relevant |
|---|---|---|---|---|---|---|---|---|---|
| Lengerich et al., Purifying Interaction Effects | AISTATS 2020, F | Main/interaction attribution is non-unique | Functional-ANOVA purification | Specified reference distribution; piecewise-constant functions for exact algorithm | Canonical decomposition without changing prediction | Does not itself identify a hidden financial value | Tag terms by observed support | [Paper](https://proceedings.mlr.press/v108/lengerich20a/lengerich20a.pdf) | Not reused; license NR |
| Van Ness & Udell, dnamite package | 2025 preprint, F; related DNAMite survival paper PMLR 259 (2025) | Additive prediction with missing values and interactions | Discretized shape functions; separate missing bins | Fitted bins/shape functions; survival censoring assumptions are task-specific | Already separates observed/missing feature importance | Missing-bin effect is not conditional hidden-value evidence | Integrate interaction terms, rather than label a bin | [Package paper](https://arxiv.org/html/2503.07642v1), [conference record](https://proceedings.mlr.press/v259/van-ness25a.html) | [Author code](https://github.com/udellgroup/dnamite), Apache-2.0 |
| Köhler et al., functional decomposition into predictor effects | npj AI 2025, A + method description | Interpreting a fixed complex predictor | Main/interaction surrogate decomposition and constraints | Approximation and distribution conventions | Faithful decomposition with controlled interactions is established | Does not provide our partial-observation contract | Make availability explicit in term evaluation | [Publisher](https://www.nature.com/articles/s44387-025-00033-7) | Not reused; license NR |
| Alvarez-Melis & Jaakkola, SENN | NeurIPS 2018, F abstract/model structure | Intrinsically interpretable nonlinear prediction | Concepts, coefficients, faithfulness/stability regularization | Concept and coefficient parameterization | Explanation belongs to the computation | Does not identify observed versus inferred information | Availability-constrained concepts | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html) | Not reused; license NR |

**Strongest rejection.** Once a centered expansion `f = sum_A f_A` and conditional
law q are fixed, splitting terms by `A subset S` is bookkeeping plus conditional
integration. A residual network without a constraint can absorb arbitrary
functions of observed inputs; reconstruction alone does not identify the parts.
Adding a mask to an additive model is especially close to the DNAMite precedent.

**Defensible difference sought.** A reference-explicit decomposition whose
inferred component has an independently estimable error, retains necessary
interactions and improves finite-sample risk could matter. But no extra estimator,
constraint or bound beyond centering and conditional expectation is presently
specified. Calling a term “observed” is not a causal statement. **NO-GO for this
method claim**, not for decompositions as a useful interface.

## B — Coherent updating under progressive verification

Candidate: learn predictions indexed by observed sets, requiring an earlier
prediction to agree with the conditional average of later predictions, while
allowing large updates on individual customers.

| Paper | Status/year | Exact problem | Method | Assumptions | Closest overlap | What is still missing | Proposed difference | Official source | Code/license if relevant |
|---|---|---|---|---|---|---|---|---|---|
| Ivanov, Figurnov & Vetrov, VAEAC | ICLR 2019, F | Arbitrary observed subsets | Conditional latent model and stochastic completions | Training/mask support; fitted conditional model | Shared inference across many masks | Finite fitted conditionals need not be jointly compatible | Explicit nested-set consistency | [Paper](https://openreview.net/pdf?id=SyxtJh0qYm) | [Author code](https://github.com/tigvarts/vaeac), MIT; old runtime not tested |
| Li, Akbar & Oliva, ACFlow | ICML 2020, F | Arbitrary conditional likelihoods | Mask-conditioned invertible transformations | Conditional density model, support and training fit | Conditional integration need not use independent models for every subset | Approximation can leave coherence error | Outcome-focused coherence | [Proceedings](https://proceedings.mlr.press/v119/li20a.html) | Proceedings links software; license NR |
| Sudak & Tschiatschek, Posterior Consistency for Missing Data in VAEs | ECML PKDD 2023, F | Inconsistent partial-input posterior families | Nested-set posterior regularizer | Generative/variational families; MCAR/MAR focus | Consistency across information sets is already a training objective | Targets latent posteriors rather than financial outcome probability | Prediction-space objective | [Official paper](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2023/research/lncs14170515.pdf), [DOI](https://doi.org/10.1007/978-3-031-43415-0_30) | [Author code](https://github.com/stschia/VAE-posterior-consistency), license NR |
| Gögl, Xing & Yau, Martingale-Consistent Self-Supervised Learning | **Preprint**, 12 May 2026 v1, F | Coarse/refined predictions under changing information | Prediction/latent martingale penalty; independent two-refinement estimator | Refinement sampler quality; finite moments; conditional independence | Direct overlap with the proposed objective and inexpensive unbiased estimator | Misspecified refinement distributions remain a limitation | Need a new correction for that limitation | [Full text, §§3.3–4](https://arxiv.org/html/2605.11846v1) | Public code/license not verified; not run |

**Strongest rejection.** Predictive tower consistency, including the two-sample
way to avoid penalizing legitimate update variance, is explicit prior art—not
merely an old probability identity rediscovered in our paper. Relabeling SSL as
credit verification does not change that mechanism.

**Defensible difference sought.** Correcting coherence against the *actual*
revelation law despite a biased imputer could be substantive. The current
construction is coherent only under its chosen q. No identified correction is
provided. A frozen coherent generative predictor is an essential baseline.
**NO-GO**; this is a prior-art reduction, not an empirical loss by a trained model.

## C — Robustness to a changed observation mechanism

Candidate: retain useful mask information, but limit harm when its relationship
with latent financial values changes. This is not ordinary covariate shift.

| Paper | Status/year | Exact problem | Method | Assumptions | Closest overlap | What is still missing | Proposed difference | Official source | Code/license if relevant |
|---|---|---|---|---|---|---|---|---|---|
| Zhu et al., StableMiss | IJCAI 2023, F | Agnostic mask-distribution shift | Joint parameterization and decorrelation | Fixed full-data law; invariant conditional target justified under MCAR/MAR | Suppressing spurious mask/feature associations | Arbitrary nonignorable shifts are not covered | Adaptive retention of informative missingness | [Proceedings/DOI](https://www.ijcai.org/proceedings/2023/525) | Code/license not verified |
| Rockenschaub et al., Robust prediction under missingness shifts | 2024 preprint, F; no later venue verified | Ignorable vs nonignorable missingness shift | Bayes-predictor analysis; NeuMISE | Stable full-data relationship; shift assumptions explicit | Directly explains when mask-dependent predictors can transfer | Nonignorable change can alter the Bayes rule | Sensitivity-aware adaptation | [Full text](https://arxiv.org/html/2406.16484v1) | Reproducibility section describes code archive; license NR |
| Li et al., DRUM | **Preprint v2**, 5 Aug 2026, F | Covariates structurally unavailable in target population | Worst-case conditional distribution, energy-distance class, bias correction | Source outcome model and target uncertainty class | Robust conditional marginalization already proposed | Structural absence differs from arbitrary per-record missingness | Handle record-specific masks and observation evidence | [Current v2](https://arxiv.org/html/2605.24212v2) | No code reused; license NR |
| Xu & Jiang, Robust Contextual Optimization with Missing Covariates | ICML 2026, A | Decisions from incomplete contextual data | Observation-consistent ambiguity sets and tractable reformulations | Declared missingness/uncertainty structure | “DRO without single imputation” is already established | Not the same financial classification interface | A classifier-specific mechanism would be needed | [Official proceedings](https://proceedings.mlr.press/v306/xu26v.html) | License NR; no claim to have audited every proof |

**Strongest rejection.** A mask-invariance penalty is close to StableMiss and can
discard real observation-process signal. A generic minimax loss over conditional
completion distributions is already a robust-learning construction. Renaming its
radius “trust in missingness” does not specify how the radius is identified.

**Defensible difference sought.** Selectively adapting to trustworthy observation
signals with measured target verification data could differ from blind invariance.
That would require a defined verification sampling design, overlap, and a new
estimator or adaptation guarantee. Current synthetic masks do not establish those
deployment assumptions. **NO-GO for the proposed unrestricted claim**; no assertion
that all MNAR sensitivity or shift methods are useless.

## D — Deterministic partial-input inference and numerical error certificates

Candidate: integrate a frozen financial predictor over hidden values using its
structure, returning a prediction and numerical approximation bound instead of
merely sampling K completions. This changes the inference algorithm, not a reason
release threshold. It deliberately differs from A's interpretability, B's
training constraint and C's distributional robustness.

| Paper | Status/year | Exact problem | Method | Assumptions | Closest overlap | What is still missing | Proposed difference | Official source | Code/license if relevant |
|---|---|---|---|---|---|---|---|---|---|
| Khosravi et al., What to Expect of Classifiers? | IJCAI 2019, F | LR with missing features | Expected prediction; generative/discriminative compatibility | Specified feature law/model class | Integrating missing inputs is already an inference method | Not arbitrary boosted-tree probability integration | Tree-specific bounded computation | [DOI/proceedings](https://www.ijcai.org/proceedings/2019/377) | Not reused; license NR |
| Khosravi et al., On Tractable Computation of Expected Predictions | NeurIPS 2019, F | Expectations/moments of discriminative circuits | Compatible circuit recursion; classification approximation | Structural compatibility; sigmoid changes tractability | Exact moments and approximate classification | General classifier expectations can be hard | Avoid compatibility restrictions at useful bounded cost | [Paper, §4](https://proceedings.neurips.cc/paper_files/paper/2019/file/fccc64972a9468a11f125cadb090e89e-Paper.pdf) | Not reused; license NR |
| Khosravi et al., Handling Missing Data in Decision Trees | ICML Artemiss **workshop** 2020, F | Trees with incomplete inputs | Density-weighted expected predictions and expected-loss fitting | Tractable density queries | Direct overlap for XGBoost-style inference | Fitted density still may miss truth | Numerical certification rather than sampling alone | [Author paper](https://starai.cs.ucla.edu/papers/KhosraviArtemiss20.pdf) | Not reused; license NR |
| Vergari et al., Compositional Atlas of Tractable Circuit Operations | NeurIPS 2021, F | Composing tractable probabilistic operations | Algebraic operations with structural conditions/hardness analysis | Appropriate circuit properties | Reusing structured integration is known | No universal free integration for arbitrary models | A genuinely cheaper restricted algorithm would be needed | [Proceedings](https://papers.neurips.cc/paper_files/paper/2021/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html) | Not reused; license NR |
| Devos, Meert & Davis, Veritas | ICML 2021, A | Tree-ensemble property verification | Search with anytime lower/upper bounds | Feasible-region optimization | Bounds/refinement are not new by themselves | Optimizes extrema, not conditional expectation | Probability-weighted refinement | [Proceedings](https://proceedings.mlr.press/v139/devos21a.html) | License NR; no code copied |

**Strongest rejection.** Leaf-mass summation for expected logits is standard
integration. Taking a sigmoid afterward is generally wrong for expected
probability. Probability-weighted box bounds would be a reasonable adaptation,
but elementary interval integration plus known tree search is not yet a distinct
efficient algorithm with a demonstrated advantage.

**Defensible difference sought.** A structure-sensitive algorithm with a useful
error/cost advantage over circuit inference, adaptive quadrature and MC could
qualify even using old parts. It must separate numerical error under q from
error in q itself. Neither a new complexity result nor a measured bottleneck
advantage is presently established. **NO-GO for claiming a new solver now**;
this does not assert that such an algorithm cannot be invented.

## Search coverage and limits

Query families included: `missing functional ANOVA decomposition`, `observed
missing neural additive`, `posterior consistency missing data`, `martingale
partial observations`, `arbitrary conditioning`, `missingness mechanism shift`,
`agnostic mask distribution`, `structurally missing distributionally robust`,
`expected predictions missing features`, `tree ensembles anytime bounds`.
Backward checks followed the direct papers to generative inference and tree
verification rather than limiting search to credit applications.

**17 work rows** above, including separate proceedings/software records where
stated. Relevant preprints through August 2026 and ICML 2026 proceedings were
checked; later unseen papers may exist. Secondary search summaries were discovery
aids only. A preprint establishes disclosed overlap, not independently verified
empirical quality. A failed novelty gate needs a reduction of the *proposed
mechanism*, not proof that a cited paper solves the whole financial application.

## Comparison without an invented novelty score

| Direction | Remaining technical question | Simple/nearest falsifier | Trivial solution guard | Data/Mac feasibility | Current gate |
|---|---|---|---|---|---|
| A | Identify useful inferred evidence without arbitrary residual allocation | Centered GA2M + the same conditional law | Fidelity plus outcome loss; preserve interactions | Small statistical models feasible; no new data needed for oracle | No extra identified estimator |
| B | Actual-law coherence despite imputer error | Coherent generative predictor + existing martingale loss | Constant predictor and wrong-q coherence controls | Small finite worlds feasible; learned comparison possible later | Direct mechanism overlap |
| C | Decide which mask signal transfers | Augmentation, StableMiss/NeuMISE, conditional DRO | Worst-environment risk plus in-domain calibration | Synthetic shift feasible; real observation-process evidence missing | Adaptation rule/identification absent |
| D | Reduce integration cost with useful error bounds | Exact small-world enumeration, circuit integration, budget-matched MC | Bound width, probability error, density misspecification | Finite oracle cheap; general compile size unknown | No distinct efficient solver established |
