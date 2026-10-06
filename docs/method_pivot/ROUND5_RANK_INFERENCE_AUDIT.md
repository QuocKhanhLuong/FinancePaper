# Round 5 — direct attribution-rank event inference

Review date: **2026-10-06**. Actual starting HEAD: `1859eff6cdf163edc917351f203bee305e02ae30`,
branch `research/method-pivot`, initially clean. This is the next action declared
in ROUND4_NEXT_DECISION.md. Historical reports and event definitions stay frozen.

## Question and present decision

Can integrating a rank-event function directly avoid repeated full attributions
with a necessary new algorithmic advantage? **NO-GO for a new-method claim from
generic contrast projection plus weighted event integration.** A synthetic
reduction/correctness audit is permitted; no new solver or financial model is
authorized by this decision. It does not prove all specialized algorithms useless.
Independent AGY review is pending at this preregistration checkpoint.

## Closest primary sources, checked this session

| Work / status | Exact overlap | Distinction and rejection boundary | Primary source / inspection |
|---|---|---|---|
| Chabrier, Crombach, Peignier & Rigotti, TopShap, IEEE Access 2024 | Avoids computing all SHAP values by pruning top-k candidates using sampling bounds | Fixed-instance attribution estimation; not the probability of future verification changing the reason set. Merely avoiding full vectors is already known | [Institutional author record](https://liris.cnrs.fr/en/member-page/lisa-chabrier), DOI [10.1109/ACCESS.2024.3489958](https://doi.org/10.1109/ACCESS.2024.3489958), [deposited manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12333390/). Indexed manuscript algorithm passages inspected; direct PMC/HAL retrieval challenged/failed, not a claim of full direct-PDF reading |
| Goldwasser & Hooker, Statistical Significance of Feature Importance Rankings, UAI 2025 | Confidence-controlled ranking/set identification with adaptive sampling | Numerical sampling randomness differs from hidden-input randomness. Their guarantees cannot simply be transferred to our completion law | [Official proceedings](https://proceedings.mlr.press/v286/goldwasser25a.html), [author preprint](https://arxiv.org/abs/2401.15800). Metadata/abstract and indexed algorithm passages; direct official PDF retrieval failed |
| Morettin, Passerini & Sebastiani, Probabilistic ML Verification via Weighted Model Integration, 2024 preprint v2 | Logical/numeric events of ML functions integrated under a supplied input distribution; bound propagation and partitioning | A compiled attribution function and a top-k event fit this representation. Encoding alone is no new inference method | [Primary full text](https://arxiv.org/html/2402.04892v2), sections III, IV, VI and Algorithm 1 inspected. Peer-reviewed successor not verified |
| Boetius, Leue & Sutter, Solving Probabilistic Verification Problems of Neural Networks using Branch and Bound, ICML 2025 | Refinines probability bounds by partitioning uncertain-input regions | Their NN-specific implementation is not an off-the-shelf tree-SHAP solver; the probability-bounding pattern itself is known | [Official proceedings](https://proceedings.mlr.press/v267/boetius25a.html), [full manuscript](https://arxiv.org/html/2405.17556v3), sections 3–5 / Appendix C inspected |
| Devos, Meert & Davis, Veritas, ICML 2021 | Generic constrained tree-ensemble verification | Deterministic extremum/witness is not probability mass; a required structural comparator, not proof that probability computation is cheap | [Official proceedings](https://proceedings.mlr.press/v139/devos21a.html); source formulation already reviewed in round 4 |
| Arenas, Barceló, Bertossi & Monet, JMLR 24 (2023) | Tractability and hardness of SHAP and pairwise comparison for specified Boolean model classes | Do not transfer monotone-DNF hardness to every shallow numeric tree workload. It warns against claiming rank queries are generically easy | [Original paper](https://jmlr.org/papers/volume24/21-0389/21-0389.pdf), introduction/results scope and comparison problem inspected |
| Wettenstein, Nadel & Boker, WOODELF-HD, 2026 preprint | Fast compiled Background SHAP point queries | A new integrator must beat modern compiled point evaluation, not only stock SHAP. Prior round's Fourier moment control already beat our moment operator | [Author preprint](https://arxiv.org/abs/2604.10569); metadata/abstract refreshed; implementation evidence remains historical |

No outside implementation was installed, copied or run in this round. Code reuse
licenses were not audited here; availability of a paper is not a code license.
No exact existing financial verification target was established by this bounded
search, but absence of a search hit is not a novelty proof.

## Exact target and reduction

Let W=(O,x_O), q_W be a **supplied conditional completion law**, and B the frozen
explainer background, distinct from q_W. Original-feature interventional SHAP
has the existing compiled form phi(z)=sum_r c_r I_r(z). Grouping over initially
observed members gives a_g(z)=sum_r a_rg I_r(z). Fix displayed candidates C from
the current explanation once, |C|=k. Let A be all groups with an initially
observed member, including competitors that were not originally displayed.
The frozen event is

```
E(z) = OR over g in C:
       [a_g(z) <= 1e-6]
       OR [sum_(h in A, h != g) 1{a_h(z)-a_g(z) > 1e-6} >= k].
R(W) = integral 1{E(z)} q_W(dz).
```

The .01 threshold selects current positive candidates only; it is not silently
applied to the post-verification sign test. This preserves revision_arrays(k).
Empty candidate support is ineligible, not a zero-risk explanation.

Each rectangle indicator is an if-then-else of axis-aligned inequalities. Signed
group contributions, contrasts, cardinality constraints and their disjunction
are a Boolean/algebraic formula. Integrating it under q is WMC for finite support
or WMI for an encodable density. For continuous observed values, condition to
obtain q_W first; do not divide two zero-mass equality events. For a donor law,
R is simply sum_l w_l E(z_l). For a product law, multiplying cell masses is valid
only under that conditional product assumption.

This is a direct **application/reduction**, not a new theorem. It neither says a
generic solver is equally efficient nor establishes polynomial complexity. The
joint rank event is a nonlinear threshold of sums: first/second moments alone
are insufficient without additional distributional assumptions.

## Proposed shortcuts and strongest rejection

**Contrast projection.** Define d_r,hg=a_rh-a_rg before bounding, so shared terms
cancel. This is linear projection/affine-expression simplification. The same
coefficient matrix lets a generic verifier exploit identical cancellation.
Independent intervals for each group can be looser, but that weak comparator is
not sufficient for a new-method claim.

**Probability branch-and-bound.** Partition completion space into proven-event,
proven-non-event and unresolved cells, accumulating q-mass. Lower probability is
proven-event mass; upper adds unresolved mass. This is a known probabilistic
verification pattern. It needs sound numeric bounds and valid region masses;
ordinary floating-point interval code is not automatically a formal certificate.

**Moment/Gaussian shortcut.** Approximating the contrast law as Gaussian from
exact moments adds an assumption; it is not exact event inference. The following
preregistered example directly falsifies moment-only identification.

**Defensible difference still missing.** An attribution-specific representation,
admissible bound or query-sharing algorithm must give a measurable cost/error
advantage over the same projected expression in generic WMC/WMI and compiled
point evaluation. No such new step is derived in the present construction.
Even fewer output coordinates is not automatic: k candidates and G available
groups can require kG sign/contrast channels, versus G grouped contributions.

## Preregistered synthetic falsifier — before code

No financial outcomes or new datasets. Build an exact piecewise tree margin

```
f(x1,x2,x3,h) = b(x1)*(2+s(h)) + b(x2)*(2-s(h)) + 2.125*b(x3)
b(x)=1{x>=.5}
```

Here s takes values [-1,-.5,0,1,2] on intervals cut at [-.75,-.25,.5,1.5].
Background B is the single all-zero row. Observe x1=x2=x3=1, hide h; impute h=-1
for the fixed current candidate set. Three singleton observed groups, k=2,
original .01 candidate threshold and 1e-6 rank/sign tolerances. A fourth hidden
player participates in SHAP but cannot be a displayed observed reason.

Compare two declared laws: qA(h=-1)=qA(h=1)=.5; qB(h=-.5)=.8, qB(h=2)=.2.
Both have E[h]=0 and Var(h)=1. The analytic expected attributions are
phi=(2+h/2, 2-h/2, 2.125, 0), derived from the fixed point-reference game.
Thus their full attribution mean/covariance agree but rank-event probabilities
need not agree. Values expected by algebra are hypotheses checked by independent
coalition enumeration, the existing tree compiler and frozen revision evaluator.

Also check an additive control (no h interaction), degenerate point-mass law,
rank ties, sign changes and an irrelevant hidden placeholder. Tests must verify
that competitors need not belong to C. Permutations of finite-support rows must
not change the answer. Candidate/weight corruption must fail explicitly.

Implementation scope: an independent **baseline/falsification harness**, not a
new rank solver. Compare compiled direct contrasts with full grouped attribution
evaluation on identical atoms; record exact numerical discrepancies. Seeds
101/102/103 may generate only fixed, bounded nonnegative finite-law weights for
additional identity checks, without tuning or selecting cases. No speedup claim
or runtime gate; no large grid, learning, WMI package or financial confirmation.
Budget <=60 CPU seconds, one thread. Save source/config/commit hashes and actual
runtime under ignored outputs. Commit this specification before the harness.

GO for further method implementation requires a new necessary mechanism beyond
the reductions above; oracle correctness alone does not pass that gate. If only
generic projection/integration survives, close this specific method claim at
NO-GO and report the useful baseline/counterexample without renaming it.
