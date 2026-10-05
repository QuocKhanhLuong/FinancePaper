# Conditional attribution moments: candidate, not established novelty

Date: 2026-10-05. This formulation concerns a numerical inference primitive,
not a new predictor, completion model, or release threshold. Point-SHAP
compilation is explicitly prior art: [WOODELF](https://doi.org/10.1609/aaai.v40i29.39630),
[WOODELF-HD](https://arxiv.org/html/2604.10569v1),
[FourierSHAP](https://papers.nips.cc/paper_files/paper/2025/hash/1b331c20064e37e204a5bcd12481bfac-Abstract-Conference.html).
The question is whether an observation-specialized joint-moment query has a
useful, nontrivial computational advantage beyond these representations.

## Method card

**A. Question.** Can the joint mean and covariance of a frozen tree model's
interventional attributions over a declared completion law be computed exactly,
without enumerating its full support, at useful cost for shallow credit trees?

**B. Failure mode.** Finite completion sampling adds numerical uncertainty to
already uncertain inputs. Covariance matters because attributions can cancel in
their sum: small raw-score variance need not mean individually stable reasons.
This is a known motivation, not a new uncertainty decomposition. Historical
K-ablation results motivate measuring cost; they do not establish a new solver's
advantage. A wrong completion law remains wrong even when integrated exactly.

**C. Inputs.** Numeric tree ensemble f, immutable empirical explainer background B,
observed coordinates x_O, mask O, semantic aggregation matrix H, and a supplied
conditional law q(X_M|x_O). q must expose rectangle probabilities. It receives no
restored values or outcomes. The pilot supplies known finite/product laws; it
does not pretend to identify a deployment law from incomplete data.

**D. Outputs.** E_q[H phi_f(X;B)] and Cov_q[H phi_f(X;B)], residual-term count,
runtime and memory. Raw-score moments follow by efficiency. These are numerical
model diagnostics, not causal effects, default probabilities, or calibrated
reason-release probabilities. No top-k event is inferred from moments alone.

**E. Assumptions.** Fixed finite B; fixed numeric axis-aligned trees; finite values;
finite second moments; exact rectangle queries under a declared q. Natural
missing values must first have a declared representation outside this prototype.
No claim for unrestricted MNAR/shift. Encoded categorical dependencies must be
preserved by q; independent dummy draws are invalid. Grouping sums already
computed feature attributions and is not a grouped-player Shapley game.

**F. Formulation.** A leaf r has value v_r and distinct path features P_r after
intersecting repeated splits. Write I_j(x_j) for its interval indicator and
b_r(C)=|B|^-1 sum_{z in B} product_{j in C} I_j(z_j). With d=|P_r|, s=|U|,
the coefficient of the projected path rectangle R_{r,U} in attribution i is

```
c[r,U,i] = v_r b_r(P_r\U) *
  { (s-1)! (d-s)! / d!       if i in U
  { -s! (d-s-1)! / d!        if i in P_r\U
  { 0                        if i outside P_r.

phi_i(x;B) = sum_r sum_{U subset P_r} c[r,U,i] 1{x in R[r,U]}.
```

Boundary cases use only their applicable branch: U empty has no positive term;
U=P_r has no negative term; a constant leaf contributes zero attribution.
This coefficient identity is a direct derivation from established path-game
linearity and dummy reduction, **not a claimed new theorem**.

Proof: for S subset P_r\{i}, the Shapley increment is
v_r [I_i product_{j in S} I_j b_r(P_r\(S union {i})) -
product_{j in S} I_j b_r(P_r\S)], with weight
|S|!(d-|S|-1)!/d!. Collect the positive monomial U=S union {i} and negative
monomial U=S. Dummy reduction removes features outside the path. Sum leaves
and trees by linearity. Repeated-feature intersection is essential.

For a query, eliminate terms inconsistent with x_O, remove their observed
coordinates, and merge equal remaining rectangles. Let the resulting grouped
coefficient rows be c_a. With p_a=q(R_a), J_ab=q(R_a intersect R_b),

```
mu = C.T p
Sigma = C.T (J - p p.T) C.
```

The moment identity is standard. The candidate algorithm specializes the
compiled function *before* computing rectangle intersections, so its integration
size depends on the residual hidden-variable rectangles, not a global grid.
Joint cross-tree terms are retained. Product laws permit factorized rectangle
masses; correlated laws require joint queries and cannot use that shortcut.

**G. Exact mechanism to test.** Observe/prune/project/coalesce the attribution
rectangle representation, then perform a joint-moment contraction under q.
The candidate difference from fast point explainers is integration of a random
input response without first generating every input. The strongest rejection is
that this is merely a standard moment operator applied to an existing compiled
representation; measured specialization benefits and an explicit complexity
advantage are required before calling it a method contribution.

**H. Reduction.** A point-mass q returns point SHAP and zero covariance. Disabling
observation specialization leaves generic rectangle integration. Enumerating a
finite joint q gives ordinary weighted exact TreeSHAP moments. Disjoint-support
covariance cancellation is valid only under a product q. None of these reductions
is a competing newly named algorithm.

**I. Limits.** Two attribution laws can share mean/covariance but differ in sign
or rank events. Nonlinear probability calibration/sigmoid cannot be moved outside
an expectation. Original prediction leaves do not determine attribution regions.
There is no general guarantee of a cheaper query: worst-case residual size is
large, and the second-moment matrix is quadratic.

**J. Training/inference.** No method training. Compile f and B from training-only
assets. At inference use only x_O, O and fitted q. Verification is exclusively an
evaluation operation. Predictor/explainer/target are unchanged.

**K. Falsifier.** Any exact-oracle mismatch, hidden-input dependence, incorrect
cross-tree covariance, or no useful accuracy/cost improvement over the strongest
available point-explainer-plus-integration baseline rejects deployment. A speedup
against stock SHAP alone does not establish novelty against WOODELF-HD.

**L. Compute estimate before benchmark.** Let L be total leaves, D maximum unique
path depth, G output groups, R residual rectangles. At most L*2^D terms, each
originally at most D nonzero coefficients. Background compile costs roughly
O(|B| L D 2^D) in a straightforward prototype. Query contraction may cost
O(R^2 (D+G)) and O(R^2+RG) memory. For 200 depth-3 trees, the loose term bound is
12,800 and a dense double joint matrix could exceed 1 GiB. Enforce a 2,000-residual
term cap and report BUDGET_EXCEEDED rather than silently sampling. This is an
estimate; no performance result is claimed here.

## Novelty decision boundary

Only a bounded numerical feasibility pilot may proceed after independent
reduction review. The possible contribution is an efficient **joint-moment
inference algorithm**. Mere correctness of the above algebra, a speedup against
an outdated comparator, or applying covariance to SHAP is insufficient.
Financial model training and confirmation remain NOT RUN unless that gate passes.
