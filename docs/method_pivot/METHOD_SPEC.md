# Formulation and reduction record — no method authorized

Status: **NO-GO before method implementation**. This file makes the rejected
mechanisms precise; it is not a renamed algorithm or a claim that any of them
has been empirically defeated. No A–D package skeleton is created.

## Shared statistical object

Let X be financial predictors, Y the later financial outcome, S the observed
field set, and W=(S,X_S). Missingness law pi_e(S|X,Y) may vary by environment e.
At serving time Y and X_notS are unavailable. Under an informative observation
process the appropriate risk is

`p_e(W) = E_e[Y | X_S,S]`,

not automatically `E[Y | X_S]`. With a fixed complete-input risk model p_f and
a chosen completion law q, define

`p_q(W) = integral p_f(X_S,h) q(dh | W)`.

This is conditional marginalization of a model, not an identified true outcome
probability unless both predictor and conditional-law assumptions are justified.
Artificial masks sampled independently of outcome do not establish natural
missingness informativeness. Natural unknowns never become verification truth.

There are four distinct shifts: P(X) (covariates), pi(S|X,Y) (observation), P(Y)
(prevalence, under its own label-shift assumptions), and P(Y|X) (outcome process).
Changing one can change induced marginals of others; a stress generator must
specify which primitive law it changes rather than conflating all four.

## Reduction A: decomposition is not identified by fidelity alone

For a fixed reference and centered expansion `f(x)=sum_A f_A(x_A)`, write

`d_S(x_S)=sum_{A subset S} f_A(x_A)`

`i_{q,S}(x_S)=sum_{A not subset S} E_q[f_A(X_A)|W]`.

Then `E_q[f(X)|W]=d_S+i_{q,S}`. This is an application of linearity to a fixed
decomposition, not a new result. Under dependent covariates the reference and
orthogonality convention matter. Without a constraint, adding any h(X_S) to
d_S and subtracting it from i leaves fidelity unchanged. An additive-plus-residual
network has exactly that problem.

Moreover `sigmoid(d_S+i)` need not equal `E_q[sigmoid(f)|W]`. A claimed probability
decomposition must choose one estimand explicitly. Observed terms can still be
proxies; model decomposition is not causal evidence provenance.

## Reduction B: coherence allows updating, but does not identify truth

For nested information sigma-fields F_S subset F_T, a coherent outcome predictor
obeys `p_S = E[p_T | F_S]`. These sigma-fields include the observation history;
the numeric feature sets alone are insufficient if the reveal selection conveys
information. The tower property is standard probability.

With fitted q, the natural penalty is

`C_q(theta)=E[(p_theta(W_S)-E_q[p_theta(W_T)|W_S])^2]`.

The supervised objective `E[BCE(Y,p_theta(W_S))]+lambda*C_q` needs BCE to prevent
constant predictions. A single refined observation inside a square adds the
refined-prediction variance and can suppress justified updates. Two independent
refinements remove that bias in expectation; this estimator is **already in
[Gögl et al. §3.4](https://arxiv.org/html/2605.11846v1)**, not our mechanism.
Zero C_q can coexist with wrong predictions under the actual law P.

### Required method card, applied to the most direct candidate B

| Item | Audit answer |
|---|---|
| A. Research question | Can partial-information outcome predictions update coherently without hiding uncertainty introduced by a misspecified revelation law? |
| B. Failure mode | Independent mask fits need not agree conditionally. Repository donor-support misses motivate law misspecification; no measured coherence failure has yet been attributed to XGB25. |
| C. Input | Current W only at inference. Complete development X,Y may supervise synthetic masking; natural unknown cells are never truth. |
| D. Output | Current outcome probability and an explicitly q-relative coherence diagnostic; neither is an explanation-correctness certificate. |
| E. Assumptions | Stable full-data law, declared sampling/revelation process, conditional support and independent refinement draws. MNAR requires more assumptions. |
| F. Objective/inference | BCE plus C_q above; model consumes W. q fitted only on permitted training data, fixed before independent evaluation. |
| G. Exact new mechanism | **None identified beyond existing prediction-space martingale consistency.** This fails the method gate. |
| H. Reduction | lambda=0 gives masking-augmented prediction; exact coherent joint inference satisfies the same identity without the new penalty. The two-draw variant overlaps prior art. |
| I. Identifiability | Training conditional risk is estimable under support and sampling assumptions; unobserved deployment law and individual hidden truth are not supplied by coherence. |
| J. Training/inference | No restored prediction, outcome or hidden truth may be serving features. A teacher depending on evolving predictor labels would need freezing/cross-fitting; none is introduced. |
| K. Falsifier | If coherence merely improves C_q while outcome loss or actual-law coherence worsens, it does not solve the proposed problem. Matching generative inference at higher cost does not establish necessity. |
| L. Compute estimate | No selected model/parameter count. A straightforward two-refinement penalty requires at least coarse + two refined evaluations and sampler work per pair; not benchmarked. |

## Reduction C: observations do not specify an unrestricted robust solution

A defensible sensitivity analysis can declare an uncertainty class Q_Gamma(W)
and report `inf_q p_q(W), sup_q p_q(W)`. A learner might minimize

`min_theta sup_{q in Q_Gamma} E_q[ell(Y,p_theta(W))]`.

Neither expression identifies Gamma or which mask relationships will persist.
Without constraints the result may be vacuous or conservative enough to destroy
discrimination. Decorrelation, conditional DRO and observation-consistent
ambiguity sets already have close precedents in the candidate table. New target
verification information could alter the identification problem, but its sampling
design and estimator must be specified before claiming adaptation. No unrestricted
MNAR recovery or guarantee is proposed.

## Reduction D: exact computation does not fix a wrong conditional law

For additive trees `f=b+sum_t sum_l v_tl * 1{X in R_tl}`,

`E_q[f|W]=b+sum_t sum_l v_tl * q(R_tl|W)`.

That is standard leaf-mass integration. Computing `E_q[sigmoid(f)|W]` is a
different query; dependence across trees remains. A partition of hidden space
with masses q(C|W), and lower/upper bounds a_C,b_C on p_f within each cell gives

`sum_C q(C|W)*a_C <= p_q(W) <= sum_C q(C|W)*b_C`.

This elementary interval bound is not a new theorem or automatically a cheap
algorithm. For any numerical approximation p_hat,

`|p_hat-p_P| <= |p_hat-p_q| + |p_q-p_P|`.

Numerical refinement can reduce the first term while leaving the second intact.
Even the exact oracle under q cannot cure completion misspecification. The proposed
D has no established new refinement algorithm or rate, so it is not implemented.

## What a future gate would have to add

An exact algorithm/constraint/estimator absent from these reductions, a reason
the closest baseline lacks it, and evidence of an outcome/informativeness/compute
benefit. A new theorem is not mandatory. Retargeting a standard identity or
demonstrating lower revision by erasing interactions is insufficient. Until that
object exists, there is no loss coefficient, architecture or predictor to tune.
