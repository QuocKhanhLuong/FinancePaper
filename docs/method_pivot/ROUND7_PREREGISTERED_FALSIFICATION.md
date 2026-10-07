# Round 7 — verification-anchored learning: analytical gate

Registered 2026-10-07, before numerical execution. Starting repository:
`a9de607946d94c087ec2ebe09f76e2e9252e62f4`, `research/method-pivot`.
This is an analytical candidate audit, not a financial benchmark or a claim of
new methodology. Independent AGY reports were read before this registration;
their suggested numbers are **unverified proposals**, not pilot results.

## One candidate to examine most deeply

**Learn partial-information predictions from a frozen verified predictor using
randomized verification to correct a misspecified completion model.** Its
possible extension jointly learns coarse/refined predictors. Select this for
falsification because completion misspecification is a genuine unresolved
assumption in FinancePaper; do not select it merely to rename AIPW.

Let W be current information including the mask, H newly verified information,
A a randomized verification indicator, and pi(W)>0 its known probability.
Assume A independent of H conditional on W. A frozen teacher gives g(W,H),
and q(W) estimates its conditional mean using an imputer. The proposed target is

    T = q(W) + A/pi(W) * (g(W,H) - q(W)).

The dangerous equal-information baseline is **ordinary AIPW pseudo-outcome
regression with the same q, pi, audits and student function class**. An exact
reduction to it defeats the new-method claim, even if it corrects imputer bias.
An unweighted verification-only mean is also compared on the homogeneous toy.

For joint training, examine the direct coherence objective

    J(theta) = E[(f_theta(W) - E[g_theta(W,H)|W])^2].

Check whether squaring an unbiased target optimizes J or adds a
theta-dependent variance penalty. Compare a conditional-moment critic with the
same information, not only imputation or XGBoost. Ordinary conditional-moment
duality is not a new theorem.

## Frozen numerical checks

1. Frozen teacher: H in {.5,1}, probabilities {2/3,1/3}; q=.9, pi=.1.
   Enumerate H and A exactly. Report mean, variance, and standard error of the
   mean over n=1000 independent intake records. Compare a fixed 100-record random
   verification mean (equal expected verification budget, different sampling
   design explicitly retained) and oracle q=E[H]. Do not present either analytic
   standard error as a measured confidence interval or training result.
2. Joint teacher: W constant, H uniform {-1,+1}, g_theta=theta*H, f=0, q=0.
   Theta in {0,.5,1}; pi in {.1,.25,.5,1}. Compare exact J, squared AIPW
   expectation, its finite-difference gradient, and unrestricted scalar
   conditional-moment critic. The signed output is a representation, not a
   credit probability. Show the effect of adding the toy supervised loss
   E[(g_theta-H)^2] with lambda=1, pi=.1.
3. Sensitivity proposal: q uniform over three states, contrasts
   (-4,8,8) and (8,-4,8), Gamma in {1,1.8,2,2.2}. Solve normalized box-constrained
   density-ratio LPs. Compare exact any-competitor failure with the different
   all-competitors criterion. A shared feasible set does not change
   inf_w min_j a_j(w) = min_j inf_w a_j(w).
4. Audit allocation: two equally frequent strata, sigma=(.4,.05), cost=(10,10),
   budgets 5 and 8. Compare uniform, unconstrained multiplier then clipping, and
   a multiplier re-solved after clipping. These are estimator-variance
   diagnostics, not classifier excess-risk or financial-gain estimates.
5. Discarded information certificate: a constant reason among a nominal
   three-label vocabulary refutes the proposed coverage bound without entropy
   assumptions. Independently enumerate interventional SHAP of uniform two-bit
   XOR; check the claimed nonzero attribution contrast.

## Boundaries and stop rule

No financial records, historical outputs, downloads, fitting, tuning, new
architectures, or new dependencies. Exact finite enumeration and small SciPy
LPs only; target <60 CPU seconds and <512 MiB, estimates before execution.
Run on the actual local machine and record environment/source/protocol hashes.
One deterministic execution plus regression tests; no sampling seeds or
customer bootstrap apply to exact finite calculations. A three-seed financial
pilot is authorized only after a genuinely distinct mechanism survives.

**NO-GO** if the candidate is algebraically the matched baseline, the asserted
advantage rests on an incorrect gradient/quantifier, or no nontrivial difference
survives. Correct equations do not by themselves pass novelty. No alternative
metric or financial test may rescue a failed gate. Preserve original worker
reports locally and publish coordinator corrections separately.

The script is a claim-falsification harness, not implementation of four new
methods. The outcome determines whether a method specification is warranted.
