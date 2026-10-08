# C5 audit with retained reference moments

2026-10-08. New branch `research/reference-moment-frontier-20261008`, base
`8611ab8`. User approved the next gate: compare general-m C5 with a baseline
retaining the reference moments used in its positive-mixture representation.
This protocol and derivation are fixed before numerical outcomes. Historical
source/config/results and inspected finance cohorts remain unchanged.

## Information and target

Use C5's exact assumptions: m>=2 observed iid Bernoulli(p) players, query A=1,
one finite hidden SHAP player with matched reference/completion law, a bounded
fixed model f in [0,1], signed interventional-SHAP pair gap, and ties as failures.
Normalize the gap by (1-p). C5 supplies positive cardinality coefficients
alpha_k,beta_k, summing to 1 and 1/2 respectively, for k=0,...,m-2:

  G(H) = sum beta_k d_k(H) + sum (alpha_k-beta_k) b_k,
  b_k=E d_k, d_k in [-1,1], t=E G=sum alpha_k b_k.

C5 retains only t after reduction. The new comparator retains the whole vector
b. These are first moments indexed by cardinality, not moments of different
orders or independent random quantities. All d_k depend on the same H.

Availability is conditional: in an implementation that computes t as
alpha dot (E d), retaining b needs no new model evaluations. We do NOT assert
that b is free for every TreeSHAP/moment implementation. The existing general
`attribution_moments` API returns mean/second/covariance, not this b vector.
General extraction cost and a real-model utility comparison are NOT RUN here.

## Sharp bounded-moment comparator

Set X_k=(1-d_k)/2 in [0,1], mu_k=(1-b_k)/2, a_k=beta_k,
c=sum (alpha_k-beta_k)b_k, M=c+sum a_k, tau=M/2.
Then G<=0 exactly when sum a_k X_k>=tau. For t>0, tau>sum a_k mu_k.
No independence between X coordinates is assumed.

For an event B of probability z, E[X_k 1_B]<=min(z,mu_k). Consequently

  z*tau <= sum a_k min(z,mu_k).

The largest z in [0,1] satisfying this inequality is the **sharp upper bound
given b**. Sorting mu and sweeping its breakpoints finds it with O(m log m)
comparisons/arithmetic operations after coefficients and moments are available;
arbitrary-precision bit cost is not constant. This is a standard bounded
first-moment optimization specialized to the C5 representation, not a claim
to invent moment inequalities or sorting.

Proof of sufficiency/attainment for 0<z<1: at the maximal feasible z choose

  X_bad,k=min(1,mu_k/z),
  X_good,k=(mu_k-z*X_bad,k)/(1-z).

Both are in [0,1], and their z/(1-z) mixture has exactly mean mu. The bad
margin is zero at the root. Because t>0, the good margin is strictly positive.
Thus two hidden states attain exactly z failure mass. Set d=1-2X and realize
each contrast by f10=(1+d)/2, f01=(1-d)/2, and equal-pair rows f=c0 in [0,1].
This is a bounded cardinality-symmetric full model with constant query prediction.
For bound 0 use a one-state model d=b with positive gap. For t<=0 the sharp
bound 1 is attained by the constant contrast d=b. Fixed preassigned hidden
laws can make the bound conservative: sharpness varies that law and model.
The theorem concerns nonpositive gaps; strict-reversal sharpness is not asserted.

An independent finite LP uses variables z and s_k=E[X_k 1_B]: maximize z,
subject to 0<=z<=1, 0<=s_k<=mu_k, s_k<=z, s_k>=mu_k+z-1, and
tau*z<=sum a_k s_k. This is a generic first-moment probability bound.

## Relationship to C5 and scalar baselines

Ordinary range-aware Markov gives z<=1-t/M (for t>0). It ignores individual
component budgets. The vector-moment optimum cannot exceed this scalar bound.
It also cannot exceed C5: conditioning on b restricts the models in C5's
optimization, while preserving its mean t and SHAP assumptions.

This does not establish that scalar Markov alone always dominates C5. Count
both directions in the audit. A result with richer information cannot be
called a same-information improvement without accounting for acquiring b.

The old C5 extremizer has b_k=1-2*z*loss_k, with `loss` returned by its
fractional-knapsack solution. Applying the vector-moment bound to those b's
must return exactly z. Hence C5 is the sharp envelope obtained when only t
is retained; the construction provides a falsifiable equality case.

## Frozen audit

Use configs/reference_moment_audit.json. For each of seven m's and three p's:
five constant b profiles, ascending/descending/alternating profiles, and three
seeded profiles from the declared rational grid. Ascending/descending use k/(m-2)
and its complement, with 1/2 for m=2; alternating uses k modulo 2. This makes
**231 profile tasks**, including all-zero/all-one moments. These are synthetic
moment conditions, not sampled people, distinct datasets or model selection.

For each profile:
1. Compare the rational sorted solution to the generic LP; compute C5's existing
   conservative inverse and scalar range-aware Markov; retain every result.
2. Construct an exact rational attaining law/contrast table and verify moments,
   bounds and event mass. No tail probability is inferred from float near-ties.
3. For m<=5, independently derive the full SHAP truth-table operator from exact
   coalition/background sums; verify the witness. Solve full-table LPs with b
   fixed and query predictions constant at the proposed maximal z, then at
   (1+z)/2 when z<1. The latter must have strictly positive minimum bad gap.
   The z=0 LP may use a zero-mass bad state as a boundary diagnostic; the
   published actual attaining witness instead has one positive-mass state.
4. Report certificate counts at the three frozen risk budgets. There is no
   empirical decision-benefit interpretation or significance test on these counts.

An additional **105 tasks** recover the old C5 extremizing moment vectors
(seven m's, three p's, five z's), compare the rational vector bound to z, and
check the LP. Total: **336 tasks**. Full-table work is NOT RUN for m>5;
compact exact moments/contrasts still run there.

Coefficient generation, generic moment LP, full-table LP, inverse evaluation,
and witnesses are different checks; do not count them as independent studies.
Record CPU, one numeric thread, elapsed time/warnings, source/dependency hashes,
tqdm/ETA and per-task JSONL progress. Deliberately pause after one task, resume,
verify complete resume, and test drift/corruption on a separate scratch run.
Stop on a correctness discrepancy or a task exceeding 180 seconds. Never
relax tolerance or replace the protocol after seeing results to rescue a claim.

## Novelty gate and publication

The generic moment-problem/LP approach is established mathematics. Bertsimas
and Popescu, SIAM J. Optim. 15(3), 780-804 (2005), study sharp probability bounds
from moments and tractable first-moment cases; sections 2 and 5 are relevant:
https://www.mit.edu/~dbertsim/papers/MomentProblems/Optimal-inequalities-in-probability-theory-A-convex-optimization-approach-SIAM15.pdf
This source supports the framework's prior status; we do not claim it explicitly
contains our exact SHAP cardinality formula. No broad first-of-its-kind claim.

If the richer comparator dominates, retain C5 as a sharp information-limited
characterization and reject incremental utility claims conditional on free b.
If scalar Markov alone fails to dominate, record that limitation too. Publish
one stage report separating RUN, earlier REPORTED, PROPOSED and NOT RUN, with
all negative findings. No finance fitting, human judgments, main merge, or
rewriting historical evidence.
