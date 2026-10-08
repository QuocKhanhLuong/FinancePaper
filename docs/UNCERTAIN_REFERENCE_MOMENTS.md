# Frozen protocol: imperfect reference moments

2026-10-08, base 43e7c76. This advances the next action in the reference-moment
report; it does not replay the archived F0/model plans. All predictors and
historical files remain unchanged. This is an exploratory synthetic audit using
previously inspected profiles, not an independent confirmatory dataset.

## Claim and proof to audit

Use exactly the C5 assumptions and signed-gap/tie-failure target from
REFERENCE_MOMENT_FRONTIER.md. Let b range over a coordinate box [l,u]. In C5,
alpha_k > beta_k > 0. Then c(b)=sum (alpha-beta)b is increasing, while
mu(b)=(1-b)/2 decreases. In the exact event-mass feasibility condition

    z * (1/2+c(b))/2 <= sum beta_k min(z,mu_k(b)),

increasing any coordinate makes feasibility harder. Consequently the sharp
worst-case bound over the box is R(l). It is attained by the previous two-state
construction at l (one state at risk endpoints). Upper endpoints cannot help
without further joint restrictions excluding l. This varies both the bounded
model and finite hidden law, not a fixed prescribed hidden law.

The range-aware scalar Markov bound is also worst at l. For positive t,
its derivative has sign determined by
alpha_j*(M-t)+t*beta_j >= 0, where M=c+1/2 and M-t>=0.
C5 and generic range use t_lower=sum alpha*l. No comparator receives the true
moment while the vector method receives an estimate.

## Fixed experiments and stop rules

1. Reuse all 231 configured profiles from the preceding stage (including
   duplicates). Expand each into radii 0,.001,.005,.01,.02,.05,.1, clipped at
   [-1,1]. These are deterministic uncertainty sets, NOT confidence intervals.
   Report every radius at budgets .01,.05,.1. No radius is selected after results.
2. At each of 1,617 boxes, verify exact attaining moments/gaps and information
   ordering, then independently optimize over b AND bad-event moments s at the
   predicted risk and at (1+risk)/2 when risk<1. This fixed-z LP minimizes
   z*tau(b)-sum beta*s with s<=z, s<=(1-b)/2,
   s>=(1-b)/2+z-1, s>=0 and l<=b<=u. A positive minimum excludes that z.
   Compare its residual against the rational lower-corner expression at 1e-9.
   LP numerical options/scaling inherit the prior documented repair.
3. Actual sampling: m=3,8,16, p=.5, all eleven profiles, N=128,2048,32768,
   32 replicates per cell; seed 20261008. Draw sufficient binomial counts from
   each profile's exact attaining one/two-state law. This is exactly the iid
   sampling experiment statistically, without materializing N repeated vectors;
   sampling probabilities are converted to float for NumPy. Save every replicate
   locally. These are synthetic repetitions, not people or independent datasets.
4. For each sample mean vector, l_k=max(-1,bhat_k-rho), where
   rho=sqrt(2 log((m-1)/delta)/N), delta=.05. One-sided Hoeffding plus a union
   bound provides simultaneous moment coverage for one configured experiment.
   Coordinate dependence is allowed; sample vectors must be iid.
5. Strong scalar control uses the SAME draws of Y=sum alpha*d and the direct
   radius sqrt(2 log(1/delta)/N), avoiding the vector union penalty. Compare
   direct C5 and generic bounds to vector C5/Markov/full-moment methods. Each
   confidence claim is standalone at 95%; do NOT take their minimum and claim
   joint 95% validity. No familywise claim across all experiment cells or models.
6. Count moment-coverage failures and risk-underbound events against exact
   synthetic truth. On simultaneous coverage, assert valid bounds exactly.
   C5 release decisions use the exact forward frontier, avoiding inverse
   rounding. Keep risk alpha and confidence error delta separate: a certificate
   at alpha=.05 with confidence .95 is not a joint 5% failure guarantee.

Failures stop the run and preserve artifacts. Per-task limit 180 seconds;
remaining tasks are NOT RUN if stopped. Resume requires identical source,
config and dependency hashes, and verifies completed task hashes. CPU only,
one numeric thread, tqdm, per-task elapsed time/ETA/warnings. Deliberately pause
after one task, resume, then test completed resume and integrity rejection.

## Interpretation fixed before execution

The primary comparison is certification at 5% for every nonzero radius and
every N; report all budgets. Separate originally interior profiles and boxes
whose lower corners are interior. A gain under deterministic radius shows
stability to that information loss, not a sample-size or deployment guarantee.
The direct-mean control can win because it pays no coordinate multiplicity cost.
If vector gains vanish under uncertainty, retain that negative result. Do not
train another financial model or tune profiles to restore gains.

The mathematical extension is monotonic robustification of an established
first-moment bound, not a new generic inequality. Hoeffding (1963),
*Probability Inequalities for Sums of Bounded Random Variables*, Theorem 2,
supplies the concentration ingredient (DOI 10.1080/01621459.1963.10500830).
No novelty priority, real-model extraction cost, human value or financial
benefit is established. Full truth-table reconstruction in this stage is NOT
RUN; the preceding definition audit remains REPORTED only.
