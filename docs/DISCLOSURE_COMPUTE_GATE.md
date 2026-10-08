# C6 computational usefulness gate — frozen before outcomes

2026-10-08. User authorized running the next substantive gate from the C6 report.
Branch `research/disclosure-compute-gate-20261008`, base `9577a20`.
Old protocols and financial cohorts remain frozen. This is an executed synthetic
mechanism study, not a new financial or human experiment.

## Question and decisive comparator

Can exact-mean disclosure certificates obtain useful coverage at lower compute
cost than enumerating hidden completions, and does C6 add value over the strongest
simple bound available from the same computation?

A pre-outcome algebra check exposes a critical comparator. In the two-observed,
one-hidden-player setting of C6, write d=f(10,H)-f(01,H), b=E_R d,
G=(d+b)/2, and t=E_Q G. Since G <= (1+b)/2, ordinary Markov applied to
(1-d)/2 gives

  Q(G<=0) <= 1-2t/(1+b).

Applying the same argument under R and the ordinary TV event bound gives

  Q(G<=0) <= epsilon+(1-b)/(1+b).

For b>-1 and t>=0 define the reference-aware baseline

  B(t,b,epsilon) = clip(min(1-t, 1-2t/(1+b),
                          epsilon+(1-b)/(1+b)), 0, 1).

The degenerate b=-1 case supplies bound 1. This baseline uses b, which is already
available when computing the exact C6 mean in this benchmark. It always matches
or improves C6. For epsilon<1-t, its second term increases with b and its third
decreases. They intersect at b*=(2t+epsilon)/(2-epsilon), and their common value
is (1-t+t*epsilon)/(1+t). Thus their minimum is at most C6 for every b. For
epsilon>=1-t the generic term supplies C6. Endpoints follow by continuity.

This does not invalidate the prior sharp theorem, whose information was only
(t,epsilon). It makes an incremental C6 coverage advantage impossible when b
is retained. This is an information-accounting falsifier, not a novel Markov
inequality. We will still run the cost/coverage benchmark to distinguish useful
cheap certification in general from an incremental contribution of C6.

## Frozen model and probability family

Two observed players have Bernoulli(1/2) reference and query (1,1). H is ONE
atomic SHAP player with 2^k possible values. Its internal encoding has k bits
in {-1,1}; these are NOT separate SHAP players. C6's scope is not extended.

Use a compact signed polynomial d(H)=sum_j w_j product_{i in S_j} H_i, with
positive integer coefficient numerators drawn uniformly from 1..9 and normalized
to sum 1. Hence |d|<=1 without clipping. Define f10=(1+d)/2, f01=(1-d)/2,
f00=f11=1/2. Prediction is constant on the query slice. No fitting or outcome
selection is involved.

Three fixed families: (a) linear, one singleton per bit; (b) sparse polynomial,
min(2k,64) distinct nonempty supports of degree 1..3 sampled with equal degree
probabilities; (c) shared gate, supports {0} and {0,j} for j>0. The simple linear
and shared-gate cases are controls for exploiting known structure.

R has independent bits P(+1)=q. Q=(1-epsilon)R+epsilon U, where U is uniform
over bit strings. TV(R,Q)<=epsilon is known by construction, not estimated.
Since supports are nonempty, E_U d=0. Thus t=(1-epsilon/2)b. Exact rational
first and second moments use independent character expectations:
E_R chi_S=(2q-1)^|S| and chi_S chi_T=chi_(S symmetric_difference T).
These standard finite-sum identities are computational infrastructure, not new
algorithmic novelty. Model generation/representation is shared by all methods.

The JSON config fixes six dimensions, three families, three seeds, three q's,
three contamination levels, three release budgets and two MC budgets: 54 model
tasks, 486 law/model cells, 1,458 cell/budget evaluations. Full enumeration is
scheduled only for k<=20 (324 cells). k=28,40 (162 cells) are scaling probes with
enumeration explicitly NOT RUN. Seeds are synthetic model instances, not people
or independent real datasets. Budgets within a model are repeated conditions.

## Comparators and measurement

1. C6 exact-mean bound.
2. Generic 1-t and the prior ordinary TV/mean-transfer baseline.
3. Reference-aware Markov/TV baseline B above, with no additional moment cost.
4. Exact-variance Cantelli, and its minimum with B. This uses more information;
   its separate second-moment computation time must be included.
5. Direct binomial tail Monte Carlo with fixed 256/2048 draws and one-sided
   Clopper-Pearson bounds. Total failure allowance 0.05 is Bonferroni-divided by
   all 486*2 planned bounds. Means/reference model are still known; sampling
   evaluates the same fixed-reference gap. MC is not treated as a deterministic
   population certificate. No sequential stopping or retrospective budget choice.
6. Structure-aware depth-first branch-and-bound with deterministic influence
   ordering and at most 5,000 visited nodes per q/model. Exact integer partial
   polynomial ranges classify whole subtrees. Unresolved subtree probability
   yields a valid risk interval, not a fabricated exact risk or zero failure.
7. Chunked exhaustive enumeration for k<=20, including optimized integer parity
   evaluation and bin-count probability aggregation. No strawman coalition
   enumeration in the timing comparison. All methods may use G=(d+b)/2.

The event threshold uses rational b and integer polynomial values, so floating
near-ties do not affect classification. Enumeration and branch probability sums
use floating arithmetic with 1e-10 comparison tolerance. The entire benchmark
uses the same prediction/reference/completion laws for all methods.

Randomize model order and method-block order using the frozen schedule seed.
Median of five repeats measures first-moment and second-moment computations;
enumeration and branch traversal run once per task and share work across the
nine laws. Report their task totals rather than pretending each law independently
paid the total. MC timing includes draws, evaluation and interval computation.
Cold import/model construction is separately excluded for all methods. Timing
ratios are local descriptive measurements, not asymptotic lower bounds or
independent statistical replications. Record CPU, numeric thread count, elapsed
time, warnings, method durations, evaluated states/nodes and unresolved mass.

## Gates, stopping and publication

Correctness: exact rational moments must agree with full enumeration at k<=20;
all deterministic bounds and branch intervals must contain enumerated risk to
tolerance. Monte Carlo upper-bound misses are reported, not asserted impossible.
A small independent SHAP-coalition test checks the grouped-player identity.

Practical cheap-certification gate: at the fixed alpha=0.05 budget, at least 25%
of k<=20 cells are certified by a mean-only bound, with at least 10x lower median
per-task cost than exhaustive enumeration on the largest enumerated dimension.
Compare MC and branch-and-bound before attributing this benefit to a new method.
The 25% and 10x thresholds are local descriptive gates, not population hypotheses.

Incremental-C6 gate: count cells certified only by C6 after including B. Algebra
predicts zero; any positive count indicates a bug/numerical issue. A zero result
is NO-GO for incremental C6 utility when reference mean b is freely available.
Do not modify the config, thresholds or model family after observing outcomes.

Abort on an invalid deterministic bound or hash mismatch. If a model task exceeds
180 seconds, stop before the next task and record the remainder NOT RUN; resume
only with unchanged source/config. All runs use unique ignored directories,
task receipts, JSONL progress, tqdm/ETA and source/dependency hashes. Deliberately
pause/resume once and verify complete resume plus scratch drift/corruption guards.
Publish one consolidated report with exact RUN, REPORTED, PROPOSED and NOT RUN
sections. Do not rerun archived finance experiments or merge main.
