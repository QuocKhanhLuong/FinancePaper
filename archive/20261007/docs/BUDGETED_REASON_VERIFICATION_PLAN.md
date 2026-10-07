# Budgeted reason verification: one prospective feasibility study

Original plan dated 2026-10-03. **2026-10-04 status:** the first
[development headroom audit](VERIFICATION_HEADROOM_RESULTS.md) is MEASURED / NO-GO;
full expected-value policy development remains NOT RUN. The prospective design
below is retained as history, including its original future-tense wording. Read the
[novelty reassessment](NOVELTY_REASSESSMENT_2026.md) first. This is not a claim of a
new generic acquisition algorithm. It does not modify existing fitted models,
reason definitions, datasets, calibration thresholds or historical results.

## 1. Hypothesis and fixed scope

> A field can be valuable for verifying reasons about other observed fields even
> when acquiring it barely changes the risk prediction. An acquisition policy
> targeting that value can improve verified reason coverage per query.

Use frozen XGBoost, the existing imputation/preprocessing and TreeSHAP reference.
Do not fit another classifier or a neural revision head. The proposed change is
the decision to request information before issuing a final reason set. Start
with **one field query**; no reinforcement learning or architecture sweep.

The zero-query release-all, rank and whole/partial-release policies are mandatory
controls. Verification can legitimately invalidate an initial reason; the policy
must retract it, not force it to remain stable.

## 2. The claim must stay the same while information arrives

Let `O0` be the originally observed fields, `M0` the artificially hidden but
recoverable fields, and `N` the naturally missing fields. Query actions come
only from `M0`. Values in `N` remain unknown throughout this experiment.

Fix the initial reason universe `G0` to semantic groups containing at least one
field in `O0`. For every partial, completed or restored record, define

\[
\Phi_g(x;O_0)=\sum_{j\in g\cap O_0}\phi_j(x).
\]

This retains the historical observed-field attribution convention: acquiring a
field changes the explanatory context, but does not add that field's own
attribution to the claims being audited. `G0`, `O0`, the model and SHAP reference
stay fixed for the episode. A separate newly discovered reason may be logged,
but cannot count as successful verification of an old claim.

Let `x*` restore every artificial hide and preserve every natural unknown. Define
the reason-verification label

\[
Y_g=\mathbf1\{\Phi_g(x^*;O_0)>.01\}.
\]

The primary released-reason failure is `1-Yg`. Retain the existing top-2
membership sensitivity, including its tie tolerance, as a **separate** target.
Do not select whichever target makes the acquisition rule look better. The
historical whole-explanation revision rate remains a separately named metric.

At any state, a released group must also be meaningfully positive under the
current point explanation. Output may contain zero or more groups. If an
initially negative group in `G0` becomes positive, report that transition
explicitly; do not call it preservation of an initially displayed reason.
Primary retained-claim analysis additionally restricts to initially positive
candidates, and is reported beside the fixed-universe communication analysis.

## 3. Optimize a decision, not an arbitrary score mixture

For an acquisition policy `pi` followed by release set `S`, use the objectives

\[
\max E|S|\quad\text{subject to}\quad
\frac{E\sum_{g\in S}(1-Y_g)}{E|S|}\le\alpha,
\qquad E\,cost(\pi)\le b.
\]

Report customer coverage `P(|S|>=1)` and `P(|S|>=2)` separately; they are not
interchangeable with reason coverage. An empty denominator gives risk N/A, not
zero. Use the existing alpha values `.05,.10,.15` and query budgets `b=0,1`
for the initial feasibility stage. Unit query cost is a transparent benchmark
assumption, not a monetary estimate. Real bundle prices require domain evidence.
Do not mix AP, query cost, coverage and reason failure into a weighted paper score.

## 4. Exactly one candidate strategy

Investigate **one-step expected gain in releasable reasons**. Denote a currently
available state by `s`, a frozen terminal release mapping by `R_lambda`, and a
train-only conditional completion family by `Q`. For an eligible query `j`,

\[
V_Q(j\mid s)=E_{v\sim Q(X_j\mid s)}
 [|R_\lambda(s\cup\{X_j=v\})|]-|R_\lambda(s)|.
\]

Choose the query with greatest positive estimated gain per unit cost; otherwise
stop. This is a standard one-step value-of-information construction, **not a
novel optimizer**. Its proposed application is the fixed reason-verification
target. No monotonicity, submodularity or global optimality is assumed.

Use the existing donor and conditional-imputer families in separate runs. Their
agreement is a sensitivity analysis, not an additional searched method family.
The terminal release rule must be identical across acquisition comparisons.
Rank-only release is the primary simple terminal control; the already measured
Stable-Core rule is a paired operational control, not automatically preferred.

Implementation details to freeze before any new financial run:

- Outer hypothetical values and inner completions need independent random streams
  conditional on the acquired value. Simply reusing an unconditional draw cloud
  after observing a hypothetical value is not conditional acquisition.
- Recondition only on observed or actually requested values. An evaluator owns
  `x*`; the acquisition policy never receives it, future SHAP, or a future label.
- Use `K=8` inner completions, matching the existing reference budget. Bound the
  outer expectation to eight values for the engineering pilot; numerical error
  must be measured separately, not mistaken for sampling customer uncertainty.
- Query every eligible missing field in the one-step development audit; do not
  prune candidates by own SHAP importance, which could discard the mechanism of
  interest. If this is too expensive, stop and report compute infeasibility.
- Train/development data fix the strategy and terminal mappings. Independent
  release calibration replays the **entire frozen query-and-stop policy** for
  each candidate setting, not just independently drawn MCAR masks.

No financial implementation has yet established that either current completion
API can recondition and run this calculation cheaply enough. That is an explicit
engineering gate rather than an assumed capability.

## 5. Baselines that could invalidate the contribution

| Baseline | Question answered |
|---|---|
| Zero-query release-all, rank and whole/top-1 policies | Is spending any verification effort necessary at the declared budget? |
| Random query and cheapest feasible query | Does personalization add value? At unit costs, cheapest is a tie/control. |
| Expected predictive-entropy reduction, same completions and predictor | Does optimizing prediction already solve the reason problem? |
| Largest absolute missing-field SHAP, following Kukar's acquisition principle | Does ordinary importance give the same benefit? |
| Largest missing-field SHAP variance, following Beebe-Wang et al. | Does the established explanation-guided acquisition rule suffice? |
| Expected reduction of reason rank instability | Does the strongest existing simple reason signal suffice? |
| Information gain on the reason-verification labels `Yg` | Does a straightforward EDDI-style target substitution match the proposed lookahead? **Strongest algorithmic falsifier.** |
| Full-information one-query oracle, evaluator only | Is there any useful headroom before fitting a policy? This is unattainable diagnostic information, not a deployable competitor. |

The literature [comparison](NOVELTY_REASSESSMENT_2026.md#closest-work-and-the-claim-each-rules-out)
also requires confronting EDFA. Its Markov-blanket priority can be compared as an
acquisition-policy adaptation if its assumptions can be implemented credibly.
It is **not** fair to present attribution failure as a reproduction of EDFA's
counterfactual-validity benchmark. State this endpoint mismatch, and do not
claim to beat EDFA without an appropriate reproduction.

Use paired customers/masks, the same compute budget and conditional sampler for
controlled policy comparisons. Report published-method adaptations as such.
No comparison may rely only on random or predictive confidence. Match the
number of displayed reasons per customer using strength/random selection as in
the completed Stable-Core study; keep a zero-query baseline at each set size.

## 6. First gate: a one-query development audit

Before a full matrix, declare the exact original-customer IDs from existing
**training/development pools only**, with hashes and no overlap with historical
assessment IDs. Freeze a bounded sample, e.g. 500 customers per available
development pool, before reading its new verification outcomes. Historical model
fits may already include these records, so this is feasibility evidence only.
An eventual confirmation requires entity-disjoint predictors, policy-development
and calibration sets. Do not label reused development data an untouched holdout.

For each partial record under the already defined missingness conditions:

1. Compute the zero-query release frontier and eligible initial reason counts.
2. In evaluator-only code, reveal each eligible field separately and evaluate the
   terminal policies against full artificial restoration.
3. Report the optimistic maximum one-query gain at matched set size and the
   fraction of customers for whom any query can help. Include the option of doing
   nothing; the oracle must not be forced to pay for an unnecessary query.
4. Compare the acquisition ordering induced by prediction, missing-feature
   importance/variance, rank reduction and reason-targeted utility.
5. Test whether query gains actually arise when prediction shifts are small,
   using all existing `.01,.02,.05` thresholds. Full-restoration shift defines a
   post-hoc analysis stratum only; it cannot route the deployed policy.

For claims of practical benefit, declare a minimum useful improvement before
this audit, in addition to paired 95% customer-cluster bootstrap intervals.
An illustrative **planning requirement**, not a fitted threshold or guaranteed
result, is five absolute percentage points of extra customer coverage at the
same budget/risk, or 20% fewer queries at matched coverage/risk. A domain owner
must justify the deployment relevance before calling either economically useful.

If even the optimistic oracle lacks that headroom, stop. In particular, if
release-all already meets risk with greater coverage, a nonzero-query solution
cannot claim a frontier improvement by ignoring that point. Do not tighten alpha
or change the revision endpoint in response.

## 7. Calibration and confirmation constraints

The complete trajectory is the statistical unit within a customer. Mask replicas,
completions, candidate queries and reasons are dependent. Bootstrap original
customers/loans; keep all replicas within the resampled cluster.

The primary micro ratio above is not the mean of customer false-discovery
proportions. Applying LTT/RCPS requires deriving a valid test for the specified
ratio and frozen policy, rather than borrowing a theorem for a different loss.
Adaptive querying and stopping must be included in the calibration replay.
Even correct exchangeable calibration is not automatically valid for a future
mortgage vintage. Retain empirical environment/temporal risk reporting and
separate any certificate from observed test intervals.

Only if the development gate passes should an implementation and a new
prospective acquisition protocol be frozen. Taiwan/Polish remain exploratory;
the earlier Freddie Stable-Core protocol is retained. A distinct acquisition
confirmation must specify its cohort and multiplicity before outcome access.
It cannot be selected after opening Freddie results. Natural unknowns never
become oracle values, and official Freddie access remains a prerequisite.

Report cost–coverage–failure frontiers, valid released reasons, customer coverage,
retractions, expected and tail query counts, AP/Brier as prediction guardrails,
wall-clock time and SHAP calls. Separate the value of a reason from causal truth
or correctness of the underlying financial prediction.

Freeze a small shared model-agnostic attribution check before confirmation, with
the same predictor and reference, to test whether the acquisition benefit is a
TreeSHAP-specific result. Report sensitivity without choosing the explainer that
produces the largest gain. Obtain domain review of reason groups and what a
verification action actually reveals before claiming analyst utility.

## 8. Decision rules and intended output

| Outcome | Research decision |
|---|---|
| No one-query headroom at the original targets/budgets | Stop acquisition development; retain the evaluation paper. |
| Prediction-based acquisition matches reason-based acquisition | The proposed decision gap has no practical evidence; no method claim. |
| Own-SHAP variance, rank reduction or reason-targeted information gain matches | Use the simple rule; frame as problem/evaluation contribution. |
| Gains vanish at matched reason count or come from changed candidates only | Reject the claimed reliability improvement. |
| Completion-family changes reverse the benefit | Narrow to completion-model sensitivity; no robust verification claim. |
| Independent temporal validation fails | Limit generality; do not tune on the confirmation cohort. |
| Material paired gains survive all controls and temporal confirmation | Consider a decision-policy contribution; algorithmic novelty still needs a specific demonstrated difference from known VOI. |

An eventual output should expose `risk_probability`, `released_reasons`,
`withheld_reasons`, `next_verification_request` or STOP, `estimated_query_cost`,
`reason_failure_target`, `calibration_scope`, and `release_status`. Attributions
and completion statistics are model diagnostics; no per-customer safety or
causal guarantee is implied. Full restored truth stays in evaluation-only files.

**Next action: run the predeclared one-query headroom audit on development data,
not another model or Stable-Core threshold search.** This document is a research
plan; it supplies no new financial performance numbers.

## Analytical appendix

This standard-library check enumerates the Shapley formula for the fixture in the
novelty report. It is an algebra check, not evidence of financial-model efficacy.

```python
from itertools import combinations, product
from math import exp, factorial

def f(x):
    a, b, c, u = x
    return 1 + .2*a + (a-b)*c + .01*u

def exact_phi(x):
    result = []
    for j in range(4):
        rest = [i for i in range(4) if i != j]
        total = 0.
        for m in range(4):
            for subset in combinations(rest, m):
                z = [x[i] if i in subset else 0. for i in range(4)]
                before = f(z)
                z[j] = x[j]
                total += factorial(m)*factorial(3-m)/factorial(4)*(f(z)-before)
        result.append(total)
    return result

p0 = 1/(1+exp(-f([1, 1, 0, 0])))
failures, shifts = [], []
for c, u in product([-1, 1], repeat=2):
    x = [1, 1, c, u]
    phi = exact_phi(x)
    expected = [.2+c/2, -c/2, 0., .01*u]
    assert all(abs(a-b) < 1e-12 for a, b in zip(phi, expected))
    assert abs(sum(phi)-(f(x)-1)) < 1e-12
    failures.append(phi[0] <= .01)
    shifts.append(abs(1/(1+exp(-f(x)))-p0))
assert sum(failures) == 2
assert max(shifts) < .001784
print("Analytical fixture verified; financial experiments NOT RUN")
```
