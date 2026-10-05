# Round 4 decision: empirical gap survives; method novelty still unestablished

Date: **2026-10-05**. Research branch only. Historical decisions are preserved.

## What is now supported

The three-restart, same-completion Taiwan development comparison passed its
predeclared gate in every restart. On grouped top-2 MCAR30 revision:

- Prediction distribution + mask detector: AP **.0991 ± .0402**.
- Same detector + four explanation statistics: **.5253 ± .0974**.
- Direct MC8: **.6139 ± .1117**.
- Rank instability: **.6012 ± .1256**.

These are restart mean/sample SD. The paired AP gains over the prediction-only
detector have positive 95% bootstrap lower bounds within each restart. The
comparison strengthens the evidence that the explanation target is not adequately
captured by the **prediction controls actually tested**, including all eight
sorted completion probabilities. See [full results](ROUND4_PREDICTION_DISTRIBUTION_RESULTS.md).

It does not prove that every predictive uncertainty method must fail. The fitting
set has only 400 customers per restart, with limited positive revision examples.
Official learned DMV was not reproduced. Taiwan is reused development evidence;
the three folds reuse people and are not external confirmation.

## Candidate method decision

**NO-GO for claiming either of the two present constructions as a new method.**

1. Directly distilling a conditional attribution density presently changes the
   response of a standard conditional density objective. It has not specified a
   necessary new mechanism beyond known amortization/density estimation.
2. Constrained search for an attribution change inside a prediction-stable
   region presently changes the query of known verification/perturbation methods.
   It has not supplied a new representation, admissible bound or search algorithm.

This decision rejects those particular constructions, not all learning-based
methods or all verification algorithms. A changed target can support a new
method if it creates and resolves a real technical difficulty; it does not do so
automatically. [Reviewed prior-art comparison](ROUND4_INDEPENDENT_MECHANISM_REVIEW.md)
states both strongest rejection and the missing defensible difference.

The earlier moment estimator's unfavorable comparison with the known Fourier
baseline, Stable-Core's limits, one-field acquisition NO-GO, and old conformal
specification's novelty NO-GO remain visible. Not-run methods are not failures.

## Strongest surviving baseline and nontriviality boundary

Rank instability remains the immediate method-claim falsifier. Both it and MC8
use completion explanations; no computational saving is assumed merely from
renaming their final statistic. Any new method must outperform the right matched
baseline in error/informativeness **or** measured computation at matched error,
without suppressing all reasons, erasing interactions, widening outputs without
bound, or relying on additional hidden truth.

No method contribution is supported by the new audit. What is supported is a
stronger **evaluation finding**, an exact synthetic counterexample, executable
baseline controls and auditable negative novelty decisions. This is not a promise
of ESWA/conference acceptance and does not satisfy the user's ultimate novelty
objective yet.

## Exactly one next research action

Audit whether **direct inference of attribution-rank event probabilities** can
avoid computing full attribution vectors for every completion, with a concrete
algorithmic advantage over the existing Fourier/moment representation and
same-completion rank baseline at matched error. The task is to derive the actual
representation/bound/estimator and its reduction before implementation—not to
rename the existing MC score or assume that this question is novel. If the
derivation reduces to known inference unchanged, record NO-GO again. No new
dataset, neural architecture or test-driven tuning is authorized by that question.

**VERIFIED:** source audit and corrected mathematics; synthetic oracle; three
bounded development runs; 181 engineering tests. **REPORTED:** earlier financial
studies and historical model fitting. **NOT RUN:** either new candidate, learned
DMV, untouched confirmation, TabM, Freddie. **ASSUMED:** the frozen completion
law and attribution reference; neither is causal truth.
