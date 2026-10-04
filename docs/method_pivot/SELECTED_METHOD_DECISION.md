# Selection decision

**NO-GO: no new method is selected from the four audited constructions.**
Decision date 2026-10-04, on `research/method-pivot`, base `95773c7`.
This is the permitted “none passes” outcome, not four proposed implementations
and not a recommendation to train them all. Historical methods remain untouched.

## Why this is a technical decision

The [candidate audit](CANDIDATE_DIRECTIONS.md) inspected 17 nearest-work entries
across four different mechanisms. The disqualifiers are specific:

1. **A:** after fixing a centered expansion, partitioning its terms by availability
   and integrating the others introduces no new estimator. Without centering,
   arbitrary observed functions can move into the residual without changing the
   output. Reconstruction by itself is insufficient.
2. **B:** the predictive conditional-expectation objective and independent
   two-refinement estimator are directly disclosed in the 2026 martingale SSL
   preprint. Earlier posterior-consistency and conditional-inference papers are
   additional precedents. Merely using financial outcomes is not a mechanism change.
3. **C:** generic decorrelation or minimax conditional prediction is not a new
   solution to deciding when informative missingness is trustworthy. That decision
   needs an explicit observation-process class or target verification evidence;
   neither a new estimator nor justified adaptive rule has been supplied.
4. **D:** expected tree predictions and tractability conditions already exist.
   Exact expected logit is not exact expected probability, and an integration
   certificate does not certify the completion distribution. No cost/error
   advantage of a distinct algorithm has been established.

We are **not** rejecting them because their building blocks are old. For example,
an independently justified correction for a biased revelation law in B, or a
new bounded-cost inference algorithm in D, could combine existing components
and still contribute. At present those are unsolved design requirements, not
implemented differences to which a paper can lay claim. “No method selected”
does not mean the four research areas are exhausted.

## Gate ledger

| Required gate | A | B | C | D |
|---|---|---|---|---|
| Exact technical difference from closest mechanism | Not supplied beyond decomposition/tagging | Direct overlap | Observation-aware adaptation unspecified | Integration/search adaptation only |
| Need in this repository | Completion dependence motivates study | Independent mask fits may be incoherent; not yet measured | Simulated-to-real mechanism gap plausible, not identified | K cost/support issues documented, not a solver bottleneck proof |
| Evidence for additional component | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Nontriviality | Requires fidelity plus predictive information | Coherent constants and wrong-q predictions are countercontrols | Unrestricted robustness can discard useful signal | Wide bounds/exact wrong-law answers do not count |
| Scope honest enough to train a novel method now? | No | No | No | No |

“Not supplied” is not “proved impossible.” “NOT RUN” is not “failed experiment.”
The novelty decision is based on reductions and unmet identification/algorithm
requirements. The finite-state checks in [the prospective audit](PREREGISTERED_PILOT.md)
can expose an error in those arguments; they are not an architecture tournament.

## What can still be claimed

The old evaluation evidence remains: financial predictions and their observed
reasons can respond differently to verification; completion explanations detect
revision better than the tested prediction controls; useful but limited release
tradeoffs exist. The new audit adds explicit requirements for any later method
claim and stops four insufficient formulations before test-driven fitting.
It adds **no supported new learning algorithm, theorem or financial result**.

No scientific venue acceptance or subjective novelty score is assigned. The
appropriate current output is a bounded evaluation/methods-validation paper,
not a new-method paper rescued by changing architecture or target.

## Execution boundary

- Permit only the declared finite-state algebra/sensitivity checks and aggregate
  artifact rechecks in this branch.
- No new financial predictor, auxiliary head, completion model, explainer,
  conformal envelope or external dataset run.
- No synthetic result may be used to retrofit A–D into a winner.
- Do not change historical reports or merge the branch to main.

One next scientific action after completing this audit: **conduct the already
motivated preregistered, blinded domain-expert assessment of grouped revision and
near-tie cases**, to establish whether the demonstrated operational differences
matter. That study tests the remaining evaluation claim; it does not manufacture
method novelty.
