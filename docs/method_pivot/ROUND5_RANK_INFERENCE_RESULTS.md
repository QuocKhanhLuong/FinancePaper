# Round 5 — exact rank-event falsifier

Executed **2026-10-06**, preregistration commit `2a40d1e2c614941341ec509ceaf2e2a487d65941`.
This is a synthetic reduction audit, **not** a new financial experiment or a
benchmark victory. No classifier was trained and no financial records were loaded.
Historical reports, predictors, completion laws and revision definitions are unchanged.

## What was actually run

The [registered margin and two completion laws](ROUND5_RANK_INFERENCE_AUDIT.md)
were evaluated with three independent paths: direct arithmetic for the margin,
exhaustive coalition SHAP, and the existing piecewise attribution compiler.
The existing `revision_arrays(k=2)` function supplied the reference event.
A finite weighted-counting control projected compiled terms to candidate signs
and pairwise contrasts and checked the same event without constructing a full
attribution output vector. This is standard linear algebra plus enumeration,
not a new inference algorithm and not an alternative completion model.

With observed x1=x2=x3=1 and hidden h, the fixed point-reference attribution is
`(2+h/2, 2-h/2, 2.125, 0)` on the declared five atoms. The current explanation
uses h=-1 and displays groups 2 and 3 (one-based). Group 1 remains a valid
competitor after verification even though it was not initially displayed.

| Quantity | Law A: h=-1/+1, equal mass | Law B: h=-.5/+2, mass .8/.2 |
|---|---:|---:|
| Hidden mean / variance | 0 / 1 | 0 / 1 |
| Full attribution mean | (2, 2, 2.125, 0) | (2, 2, 2.125, 0) |
| Attribution covariance, upper-left 2×2 | [[.25,-.25],[-.25,.25]] | same |
| Remaining covariance entries | 0 | 0 |
| Margin at every completion | 6.125 | 6.125 |
| Verified top-2 revision probability | **.50** | **.20** |
| Exact-law rank-instability control | .25 | .10 |
| Exact-law sign-instability control | 0 | 0 |
| Mean variance over the three observed contributions | 1/6 | 1/6 |

The rank/sign/variance rows are descriptive exact-law adaptations of the frozen
completion controls, logged after the primary falsifier first ran. They do not
change the target or gate; they are not finite K8 performance measurements.
Here one of two displayed reasons exits whenever revision occurs, so the rank
score is exactly half the event probability. **The toy does not establish an
advantage over rank instability.** It instead preserves that dangerous baseline.

The two laws have exactly matching computed first/second attribution moments,
yet revision probability differs by **.30**. Thus those moments alone cannot
identify this event probability without additional assumptions. A Gaussian
shortcut would impose such an assumption. This is an instantiated standard
moment non-identification argument, **not a new theorem**.

## Numeric and engineering checks

- Maximum compiler-versus-coalition discrepancy: `4.440892098500626e-16`.
- Difference between the two means and between the two covariance matrices:
  `0.0` at recorded float64 precision.
- Direct contrasts and frozen full-attribution revision events agree on all
  five atoms: `[false, false, false, true, true]`.
- Seeds 101/102/103 generate additional probability weights only. All three
  weighted-event identities and moment/enumeration identities pass; these are
  not three training restarts or independent financial samples.
- Additive and point-mass controls, split boundaries, sign changes, ties,
  corruption/ineligibility guards, ignored hidden placeholders and support-row
  permutations are covered by **20 new tests**, including the reviewer-event
  counterexample at exact ties.
- Combined suite: **201 passed, 0 failed, 0 skipped**, 13.31 seconds. Three
  upstream SHAP/matplotlib deprecation warnings, no suppressed failures.

The supplied four-equiprobable-atom Fourier moment comparator does not support
these nonuniform laws; it was **not run** under falsely relabeled assumptions.
Any algorithm returning only identical exact moments has the same identification
limitation. No claim is made against Fourier event computation or characteristic
functions retaining more than first/second moments. Its historical round-two
speed advantage remains valid in its registered setting.

## Cost and limits

Current hardware verified: **Apple M4 Pro, 24 GiB**, macOS 26.2, CPU execution,
Python 3.11.16, NumPy 2.4.6. The final tiny audit used **0.003207 CPU seconds**
(one thread); imports and process startup are excluded. This is a budget receipt,
not a repeated latency benchmark or a speedup estimate. The declared budget was
60 CPU seconds. A direct query used six sign/contrast channels versus three
grouped attribution channels: lower output dimension is not automatic.

This finite oracle visits every support atom and materializes support×rectangle
indicators. It is intentionally bounded and offers no scalability evidence.
Floating-point comparisons near a rank/sign boundary can differ across algebraic
evaluation orders; the observed agreement is not a formal numerical certificate.
The supplied completion laws are known in this toy; no data-estimated conditional
law or misspecification robustness was tested. Background SHAP is not causal.

## Artifacts and reproduction

Ignored artifacts available in this environment:

- `outputs/method_pivot/round5/falsifier/results.json`: initial registered run.
- `outputs/method_pivot/round5/final_falsifier/results.json`: same primary run
  with descriptive frozen rank/sign/variance controls added.
- `outputs/method_pivot/round5/reviewed_falsifier/results.json`: final receipt
  after the additional tie-event regression test, unchanged primary result.
- `outputs/method_pivot/round5/reviewed_falsifier/pytest.log` and `pytest.xml`:
  final full suite (initial 200-test logs remain in `falsifier/`).

The final JSON SHA-256 is
`b4206810d229f48dfe736f77fdbcb9410d4a6184dd9115781d9143e885ec0925`.
Its manifest records the preregistration HEAD, the worktree changes including
untracked implementation files, UTC, versions, seeds and hashes of seven source/protocol/
lock files. All seven hashes were independently checked against the executed tree.
The reports/results were not present at preregistration; hashes, rather than a
false claim of a clean implementation commit, identify the executed sources.

```bash
uv run --frozen --extra temporal pytest -q tests/test_rank_event_audit.py
uv run --frozen --extra temporal pytest -q
uv run --frozen --extra temporal python scripts/run_rank_event_audit.py \
  --output outputs/method_pivot/round5/reproduction
```

The CLI refuses to overwrite an existing result. Raw JSON/logs remain ignored;
this aggregate report and the source/tests are public reproducibility artifacts.

## Status ledger

| Status | Evidence |
|---|---|
| VERIFIED | Current source, hardware, finite oracle, registered probabilities, 201-test receipt and source hashes |
| REPORTED / historical | Taiwan and Polish revision detection and round-two speed results; not rerun in round 5 |
| NOT RUN | Generic WMI/ProbBounds/TopShap implementations, financial rank-inference benchmark, new predictive model, Freddie, TabM |
| ASSUMED in toy | Supplied conditional finite law, fixed point background, fixed compiler and current imputation |

The [decision](ROUND5_NEXT_DECISION.md) concerns the mechanism, not merely whether
the code computes the right answer. Correctness passed; a new-method claim did not.
The [review receipt](ROUND5_ORCHESTRATION_REVIEW_RECEIPT.md) documents actual AGY
participation and the coordinator's corrections rather than treating agent
agreement as scientific validation.
