# Method-pivot results

Measured **2026-10-04**, 16:38:28 UTC / 23:38:28 Asia/Ho_Chi_Minh.
Method decision: **NO-GO before method implementation**. Financial method
training/confirmation: **NOT RUN**. The prospective finite-state audit is in
[PREREGISTERED_PILOT](PREREGISTERED_PILOT.md), committed at `d3a0c82` before
execution. Successful run source: `95f7bcad49ea75cc9031170925df2d496cc8712c`,
clean working tree. Historical artifacts are not new model results.

## What actually ran

- Two exactly enumerated eight-state populations, one additive and one with an
  interaction; Bernoulli outcome probabilities are specified, not fitted.
- 216 nested-subset/state comparisons per population, 432 total.
- Exact conditional inference, constant prevalence, mean-filled point inference,
  and exact inference under deliberately misspecified completion law q.
- A declared change in the hiding process, holding full-data P(X,Y) fixed.
- Read-only checks of **five existing aggregate CSVs**, including their hashes.
- **14 mathematical identity/witness checks passed**. These are assertions about
  finite examples, not empirical superiority of a new method.
- **16 new unit tests passed in .08 s**. Full suite: **137 passed, 0 failed,
  0 skipped in 14.19 s**, with three existing SHAP/Matplotlib deprecation warnings.

No predictor, completion model or release policy was fitted. No SHAP computation
was part of the finite-state audit. The full test suite separately executes its
existing small fixtures; that is not financial experiment replication.

## Expected outcome loss under the true synthetic population

Observe x1, hide x2 and x3. Each row is an exact population expectation over
states and Bernoulli outcomes, not a sample mean or a Taiwanese default score.
There are no sampling CIs/seeds because every state and probability is enumerated.

| Mechanism | Inference/control | Expected Brier | Expected log loss |
|---|---|---:|---:|
| Additive | Correct conditional oracle | .197534 | .579357 |
| Additive | Constant prevalence | .227353 | .647144 |
| Additive | Mean-filled point | .197788 | .580287 |
| Additive | Exact integration under wrong q | .205883 | .598505 |
| Interaction | Correct conditional oracle | .205668 | .598330 |
| Interaction | Constant prevalence | .229220 | .650991 |
| Interaction | Mean-filled point | .206939 | .602650 |
| Interaction | Exact integration under wrong q | .261236 | .754736 |

The constant, correct-law and wrong-law families all have self-tower error
<=1.12e-16. Wrong-q actual-law coarse/full coherence MSE is .008349 additive
and .055568 interaction. In the interaction world, exact wrong-law integration
is **worse than a constant predictor** on both outcome losses. This falsifies
“coherence/exact integration alone ensures reliable risk,” not the usefulness of
properly specified coherent learning or the performance of any cited paper.
The mean-fill comparator is close to the oracle here; no large benefit of a new
method is manufactured from this fixture.

## Mechanism checks

| Diagnostic | Additive | Interaction | Meaning |
|---|---:|---:|---|
| Single-refinement squared disagreement at the oracle | .018815 | .089569 | Penalizes legitimate update variance even when coarse prediction is correct |
| Independent-pair product expectation at the oracle | 0 | -8.67e-19 | Zero up to floating-point error; existing unbiased construction |
| Same product under biased coarse prediction | .008349 | .055568 | Equals actual conditional squared bias, not only at a zero-loss point |
| Maximum difference: expected probability vs sigmoid(expected logit) | .022365 | .047917 | Logit reconstruction does not supply probability marginalization |
| Reconstruction error after moving 2*x1 between components | 5.55e-16 | 8.88e-16 | Fidelity survives a large arbitrary allocation change without centering |
| Mean absolute change in hidden-case oracle risk after the fixed mechanism flip | .176357 | .423532 | Observation-process change alone can change the correct partial risk |

All effects, including the near-zero mean-fill loss gap, are retained. These are
counterexamples under fixed assumptions, not estimated effect sizes in a financial
population. The MNAR example conditions on observed x1,x3 and x2 missing; it does
not claim source and target have identical observable population distributions.

Illustration: in the interaction population with observed x1=+1, feasible hidden
states give full-input probabilities from **.075858 to .937027**. Any one number
has worst-case error at least **.430584** against those full-input responses.
That does not prevent the conditional oracle from being calibrated for outcomes;
it prevents claiming to know an individual's unrevealed state exactly.

## Historical aggregates checked, not experiments rerun

| Item | Existing artifact confirmed |
|---|---|
| Group-top2 Region B, MCAR30, raw shift <=.02 | Taiwan 42/727 = 5.7772%; Polish 85/552 = 15.3986% |
| Group-top2 detector AP | Taiwan prediction-only .144737, MC8 .692643, rank .695400; Polish .262246, .857991, .821935 |
| Paired detector differences | Existing rows retained, including Taiwan MC-minus-rank CI crossing zero |
| Meaningful Stable-Core vs release-all | Failed/released denominators and four aggregate ratios agree with stored tables |
| Retained-candidate customer ceiling | All 12 stored elementary-bound rows have zero possible gain over the best feasible zero-query control |

All five files were available. No historical training, prediction generation,
bootstrap or attribution cache was rerun by this audit. Other historical findings
remain **REPORTED**, with the scope recorded in the evidence ledger. Envelope,
TabM, Freddie and new learned A–D methods remain **NOT RUN**.

## Compute and provenance

Finite calculation plus aggregate checks: **.01856 s CPU** inside the process;
excludes interpreter startup, final serialization and full-suite tests. One run,
not a repeated latency benchmark. Actual peak RSS was not measured. No GPU used,
no trainable parameters, no new financial data download. Live host was Apple
M4 Pro / 24 GiB; MPS available, CUDA unavailable. No CPU fallback was necessary.

Local, ignored evidence:

```text
outputs/method_pivot/audit.json
outputs/method_pivot/audit.log
outputs/method_pivot/audit_initial_failure.log
outputs/method_pivot/unit_tests.log
outputs/method_pivot/full_tests.log
```

Hashes frozen in the successful audit:

| Object | SHA256 |
|---|---|
| Prospective protocol | `69acf5a16cc045dc1bdc222a9981c354ed9d8e5ff6c470c98256a3544cc62bf0` |
| Config | `ce7cce492f170b737250ab1ac37de0a0ca866d2e5c5bf94aa2dae2975263517e` |
| Audit source | `999ade43ba0375db78c776e360d57871bd6b999069a633c92a80cb5983b95bf5` |
| Successful audit JSON | `ebc06497ce78eef7b4fc36e076ec08aeefc326a5d79e8e1ebab7d63ba9311b16` |
| Full-suite raw log | `2c472f462632edbec8101f206d59ee632c9439058fe08cd0b2ddbe011d0bc3ef` |

Actual commands used the existing environment's `.venv/bin/python`. Portable
locked-environment equivalents (choose a fresh output path if the first exists):

```bash
uv run --frozen --extra temporal python scripts/audit_method_pivot.py --historical
uv run --frozen --extra temporal pytest -q tests/test_method_pivot_audit.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal pytest -q
```

The locked `uv run --frozen --extra temporal` audit command was also executed
with `--output outputs/method_pivot/audit_cli_verified.json`; all 14 checks
passed again. This verifies the documented CLI, not another independent experiment.

The runner refuses to overwrite an existing result. With absent historical CSVs
it labels them UNAVAILABLE and still runs the separately identified synthetic
audit; it never fabricates historical reproduction. No ignored outputs are committed.

## Interpretation and decision

The checks reveal no contradiction in the reduction arguments. They support
the need to distinguish inference accuracy **under q**, q's statistical validity,
and individual irreducible uncertainty. They do not establish a new remedy.
The decision remains **NO-GO for a new method from these formulations**.
No synthetic metric was promoted to a financial result, no failed target was
changed, and no new cohort was opened to seek a more favorable outcome.

## Initial audit failure, before rerun

The first audit at `d212733` stopped when the **new checker** interpreted the
historical CSV's `conservative` field as `False/True`; the actual encoding is
`0/1`. This is a parsing bug in the new audit, not a change to the historical
policy or its results. Added explicit validated boolean parsing and regression
fixtures for both encodings. Generator, protocol and mathematical checks are
unchanged. The initial test-log launcher also failed because the failed audit
had not yet created its output directory; no test had run at that point.
Failure receipt: local `outputs/method_pivot/audit_initial_failure.log`.
