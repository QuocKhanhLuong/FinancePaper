# Round 6 preflight incident and narrow amendment

Recorded 2026-10-06 after preregistration commit `ad155bf`, before any completed
structural row or aggregate. The first attempt is retained under ignored
`outputs/method_pivot/round6/structural/` with `manifest.json`, `failure.json`
and parent `runner.log`. It stopped with:

```
ValueError: Completion metadata must be observed=1 and delta=1/5
```

This is an error in the **new audit's contract and preregistration wording**,
not an identified defect in the frozen predictor or historical preprocessing.
`src/financepaper/data/temporal.py::elapsed_delta` sets the first month to zero
because the pre-history is unknown; subsequent complete months have elapsed gap
one. `TemporalBatch.flatten(True)` divides this by five. Complete-state metadata
therefore has all observed masks equal to one, April deltas zero, and May–September
deltas one fifth. The original round-six protocol is retained as written; this
amendment supersedes only its sentence asserting that all deltas equal one fifth.

The audit now requires an explicit metadata contract. The Taiwan runner derives
it from the existing `TEMPORAL_FIELDS` and `elapsed_delta` with an all-observed
mask, rather than hardcoding a universal delta. A regression checks both first
and later months and rejection of the erroneous first-month value. Only the
new audit implementation changes; no historical source, weights, background,
donors, split, event, tolerance, eligibility, cohort, numerical guard, primary
screen or budget changes.

Information exposed before the stop: compilation produced 3,013 terms in about
0.064 seconds; the first record failed its encoding check. No successful
per-record reduction, event comparison, timing aggregate or headroom decision
existed. No verification file or financial outcome was used. The legacy trusted
predictor bundle was deserialized; it contains outcome objects that are not used
by this audit. This is not a claim that such objects were never present in memory.

The corrected attempt uses a new output directory
`outputs/method_pivot/round6/structural_amended/`; the failed attempt is not
overwritten. This amendment and corrected harness are committed before rerun.
