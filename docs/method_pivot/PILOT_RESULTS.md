# Method-pivot results

Method decision: NO-GO before implementation. Financial training/confirmation:
**NOT RUN**. The prospective finite-state audit is recorded in
[PREREGISTERED_PILOT](PREREGISTERED_PILOT.md); results will be populated only after
actual execution. Historical artifacts are not new model results.

## Initial audit failure, before rerun

The first audit at `d212733` stopped when the **new checker** interpreted the
historical CSV's `conservative` field as `False/True`; the actual encoding is
`0/1`. This is a parsing bug in the new audit, not a change to the historical
policy or its results. Added explicit validated boolean parsing and regression
fixtures for both encodings. Generator, protocol and mathematical checks are
unchanged. The initial test-log launcher also failed because the failed audit
had not yet created its output directory; no test had run at that point.
Failure receipt: local `outputs/method_pivot/audit_initial_failure.log`.
