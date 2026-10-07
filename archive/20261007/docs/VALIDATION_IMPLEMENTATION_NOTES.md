# Validation implementation incidents

## 2026-10-03 — report script compile check, before external evaluation

`python -m py_compile scripts/report_decisive_validation.py` caught invalid Python
conditional starred-tuple syntax in `ratio_ci`. The report script had never run;
no metric, figure or decision was produced by it. Scientific stages were paused
before external assessment. This note was written before correcting the syntax.
Correction: explicitly assign the two interval endpoints, then return the tuple.
No changes to frozen sources, predictors, masks, groups, events, scores, calibration
families or thresholds. Compile and a synthetic interval check must pass before
report generation. This is an unexecuted reporting implementation error, not an
error discovered in historical research results.

## Post-freeze calibration-assumption audit

The Polish split correctly keeps identical feature records together; release
calibration nevertheless contains 590 statements but 585 distinct feature
clusters (five duplicate rows). The predeclared binomial threshold procedure
counts statements, while the evaluation bootstrap clusters exact duplicates.
Consequently its iid Bernoulli interpretation is not established for Polish.
This is an assumption limitation of the frozen policy, not evidence of independent
company-level certification. No thresholds, methods or splits are changed after
external assessment. Report the conservative procedure as a sensitivity rule and
its empirical external coverage/risk, **not a finite-sample corporate-level risk
guarantee**. Any later confirmatory policy must calibrate at the independent
entity/cluster level prospectively. Also, company identity beyond exact duplicate
features is unavailable. The other duplicate counts are train35, selector-train4,
default-calibration2, revision-calibration1 and assessment13.

## Supplementary report compile check

The first invocation of the new supplement script stopped at parsing: a missing
closing parenthesis in its manifest writer. Nothing in the script executed and
no supplemental result was written. Recorded before adding the parenthesis and
rerunning compile. Frozen scientific sources/artifacts were unaffected.
