# Method pivot: evidence ledger

Audit date: **2026-10-04**, Asia/Ho_Chi_Minh. Base actually inspected:
`95773c74f757be9b99afbccb48bcd9d17fd305d7`; fetched `origin/main` matched.
Work branch: `research/method-pivot`. No merge into main is authorized.

This is a controlled research audit, not a replacement for
[FINAL_RESEARCH_RESULTS](../FINAL_RESEARCH_RESULTS.md). Literature and algebra
can reject a proposed novelty claim; they cannot establish that an unrun model
loses empirically. A negative result applies to its tested target and protocol.

## Evidence levels

| Item | Reported result | Checked in this pivot | Scope |
|---|---|---|---|
| Group-top2, MCAR30, raw probability shift <= .02 | Taiwan 42/727; Polish 85/552 | Existing `region_B_intervals.csv` rows and file hash | Denominator is prediction-stable **and explanation-eligible**, not all customers |
| Completion evidence vs prediction-only detector | Group AP .6926 vs .1447 Taiwan; .8580 vs .2622 Polish | Existing detector aggregate table; audit receipt generated below | Does not exhaust all uncertainty/learning methods |
| MC vs rank | No general MC advantage | Existing paired-difference table | Both need completion explanations; rank does not remove shared SHAP cost |
| Stable-Core | Lower reason failure, no general coverage superiority at registered budgets | Existing aggregate reason counts/coverage | A meaningful-positive reason failure is not the historical top-k revision event |
| One-field acquisition | Strongest zero-query policy reaches retained-candidate customer ceiling | Existing elementary-bound table | Not a rejection of all acquisition objectives, budgets or policies |
| Conformal envelope | Novelty gate NO-GO | Read audit and reduction | **No envelope experiment**; not an empirical failure |
| TabM / Freddie | NOT RUN | Historical execution status; no files supplied by user | No inferred architecture or mortgage result |

All historical artifact checks are **read-only rechecks of existing aggregate
outputs**, not regenerated predictions, refitted models or independently replicated
statistics. They are present locally under ignored `outputs/`. Their hashes,
shapes and selected values are recorded by `scripts/audit_method_pivot.py` when
called with `--historical`. Missing files in another checkout are reported as
unavailable, never replaced with invented results. This pivot does not certify
every row-level cache or re-run the previous 232-value consolidation audit.

## Material read and relevant implementation

Read README and the final results/decision, novelty reassessment, conformal audit,
SOTA comparison, revision metric audit, Stable-Core results, verification headroom,
MC failure analysis and release policy validation. Relevant contracts inspected:

- `experiments/revision_study.py`: training-only fitting; 21,000 development-pool
  customers, three outer folds, exclusion of the original 9,000 test/reserve.
- `reliability/donors.py`: observed-value nearest donors, finite K, training reference.
- `reliability/conditional_completion.py`: training-only per-field conditional
  forests; not a guaranteed compatible joint distribution; natural NaNs restored
  to unknown at the output, distinct from artificial verification cells.
- `reliability/reason_sets.py`: current evidence separate from restored labels;
  meaningful and ranked targets distinct.
- `configs/revision_study.yaml`, `configs/verification_headroom.yaml`, existing
  temporal/decisive study conventions and `pyproject.toml`.

No historical predictor, explanation definition, grouping, calibration, threshold,
split or configuration is changed. No defect in those inspected contracts is
being asserted. Cross-explainer differences and donor-model limitations are
scientific limitations already documented, not silently repaired bugs.

## Four different sources of uncertainty/error

1. **Reducible estimation/computation error:** too few donors, a poorly estimated
   conditional distribution, finite optimization or integration error.
2. **Irreducible partial-information uncertainty:** two hidden completions have
   positive conditional probability and different outcomes/model responses.
   Knowing their distribution does not identify which completion is this person.
3. **Legitimate updating:** verified repayment history can change risk and reasons.
   Enforcing identical predictions before and after revelation would erase evidence.
4. **Pipeline/assumption sensitivity:** arbitrary reference, imputation support,
   rank ties, incompatible conditionals or a changed observation process.

For a fixed observed state w and two feasible hidden states h1,h2 with responses
p1 != p2, any single output a satisfies
`max(|a-p1|, |a-p2|) >= |p1-p2|/2`. This elementary bound is an illustration,
**not a new theorem**. It does not prevent calibrated conditional prediction.

The repository's Taiwan case 599 (all eight sampled PAY_0 values 0; verified value
2) motivates completion-support error. It does not establish that a new training
objective fixes it. Perfect integration of a wrong completion law can still be wrong.

## Claims that the pivot must not inherit

- One weak revision selector does not disprove learning of conditional outcomes,
  latent states or different explanation targets.
- One-field retained-reason headroom failure does not disprove multi-field value
  of information or discovering new reasons.
- Standard conformal containment is not empirical reason-level failure control.
- Additive observed-attribution invariance is not a correctness certificate.
- Same semantic names across predictors do not imply the same verified explanation.
- Inspected Taiwan and Polish records remain development evidence. New masks,
  seeds or folds do not erase historical selection. Exact Polish duplicates are
  identifiable; independent company identities are not.

## Data and compute boundary

Freshly checked official UCI pages on the audit date: [Taiwan](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
DOI `10.24432/C55S3H`, 30,000 records, 23 predictors, CC BY 4.0;
[Polish](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
DOI `10.24432/C5F600`, CC BY 4.0, selected fifth-year file 5,910 statements,
410 one-year bankruptcies, natural missingness. The landing-page headline count
is not the selected-file count. Attribution/change notices are required by the
license; repository policy remains no raw/processed rows or model artifacts committed.

No new dataset was downloaded. **No confirmation dataset is selected in this
NO-GO audit.** Freddie access/use is UNRESOLVED for a new run here: no supplied
official files or newly completed terms audit; previous chat conclusions are not
substituted for authorization. Taiwan/Polish are legally usable development
benchmarks, not new independent confirmation.

Live runtime: Apple M4 Pro, 24 GiB; Python 3.11.16, PyTorch 2.14.1,
MPS available, CUDA unavailable. The only new numerical work is deterministic
finite-state CPU checking. No training/GPU/attribution budget is consumed.
