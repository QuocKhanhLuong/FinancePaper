# F1 review kit — actual software audit, no expert or human evidence

Run `review_20261007T154931Z`, created `2026-10-07T15:49:32.006099+00:00`, CPU / Python
`3.11.16`. Base `eb6689c13bcc95321144f3477d0bf1280931214d`; branch
`research/decision-value-review-kit-20261007`. This is a new preparation stage, not a rerun of F0.

## VERIFIED / REPRODUCED

- Verified the original vignette stage hashes. Existing cases, labels, assignment
  and F0 outputs were read-only; no new predictor or policy was fitted.
- Independently enumerated all 24 feature orders for each of
  490 current/full vector comparisons across
  245 synthetic pool cases. Maximum absolute difference
  from the closed-form observed-group contributions: 4.44e-16.
- Enumerated feasible H values independently of `independent_answer`: all
  245 pool labels agree with the stated entailment rule.
- Checked 16 selected cases and
  256 public renderings against the frozen
  current-only renderer and allocation. No repeated case per slot.
- Exported one self-contained HTML preview from `slot_000.jsonl` only, with
  strict field validation and no private-label input to the exporter. No answers
  or human timings are collected. Browser/test QA is recorded separately in
  `validation_receipt.json`; creating HTML alone is not browser validation.
- Stage logs, actual elapsed time, tqdm/ETA, seeds/hashes and resume receipts
  remain in the ignored run directory. The original bundle remains unchanged.

## Deterministic baseline result — not simulated human performance

| cohort | baseline | n_cases | correct | accuracy | evidence |
| --- | --- | --- | --- | --- | --- |
| synthetic_pool | always_review | 245 | 126 | 0.5142857142857142 | deterministic_software_check_not_human_accuracy |
| synthetic_pool | always_supported | 245 | 119 | 0.4857142857142857 | deterministic_software_check_not_human_accuracy |
| synthetic_pool | current_record_rule | 245 | 245 | 1.0 | deterministic_software_check_not_human_accuracy |
| selected_fixture | always_review | 16 | 9 | 0.5625 | deterministic_software_check_not_human_accuracy |
| selected_fixture | always_supported | 16 | 7 | 0.4375 | deterministic_software_check_not_human_accuracy |
| selected_fixture | current_record_rule | 16 | 16 | 1.0 | deterministic_software_check_not_human_accuracy |

The current-record rule uses only A and H, already present in arm 1. Thus the
answer is entirely recoverable without a score, explanation or completion
distribution. The selected fixture has 7 supported / 9 review cases; this
balanced-by-computational-stratum sample is not population prevalence.

**Interpretation:** this fixture can test presentation/comprehension effects.
It cannot demonstrate new factual information supplied by uncertainty disclosure.
Human difficulty, ceiling effects and actual benefit remain unknown. Do not tune
the task to make the more complicated arm win after observing participant data.

## Visible information length

Variable visible text, whitespace-delimited words, excludes JSON schema and
shared page chrome; this differs deliberately from the earlier JSON-length proxy.

| arm | min | mean | max |
| --- | --- | --- | --- |
| 1 | 57 | 57.0 | 57 |
| 2 | 75 | 75.0 | 75 |
| 3 | 88 | 92.5 | 94 |
| 4 | 130 | 134.5 | 136 |

Arm 4 remains longer than arm 3. The fixed page layout does not prove equal
reading effort. No filler, new arm or revised case labels were introduced.

## REPORTED, PROPOSED and NOT RUN

F0 utility, historical model metrics and runtimes are **REPORTED from the prior
stage**, not recomputed here. The F1 answer is mathematically checkable, but its
professional relevance and the rubric have not been approved by an expert.

The new [review guide](https://github.com/QuocKhanhLuong/FinancePaper/blob/research/decision-value-review-kit-20261007/docs/DECISION_VALUE_REVIEW_GUIDE.md)
contains concrete relevance/rubric/effort questions. The reviewer preview stays
local at `runs/decision_value_pilot/review_20261007T154931Z/expert_preview/index.html`; it was not sent to anyone.

**HUMAN STUDY, expert approval, ethics/consent, power simulation, final SESOI and
sample-size justification: NOT RUN / PENDING.** No AI ratings replace people.
The fixture remains **UI/software preparation only** until its relevance gate
is resolved. No algorithm novelty, financial return or real-world utility claim.

**One next action:** supervisor/domain expert reviews the task and proposed
rubric using the local preview and guide, and decides whether to retain it for
a separately approved usability pilot.
