# F1 review kit — expert review still pending

Continuation of `eb6689c`, 2026-10-07. This stage audits the synthetic fixture and
prepares a local, single-slot display for a supervisor/domain expert. It does not
recruit anyone, collect answers, approve a rubric or repeat F0. Original reports,
configs, policies and case selection are unchanged. HUMAN STUDY NOT RUN.

## What the preview contains

Open the generated `expert_preview/index.html` in a browser. It contains exactly
one preassigned version of each of the 16 synthetic cases, in the existing order.
Previous/Next and arrow keys navigate cases. There are no answer controls,
recording, telemetry, network requests or saved ratings. The two possible choices
are displayed as task instructions only. The interface retains English wording
from the frozen bundle; this guide is investigator-facing and may be discussed
in the expert's preferred language. Translation is not yet validated.

Only the selected slot JSONL enters the preview exporter. Full records, answers,
restored probabilities, arm numbers and computational strata are excluded by a
strict schema. The exporter escapes script delimiters and renders text with
`textContent`. Do not give one reviewer the whole original expert directory or
the private files. Navigation back within the same assigned version is allowed
in this design review; this is not a timed confirmatory assessment.

## The concrete decision for the expert

Judge whether the task measures a useful evidence-support decision and whether
the independent rubric is understandable, before reviewing computational scores.
The statement is "A covers H (A>=H)". H belongs to {0,1}; a missing value leaves
both possible. Mark supported only if the statement holds in every feasible
world; otherwise review. This is a proposed mathematical rubric, not a
professional lending rule or a rubric already approved by an expert.

The answer is fully determined by A and observed/missing H, already visible in
every arm. Therefore an exact rule solver succeeds without risk scores or
explanations. This is a *software check*, not measured human accuracy. A human
experiment here could study comprehension, attention or distraction due to
presentation. It cannot establish additional factual information from arm 4.
Do not claim actual human ceiling effects or no human benefit without data.

The expert should resolve these items in a dated review, separately from any
future participant dataset:

1. Is this evidence-support task relevant enough to justify a human pilot, or
   should it remain only a UI/software fixture?
2. Is universal entailment over {0,1} the appropriate independent rubric? Is
   "review" an acceptable common response to uncertainty and contradiction?
3. Are the score, numerical group contributions and completion frequencies
   understandable without being mistaken for observed facts or correctness?
4. Does the longer arm-4 display create a reading-effort confound? What common
   layout/reading window would be acceptable for a separate usability pilot?
5. If the task is retained, what smallest accuracy gain and maximum added time
   would matter? These are currently unset and cannot be inferred from SHAP.

No completed answers or expert signature are supplied by the agent. Review
status: **PENDING**. No person has been contacted. Any resulting rubric changes
must create a new version/new cases before data collection; never rewrite this
fixture's historical outputs or optimize against confirmatory judgments.

## Preparation and gate

The frozen primary comparison remains arm 4 versus arm 3. No new arm, policy
threshold or model is introduced. Visible-content length is audited separately
from the older serialized-JSON word proxy; fixed layout alone does not establish
equal cognitive effort. No extra filler text is added to force equal length.

Only after expert relevance/rubric review should the team finalize SESOI and
time cost, validate a crossed reviewer/case analysis in power simulations, obtain
appropriate consent/ethics clearance, and preregister before recruitment. The
current 16 slots are software allocation fixtures, not a sample-size justification.

## Reproduce locally

```bash
# Existing lockfile-compatible environment; this performs no fitting.
.venv/bin/python scripts/run_decision_review.py \
  --bundle-run runs/decision_value_pilot/20261007T151716Z \
  --output runs/decision_value_pilot/NEW_REVIEW_RUN --stop-after audit
.venv/bin/python scripts/run_decision_review.py \
  --bundle-run runs/decision_value_pilot/20261007T151716Z \
  --output runs/decision_value_pilot/NEW_REVIEW_RUN --resume
```

Run directories contain manifests, SHA256 stage receipts, progress JSONL, tqdm/ETA
and one RESULTS report. Resume verifies code/input/output hashes. Optional
`--publish reports/decision_value_review/NEW_STAGE` copies an explicit allowlist
of aggregates only; no case bundles, private labels or logs are published.
