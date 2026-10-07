# Decision-value pilot: frozen audit and preregistration draft

Version 1, 2026-10-07. F0 is an exploratory artifact reanalysis; F1/F2 are software
preparation and a protocol draft. HUMAN STUDY NOT RUN. No new predictor, policy
fit, SHAP computation or assessment-based model selection is authorized here.
Governing instructions: `prompts/ASTRA_NEXT_RUN.md` and `docs/NEXT_EXPERIMENTS.md`.

## F0 estimand and fixed analysis

Use frozen `outputs/stable_core_study` outer artifacts, the historical empirical
alpha=.10 operating point and meaningful-positive reason target. All initially
observed members are held fixed. A candidate has current grouped attribution
>.01 raw logit; it fails if its restored attribution is <=.01. These are model
quantities, not economic truth. No threshold is reselected from these outcomes.

Keep release-all, strongest top-1, mask warning, whole-MC1/2/all, rank, donor,
conditional and both-family Stable-Core. Top-1 selects the strongest eligible
current reason, retaining cases with only one candidate. Historical whole-MC2
requires two candidates: its coverage denominator still contains *all* cases.
Mask warning releases the same candidates as release-all and adds disclosure;
there is no assumed human response to it. Rank here is the historical
rank-support release rule, not a rerun of revision-detection AP.

Compute counts by dataset/condition and descriptive overall mixture: distinct
records, customer-condition cases, independent resampling clusters, candidate,
released, valid and failed reasons, coverage >=1 and >=2 with all-case denominators.
Whole-explanation **any meaningful failure among released reasons** goes in a
separate table; it is not historical grouped top-2 revision and is not substituted
for that target. Historical revision AP/rates are REPORTED only in this stage.

Utility is `valid - r*failed`, `r=[0,.5,1,2,5,10]`, b=1. Report totals and utility
per customer-condition case, not per unique person; all conditions remain paired.
Compute/delay prices are UNKNOWN, never assumed measured zero or converted from
milliseconds to money. Display the frontier in (failed, valid) space and all
grid maximizers, including ties. This is descriptive, not a deployment selector.
Break-even against release-all is lost valid reasons / avoided failures when
the denominator is positive. Preserve negative findings and zero-release risk
as undefined. Check the reported 893/505 and 254/261 against row-level counts.

Use 1,000 percentile bootstrap draws, seed 20261007, resampling Taiwan customers
within each historical disjoint fold; resample Polish exact-feature clusters.
All masks and all policies for a sampled cluster share its multiplicity. Report
paired utility differences against release-all. These intervals condition on
already fitted policies and do not include fitting uncertainty. Polish clusters
are not verified company identities; no population guarantee is implied.

Hash-check assessment and calibration receipts and every input used; reconstruct
release masks from frozen policies *before* loading restored labels; recompute
meaningful labels and reconcile historical count tables. A missing/hash-drifted
artifact stops the stage. Recover at historical commit 95773c7 in an isolated
worktree if needed; never synthesize customer rows from aggregate reports.
Historical runtimes are carried as REPORTED seconds per 16-case batch, five
repetitions on the first 16 fold-0 MCAR30 records only. They are contextual values,
not timings for every condition or for all cases. No newly measured online latency
claim is made. This run separately
records actual CPU audit elapsed time, stage JSON logs and tqdm ETA.

## F1 synthetic task contract (requires expert validation)

The prepared bundle is a **synthetic ledger task**, not UCI customer vignettes or
external banking validation. This gives a checkable answer without inventing
professional judgments on benchmark records. One unique ledger has resource A,
context B/C and requirement H. H belongs to the stated set {0,1}; when hidden,
the current record shows null. The task asks whether the available record
supports the statement A>=H, with choices `supported` or `review`.

The independent rubric is logical entailment: answer supported exactly when
A>=max(H consistent with current information), otherwise review, including
both contradicted and insufficiently supported statements. It depends only on
A and observed/missing H. Neither risk score, SHAP, completion counts, restored
outcomes nor the intervention defines correctness. Full records are stored
privately; an answer about *available evidence* can differ from the hidden
statement's eventual truth. Expert review must establish whether this task is
useful and understandable; its business validity is currently NOT ESTABLISHED.

For a transparent software stress fixture, the fixed simulated risk score is
sigmoid(a*(1-h)+b*h+c), with missing H imputed to zero. Exact zero-reference
Shapley contributions of observed groups A/B/C are a*(1-h/2), b*h/2 and c.
H's contribution is omitted when hidden; displayed terms need not sum to score.
This analytic example is not a newly trained classifier or novel method.
Eight hypothetical H values are either balanced 0/1 or all zero, intentionally
allowing confidently wrong completion evidence. This is a biased synthetic
sampler, not a data-calibrated probability distribution.

Four arms hold the question, available record and current prediction fixed:

1. Risk score only (no model explanation).
2. Risk plus neutral numerical group contributions.
3. Arm 2 plus a generic missing-information warning.
4. Arm 3 plus fixed-mapping completion support counts and an incompleteness caveat.

The common candidate statement is part of the decision question even in arm 1;
it is not a claim generated from positive SHAP. Text never infers "low income"
or "poor payment" from a sign. Arm 4 is more informative and longer; word counts
are exported, and equal effort/time is NOT established. Before collection, a
separate usability pilot must fix layout, reading window and overload checks;
do not conceal this length difference as a matched-effort experiment.

Sample four cases from each mutually exclusive fixture stratum: MC false negative
first, then near-tie (second/third current contribution gap <=.02), then revised,
then stable. Revision checks initial positive top-2 membership/sign against full
information, allowing 1e-6 ties; it is private computational metadata only, not
the primary answer. Sampling is seeded uniform within each stratum, with exact
inclusion probability 4/pool-stratum-size. The balanced sample cannot estimate
population prevalence. No case is duplicated across ledger A/B/C values.

Prepare 16 allocation *slots*, not participants, in a randomized balanced rotation:
each slot sees every selected case once, one quarter in each arm; each case appears
equally in all arms across slots. Randomize order independently for each slot.
Only that slot's JSONL is shown to its assigned reviewer. Never distribute the
whole expert directory to one reviewer. No model identity, arm number, computational
stratum, MC/rank label, restoration score or private answer appears in those files.
The exporter takes a typed current-only object and explicitly projects allowed
fields. Private evaluation/assignment files stay outside the expert directory.
Raw policy scores/selection labels are not shown; the intended aggregate counts
in arm 4 are its declared intervention. Tests poison private information and check
it cannot alter public rendering; preserve full/current separation on future import.

## F2 preregistration draft — not registered and not open for recruitment

Primary hypothesis: arm 4 improves **decision accuracy against the independently
locked rubric** over arm 3. Primary estimand: marginal accuracy difference on the
locked case distribution, not default prediction, confidence or financial return.
Secondary endpoints: decision time, calibration of self-reported confidence,
review requests and expert rubric agreement. Comparisons with arms 1/2 and strata
are secondary; use Holm adjustment within that secondary family. No Stable-Core
secondary arm is added in this draft.

Proposed confirmatory analysis: logistic mixed model with fixed arm and stratum,
crossed reviewer and case intercepts and arm slopes where identifiable; report
the marginal 4-minus-3 accuracy difference and its 95% interval. A locked fallback
and a two-way reviewer/case bootstrap sensitivity analysis must be specified and
validated in simulation before registration. Individual ratings are not independent.
The [lme4 documentation](https://lme4.github.io/lme4/reference/lme4-package.html)
describes support for GLMMs and crossed random effects; no such human model has
been fitted in this run.

Power status: NOT RUN for the intended crossed design. Supervisor/partner must
first choose the smallest meaningful accuracy gain and acceptable time cost;
none is estimated from these software fixtures or historical SHAP counts.
Before registration, simulate the exact assignment and planned model at that gain
and zero effect, varying baseline accuracy, reviewer/case heterogeneity, random
slopes and missingness; use >=1,000 replicates per scenario, report Monte Carlo
intervals, convergence failures, null rejection rate and achieved power. Select
reviewer and case counts for two-sided alpha .05, power >=.80 and an agreed
precision target; stress-test effect-size and attrition assumptions. A draft
scenario grid (not evidence-based inputs): baseline .60/.75, gains .05/.10,
reviewer/case logit SD .3/.7, dropout 0/.10. These placeholders must be replaced
or accepted explicitly before simulation is treated as sample-size justification.
The 16 slots are a software test layout, not a powered study or sample-size claim.

Before any collection: experts validate rubric and relevance blind to policy
scores/restored outcomes, sign a versioned rubric; supervisor fixes SESOI, costs,
design and power assumptions; obtain institutional consent/ethics review as
applicable; preregister and hash protocol, cases, UI and allocation. Use separate
usability cases/reviewers, excluded entirely from confirmatory data. Do not tune
policy or case thresholds using confirmatory judgments.

Precollection exclusions: no consent/ineligible expertise/duplicate participation
or documented rendering failure; do not exclude low accuracy, slow answers or
unfavorable arm effects. All randomized participants remain in the accounting;
record missing/withdrawn ratings by arm. Prespecify MAR model-based handling plus
worst-case sensitivity for missing outcomes; no invented replacements or AI ratings.
Do not stop on observed significance: stop at preregistered reviewer AND case
targets or a documented ethics/technical stop. Keep failed arms and null results.

Gate: continue only with accuracy benefit over generic warning exceeding the
agreed meaningful margin at acceptable time burden. Confidence alone does not
pass. If experts reject task relevance, retain only algorithmic sensitivity
claims; no threshold rescue on assessment. No people/approval means software
and protocol only, HUMAN STUDY NOT RUN.

## Execution and publication

From a lockfile-compatible environment (CPU):

```bash
uv sync --frozen --extra temporal
uv run --frozen --extra temporal python scripts/run_decision_value_pilot.py --output runs/decision_value_pilot/NEW_RUN
# An interrupted run can continue; input/code/config drift is refused:
uv run --frozen --extra temporal python scripts/run_decision_value_pilot.py --output runs/decision_value_pilot/NEW_RUN --resume
```

Completed stage receipts protect outputs by SHA256. Incomplete stages rerun;
bootstrap and assignment seeds make them deterministic. `--stop-after taiwan`
supports testing a real partial-run resume. Logs, NPZ, row data, case bundles,
empty rating forms and models remain ignored/local. Commit code, tests, this
protocol, aggregate CSV/figures and one aggregate RESULTS report only. Do not
merge main. Source archives and historical configs/results remain unchanged.

Data attribution (official pages checked 2026-10-07): Yeh, I. (2009),
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
DOI 10.24432/C55S3H; Tomczak, S. (2016),
[Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
DOI 10.24432/C5F600. UCI lists CC BY 4.0; this run publishes derived aggregate
audit results with attribution, not raw rows. No new terms were accepted.
