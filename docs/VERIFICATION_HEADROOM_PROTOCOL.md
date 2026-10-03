# One-query verification headroom audit

Prospectively specified 2026-10-04. This is the first gate in
[the acquisition plan](BUDGETED_REASON_VERIFICATION_PLAN.md), not a learned
acquisition policy or a confirmation experiment. No predictor is retrained.
Freeze this document, code, config, model, historical release mappings, exact
customer IDs and masks before evaluating any new query outcomes.

## Cohort gate

The union of the three historical Taiwan outer prediction cohorts contains all
21,000 records in the train/development pool. Each fold's `predictions.csv`
confirms 7,000 evaluated customers. Excluding **all** assessment IDs leaves zero
eligible Taiwan development records, not merely fewer explanation cases.
Taiwan therefore has **no new run in this audit**. Do not relabel another fold's
assessment customers as fresh development data or open the reserved old test.

Use the existing Polish `selector_train` partition, 590 statements in 586
exact-feature clusters. Exclude any cluster in historical outer assessment,
predictor training, default calibration or release calibration. Choose the
smallest record ID per remaining cluster, then sample 500 clusters without
replacement with seed 20261004, sort their representative IDs. The cohort was
used in earlier selector development: explicitly exploratory development reuse,
not a new external confirmation. The frozen XGBoost was not fitted on it.
Labels play no role in sampling. No Freddie files are available; Freddie NOT RUN.

Use the existing MCAR generator with seed 20261004 on these sorted records:
10% and 30% use nested uniforms and hide originally observed cells only. Natural
unknowns remain NaN in every state. There are 1,000 customer-condition episodes,
500 independent exact-feature clusters. No new missingness family is introduced.

## Claims, actions and frozen mappings

Use the historical Polish XGBoost, preprocessing, probability calibration,
raw-logit interventional TreeSHAP background and seven semantic groups. Freeze
initial observed fields O0, initially meaningful positive groups (`phi > .01`),
and group availability. Sum only O0 fields at **every** state, including full
restoration. A query never adds its own attribution to the audited claims.

Primary survival means fully restored grouped attribution > .01. The separate
ranked sensitivity additionally requires the historical top-2 membership with
1e-6 tie tolerance. Do not substitute it as the primary outcome.

Enumerate STOP plus **every single artificially hidden field**, revealing only
that field's actual value in evaluator-only code. Compute the same point
explanation and recondition training-donor K8 completions on that value. Other
artificial hides remain unknown. Independent deterministic streams by condition
and fixed 128-state batch, in field-major order; never reuse an unconditional
completion cloud after a query.
All completion inputs exclude target labels and unrevealed true values.

For each state, retain only initially positive candidate groups that are still
positive in that state. Use existing empirical rank and donor Stable-Core
terminal mappings for alpha .05/.10/.15, loaded unchanged from the historical
independent release calibration. Log zero-query release-all, strongest top-1,
rank, whole-MC top-1/top-2/all, and donor Stable-Core. No threshold is refitted.
Reusing these mappings after acquisition **does not calibrate a deployment
trajectory**: the post-query policies and oracle are diagnostics only.

## Precisely what the oracle means

Each terminal rule produces a fixed set for each actual reveal action. An oracle
may choose STOP, one action, or abstain, knowing the verification labels. It
cannot edit the rule's reason set. On each environment separately, solve a
multiple-choice binary optimization with at most one set per customer and
`sum(failed reasons) <= alpha * sum(released reasons)`. Empty risk is N/A.
Report two lexicographic optima: (1) reason count, then customer count, then fewer
queries; (2) customer count, then reason count, then fewer queries. This is a
transparent tie break, not a combined research score. Require solver optimality.

Run the **same label-informed oracle with query budget zero**. An improvement
over an ordinary zero-query rule may arise from oracle knowledge alone;
the paired zero-oracle/one-oracle difference isolates available query actions
within this frozen terminal family. These are in-sample optimistic bounds, not
deployable performance or a global bound over every possible acquisition rule.

A stronger elementary bound also applies: a retained-claim policy can cover no
more customers than have an initial candidate, and can release no more than all
initial candidates. If release-all/strongest top-1 already reaches that customer
ceiling within a declared risk budget, a new retained-claim query policy cannot
improve customer coverage at that budget. This bound must not be omitted.

## Metrics and decision fixed before outcomes

Report reason failure, reason coverage, customers with >=1 and >=2 reasons,
mean count, query fraction; same-size strength/random controls at the chosen
state; initial candidates retracted after a query; and initially nonpositive
groups that become positive (excluded from primary gains). Preserve unit query
cost as a benchmark convention, not a monetary estimate. Log runtime and
attributed rows/calls. Report full-restoration prediction shift strata .01/.02/.05
only as evaluator diagnostics, never policy inputs. Record candidate-field
ordering from realized prediction change, initial own-SHAP importance/variance,
rank support improvement and realized verified-reason gain. The first and last
are explicitly hindsight diagnostics, **not** implementations of predictive
entropy VOI or EDDI information gain.

Use 1,000 paired bootstrap draws of the 500 customer clusters, carrying both
conditions together. Intervals on optimized oracle actions are conditional on
the selected actions and optimistic; they do not validate a learned policy or
cover optimization uncertainty. Give exact elementary coverage bounds beside
them. Do not resample queries, reasons or completions as customers.

**GO to acquisition-policy development only if** at a primary meaningful target
budget there is at least +5 absolute percentage points of customer coverage
headroom beyond a feasible zero-query baseline, query-enabled versus zero-query
oracle improvement is also at least +5 points, and the conditional paired
interval excludes zero. This is a predeclared feasibility screen, not an
economically validated benefit threshold. Report all three alphas and both
environments. A ranked-only win cannot rescue a primary failure. If the primary
gate fails, stop: do not implement the full 8x8 expected-value policy, perform a
conditional-imputer acquisition sweep, change alpha, add reasons or search a new
classifier to obtain novelty. Existing evaluation results remain valid within
their stated scope.

## Reproduction

Requires the local official Polish file and historical fitted artifacts. No
download is performed. Row-level arrays, manifests, figures and model artifacts
remain ignored under `outputs/verification_headroom/`.

```bash
uv run --frozen --extra temporal python scripts/run_verification_headroom.py freeze
uv run --frozen --extra temporal python scripts/run_verification_headroom.py evaluate
uv run --frozen --extra temporal python scripts/run_verification_headroom.py report
```
