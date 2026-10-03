"""Generate the public aggregate report; never publish customer-level artifacts."""
from pathlib import Path
import json
from hashlib import sha256
import numpy as np
import pandas as pd
from financepaper.experiments.decisive_validation import validate_hashes


root=Path('outputs/verification_headroom')
out=root/'analysis'
receipt=json.loads((out/'report_receipt.json').read_text());validate_hashes(receipt['artifacts'])
audit=json.loads((root/'audit_receipt.json').read_text())
assert audit['status']=='PASS'
f=pd.read_csv(out/'metrics.csv');p=pd.read_csv(out/'paired_query_gain.csv')
b=pd.read_csv(out/'elementary_bounds.csv')
decision=json.loads((out/'decision.json').read_text())


def markdown(headers, rows):
    escaped=lambda row:[str(value).replace('|','\\|') for value in row]
    return '\n'.join(['| '+' | '.join(escaped(headers))+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(escaped(row))+' |' for row in rows]])


def pct(x):return f'{100*x:.2f}%'


def interval(row,metric):
    return f'{pct(row[metric])} [{pct(row[metric+"_lo"])}, {pct(row[metric+"_hi"])}]'


main=[];oracle=[];matched=[];decoupling=[];runtime=[]
methods=['zero_release_all','zero_top1','zero_rank','zero_stable_donor',
         'oracle_rank_customers_b1','oracle_stable_donor_customers_b1']
for _,r in f[(f.target=='meaningful')&(f.alpha==.1)&f.method.isin(methods)].iterrows():
    main.append([r.condition,r.method,f'{int(r.failures)}/{int(r.reasons)}',interval(r,'risk'),
                 interval(r,'customer_ge1'),pct(r.reason_coverage),pct(r.query_fraction)])
for _,r in p[(p.objective=='customers')&(p.metric=='customer_ge1')].iterrows():
    oracle.append([r.condition,r.target,pct(r.alpha),r.method,
                   f'{100*r.difference:.2f} [{100*r.lo:.2f}, {100*r.hi:.2f}]'])
for _,r in f[(f.alpha==.1)&f.method.str.startswith('oracle_stable_donor_reasons_b1')].iterrows():
    matched.append([r.condition,r.target,r.method.replace('oracle_stable_donor_reasons_b1','selected'),
                    interval(r,'risk'),int(r.reasons)])
for c in ['mcar10','mcar30']:
    with np.load(root/c/'states.npz') as z:initial=z['candidate'][:500].copy()
    with np.load(root/c/'verification_only.npz') as z:
        survival=z['meaningful'];shift=z['probability_shift']
    failure=(initial&~survival).any(1)
    for cutoff in [.01,.02,.05]:
        stable=shift<=cutoff
        # Include customers without an eligible query or candidate; same denominator
        # as the registered 500-customer cohort, not the query-order diagnostic subset.
        decoupling.append([c,cutoff,int(stable.sum()),int((stable&failure).sum()),
                           pct((stable&failure).sum()/stable.sum())])
    r=json.loads((root/c/'runtime.json').read_text())
    runtime.append([c,r['states'],r['eligible_queries'],r['attributed_rows']+r['verification_attributed_rows'],
                    f"{r['completion_seconds']:.2f}",f"{r['attribution_seconds']:.2f}",f"{r['total_seconds']:.2f}"])
freeze_hash=sha256((root/'protocol_freeze.json').read_bytes()).hexdigest()
text=f'''# One-query verification headroom: development results

Measured 2026-10-04 (Asia/Ho_Chi_Minh). **{decision['decision']}: stop the proposed
acquisition-method development at its first gate.** A query can recover some
coverage withheld by donor Stable-Core, but the strongest zero-query baselines
already cover every customer with an initial candidate at the declared budgets.
This result does not invalidate prediction–explanation decoupling; it invalidates
the proposed customer-coverage justification for this particular next method.

## What actually ran

The [prospective protocol](VERIFICATION_HEADROOM_PROTOCOL.md), source, masks,
model and unchanged calibrated thresholds were frozen before new query outcomes.
Polish: **500 distinct exact-feature clusters**, reused selector-development data,
MCAR10/MCAR30, 1,000 episodes. Frozen XGBoost, seven semantic groups, historical
raw-logit interventional TreeSHAP, donor K8. Every possible actual single-field
reveal plus STOP: **12,537 query actions, 13,537 states**. No predictor or release
threshold was refitted. Source complete truth is confined to evaluator code;
natural missing values were never restored. The donor pool is training-only.

Taiwan has **no new run**: all 21,000 train/development-pool records are in the
union of the three previously evaluated outer prediction folds. Excluding all
historical assessment IDs leaves zero eligible records. We did not open the old
test/reserve or label a previously evaluated fold as fresh development.
Freddie is **NOT RUN**, because official files have not been supplied.

Freeze timestamp: `2026-10-03T17:14:42.181990+00:00`.
Freeze SHA256: `{freeze_hash}`.
Cohort/masks SHA256: `6fc0d47d9919cf30814a3ebc3389ae82a795f05d9ac2a098984a803efa3670fc`.
These are local reproduction artifacts, not an independent preregistration.

## Primary target and the decisive zero-query comparison

Keep the initial candidate reasons and initially observed attribution summands
fixed. A reason survives when its fully restored grouped contribution remains
above .01 raw logit. The primary risk is failed reasons / released reasons.
No requirement to display two reasons, no causal/correctness claim.

At alpha=10%, the table includes 95% customer-bootstrap intervals. Oracle rows
use true verification labels for action selection and are unattainable bounds.

{markdown(['Condition','Method','Failed / released','Reason risk [95% CI]','Customer >=1 [95% CI]','Reason coverage','Query fraction'],main)}

The elementary retained-claim ceiling is **93.4% for MCAR10 and 95.0% for
MCAR30**: the remaining customers have no initial candidate to preserve.
Top-1 reaches that ceiling without querying, with primary risk 0.43% and 2.53%.
Its upper bootstrap endpoints are 1.08% and 4.19%; these are empirical intervals,
not deployment certificates. Thus the maximum extra customer coverage over this
baseline is **zero at all three declared budgets (5/10/15%)** in this sample.
The same elementary customer bound applies to the ranked sensitivity here.

Top-1 releases fewer reasons, so it is not a reason-coverage dominator. At 10/15%
primary budgets, however, release-all also meets empirical risk and attains
**100% reason coverage**. At MCAR30/5%, release-all fails the budget (5.53%);
do not hide that exception or call pooled success environment-wise control.
Querying may trade computation against number/quality of reasons there, but
the predeclared +5-point customer-coverage hypothesis still fails.

## Separating query value from oracle knowledge

Both b=0 and b=1 oracles know verification labels and may abstain. They must use
the terminal mapping's entire output for their chosen state; no oracle deletion
of individual failing reasons is allowed. Binary optimization proved optimality
for all **96** finite-family solutions. These are exact within the enumerated
action/terminal family, not globally optimal deployable acquisition policies.

The table reports b=1 minus b=0 customer coverage, in **percentage points**.
Intervals resample customers while keeping selected oracle actions fixed; they
are optimistic and do not cover optimization uncertainty.

{markdown(['Condition','Target','Alpha','Terminal rule','Query gain pp [conditional 95% CI]'],oracle)}

For meaningful reasons, donor Stable-Core gains only **0.2 pp MCAR10** and
**1.6 pp MCAR30**, below the predeclared 5 pp requirement. Its best customer
coverage remains below the no-query top-1 ceiling. At 10/15% the rank terminal
chooses STOP throughout; there is no coverage or reason-count headroom over
release-all. Ranked-target gains cannot replace the failed primary endpoint.

## Matched-size controls and legitimate changes

For the reason-count-maximizing b=1 donor Stable-Core oracle, strength/random
controls release exactly the same number of reasons for each customer at the
same selected state. Their actions are inherited from the oracle, so these are
identity-selection diagnostics, not deployable acquisition competitors.

{markdown(['Condition','Target','Selection','Failure [95% CI]','Released count'],matched)}

Some identity-selection benefit survives matched counts; it does not establish
coverage superiority or a novel acquisition algorithm. The optimizer maximizes
count subject to alpha, not minimum failure; querying can increase failure
while staying inside the empirical constraint. Do not call it uniformly safer.

Across candidate actions, 23/500 MCAR10 and 58/500 MCAR30 customers have an
initially nonpositive group become positive. These new claims are excluded from
primary gains. Counting them as preserved reasons would change the estimand.
Retractions of initially positive but ultimately failing claims are legitimate.
The query-order file retains these diagnostics and full selected-action arrays
locally. Initial own-SHAP importance/variance, realized prediction change and
rank-support change were compared with realized valid-candidate count. The
utility is mostly tied under this fixed universe (only 4/25 customers have
nonconstant utility versus own-importance at MCAR10/30); no ranking-superiority
claim is supportable. Expected entropy VOI, EDDI-style information gain and a
trained policy were **not implemented**, because the headroom gate failed.

## The phenomenon still appears in development

All 500 customers per condition are included below, including episodes with no
query available. Failure means at least one initially meaningful positive group
loses that property after full artificial restoration. This is not the historical
strict feature-top-3 event, nor the ranked target.

{markdown(['Condition','Prediction shift cutoff','Prediction-stable N','Region B N','P(reason failure | stable prediction)'],decoupling)}

This distinguishes two claims: some reasons fail despite a stable prediction,
yet a simple strongest-reason output can already provide broad useful coverage.
The former does not entail the need for the proposed acquisition algorithm.
These reused development results are not additional untouched external evidence.

## Compute, checks and artifacts

{markdown(['Condition','States incl. STOP','Actual query options','Attributed rows incl. restoration','Completion seconds','State attribution seconds','Total seconds'],runtime)}

Total approximately **66.90 seconds** on local CPU for evidence generation.
The 122,833 attributed rows include 1,000 full-restoration rows. This is one
end-to-end audit timing, **not** a repeated-latency benchmark; it does not estimate
the much more expensive proposed 8x8 hypothetical lookahead inference cost.
TreeSHAP dominates these timings. No GPU was required.

The independent checker verified source hashes, cohort disjointness, exhaustive
query coverage, natural-mask preservation, fixed attribution universe, all 96
oracle solutions/risk constraints, matched-set counts during reporting, and 24
independently recomputed attribution states. No mismatch was found. **121 tests
passed**, including brute-force comparisons of the optimizer with small exact
fixtures. The first unrestricted-thread full-suite run hung inside an existing
PyTorch/OpenMP sqrt call and was terminated; the complete suite passed with
OMP/MKL/VECLIB threads set to one. The financial runner already fixes these
variables, so no experimental definition/result was changed to address this
test-runtime issue.

```bash
uv run --frozen --extra temporal python scripts/run_verification_headroom.py freeze
uv run --frozen --extra temporal python scripts/run_verification_headroom.py evaluate
uv run --frozen --extra temporal python scripts/run_verification_headroom.py report
uv run --frozen --extra temporal python scripts/audit_verification_headroom.py
uv run --frozen --extra temporal python scripts/write_verification_headroom_report.py
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
  uv run --frozen --extra temporal pytest -q
```

Freeze/evaluate refuse overwriting completed studies. Local aggregate CSVs:
`analysis/metrics.csv`, `paired_query_gain.csv`, `elementary_bounds.csv`,
`solver_receipts.csv`; customer diagnostics remain in ignored `outputs/`.
`analysis/headroom.png` and `.pdf` compare primary alpha=10% coverage/failure.
Audit/analysis receipts include artifact hashes. No raw, processed rows, model
artifacts, or customer-level outputs are committed.

## Research and novelty decision

**NO-GO for one-query retained-reason acquisition as the next method contribution.**
Do not tighten alpha, change meaningful to ranked, add new reason candidates,
or drop top-1 to create apparent headroom. Do not implement the full nested
lookahead or a conditional-imputer acquisition grid after this failed gate.

The [nearest-work reassessment](NOVELTY_REASSESSMENT_2026.md) remains in force:
selective attribution, SHAP uncertainty, and explanation-guided acquisition
already exist; retargeting standard value of information is not automatically
new. This audit adds a useful negative result and a reason-coverage ceiling
diagnostic, **not enough evidence for a new algorithm paper**. There is still an
evaluation-paper story about verification targets, predictive/explanatory
decoupling, semantic grouping and simple policy baselines, with narrower claims
than a novel safe-explanation method. ESWA remains **promising but incomplete
for an evaluation paper; unsupported as a new-method claim**.

**One next action:** prepare an evaluation-paper manuscript outline with an
explicit claim–evidence table incorporating the negative Stable-Core and
acquisition results. Do not manufacture another method to satisfy a novelty
label. Large-scale confirmation remains unavailable until official data arrive.
'''
Path('docs/VERIFICATION_HEADROOM_RESULTS.md').write_text(text)
print('Wrote docs/VERIFICATION_HEADROOM_RESULTS.md from validated aggregate artifacts')
