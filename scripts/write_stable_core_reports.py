"""Render aggregate reports from the frozen study; never fit/select methods."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path("outputs/stable_core_study")
A=ROOT/"analysis"


def table(frame,columns,percent=()):
    lines=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for _,row in frame.iterrows():
        cells=[]
        for column in columns:
            value=row[column]
            if pd.isna(value):cells.append("N/A")
            elif column in percent:cells.append(f"{100*value:.2f}%")
            elif isinstance(value,(float,np.floating)):cells.append(f"{value:.4f}")
            else:cells.append(str(value))
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)


def main():
    m=pd.read_csv(A/"metrics.csv");d=pd.read_csv(A/"paired_differences.csv")
    primary=m[(m.target=="meaningful")&(m.alpha==.1)&(m.conservative==0)&(m.condition=="overall")]
    ranked=m[(m.target=="ranked")&(m.alpha==.1)&(m.conservative==0)&(m.condition=="overall")]
    methods=["release_all","whole_legacy2","whole_mc1","whole_mc_all","rank","variance","frequency","strength",
             "lower_quantile","stable_donor","stable_conditional","stable_both"]
    cols=["dataset","method","reason_risk","reason_coverage","customer_ge1","customer_ge2","mean_reasons"]
    pct=["reason_risk","reason_coverage","customer_ge1","customer_ge2"]
    selected=primary[primary.method=="stable_both"]
    ci=selected[["dataset","reason_risk","reason_risk_lo","reason_risk_hi","customer_ge1","customer_ge1_lo","customer_ge1_hi"]]
    paired=d[(d.alpha==.1)&(d.method=="stable_both")&(d.metric=="reason_risk")&d.baseline.str.contains("matched")]
    paired=paired.assign(difference_pp=paired.difference*100,lo_pp=paired.lo*100,hi_pp=paired.hi*100)
    common=pd.read_csv(A/"common_eligibility.csv")
    common=common[(common.subset=="at_least_two_candidates")&common.method.isin(["whole_legacy2","rank","stable_both"])]
    runtime=pd.read_csv(A/"runtime_summary.csv")
    pred=pd.read_csv(A/"prediction_metrics.csv")
    pred=pred[pred.probability=="calibrated"].groupby(["dataset","condition"])[["average_precision","roc_auc","brier","log_loss"]].mean().reset_index()
    conservative=m[(m.conservative==1)&(m.method!="release_all")&(~m.method.str.contains("matched"))]
    assert not (conservative.released_reasons>0).any()
    text="""# Stable-Core v2: measured results and falsification

Measured 2026-10-03. **No general coverage superiority or algorithmic novelty is
supported.** Variable-size stable reason sets are useful as a conservative
operational option, but they do not replace the strongest simple policies.

## Provenance, target and denominators

Protocol/source/input hashes preceded the run; all conditional models and release
policies were frozen before new assessment computation. XGBoost 25-view predictors,
preprocessing, semantic groups, masks and TreeSHAP references stayed unchanged.
Taiwan: 2,400 distinct assessment customers across three disjoint 800-customer
folds, four conditions, 9,600 customer-condition records. Polish: 1,182 assessment
statements, two conditions, 2,364 records; exact duplicate-feature clusters are
the bootstrap unit. These datasets were inspected previously: **exploratory
extension**, not independent confirmation. Freddie: **NOT RUN**, no official files.

Primary reason failure means restored observed-group contribution <=0.01 in raw
logit units. Ranked sensitivity additionally requires top-2 membership. Neither
is causal correctness, model correctness or expert validation. Current candidates
are **all** available groups with contribution >0.01; output size is variable.
These new reason-level definitions do not overwrite the historical whole-event.

Tables aggregate an equal mixture of declared environments, with reason-weighted
failure ratios. Customer coverage includes customers with no candidate reasons.
The 10% values below are calibration budgets, not guarantees. Tables for every
alpha .05/.10/.15, environment and conservative policy are in generated CSVs.

## Main target: meaningful positive reasons, 10% budget

"""+table(primary[primary.method.isin(methods)],cols,pct)+"""

Release-all already achieves 2.56%/3.90% aggregate reason risk with greater coverage
than Stable-Core on Taiwan/Polish. Thus the primary claim of greater coverage at
the same 10% or 15% budget **fails against this simple baseline**. The same endpoint
is substantially easier than preserving a top-k ranking. At 5%, Polish MCAR30
release-all and several empirically calibrated simple policies reach **5.58%**:
pooled success must not be mistaken for environment-specific risk control.

Stable-Core both retains 93.64%/92.75% of candidate reasons and reaches
90.82%/92.64% customer coverage with 0.286%/0.243% reason failure. It trades some
coverage and compute for much lower observed failure; this is a useful point,
not dominance at the declared budgets. No new sub-5% budget was tuned afterward.

95% paired customer-cluster bootstrap intervals (1,000 draws):

"""+table(ci,list(ci.columns),list(ci.columns[1:]))+"""

## Ranked sensitivity, 10% budget

"""+table(ranked[ranked.method.isin(methods)],cols,pct)+"""

Stable-Core both improves coverage relative to the exactly-two-reason whole policy,
but **rank-only and whole top-1 reach more customers**. Rank-only also releases
more reasons while satisfying the 10% empirical budget in each assessed environment.
It is not displaced by the more elaborate Stable-Core rule. Stable-Core has lower
observed reason risk than rank-only; their utilities differ, so neither claim
universal domination nor hide the strong baseline.

At the 10% operating point, every meaningful-target Stable-Core fit selects
`(tau_sign,tau_rank,q)=(0,0,.01)`: it reduces exactly to a positive lower-quantile
filter. Every ranked fit selects `(0,.5,.01)`. The extra sign-support condition is
inactive. This is direct empirical evidence against novelty of the compound rule,
not a reason to invent a more complex rule after assessment.

Crucial limitation: 28.15% of Taiwan and 42.52% of Polish current candidate reasons
already lie outside top-2 **before** verification. Ranked release-all failure is
therefore not a pure revision rate. It is a ranked-claim acceptance diagnostic.
Do not cite the 29.37%/43.73% ranked release-all failures as the original decoupling
phenomenon. Historical within-model top-2 revision results remain the reference.

## Eligibility and shorter-explanation controls

The common subset below requires >=2 current candidates, removing the eligibility
advantage from letting previously ineligible one-reason customers receive output:

"""+table(common,["dataset","target","method","n","customer_coverage","reason_risk"],["customer_coverage","reason_risk"])+"""

On the meaningful target, whole top-2 covers 100% of this common subset; Stable-Core
both covers 99.74%/99.62%. Hence its apparent aggregate customer gain over exactly
top-2 is largely candidate eligibility, not improved abstention. On ranked Polish
it retains a smaller real gain over whole top-2, but rank-only still covers 100%.

Paired differences in reason risk: Stable-Core both minus a control with **exactly
the same count on every same customer**; negative is better. Values below are
percentage-point differences, not relative percentage changes:

"""+table(paired,["dataset","target","baseline","difference_pp","lo_pp","hi_pp"])+"""

For meaningful reasons, the reductions versus matched strength are 0.47 pp on
Taiwan (95% CI 0.37–0.60 pp reduction) and 0.68 pp on Polish (0.48–0.92 pp).
Thus some **reason identity selection benefit survives matching set size**.
This does not establish superiority to optimally calibrated rank/frequency policies.
Size-matched controls are diagnostics, not lower-cost deployed alternatives: their
sizes are taken from the Stable-Core policy without consulting verification truth.

## Multiple completion families and risk-control limits

Donor-only -> both changes meaningful reason risk from 0.53% to 0.29% on Taiwan,
and 0.54% to 0.24% on Polish. Customer coverage decreases by about 0.57/0.42 pp.
Both-family agreement therefore does not collapse coverage in this study. The
family-specific policies were independently calibrated, so all disagreement/risk
tables must be read with their selected thresholds. Agreement is not worst-case
robustness: both completion families can share unsupported regions.

All **conservative finite-grid Hoeffding policies abstain completely**, at all
three budgets and both targets. The tested sample sizes and bound do not certify
useful coverage. This is a failure of the strong risk-control claim, not evidence
that empirical coverage is zero or that every possible risk bound would fail.
Do not replace the bound or threshold grid after seeing this result.

Stable-Core both releases 0–7 reasons on both datasets. Its meaningful-target
empty fractions are 9.18%/7.36%, including intrinsically candidate-empty records.
Customer any-released-reason failure is 0.65%/0.73%, distinct from micro reason
failure. Per-group tables, coverage>=2, count distributions and family disagreement
are all exported; no successful cases were selected for the headline.

## Unchanged predictive performance

These metrics describe the same frozen predictor on this explanation cohort;
Taiwan values are means over the three folds. They are not the larger historical
4,500-customer test or full mortgage performance. Every explanation policy shares
these predictions; none improves AP through a new risk model.

"""+table(pred,list(pred.columns))+"""

## Runtime

CPU end-to-end serving on the first 16 fixed MCAR30 customers, one warm-up,
five interleaved repetitions, including current TreeSHAP, completion generation,
completion TreeSHAP and policy scoring. Model loading/fitting excluded. Seconds
are per batch, not an individual request SLA:

"""+table(runtime,["dataset","method","total_seconds_mean","total_seconds_std"])+"""

Both-family Stable-Core is approximately 3.5x/10.1x donor rank-only cost on
Taiwan/Polish. Donor Stable-Core and donor rank-only have essentially the same
attribution cost. Strength-only needs current SHAP but no completion attributions.
No cached-scoring versus uncached-inference comparison is used.

## Figures and reproducibility

Generated under `outputs/stable_core_study/analysis/`, each PNG plus PDF:

- A_reason_risk_coverage: declared operating points, no test-selected threshold.
- B_customer_risk_coverage: >=1-reason coverage versus verified risk.
- C_whole_vs_partial: whole/top-1/rank/partial comparison.
- D_reason_set_sizes: variable cardinality including zero.
- E_failure_by_group: all semantic groups, no group selection.
- F_completion_families: donor, conditional, both.

`metrics.csv` includes all risk budgets, conservative results and per-condition
95% CIs; `paired_differences.csv` uses the same customer resamples. Additional
outputs contain class-wise prediction metrics, common eligibility, pre-existing
rank exclusions, completion truth support and five timing repetitions. Raw/cached
row outputs, figures and models remain ignored; this report is an aggregate summary.

The software audit recomputed **2,009,952 customer-policy decisions** and
**861,408 matched-count comparisons** over 11,964 customer-condition records,
with zero observed/natural cell changes and zero invalid attributions. These are
checks, not independent samples. Test suite: **111 passed**, three pre-existing
SHAP/matplotlib deprecation warnings. Root-run software checks are not peer review.

Decision: retain partial reasons as an **optional operational policy**, reject a
new-algorithm/superior-coverage claim, and do not replace the whole-policy baseline.
The independent mortgage confirmation remains blocked by missing official files.
"""
    Path("docs/STABLE_CORE_RESULTS.md").write_text(text)

    decision="""# Stable-Core v2: final decision

**Option 2: operational strategy, not a demonstrated algorithmic contribution.**
The [measured results](STABLE_CORE_RESULTS.md) reject broad coverage superiority
against strong simple baselines. Keep the implementation and negative evidence;
do not tune another threshold family on these assessment results.

1. **Better than whole abstention?** Sometimes compared with an exactly-top-2
   policy, but not against the strongest whole/top-1/rank controls. Much of the
   aggregate benefit is newly allowing one-reason candidates. On common >=2
   eligibility, meaningful-target coverage is slightly lower than whole top-2.
2. **Useful coverage?** Yes empirically: both-family sets explain 90.82% of Taiwan
   and 92.64% of Polish customer-condition cases; reason coverage 93.64%/92.75%.
3. **Multiple completion models?** Yes in these bounded experiments: agreement
   lowers observed false stability with modest coverage loss. Both can still be
   misspecified; this is not a mathematical worst-case guarantee.
4. **Large external validation?** UNKNOWN / NOT RUN. No official Freddie files
   exist locally. No mortgage efficacy, runtime, transfer or risk-control claim.
5. **Beats rank instability?** NO for the declared fixed-budget coverage objective.
   Rank-only provides greater coverage under the ranked target. Stable-Core
   exchanges coverage for lower failure; no universal winner is established.
6. **More than fewer reasons?** YES for reason identity: at identical per-customer
   set sizes, both-family selection reduces meaningful reason failure versus
   strength matching by 0.47/0.68 percentage points (paired CIs exclude zero).
7. **Defensible algorithmic novelty?** NOT ESTABLISHED. Stable subsets, uncertainty
   filtering and calibrated set-valued decisions have close prior work. Do not
   claim a new generic uncertainty or conformal algorithm. At the 10% primary
   operating point the fitted rule reduces to `q10 > .01`; sign/rank thresholds
   are both zero. The compound rule does not demonstrate a distinct mechanism.
8. **Applied-policy value?** YES, with limits: a tested variable-size output,
   verification target, paired baselines and explicit cost/risk tradeoffs strengthen
   the evaluation paper. Primary meaningful failure is already low enough that
   release-all satisfies the 10/15% aggregate budgets; report that plainly.
9. **Inference output?** Risk raw/calibrated; stable and withheld candidate groups;
   per-family sign/top-2 frequencies and magnitude quantiles; NONE/PARTIAL/
   ALL_CANDIDATES; target, declared budget, calibration scope and thresholds. Never
   show verification truth or an individual-customer safety guarantee.
10. **Replace whole release?** NO. Retain it and top-1/rank/frequency controls.
    Offer stable subsets as a conservative operating option, with the reason-level
    target and lack of guarantee visible to the downstream user.

## Critical limitation

Every conservative finite-grid bound produced empty sets. Useful coverage here
comes from empirical calibration, and Polish MCAR30 has a 5% budget violation
for several simple policies. These results do not support a deployment risk
guarantee. Ranked release-all counts also include candidates already below top-2;
do not misreport them as pure post-verification revision.

## Next action

Acquire the exact officially licensed Freddie Release 47 files and run the
already declared temporal confirmation with the v2 policy and strong baselines
frozen. No further Taiwan/Polish method search is justified by this result.
"""
    Path("docs/STABLE_CORE_DECISION.md").write_text(decision)


if __name__=="__main__":main()
