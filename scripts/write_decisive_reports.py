"""Render result tables with the explicitly bounded, evidence-reviewed narrative."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("outputs/decisive_validation/analysis")
DOC=Path("docs")


def table(frame):
    def cell(x):
        if isinstance(x,(float,np.floating)):
            return "undefined" if not np.isfinite(x) else f"{x:.4f}"
        return str(x).replace("|","/")
    return "| "+" | ".join(map(str,frame.columns))+" |\n| "+" | ".join(["---"]*len(frame.columns))+" |\n"+"\n".join("| "+" | ".join(cell(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))+"\n"


def interval(point,low,high):
    return "undefined" if not np.isfinite(point) else f"{100*point:.2f}% [{100*low:.2f}, {100*high:.2f}]"


def policy_table(frame):
    f=frame.copy()
    f["Coverage [95% CI]"]=[interval(a,b,c) for a,b,c in zip(f.coverage,f.coverage_low,f.coverage_high)]
    f["Revision [95% CI]"]=[interval(a,b,c) for a,b,c in zip(f.risk,f.risk_low,f.risk_high)]
    return table(f[["variant","alpha","family","condition","Coverage [95% CI]","Revision [95% CI]"]])


def main():
    metrics=pd.read_csv(ROOT/"B_detector_metrics.csv")
    policy=pd.read_csv(ROOT/"D_policy_intervals.csv")
    phenomenon=pd.read_csv(ROOT/"A_phenomenon_sensitivity.csv")
    pairs=pd.read_csv(ROOT/"paired_fold_mean_differences.csv")
    baseline=pd.read_csv(ROOT/"release_all_eligible.csv")
    regions=pd.read_csv(ROOT/"region_B_intervals.csv")
    timing=pd.read_csv(ROOT/"runtime_repetitions.csv")
    def metric_table(dataset,variant,methods,condition="mcar30"):
        f=metrics[(metrics.dataset==dataset)&(metrics.variant==variant)&(metrics.condition==condition)&metrics.method.isin(methods)&metrics.probability.eq("calibrated_score")]
        rows=[]
        for name in methods:
            g=f[f.method==name];row={"Detector":name}
            for col in ("roc_auc","average_precision","brier","log_loss"):
                row[col]=f"{g[col].mean():.4f} ± {g[col].std():.4f}" if dataset=="taiwan" else f"{g[col].mean():.4f}"
            rows.append(row)
        return table(pd.DataFrame(rows))
    provenance="""Measured 2026-10-03. Source/reference `04bf054`; all new method choices were
hashed before assessment. Taiwan: three frozen internal folds, 800 explanation
customers each, historically inspected benchmark. Polish: one frozen independent
1,182-statement assessment, corporate bankruptcy within one year. No external
method selection. All raw data, models, CSV/NPZ/JSONL and figures remain ignored
under `outputs/decisive_validation/`. Reproduce with README commands. Historical
reports are unchanged. Software audits are root-run, not independent peer review.
"""
    semantic="# Domain-grouped reason validation results\n\n"+provenance+"""
## Verdict

**Keep the phenomenon, narrow its magnitude and operational interpretation.**
Grouping removes much feature-level turnover, but Region B remains under every
reported group size. These are model-reason changes, not human-validated errors.
Groups/tolerances were specified in [DOMAIN_REASON_GROUPS.md](DOMAIN_REASON_GROUPS.md).
Signed observed-member aggregation is primary; whole-group sums are a separate
sensitivity including imputed contributions. No metric replaces the old event.

## Table A: XGBoost, raw probability shift <= .02

Rates below are fractions. Eligibility differs by definition; denominators must
not be suppressed. Taiwan includes MCAR10/30/MAR30; Polish only MCAR10/30.

"""
    f=phenomenon[phenomenon.model.isin(["xgb25","xgb"])&phenomenon.variant.isin(["feature3","group1","group2","group3"])&phenomenon.condition.isin(["mcar10","mcar30","mar30"])&phenomenon.cutoff.eq(.02)&phenomenon.tolerance.eq("event")]
    semantic+=table(f[["dataset","variant","condition","eligible","stable_n","B","revision_among_stable","overall_revision"]])
    semantic+="\n## Prediction cutoffs and confidence intervals\n\nAll .01/.02/.05 cutoffs are reported, not selected. Primary group top-2 illustration:\n\n"
    f=regions[regions.variant.eq("group2")&regions.probability_scale.eq("prob_shift")&regions.condition.isin(["mcar10","mcar30","mar30"])]
    semantic+=table(f[["dataset","condition","cutoff","stable_n","B","B_given_stable","low","high"]])
    semantic+="""
Full A/B/C/D counts and percentages for **all** variants/cutoffs/rank gaps are in
`analysis/A_phenomenon_sensitivity.csv`; `region_B_intervals.csv` adds clustered
95% intervals and the calibrated-probability sensitivity. Stable here is an
oracle restoration diagnostic, **not a deployable confidence or correctness label**.

## Rank-gap sensitivity, MCAR30, cutoff .02

The sign boundary remains 1e-6; only rank gap changes. Units are raw-logit
attributions. No normalization was introduced to make tolerances look favorable.

"""
    f=phenomenon[phenomenon.model.isin(["xgb25","xgb"])&phenomenon.variant.isin(["feature3","group1","group2","group3"])&phenomenon.condition.eq("mcar30")&phenomenon.cutoff.eq(.02)]
    semantic+=table(f[["dataset","variant","tolerance","overall_revision","B","revision_among_stable"]])
    semantic+="""
Taiwan grouped Region B becomes small under the widest .02 gap: 1.78%, 2.89%,
2.58% among stable eligible customers for group1/2/3, versus 8.01%, 10.87%, 11.04%
on Polish. Thus the original feature-level magnitude was substantially driven
by fine rankings. It did not collapse to zero, but “frequent serious errors” is
not established. The unvalidated financial/human importance of a small logit gap
must not be inferred from these percentages.

Whole-group top2 gives higher MCAR30 revision (Taiwan20.04%, Polish53.13%) because
it adds imputed/restored contributions. This is a different estimand, **not** the
preferred result. The observed subset is fixed at both endpoints.

## Common eligibility and analysis controls

"""
    common=pd.read_csv(ROOT/"common_eligibility_grouping.csv")
    semantic+=table(common[common.condition.eq("mcar30")])
    f=phenomenon[phenomenon.model.isin(["lr","additive","vanilla","mask_delta"])&phenomenon.cutoff.eq(.02)&phenomenon.tolerance.eq("event")&phenomenon.condition.eq("mcar30")]
    semantic+="\nSaved analysis controls, no refitting (absolute GRU/tree comparisons retain different explainer estimands):\n\n"+table(f[["dataset","model","variant","eligible","stable_n","B","overall_revision"]])
    semantic+="""
Additive controls' zero observed-reason revision is structural and does not
establish superior explanations. Group3 eligibility is only 36.7% on Taiwan
MCAR30; lower revision or selected coverage must be interpreted with that loss.
No timing groups, demographics or Polish ratios were regrouped after outcomes.

Figures: `analysis/01_decoupling.{png,pdf}`, `02_feature_group_revision`,
`06_external_regions`. A=both stable; B=prediction stable/reasons revised;
C=prediction shifts/reasons stable; D=both change.
"""
    (DOC/"DOMAIN_REASON_VALIDATION_RESULTS.md").write_text(semantic)

    kdoc="# Monte Carlo K ablation\n\n"+provenance+"""
## Fixed method and Table B

K1/2/4 are prefixes of the original K8 stream; K16 appends an independent eight
draws. Frozen predictor, same customers/masks/TreeSHAP background/current reasons.
The historical K8 scores were checked exactly in all old Taiwan environments.
Default calibration is unchanged. New revision calibration uses the declared
four-environment mixture, so Brier/log-loss need not equal the old five-environment
report (e.g. Taiwan MCAR30 MC8 Brier .0882 here versus .0867 historically).

Taiwan MCAR30, feature top3: fold mean ± SD, not three independent studies.

"""
    methods=["prediction_only","learned_selector","prediction_variance","missing_fraction","attribution_variance","rank_instability","sign_instability","mc8"]
    kdoc+=metric_table("taiwan","feature3",methods)
    kdoc+="\n## Table C: all K, MCAR30\n\nTaiwan feature top3:\n\n"+metric_table("taiwan","feature3",[f"mc{k}" for k in (1,2,4,8,16)])
    kdoc+="\nTaiwan group top2:\n\n"+metric_table("taiwan","group2",[f"mc{k}" for k in (1,2,4,8,16)])
    kdoc+="\nPolish feature top3 (one fixed assessment, no invented seed SD):\n\n"+metric_table("polish","feature3",[f"mc{k}" for k in (1,2,4,8,16)])
    kdoc+="\nPolish group top2:\n\n"+metric_table("polish","group2",[f"mc{k}" for k in (1,2,4,8,16)])
    kdoc+="""
All conditions, group1/2/3, raw/calibrated Brier and log loss are retained in
`B_detector_metrics.csv` / `C_K_ablation.csv`; the raw MC fraction is a completion
sensitivity estimate, not a calibrated posterior. At K1, sample prediction variance
is necessarily zero, so the prediction-variance comparator stays at its frozen K8
budget rather than pretending equal uncertainty information.

## Smallest useful K

The **predeclared** near-reference criterion was 95% of K16's AP gain above event
prevalence, averaged over Taiwan MCAR10/30/MAR30, within folds. Results:

"""
    gain=pd.read_csv(ROOT/"K_relative_gain.csv")
    kdoc+=table(gain.groupby("method").relative_gain.agg(["mean","min","max"]).reset_index())
    kdoc+="""
No cheaper K meets that criterion: K8 captures **93.18%**, K4 **80.85%**.
K16 is the smallest tested budget meeting its own reference criterion; that is
not evidence of convergence. K4 already detects useful signal, but “95% of the
benefit at a fraction of the cost” would be false here. Keep **K8 as the frozen
operational reference** and describe it as a budget compromise, not an optimum.
External performance does not select K. Increasing K also does not repair a
misspecified donor distribution, and Brier/log loss need not improve monotonically.

## End-to-end timing, five repetitions after one warm-up

Apple M4 Pro, 24 GiB unified memory, CPU single-thread inference. Persistent
model/background/explainer already loaded; includes transforms, current SHAP,
donors, completion predictions/SHAP, score and calibration/policy call. Excludes
disk loading. Repetitions measure timing variation on a fixed batch, not customer
population latency quantiles. Single-record results use one fixed diagnostic case.

"""
    tr=[]
    for size in (1,128):
        sub=timing[timing.batch_size==size];base=sub[sub["mode"].eq("1")].seconds.mean()
        for mode in ("1","2","4","8","16","learned"):
            g=sub[sub["mode"].eq(mode)]
            tr.append(dict(batch=size,method=mode,latency_ms=f"{g.seconds.mean()*1000:.3f} ± {g.seconds.std()*1000:.3f}",
                relative_K1=g.seconds.mean()/base,SHAP_calls=int(g.shap_calls.iloc[0]),prediction_calls=int(g.prediction_calls.iloc[0]),completion_calls=int(g.completion_calls.iloc[0])))
    kdoc+=table(pd.DataFrame(tr))
    kdoc+="\nTiming breakdown (batch128, seconds; each stage mean ± SD):\n\n"
    tr=[]
    for mode in ("1","2","4","8","16","learned"):
        g=timing[timing.batch_size.eq(128)&timing["mode"].eq(mode)];row={"method":mode}
        for stage in ("completion","transform","prediction","shap","scoring"):
            row[stage]=f"{g['seconds_'+stage].mean():.4f} ± {g['seconds_'+stage].std():.4f}"
        tr.append(row)
    kdoc+=table(pd.DataFrame(tr))
    kdoc+="""
Explicit batched SHAP calls are K+1 (current plus completions), completion sampler
calls one; K16 uses two eight-draw RNG blocks inside that call. TreeSHAP's own
additivity work is included in SHAP time. Most cost is SHAP, not the revision
head arithmetic. The generic selector still needs current SHAP and eight cheap
prediction draws. Its approximately .097 s/batch128 is a real deployment advantage
over MC4 .369 s and MC8 .648 s, with weaker detection. It is retained visibly.

Coverage at alpha .05/.10/.15 for **each** K is in `D_policy_intervals.csv` (pooled,
empirical and conservative; all customer-level CIs); no test-picked threshold or
oracle “coverage at risk” is reported as a deployable rule. Figure03 shows detection
and runtime; Figure04 uses whole-score ties for descriptive risk–coverage curves.

## Why the score works, and its limits

MC8 and rank instability are almost the same signal (Spearman .998 Taiwan and
.997 Polish feature MCAR30). Within-fold AP difference from rank instability on
Taiwan is .0028 [−.0010,.0082]; Polish feature difference is −.0002 [−.0024,.0019].
**No general superiority over rank instability is established.** Both need the
same completion explanations. MC adds substantial detection over scalar SHAP
variance/predictive uncertainty, not a novel uncertainty mechanism. See the
[failure audit](MONTE_CARLO_FAILURE_ANALYSIS.md).

Use `paired_fold_mean_differences.csv` for comparisons to fold-mean tables.
`paired_AP_differences.csv` is explicitly a pooled cross-fold score-ranking
diagnostic; differing fold calibrators can make its AP difference different.
Do not use that pooled difference to claim an algorithmic gain absent within folds.
"""
    (DOC/"MONTE_CARLO_K_ABLATION.md").write_text(kdoc)

    pdoc="# Release policy validation\n\n"+provenance+"""
## Decision and assumptions

**Robust-environment calibration is the most defensible declared stress policy,
not a distribution-free or mechanism-specific deployment guarantee.** One common
threshold is fit against every simulated calibration environment and is used at
inference without a mechanism label. Pooled calibration is the reference; observable
strata use total missing fraction <=.15, (.15,.30], >.30 and never the hidden
MCAR/MAR mechanism. Small strata fail closed. Policies were fixed before each
assessment; alpha5/10/15% all remain visible.

Separate revision Platt pool and release pool. Empirical risk constraints and
simultaneous finite-grid binomial-upper-bound rules are reported separately.
The grid is unchanged 0,.01,...,1; ties are not outcome-sorted. Overall means
equal environment mixture, excluding complete cases, unlike the historical
five-condition pooled report. Historical thresholds were not overwritten.

Intervals are 1000 paired customer bootstrap draws, stratified by Taiwan outer
fold; Polish resamples exact-feature duplicate clusters. These are conditional
on trained models and fitted thresholds, not uncertainty over repeated studies.
Multiple masks of one customer are never independently resampled. Zero releases
have **undefined** risk. Bounds across environments use Bonferroni correction;
no independence between environments is required, but independent customers within
an environment are an assumption.

Polish release calibration has 590 statements/585 exact-feature clusters and no
company IDs. The frozen binomial rule counts statements. Therefore even its
“conservative” label is a procedure description, **not corporate-level certification**.
See [implementation/assumption audit](VALIDATION_IMPLEMENTATION_NOTES.md).

## Table D: main MC8 feature-top3 policies, all risk targets

Conservative-rule results; coverage denominator includes all sampled customers.

"""
    for ds in ("taiwan","polish"):
        pdoc+=f"\n### {ds}\n\n"+policy_table(policy[policy.dataset.eq(ds)&policy.method.eq("mc8")&policy.variant.eq("feature3")&policy.conservative])
    pdoc+="\n## Grouped top2 at 10% target\n\n"
    for ds in ("taiwan","polish"):
        pdoc+=f"\n### {ds}\n\n"+policy_table(policy[policy.dataset.eq(ds)&policy.method.eq("mc8")&policy.variant.eq("group2")&policy.alpha.eq(.1)&policy.conservative])
    pdoc+="""
Grouped top1/top3, all grouped alpha targets, all K and detector controls are in
`analysis/D_policy_intervals.csv`; this includes every empirical/conservative
result, not just successful thresholds. `paired_policy_differences.csv` gives
paired robust-minus-pooled/stratified differences. Plot05 shows alpha10% with
undefined/no-release conditions labelled explicitly.

## What failed

The empirical pooled rule misses environment targets: feature MCAR30 alpha10%
gives Taiwan10.78% and Polish10.08%; Polish grouped top2 reaches12.43%. These are
observations, not proof of a guarantee violation since no such guarantee exists.
Even empirical robust Polish feature MCAR30 is10.08%, slightly above10%.
At alpha5%, conservative robust feature policies release **nothing** on either
dataset; Polish grouped top2 also releases nothing. This operating point is not
practical with the present calibration sample sizes and coarse MC scores.
Observable-stratified conservative feature policy releases nothing on Polish
MCAR30 even at10%; conditioning fragments the data and can lose most coverage.
Robust is more cautious, not uniformly Pareto-superior to pooled calibration.

## Necessary baseline: release all currently eligible reasons

No MC computation or learned threshold. MCAR30 only below; full environments in
`release_all_eligible.csv`. These are observed descriptive rates, not calibrated rules.

"""
    f=baseline[baseline.condition.eq("mcar30")].copy()
    f["coverage [CI]"]=[interval(a,b,c) for a,b,c in zip(f.coverage,f.coverage_low,f.coverage_high)]
    f["revision [CI]"]=[interval(a,b,c) for a,b,c in zip(f.risk,f.risk_low,f.risk_high)]
    pdoc+=table(f[["dataset","variant","coverage [CI]","revision [CI]"]])
    pdoc+="""
Taiwan group1/group2 already have low unconditional revision. At a10% target,
it would be misleading to credit MC for that baseline semantic aggregation effect.
Polish grouped top2 is more decisive: unselected78.34% coverage/19.01% revision,
versus robust conservative58.21% coverage/2.18% revision (95% CI1.17–3.29%) under
MCAR30. This demonstrates a useful tradeoff; it does not certify unobserved shifts
or establish an optimal policy. The current method is useful at10/15% in the
declared environments; a universal5% operational claim is rejected.
"""
    (DOC/"RELEASE_POLICY_VALIDATION.md").write_text(pdoc)

    edoc="# External validation results — Polish corporate bankruptcy\n\n"+provenance+"""
## Provenance and independent assessment

[Official UCI source](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
[DOI10.24432/C5F600](https://doi.org/10.24432/C5F600), CC BY4.0, creator Sebastian
Tomczak. Download/license verified before use. `5year.arff` is 5,910 statements,
64 numerical ratios, 410 bankruptcy positives, 5,500 negatives, **one-year horizon**.
The UCI aggregate header is not this file's schema. ARFF SHA256:
`cb3f6f250ac46bd8d18e9a222f489fe8ee3e396fcec18959f5a0ef8e8169b2fc`.
Archive SHA256: `17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9`.

4,666 natural missing cells, 2,879 records with natural missingness, 3,031 complete
records, no infinite values. Attr37 is missing in2,548 statements. There are60
duplicate-feature rows; all exact duplicates were grouped before splitting.
Partitions (rows/positives): train2961/205; selector590/36; default-cal294/17;
revision-cal293/20; release-cal590/40; assessment1182/92. Assessment has627 originally
complete and555 naturally incomplete statements. No company ID means residual
identity or temporal overlap cannot be ruled out. No five-file pooling.

Donors are the1,544 complete **predictor-training** statements, never the target
or another partition. Only artificial MCAR10/30 cells have verifiable truth.
Natural unknowns remain NaN in partial, completion and restored raw records and
are handled by the frozen train-median transform. There is no MAR external test.

The [protocol](EXTERNAL_DATASET_PROTOCOL.md), all group assignments, K candidates,
model/selector recipes and calibration families were frozen before the external
assessment marker. No method or threshold was changed afterward. One external
split supports transfer of the phenomenon, not a general consumer-default claim.

## Table E: predictive controls, separate target

Fixed .5 classification threshold; raw and calibrated results are both retained.
The main question is explanation revision, not external classifier superiority.

"""
    pm=pd.read_csv(ROOT/"E_external_prediction.csv")
    edoc+=table(pm[["model","condition","probability","roc_auc","average_precision","recall","f1","brier","log_loss","ece"]])
    edoc+="""
Post-hoc default calibration **worsens** XGB Brier/log loss in all three conditions:
MCAR30 Brier .0499 raw -> .0523 calibrated. Calibration used only294 statements
with17 positives and was not refit after this result. No calibration-improvement
claim is made. Classwise precision/recall/F1/support are in E_external_prediction.csv.

## Transfer of decoupling, raw |delta p| <= .02

"""
    f=phenomenon[phenomenon.dataset.eq("polish")&phenomenon.model.eq("xgb")&phenomenon.variant.isin(["feature3","group1","group2","group3"])&phenomenon.cutoff.eq(.02)&phenomenon.tolerance.eq("event")&phenomenon.condition.ne("complete")]
    edoc+=table(f[["variant","condition","eligible","A","B","C","D","revision_among_stable","overall_revision"]])
    edoc+="""
All three cutoffs and rank gaps are in the [group validation report](DOMAIN_REASON_VALIDATION_RESULTS.md).
At the widest .02 logit rank gap, grouped top2 MCAR30 still has60/552 stable cases
with revision (10.87%). The coarse financial groups attenuate, but do not erase,
the external phenomenon. Group2 MCAR30 RegionB at the frozen event is15.40%
[12.30,18.41] among stable eligible statements.

## Detectors, MCAR30

Feature-top3 target:

"""+metric_table("polish","feature3",methods)+"\nGroup-top2 target:\n\n"+metric_table("polish","group2",methods)
    edoc+="\nPaired AP/AUROC differences (MC8 minus prediction-only), 1000 duplicate-cluster draws:\n\n"
    edoc+=table(pairs[pairs.dataset.eq("polish")&pairs.condition.eq("mcar30")&pairs.second.eq("prediction_only")&pairs.variant.isin(["feature3","group1","group2","group3"])][["variant","metric","difference","low","high"]])
    edoc+="""
Rank instability ties MC on the feature target; this is an important simple
completion-explanation comparator, not a negative result to hide. The learned
selector remains weaker. Grouped selector results test a **fixed feature-target
selector recalibrated** to grouped labels, not a newly trained group-optimal selector.

## Natural-missingness sensitivity

"""+table(pd.read_csv(ROOT/"external_natural_missingness_strata.csv").query("condition == 'mcar30'"))
    edoc+="""
Natural unknowns do not eliminate RegionB: grouped top2 is43/311 (13.83%) in
originally complete and42/241 (17.43%) in naturally incomplete stable statements.
This is observational stratification, not a causal missingness effect. Restoration
is partial: it returns to the original natural-missing record, not all-information
truth. Complete-case donors may be a selected population, and shared financial
ratio constraints can be broken by mixing query and donor subsets. Neither is
repaired after assessment.

## External success/failure gate

All four bounded transfer criteria are met: RegionB persists; prediction-only
detection is weaker; MC discriminates revision; useful selected coverage remains
at10/15% declared budgets. This is **qualified transfer**, not universal risk control:
MCAR only, one split/target, dictionary-grounded groups without human validation,
calibration deterioration for default, no useful robust5% coverage, and no
independent-company calibration guarantee. Taiwanese and Polish prediction metrics
are never pooled. Figure06 presents the two targets separately.
"""
    (DOC/"EXTERNAL_VALIDATION_RESULTS.md").write_text(edoc)

    failure="# Completion Monte Carlo: mechanism and failure audit\n\n"+provenance+"""
## What is actually measured

Correlation and fixed-missing-count detection are in `MC_correlations.csv` and
`fixed_missing_count_detection.csv`. MC is an operational any-reason revision
frequency; rank instability is the average fraction of reasons exiting. Their
near-equivalence is expected when most events involve one rank exit. Scalar
attribution variance can be large far from any decision boundary, while a small
near-tie shift can change top-k membership; target alignment explains the large
gain over variance. This is a descriptive mechanism, not causal identification.

The current score does not prove correct reasons. Rank-instability AP is
statistically indistinguishable from MC in Taiwan feature/group2 and Polish
feature-level comparisons. Polish group2 gives a small MC advantage; do not
generalize it into algorithmic novelty. Group-conditioned means do not identify
causal importance because multiple groups may be missing together.

## Support and validity

Donor IDs were independently checked against predictor-training pools. No target,
hidden query truth or assessment row enters donor selection. Observed values and
natural NaNs were checked at every draw. Taiwan categorical values come from
valid training donor codes; numeric completed values come from actual train rows,
not extrapolation. This ensures marginal validity, **not accounting-consistent
joint query/donor ratios or a calibrated conditional distribution**. Eight draws
typically use about7.0–7.2 distinct donors; diversity alone is not coverage.

At K8 MCAR30, numerical truth is inside the sampled min/max for81.02% of Taiwan
artificial numeric cells; true category appears for94.74% of categorical cells.
Polish numeric inclusion is75.09%. These are descriptive empirical supports,
not confidence intervals; cell counts are not independent-sample inference.
Truth-percentile values and all support checks are exported in a
**verification-only** table. Natural unknowns never contribute to this audit.

## Failure bins, not release decisions

High MC>=.5 with no verified feature revision: Taiwan MCAR30 n91; Polish n58.
Low MC<=.125 with revision: Taiwan n64; Polish n26. Definitions were fixed before
outcomes; they are not optimized classification thresholds. Near-tie fraction
(current third/fourth gap<=.01) is about56%/31% for Taiwan/Polish high-score false
positives, so **not all** false positives are ties. For low-score false negatives,
sampled truth support is lower (Taiwan about79%, Polish66%); consistent with missed
hidden regions, but not proof of the source of every error. No donor tuning follows.

## Deterministic representative cases inspected

The export contains first-five record IDs in each declared case bin/condition/fold,
280 cases total. These examples are the lowest IDs for each MCAR30 type, not
hand-picked successes. IDs are dataset row audit indices, not identities.

| Dataset / row | Type | Raw probability partial -> restored | MC8 | Inspection |
|---|---|---|---|---|
| Taiwan55, fold2 | High MC, no revision | .619932 -> .620146 | .75 | Third/fourth margin .00464; displayed scores barely move; all true hidden fields supported by sampled range/category. Broad hypothetical changes need not occur in the one realized verification. |
| Taiwan599, fold0 | Low MC, revision | .155584 -> .655144 | 0 | True hidden September PAY_0=2, all eight donor PAY_0=0. Missed critical category despite support for six of seven hidden fields and eight distinct donors. |
| Taiwan162, fold0 | Stable prediction, revised reasons | .366477 -> .349182 | .625 | BILL_AMT1 is an observed reason whose contribution/ranking changes when other bills are restored; delta p .01730. |
| Taiwan76, fold0 | Prediction shifts, stable reasons | .273560 -> .343218 | .125 | PAY_3, LIMIT_BAL, PAY_AMT6 stay positive/top3 despite delta p .06966. |
| Polish92 | High MC, no revision | .013761 -> .002898 | .75 | Margin .00610; true verification preserves Attr16/23/21 reasons despite many risky possible completions. |
| Polish442 | Low MC, revision | .237042 -> .289145 | .125 | True hidden net-profit/asset ratio −.26989 lies below sampled minimum −.22076; debt/equity-related reasons change rank. Seven distinct donors do not cover that region. |
| Polish2 | Stable prediction, revised reasons | .016104 -> .012474 | 1 | Delta p .00363; Attr10 rank changes near a .00528 boundary. Illustrates why semantic/tolerance checks matter. |
| Polish24 | Prediction shifts, stable reasons | .568735 -> .356223 | 0 | Attr41/39/35 remain the reason set despite delta p .21251; prediction and reason stability differ in both directions. |

Full before/restored scores, hidden fields, true values, sampled extrema and
donor diversity are in `representative_cases.csv`. Observed case associations
do not establish a causal explanation for the financial outcome.

## Audit receipt

Independent root recomputation checked66 cache batches,43,965 current rows,
703,440 completion rows,295,365 event rows and6,891,264 policy-release decisions.
No invalid current, restored or completion attributions were found; donor/source
provenance and artificial-only restoration passed. All results retain the original
frozen TreeSHAP estimand. Cross-explainer invariance is **not** established; prior
GRU IG/reference/permutation sensitivity remains a limitation in the historical
metric audit, not new confirmation here.
"""
    (DOC/"MONTE_CARLO_FAILURE_ANALYSIS.md").write_text(failure)
    print("Five measured report documents rendered; final decision is separately authored.")


if __name__=="__main__":
    main()
