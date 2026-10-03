# Final decision after the decisive validation stage

**Continue, with a narrowed evaluation + selective explanation policy paper.
ESWA assessment: promising but incomplete.** No new predictive architecture or
neural revision head is justified by this study. K8 remains the frozen reference,
not an externally selected optimum. No publication acceptance is promised.

Evidence measured 2026-10-03 against `04bf054`: three previously inspected Taiwan
internal folds, 2,400 distinct explanation customers; one independently frozen
Polish bankruptcy assessment, 1,182 statements/1,169 exact-feature clusters.
All models, groups, K candidates, event definitions and calibration families
were fixed before external outcomes. The original 9,000 Taiwan test/reserve
customers remain excluded. Historical result documents were not overwritten.

## Answers to the fifteen decision questions

1. **Does feature-level revision survive grouping?** Yes, at reduced magnitude.
   Taiwan MCAR30 revision is17.92% for feature top3, versus4.34%/8.18%/9.76% for
   observed group top1/2/3. Eligibility is92.08%/68.25%/36.71% for those grouped
   definitions; these are different denominators. Under a .02 logit rank gap,
   grouped revision remains2.53%/4.03%/4.77%. This rejects complete collapse but
   shows that fine rankings inflate the original magnitude. Whole-group sums,
   which include imputed members, are explicitly separate and are not substituted.

2. **How large is Region B?** At raw |delta p|<=.02 and MCAR30, Taiwan feature3
   has116/898 stable eligible customers (12.92%); grouped top1/2/3 has39/1011
   (3.86%),42/727 (5.78%),26/349 (7.45%). Group2 95% customer-bootstrap interval
   is4.14–7.51%. MAR30 group2 is35/814 (4.30%, CI2.95–5.69%). All .01/.02/.05
   cutoffs, A/B/C/D counts/percentages, calibrated-probability sensitivity and
   epsilon0/.005/.01/.02 appear in the [semantic report](DOMAIN_REASON_VALIDATION_RESULTS.md)
   and generated tables. Prediction stability is not prediction correctness.

3. **Is it reproduced externally?** Yes, for the bounded external financial-risk
   target. Polish MCAR30 feature3 RegionB is273/795 (34.34%); group1/2/3 is83/724
   (11.46%),85/552 (15.40%, CI12.30–18.41%),61/335 (18.21%). Even group2 with
   rank gap .02 has60/552 (10.87%). This is corporate bankruptcy within one year,
   not Taiwan consumer default or another independent consumer-credit cohort.
   One fixed split and MCAR10/30 do not establish general transfer across mechanisms.

4. **Is MC stronger than prediction uncertainty?** Yes on the tested targets.
   MCAR30 feature AP: Taiwan .7199 vs prediction-only .2192; Polish .9222 vs .4562.
   Group2 AP: Taiwan .6926 vs .1447; Polish .8580 vs .2622. Paired within-fold
   AP gains for group2 are .5479 [ .4646,.6230 ] and .5957 [ .5294,.6518 ].
   The standard HGB prediction-only detector includes probability, entropy,
   completion prediction variance and missing fraction. This does not exhaust
   every uncertainty method; it refutes their equivalence for these strong fixed
   controls. Generic learned selectors remain weaker, not removed from reports.

5. **Smallest useful K?** K1 already contains signal; K4 is an economical useful
   detector, but neither K4 nor K8 meets the prospectively declared near-reference
   criterion. Across Taiwan conditions/folds, K4 captures80.85% and K8 captures
   93.18% of K16's AP gain above event prevalence. Only K16 meets95% of that tested
   reference; it is not a convergence oracle. Retain K8 as the pre-existing budget
   compromise. Batch128 CPU latency: K4 .369±.0015 s, K8 .648±.0029 s,
   K16 1.205±.0044 s; generic selector .097±.0006 s. Five warm timing repetitions,
   no rented GPU, Apple M4 Pro. See [K ablation](MONTE_CARLO_K_ABLATION.md).

6. **Can risk be controlled at useful coverage?** Empirically, at10/15% in the
   declared environments. For grouped top2 MCAR30, conservative robust alpha10%
   gives Taiwan61.46% coverage/3.25% revision [2.32,4.20], and Polish58.21%/2.18%
   [1.17,3.29]. Feature3 yields39.88%/3.45% and44.67%/2.46%. At5%, robust feature
   policies release nothing in both datasets; Polish group2 also releases nothing.
   Thus a useful universal5% rule fails. Taiwan group1/group2 already have low
   unconditional revision: their release-all baseline must prevent attributing
   aggregation's benefit to MC. On Polish group2, release-all is78.34% coverage
   at19.01% revision, so selection provides a clearer tradeoff.

7. **Most defensible calibration policy?** Robust across the simulated calibration
   environments considered, with explicit coverage cost. One common threshold
   does not need a true mechanism label at inference. Observable stratification
   fragments the calibration sample and can reject nearly everything. Empirical
   pooled thresholds exceed condition-specific budgets, including12.43% at an
   intended10% for Polish group2 MCAR30. Conservative pooled is often higher
   coverage and performs acceptably here; robust is not uniformly better on
   all objectives. No distribution-free or unseen-mechanism guarantee is established.
   Polish's statement-level binomial rule also lacks independent-entity certification
   because duplicates/company identity are unresolved; bootstrap clusters duplicates.

8. **Main MC failure modes?** Near-tie false positives, plausible but unrealized
   completions, and missed hidden regions/categories. Example Taiwan row599:
   hidden PAY_0=2 while all eight donors have0; MC=0 despite revision and a large
   probability shift. Polish low-score revision cases have lower sampled truth
   support. Eight distinct donors do not ensure support of a consequential field.
   Numeric/categorical provenance is valid, but mixed query/donor financial ratios
   need not satisfy accounting identities. See [all four case types and audit](MONTE_CARLO_FAILURE_ANALYSIS.md).

9. **Does natural missingness change the conclusion?** RegionB persists both with
   and without it. Polish group2 MCAR30: originally complete43/311 stable cases
   revise (13.83%), naturally incomplete42/241 (17.43%). Natural NaNs never receive
   restoration truth; only artificial cells are restored. These are observational
   strata, not evidence that natural missingness causes greater revision. Complete-
   case training donors may be biased. The external endpoint is the original
   naturally incomplete record, not full information in an absolute sense.

10. **Dependence on explainer?** Still a material limitation. New primary Taiwan
    and Polish comparisons use the same raw-logit interventional TreeSHAP convention,
    within each frozen model/reference. Existing GRU controls use conditional IG;
    their absolute rates cannot rank explanation correctness against TreeSHAP.
    Historical alternate-IG/permutation sensitivity is retained, not inflated into
    explainer-invariant confirmation. Signed sums of feature SHAP are not recomputed
    coalition SHAP, and no attribution is asserted to be causal truth.

11. **What could be novel?** A narrowly specified verification-based, originally
    observed reason-revision evaluation and its empirical separation from predictive
    confidence; a semantically stress-tested cross-financial-risk benchmark; and
    an explicitly bounded release-policy/compute evaluation using completion evidence.
    These are candidate contributions pending manuscript-level literature review,
    not a first-ever claim. Prior [Selective Explanations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html)
    addresses selective explanation quality; missing-data SHAP/uncertainty and
    [Calibrated Explanations](https://doi.org/10.1016/j.eswa.2024.123154) already
    establish neighboring ideas. Use the repository's [verified SOTA comparison](SOTA_COMPARISON_2026.md)
    and [related-work gap](RELATED_WORK_GAP.md) rather than generic novelty claims.

12. **What must not be claimed?** New GRU/classifier, generic MC uncertainty novelty,
    Bayesian posterior, causal/correct reasons, optimal K, all-metric superiority,
    human-validated financial reasons, guaranteed10% deployment risk, or general
    consumer-default replication. MC is nearly equivalent to rank instability:
    within-fold Taiwan feature AP difference .0028 [−.0010,.0082], Polish feature
    −.0002 [−.0024,.0019]. A claim that this estimator universally beats that simple
    explanation baseline fails. Polish default Platt calibration also worsens
    Brier/log loss; no improvement is asserted or tuned afterward.

13. **Ready for ESWA?** **Promising but incomplete.** External grouped decoupling
    and meaningful selected coverage materially improve the evidence. Yet economic
    grouping is dictionary-grounded, not expert validated; one external cohort,
    explainer dependence, complete-case completion assumptions, absence of entity
    calibration and coarse low-risk coverage limit an operational recommendation.
    A submission must foreground those limits and strong simple baselines.

14. **What paper is supported now?** A reproducible applied evaluation/benchmark
    or methods-validation paper on explanation release under simulated incomplete
    financial inputs. It supports neither a novel neural architecture paper nor
    a validated lending deployment system. No evidence-based acceptance prediction
    for a particular journal is possible.

15. **Continue, narrow or stop?** **Continue, narrowed to evaluation + selective
    explanation.** Stop architecture search, neural-head claims and attempts to
    force a universally winning model. The single next action is a preregistered,
    blinded domain-expert review of grouped revision/near-tie cases to establish
    whether the operational changes matter to financial analysts. Do not adapt
    this external experiment in response to those future judgments.

## Final proposed contributions (at most three)

- A frozen verification benchmark shows that prediction stability can coexist
  with observed-reason revision, surviving semantic grouping and one external
  corporate financial-risk dataset, with explicit metric sensitivity.
- Explanation-specific completion evidence detects that target substantially
  better than predictive-confidence controls; rank instability remains an
  essentially equivalent simple explanation baseline.
- Independent-calibration release policies quantify useful coverage/risk/compute
  tradeoffs in declared missing environments, including zero-coverage and
  calibration failures rather than claiming a deployment guarantee.

## Recommended pipeline

```text
Incomplete record + observed mask
  -> frozen XGBoost + current TreeSHAP
  -> observed feature/group reasons (declared definition)
  -> training-only donor completions, reference K=8
  -> same XGBoost/TreeSHAP -> MC revision sensitivity
  -> independent revision calibration
  -> common threshold calibrated across simulated environments
  -> release / withhold

Artificial truth -> separate verification-only evaluator, never serving input
```

## Validation and artifacts

84 tests passed before experiments. New independent recomputation checked66 cache
batches,703,440 completion rows,295,365 event rows and6,891,264 release decisions;
all matched. No invalid attribution endpoints/completions. Historical audit also
passed88,800 reason rows and525,000 prediction rows. Six PNG/PDF figure pairs,
all alpha/K/metric tables,1000-draw paired intervals,280 deterministic cases,
donor audits, source/data/partition/policy hashes and current/verification JSONL
are generated locally and intentionally not committed. See the README for commands.
Two unexecuted reporting syntax errors and the Polish calibration-assumption
limitation are transparently recorded in VALIDATION_IMPLEMENTATION_NOTES.md.

AI-assisted code, literature verification, analysis and drafting are disclosed.
The audit is reproducible software checking, not independent human scientific review.
