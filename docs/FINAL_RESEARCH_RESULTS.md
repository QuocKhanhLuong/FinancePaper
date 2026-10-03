# Final consolidated research results

**Evidence cutoff: 2026-10-04. Decision: continue as an evaluation and selective
explanation policy paper; no supported new-algorithm claim. ESWA assessment:
promising but incomplete.**

Prediction–explanation decoupling survives semantic grouping and one external
corporate-bankruptcy assessment. Completion-based explanation evidence detects
revision much better than the tested prediction-only controls. However, rank
instability is a strong, often equivalent comparator. Stable-Core reduces reason
failure without establishing superior coverage at the declared budgets. The
acquisition headroom gate failed; the conformal-envelope novelty gate is NO-GO.

This report consolidates **completed experiments**, not a new experiment or a
claim that all requested validation is complete. It was checked against the
local aggregate CSVs and historical reports on repository base
`79e73619f6212ca7d2126b4c466540a2890730f5`. No predictor, mask, attribution,
target, calibration rule or threshold was changed. No new assessment cohort was
opened. Historical reports remain unchanged.

## 1. Evidence populations and endpoints

| Study | Evaluation unit and size | Status and interpretation |
|---|---|---|
| Historical predictor follow-up | Five restarts; fixed 4,500 Taiwan prediction records and 400 explanation records | Exploratory inspected test; retained in [historical results](ROBUSTNESS_FOLLOWUP_RESULTS.md), not pooled with the later folds |
| Revision study and decisive Taiwan validation | Three internal folds; 21,000 distinct outer prediction customers in total; 2,400 distinct explanation customers | Historically studied development pool; 9,000 old test/reserve records excluded; not fresh confirmation of the adaptive research process |
| Decisive Polish validation | 1,182 assessment statements; 1,169 exact-feature clusters | Independently frozen external assessment at that stage; one split, MCAR10/30 only |
| Stable-Core v2 | Same 2,400 Taiwan customers across four conditions, 9,600 episodes; same 1,182 Polish statements across two conditions, 2,364 episodes | Exploratory reuse of inspected cohorts; episodes are not independent customers |
| One-query headroom | 500 Polish development clusters; 1,000 episodes; 12,537 actual single-field reveal actions | Development reuse; hindsight oracle is an upper bound, not a deployed policy |
| Freddie Mac large-scale validation | Planned cohort only; eligible counts unknown | **NOT RUN: official files have not been supplied** |

Taiwan predicts next-month consumer credit-card default. Polish predicts
corporate bankruptcy within one year using `5year.arff`; its prediction metrics
are reported separately. Natural Polish missing values remain unknown. Only
artificially hidden, originally observed cells are restored. Exact-feature
clustering reduces duplicate leakage, but the absence of company identifiers
prevents certification of independent corporate entities.

The primary attribution convention for tree analyses is fixed-background
interventional TreeSHAP in **raw logit units**. Semantic contributions are signed
sums over the **same initially observed feature members** before and after
restoration. A group with no initially observed members is unavailable. These
are sums of feature SHAP values, not newly computed coalition SHAP values.

Two endpoints must remain distinct:

- **Historical whole-explanation revision:** initially eligible positive top-k
  observed reasons require revision when a displayed reason changes sign or
  leaves the top-k under the frozen tolerance rules. Semantic top-2 is the main
  grouped illustration; top-1/top-3 remain sensitivity analyses.
- **Stable-Core meaningful-reason failure:** a currently positive candidate group
  has restored contribution at or below .01. The output can have any size.
  Failure is counted per released reason, not per explanation. This easier
  endpoint cannot be substituted for historical ranking revision.

Full contracts: [reason groups](DOMAIN_REASON_GROUPS.md),
[metric audit](REVISION_METRIC_AUDIT.md),
[Stable-Core protocol](STABLE_CORE_STUDY_PROTOCOL.md).

## 2. Prediction controls: keep XGBoost, no architecture victory

Taiwan revision-study folds; **mean ± sample SD**, calibrated default
probabilities. Each fold has 7,000 outer prediction customers. AP is higher-is-better;
Brier is lower-is-better. These are not the historical five-restart test results.

| Predictor | Clean AP | MCAR30 AP | MAR30 AP | MCAR30 Brier |
|---|---:|---:|---:|---:|
| Logistic Regression 25-view | .5359 ± .0181 | .5036 ± .0152 | .4952 ± .0242 | .1437 ± .0022 |
| Additive-tree control | .5368 ± .0193 | .5061 ± .0175 | .4894 ± .0226 | .1435 ± .0019 |
| Vanilla GRU + masking | .5326 ± .0122 | .5042 ± .0102 | .4872 ± .0168 | .1429 ± .0012 |
| Mask/delta GRU + masking | .5327 ± .0132 | .5051 ± .0061 | .4916 ± .0184 | .1429 ± .0018 |
| XGBoost 25-view | .5465 ± .0222 | .5117 ± .0195 | .5035 ± .0243 | .1417 ± .0025 |

The recurrent branch remains a function-class control. It is a mask/delta-input
GRU, not GRU-D with learned decay. Its earlier gains were not consistent across
protocols; no all-metric winner or benchmark-SOTA claim is supported. Linear and
additive models' zero observed-reason revision is structural under the chosen
attribution convention, not evidence of explanation correctness.

On the separate Polish assessment, XGBoost MCAR30 default AP/AUROC is
.5458/.8984. Default Brier worsens from .0499 raw to .0523 after calibration;
log loss worsens from .1754 to .1874. Calibration is not automatically beneficial.
Explanation policies leave all predictor outputs unchanged.

Sources: [revision-study results](REVISION_AWARE_RESULTS.md),
[external predictive controls](EXTERNAL_VALIDATION_RESULTS.md).

## 3. Main finding: stable predictions can have revised reasons

“Stable” means **raw probability shift after restoration <= .02**, not that the
prediction is correct or that deployment uncertainty is low. Below, the numerator
is Region B; the denominator contains prediction-stable, explanation-eligible
cases. All .01/.02/.05 cutoffs were evaluated; .02 is an illustration, not an
assessment-selected optimum.

| Dataset | Condition | Feature top-3: B / stable | Group top-2: B / stable | Group top-2 rate [95% CI] |
|---|---|---:|---:|---:|
| Taiwan | MCAR10 | 101/1,657 = 6.10% | 30/1,365 | 2.20% [1.46, 3.02] |
| Taiwan | MCAR30 | 116/898 = 12.92% | 42/727 | **5.78% [4.14, 7.51]** |
| Taiwan | MAR30 | 72/1,040 = 6.92% | 35/814 | 4.30% [2.95, 5.69] |
| Polish | MCAR10 | 141/969 = 14.55% | 47/721 | 6.52% [4.61, 8.39] |
| Polish | MCAR30 | 273/795 = 34.34% | 85/552 | **15.40% [12.30, 18.41]** |

Grouping removes much fine-grained turnover, but does not erase the phenomenon.
At a wider .02 raw-logit rank gap, group-top2 Region B falls to **2.89% Taiwan**
and **10.87% Polish** at MCAR30. The numerical magnitude is sensitive to what
counts as a material ranking change. Expert-validated financial importance is
still absent; these percentages do not establish harmful lending decisions.

Under MCAR30, overall eligible group-top2 revision is 8.18% Taiwan and 19.01%
Polish. Those denominators differ from the stable-only column above. Group-top1
and top3 also retain nonzero Region B; top3 has much lower eligibility. Originally
complete and naturally incomplete Polish statements both exhibit the phenomenon.

Sources: [all group sizes, cutoffs, tolerances and common-eligibility controls](DOMAIN_REASON_VALIDATION_RESULTS.md),
[external assessment](EXTERNAL_VALIDATION_RESULTS.md).

## 4. Revision detection: completion evidence wins over prediction controls

MCAR30, **revision detection**, not default prediction. Values below are Taiwan
fold means or the single Polish assessment. Brier uses independently calibrated
revision scores from the decisive stage. Fold SDs are in the source reports.

| Target | Dataset | Detector | AUROC | AP | Brier | Log loss |
|---|---|---|---:|---:|---:|---:|
| Feature top-3 | Taiwan | Prediction-only learned control | .5728 | .2192 | .1464 | .4675 |
| Feature top-3 | Taiwan | Generic learned selector | .7963 | .4574 | .1209 | .3832 |
| Feature top-3 | Taiwan | Rank instability | .8982 | .7171 | .0977 | .3163 |
| Feature top-3 | Taiwan | MC K8 | .9000 | .7199 | .0882 | .2905 |
| Feature top-3 | Polish | Prediction-only learned control | .5662 | .4562 | .2358 | .6667 |
| Feature top-3 | Polish | Generic learned selector | .7719 | .6380 | .1924 | .5623 |
| Feature top-3 | Polish | Rank instability | .9511 | .9224 | .1129 | .3663 |
| Feature top-3 | Polish | MC K8 | .9514 | .9222 | .0915 | .2933 |
| Group top-2 | Taiwan | Prediction-only learned control | .6004 | .1447 | .0745 | .2781 |
| Group top-2 | Taiwan | Generic learned selector | .5774 | .1366 | .0752 | .2819 |
| Group top-2 | Taiwan | Rank instability | .9171 | .6954 | .0472 | .1613 |
| Group top-2 | Taiwan | MC K8 | .9207 | .6926 | .0428 | .1507 |
| Group top-2 | Polish | Prediction-only learned control | .6059 | .2622 | .1506 | .4758 |
| Group top-2 | Polish | Generic learned selector | .5947 | .2559 | .1516 | .4796 |
| Group top-2 | Polish | Rank instability | .9240 | .8219 | .0666 | .4189 |
| Group top-2 | Polish | MC K8 | .9513 | .8580 | .0622 | .2125 |

The prediction-only control uses current probability, entropy, completion
prediction variance and missing fraction. Grouped learned-selector rows reuse
the frozen feature-target selector with grouped recalibration; they are **not**
a newly trained optimal grouped selector. AP cannot be compared across datasets
as if revision prevalence and difficulty were equal.

For group-top2, paired MC-minus-prediction-only AP differences are
**+.5479 [.4646, .6230] Taiwan** and **+.5957 [.5294, .6518] Polish**.
Against rank instability they are **−.0028 [−.0322, .0263] Taiwan** and
**+.0361 [.0129, .0605] Polish**. Feature-top3 MC-minus-rank differences include
zero on both datasets. Thus explanation evidence adds information, but MC does
not generally outperform rank instability. Both use completion explanations;
rank scoring does not eliminate their shared SHAP cost.

This stage's Taiwan MC8 Brier .0882 is not the earlier revision study's .0867:
the declared revision-calibration environment mixture changed between stages.
The historical number is preserved, not overwritten or averaged with this one.

Sources: [detector/K report](MONTE_CARLO_K_ABLATION.md),
[external detector tables](EXTERNAL_VALIDATION_RESULTS.md),
[failure analysis](MONTE_CARLO_FAILURE_ANALYSIS.md).

## 5. Release policy: useful tradeoffs, no universal risk guarantee

Group-top2, MCAR30. MC policies use the **conservative finite-grid rule at a 10%
calibration budget**; release-all is an uncalibrated descriptive reference.
Coverage denominator includes every explanation-sampled customer, including
those without two eligible reasons. Risk is whole-explanation revision among
released cases. Brackets are 95% customer/duplicate-cluster bootstrap intervals.

| Dataset | Policy | Customer coverage [95% CI] | Released-explanation revision [95% CI] |
|---|---|---:|---:|
| Taiwan | Release all eligible | 68.25% [66.50, 70.13] | 8.18% [6.87, 9.50] |
| Taiwan | MC8 pooled calibration | 64.12% [62.29, 66.17] | 5.07% [3.99, 6.20] |
| Taiwan | MC8 observable strata | 38.88% [37.00, 40.79] | 1.61% [0.79, 2.43] |
| Taiwan | MC8 robust-environment calibration | 61.46% [59.62, 63.38] | 3.25% [2.32, 4.20] |
| Polish | Release all eligible | 78.34% [75.95, 80.66] | 19.01% [16.43, 21.63] |
| Polish | MC8 pooled calibration | 68.02% [65.17, 70.73] | 7.84% [5.97, 9.67] |
| Polish | MC8 observable strata | 27.75% [25.21, 30.20] | .91% [0.00, 2.01] |
| Polish | MC8 robust-environment calibration | 58.21% [55.34, 61.17] | 2.18% [1.17, 3.29] |

Taiwan's release-all grouped risk already lies below 10% descriptively. MC must
not take credit for semantic aggregation alone. Polish provides clearer evidence
of a useful coverage/risk tradeoff. Robust calibration uses one common threshold
and no unknown mechanism label at inference; it is robust only across the
simulated calibration environments considered, with a coverage cost.

Failures remain important: empirical pooled group-top2 calibration reaches
12.43% Polish MCAR30 revision at a 10% budget. At 5%, conservative robust
feature-top3 releases nothing on both datasets, and Polish group-top2 also
releases nothing. Empty releases have **undefined risk**, not zero risk.
Polish statement-level calibration is not a corporate-entity certificate.
These results do not establish distribution-free or unseen-environment control.

Source: [all policies, environments and 5/10/15% budgets](RELEASE_POLICY_VALIDATION.md).

## 6. Stable-Core: lower failure, failed superiority hypothesis

This table uses the **meaningful-positive reason target**, a 10% empirical
calibration budget, and the declared mixture of environments. It does not use
the whole-explanation top-k event in Sections 3–5. Taiwan/Polish cohorts have
already been inspected. Reason failure is micro-averaged over released reasons;
customer coverage is over customer-condition episodes.

| Dataset | Policy | Failed / released reasons | Reason failure [95% CI] | Reason coverage | Customers with >=1 reason |
|---|---|---:|---:|---:|---:|
| Taiwan | Release all | 564/21,994 | 2.56% [2.27, 2.87] | 100.00% | 92.61% |
| Taiwan | Whole top-1 MC policy | 102/8,891 | 1.15% [.87, 1.42] | 40.42% | 92.61% |
| Taiwan | Donor Stable-Core | 111/21,031 | .53% [.41, .65] | 95.62% | 91.40% |
| Taiwan | Conditional Stable-Core | 112/20,829 | .54% [.42, .67] | 94.70% | 91.14% |
| Taiwan | Both-family Stable-Core | 59/20,596 | **.29% [.21, .38]** | 93.64% | **90.82%** |
| Polish | Release all | 277/7,100 | 3.90% [3.39, 4.41] | 100.00% | 94.16% |
| Polish | Whole top-1 MC policy | 27/2,226 | 1.21% [.72, 1.79] | 31.35% | 94.16% |
| Polish | Donor Stable-Core | 36/6,722 | .54% [.38, .72] | 94.68% | 93.06% |
| Polish | Conditional Stable-Core | 27/6,640 | .41% [.27, .56] | 93.52% | 92.94% |
| Polish | Both-family Stable-Core | 16/6,585 | **.24% [.14, .37]** | 92.75% | **92.64%** |

Both-family agreement retains useful coverage and lowers observed failure.
At matched per-customer reason counts, it reduces failure relative to strength
selection by **.4710 percentage points [.3687, .5958] Taiwan** and **.6834
[.4771, .9173] Polish**. Some reason-identity benefit therefore survives the
shorter-explanation control.

Nevertheless, release-all already meets the pooled 10/15% budgets with higher
coverage. Under the ranked sensitivity, rank-only and top-1 reach more customers.
Every fitted meaningful-target Stable-Core rule selects `(0,0,.01)`, reducing
to a positive lower-quantile filter; the additional sign/rank conditions do not
provide a demonstrated new mechanism. All tested conservative v2 finite-grid
policies abstain completely at 5/10/15%. The empirical policies are not certificates.

Two-family serving costs about 3.5x Taiwan / 10.1x Polish donor rank-only cost on
the fixed timed batches. Both completion families may share misspecification.
**Retain Stable-Core as an optional lower-failure operating point, not as a
replacement, universal Pareto improvement, or new uncertainty algorithm.**

Source: [Stable-Core results, ranked and matched-size controls](STABLE_CORE_RESULTS.md).

## 7. Compute and stopped branches

Taiwan MCAR30 feature-top3 detection; AP is a fold mean. Timing is an independent
fixed 128-case CPU batch on Apple M4 Pro, 24 GiB, single numerical thread, five
warm repetitions; model loading excluded. Mean ± SD is in milliseconds.

| K | Revision AUROC | Revision AP | Batch latency, ms | Cost relative to K1 |
|---|---:|---:|---:|---:|
| 1 | .7548 | .4472 | 159.642 ± .700 | 1.0000x |
| 2 | .8287 | .5733 | 229.371 ± .734 | 1.4368x |
| 4 | .8752 | .6567 | 368.810 ± 1.495 | 2.3102x |
| 8 | .9000 | .7199 | 648.082 ± 2.913 | 4.0596x |
| 16 | .9129 | .7578 | 1,205.498 ± 4.448 | 7.5512x |

K8 remains the frozen budget compromise. It captures 93.18% of K16's AP gain
above prevalence across the declared Taiwan conditions; K4 captures 80.85%.
Neither meets the predeclared 95% near-reference criterion. K16 is a reference,
not proof of Monte Carlo convergence. The weaker learned selector costs
97.374 ± .596 ms/batch. Most MC cost is SHAP; more draws cannot repair a biased
completion distribution. See [timing protocol](MONTE_CARLO_K_ABLATION.md).

| Branch | Final status | Decisive evidence |
|---|---|---|
| Neural revision head / architecture search | Stopped; no supported rationale | Generic selector loses to explicit completion evidence |
| Stable-Core algorithmic-superiority claim | Not supported | Strong simple policies retain more coverage at the declared budgets |
| One-field acquisition | **Measured NO-GO** | On 500 Polish development clusters, oracle gain over donor Stable-Core is only .2 pp MCAR10 / 1.6 pp MCAR30, below the predeclared 5 pp gate; zero-query top-1 already reaches the retained-candidate customer ceiling |
| Verification-conformal envelopes as a new method | **Novelty-audit NO-GO; NOT RUN** | Inner/outer multilabel conformal sets and calibration of explanation outputs already exist; current specification supplies no demonstrated mechanism beyond retargeting them |
| TabM / further predictor family expansion | **NOT RUN** | Envelope development stopped at its research gate |
| Freddie temporal large-scale confirmation | **NOT RUN** | Official licensed files absent; no mortgage result exists |

The envelope decision is **not** empirical evidence that envelopes fail.
The exact future-verification financial target was not found as an identical
implementation in the bounded prior-art search, but that absence cannot establish
originality. The one-query oracle is evaluator-only; its gains are not inference
performance. No failed primary target was replaced by a more favorable sensitivity.

Sources: [headroom experiment](VERIFICATION_HEADROOM_RESULTS.md),
[conformal novelty audit](CONFORMAL_EXPLANATION_NOVELTY_AUDIT.md),
[large-scale execution status](LARGE_SCALE_VALIDATION_RESULTS.md).

## 8. Metrics: what the numbers actually mean

| Metric | Meaning and denominator | Interpretation |
|---|---|---|
| Default AP / AUROC | Ranking of the true credit-risk outcome | Higher is better; AP depends on outcome prevalence; not explanation reliability |
| Revision AP / AUROC | Ranking of the frozen verification-revision event | Higher is better; same names as prediction metrics, different labels |
| Brier | Mean squared error between a probability and its binary outcome | Lower is better; specify default vs revision and raw vs calibrated; not a pure calibration-only metric |
| Log loss | Negative log probability assigned to the observed binary outcome | Lower is better; strongly penalizes confidently wrong probabilities |
| ECE | Weighted mismatch between mean probability and event rate within bins | Lower is generally better, but bin-dependent; never a guarantee |
| Recall / F1 | Positive-outcome detection / precision–recall balance at a declared threshold | Threshold-dependent; do not substitute them for calibration or AP |
| Probability shift | Absolute partial-versus-restored probability difference | Measures restoration sensitivity, not correctness; restoration truth is evaluator-only |
| Region B rate | Revised explanations / stable, eligible cases | Quantifies decoupling; not the fraction of all customers |
| Whole-explanation risk | Released explanations with at least one revision / released explanations | Historical all-or-nothing endpoint; not micro reason failure |
| Reason failure risk | Failed released reasons / all released reasons | Stable-Core endpoint; customers with more reasons contribute more claims |
| Customer coverage >=1 / >=2 | Customer-condition episodes receiving at least one / two reasons divided by all episodes | Higher is useful only at acceptable risk; includes candidate-empty cases in denominator |
| Reason coverage | Released candidate reasons / all current candidate reasons | Measures retained information; can differ sharply from customer coverage |
| Risk–coverage curve | Observed failure versus retained coverage as a score threshold varies | Descriptive curve; choosing the best assessment point is not a deployable calibrated policy |
| 95% paired bootstrap CI | Resample customers or exact-feature clusters, keeping masks paired | Conditional on fitted models/policies; not full research-process uncertainty or a deployment certificate |

All principal intervals use 1,000 draws. Completions, masks and repeated timing
measurements are not independent customer samples. A zero-release policy has
undefined risk. A calibrated 10% budget is a rule-fitting target, not proof that
every future subgroup will have at most 10% failure. Raw completion frequency is
not a Bayesian posterior. A future joint envelope guarantee, if implemented,
would also be different from controlling micro reason failure.

## 9. Final paper contribution and usable pipeline

The defensible contribution is limited to these three evidence-based statements:

1. **Verification-based evaluation:** frozen within-model explanations can change
   despite small restoration-induced prediction shifts; the effect survives
   semantic grouping and one external corporate financial-risk assessment, with
   explicitly measured rank/tolerance sensitivity.
2. **Separation of reliability signals:** completion-based explanation evidence
   detects this revision target substantially better than the tested predictive
   uncertainty controls; rank instability remains a strong explanation baseline.
3. **Operational tradeoff evidence:** independently calibrated release policies
   and partial reason sets expose a measurable risk/coverage/compute tradeoff,
   including empty-policy failures and negative results against simple controls.

These support an applied evaluation or methods-validation contribution. They do
not establish a first-ever formulation, a new conformal theorem, new SHAP/MC
algorithm, causal explanations, human-validated reasons or superiority on all
metrics. The literature basis and its limits are in the
[SOTA comparison](SOTA_COMPARISON_2026.md) and
[latest novelty audit](CONFORMAL_EXPLANATION_NOVELTY_AUDIT.md).

The final research reference remains:

~~~text
Incomplete record + separate natural/artificial missing masks
  -> frozen XGBoost + fixed-background TreeSHAP
  -> semantic reasons over initially observed members
  -> K8 training-only donor completions and their explanations
  -> MC revision score, with rank instability as a mandatory comparator
  -> independently fitted release threshold
  -> release the eligible reason list or withhold it

Optional operating point: calibrated Stable-Core subset
  (reported with reason-level risk and matched-size controls)

Offline evaluation only:
restore artificially hidden truth -> recompute frozen explanation -> score revision
~~~

Normal inference reports risk probability, missing-information indicators,
released reasons/status and the applicable revision score/policy. Restored
probabilities, true revision labels and oracle actions never enter inference.
No conformal inner/outer coverage claim belongs in this output.

**Final recommendation: continue the narrowed evaluation paper; stop method
expansion under the rejected claims.** ESWA is **promising but incomplete** as an
applied evaluation paper and unsupported as a novel-algorithm paper. The largest
unresolved scientific issue is whether the remaining grouped/near-tie revisions
matter to domain users. One external split, explainer dependence, missing company
identifiers, completion misspecification and absent large-scale confirmation
further limit generality. No acceptance prediction is made.

**Single next scientific action:** conduct the previously recommended preregistered,
blinded domain-expert assessment of grouped revision and near-tie cases, without
retuning the frozen predictor, target or policies on those judgments.

## 10. Verification receipt and artifact map

This consolidation re-read the linked reports and programmatically checked 232
table values/counts against the existing aggregate CSVs, including detector means,
Region B intervals, release results, Stable-Core counts and repeated timings.
Paired AP differences were also inspected in their source CSV; all local report
links resolve. It did not rerun training or claim
new statistical replication. The historical headroom stage records **121 tests
passed**; that is a prior receipt, not a new test run for this documentation change.

| Evidence | Local aggregate artifact, under ignored outputs/ |
|---|---|
| Decoupling | decisive_validation/analysis/region_B_intervals.csv |
| Detector means and calibrated losses | decisive_validation/analysis/B_detector_metrics.csv |
| Paired detector differences | decisive_validation/analysis/paired_fold_mean_differences.csv |
| Release calibration | decisive_validation/analysis/D_policy_intervals.csv |
| Unselected reference | decisive_validation/analysis/release_all_eligible.csv |
| Repeated timings | decisive_validation/analysis/runtime_repetitions.csv |
| Partial-reason policies | stable_core_study/analysis/metrics.csv |
| Acquisition headroom | verification_headroom/analysis/metrics.csv |

Reproduction commands and original freeze receipts remain in [README](../README.md)
and the linked stage reports. Raw datasets, row-level outputs, model artifacts
and caches are not committed. This consolidated report contains aggregate results
only and preserves all negative findings and NOT RUN boundaries.
