# Verification-conformal explanation envelopes: novelty audit

Research cutoff: **2026-10-04, Asia/Ho_Chi_Minh**. Repository inspected:
main/origin/main, **0e612d977d29a3238237eef5248dad93a9000323**.
This is a literature, target-contract and mathematical audit. No new financial
experiment, predictor fitting, conformal calibration or test-set inspection
was performed. Historical definitions and results are unchanged.

## Decision

**NO-GO**

This decision applies to **developing the currently specified envelope as a new
method contribution**. The proposed future-verification target remains a
legitimate applied research question. The audit did not find an inspected paper
implementing the exact combination of artificial financial-feature verification,
the repository's observed-member semantic reason target, and inner/outer
conformal sets. That bounded search finding is not a first-ever claim.

The reason to stop is more specific than “conformal prediction already exists”:
the requested statistical construction is already a generic multilabel method,
and conformally predicting explanation outputs is also established. With the
predictor/explainer frozen, substituting verified reason membership as its label
does not yet specify an additional estimation, calibration or inference mechanism.
There is no demonstrated advantage over that retargeted baseline. Adding TabM or
renaming the wrapper would not supply one.

Per the requested gate, methodological implementation stops here. The generic
baseline is specified analytically below to make the collision reviewable;
neither it nor a purported new envelope algorithm was implemented or evaluated.
This is **not an empirical finding that envelopes fail** and does not overturn
the measured prediction–explanation decoupling.

## 1. Repository evidence that constrains this proposal

The review covers README, the original concept/pipeline/protocol and SOTA
documents, the final decision, novelty reassessment, domain-group definitions
and results, K/runtime audit, completion failures, release-policy validation,
Stable-Core v2, and the latest verification-headroom report.

| Frozen evidence | Implication for an envelope study |
|---|---|
| Group-top2 Region B at MCAR30 and raw probability shift <=.02: Taiwan 42/727 (5.78%); Polish 85/552 (15.40%) | There is a target worth studying. These are conditional, eligible-customer denominators, not all-customer failure rates. |
| Group-top2 MC8 detection AP .6926/.8580 versus prediction-only .1447/.2622 | Completion evidence is informative on these particular targets; it is not a conformal guarantee. |
| Rank instability often matches MC; feature-level paired AP differences include zero | A completion-based envelope must face a conformalized rank baseline, not just prediction confidence. |
| Meaningful-positive release-all reason failure 2.56%/3.90% across the Stable-Core study | Existing 10/15% micro-reason budgets leave little aggregate reason-coverage headroom. This does not imply low joint set miscoverage. |
| Stable-Core lowers failure but loses coverage; every conservative v2 grid policy is empty | Do not rebrand empirical thresholds as conformal coverage or assume a new wrapper solves sample scarcity. |
| One-query Polish development oracle gains only .2/1.6 customer-coverage points over donor Stable-Core | The acquisition branch stays stopped. Its retained-candidate ceiling is not a theorem about the new outer-set target. |
| Historical Taiwan/Polish cohorts have been inspected; official Freddie files are absent | No fresh confirmation or large-scale result can be claimed here. |

Sources: [final decision](FINAL_PAPER_DECISION.md),
[grouped validation](DOMAIN_REASON_VALIDATION_RESULTS.md),
[Stable-Core results](STABLE_CORE_RESULTS.md),
[headroom results](VERIFICATION_HEADROOM_RESULTS.md),
[completion failures](MONTE_CARLO_FAILURE_ANALYSIS.md).
These numbers are historical receipts, not reruns in this audit.

### Source contracts inspected

| Source under src/financepaper | Relevant contract |
|---|---|
| models/logistic.py, models/xgboost.py; historical experiment configurations | Existing regularized LR and frozen XGB25 remain controls. Twenty-five views concern training augmentation, not 25 independent test customers. |
| models/temporal_gru.py, models/missing_aware_gru.py, models/logit_blend.py | The retained recurrent model is a mask/delta-input GRU, **not GRU-D**; there is no learned decay. Historical blends remain analysis history. |
| training/fit.py, losses.py, calibration.py, device.py | Development-only early stopping, separate loss ablations, positive-slope Platt calibration, MPS/CPU support. Probability calibration is not explanation-set calibration. |
| explanations/shap_values.py | Fixed-background independent linear attribution and interventional TreeSHAP in raw logit units, with additivity checks. |
| explain/integrated_gradients.py, explain/permutation.py | IG integrates values while endpoint masks/deltas stay fixed. The existing common permutation control also conditions on endpoint metadata. |
| reliability/validation.py: aggregate_groups | Signed sums over the same initially observed members at all endpoints; a group with no such members is unavailable. |
| reliability/reason_sets.py: evidence, verified, rank_membership | Serving evidence and restored labels are separate. Meaningful target uses >.01; ranked sensitivity adds tolerant top2. Current candidates are a separate mask. |
| evaluation/metrics.py, revision.py, revision_audit.py, temporal_metrics.py, verification_headroom.py | Prediction, historical revision, bootstrap and hindsight-oracle quantities have different denominators and roles. The oracle is evaluator-only. |
| reliability/policy.py, validation.py, reason_sets.py | Existing empirical/conservative release procedures are not implementations of joint conformal inner/outer prediction. |

Configurations inspected: pilot.yaml, temporal_pilot.yaml,
robustness_followup.yaml, revision_study.yaml, decisive_validation.yaml,
stable_core_study.yaml, verification_headroom.yaml and freddie_validation.yaml.
Source inspection is not a rerun or certification of every historical artifact.
No newly established implementation bug is asserted.

## 2. Search scope and verification limits

This is a targeted prior-art audit, **not an exhaustive systematic review**.
Primary records were checked on JMLR, PMLR, NeurIPS/ICLR proceedings, publisher
DOI pages and arXiv; official TabM code/license was checked separately.
Search engines were discovery tools, not authorities for the conclusions.

Queries included the following, followed by exact-title and citation searches:

| Area | Representative search strings |
|---|---|
| Multilabel | conformal multilabel prediction inner outer sets; “inner” “outer” “sets” “multilabel” conformal; “Conformal” “Binary Relevance” multilabel; “Classification with Valid and Adaptive Coverage” |
| Explanation calibration | “conformal” “explanation” “sufficient”; “conformal” “explanation sets”; “conformal” “feature attribution” confidence regions; “conformal” “attribution” “intervals”; “GeoXCP” |
| Missing/future information | “conformal” “explanations” “missing” “verification”; “conformal” “explanations” “missing covariates”; “explanation” “future” “feature revelation”; “post-verification” “explanation”; “prediction sets” “explanations” “missing” |
| Set-valued attribution | “set-valued explanations”; “conformal” “lower” “upper” “explanations”; “SHAP” “missing” “interval” explanation uncertainty conformal; “set-valued” “attribution” missing |
| Current controls | “TabM” ICLR 2025 official github license; exact titles of Selective, Calibrated, Guarded Explanations and imputation-uncertainty papers |

The decisive JMLR inner/outer equations and algorithms were inspected in the
full paper. The imputation-uncertainty simulation target and the Conformal
Shapley/Reveal-IG definitions were inspected in primary full text. COPA 2023
has an accessible primary PDF. For COPA 2024/2026 explanation papers, the claims
below use official proceedings abstracts and indexed primary text; direct
retrieval of some GitHub-hosted PDFs failed. GeoXCP's publisher full-text endpoint
was intermittent: publisher-indexed Section 3.2 and Appendix A support the limited
target comparison. No inaccessible paper is represented as fully reviewed.
Search absence cannot prove originality.

## 3. Nearest-work comparison

“Future verification” means actual additional information becomes known, not
merely evaluating another SHAP coalition. “Post-verification target” means the
specific supervised restored-reason membership requested here. “No” describes
the inspected method, not every possible adaptation of it.

| Work | Explanation output | Missing input? | Future verification? | Set-valued? | Conformal/calibrated? | Post-verification target? | Difference from us |
|---|---|---|---|---|---|---|---|
| [Cauchois, Gupta & Duchi, JMLR 22(81), 2021](https://jmlr.org/papers/v22/20-753.html) | Generic multilabel output | Not required | Generic unknown label | **Inner/outer label sets** | **Joint conformal containment** | Not instantiated that way | Eq. 2 and Section 3 already supply the proposed mathematical output/guarantee. Financial verification changes the target, not this construction. |
| [Lambrou & Papadopoulos, COPA 2016](https://doi.org/10.1007/978-3-319-33395-3_7) | Binary-relevance multilabel predictions | Not central | No | Label regions | Conformal; Hamming-loss confidence | No | Per-reason binary conformal classifiers are an established baseline, not a new head. |
| [Romano, Sesia & Candès, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html) | Multiclass prediction sets | Not central | No | Class sets | Adaptive conformal scores | No | APS concerns one categorical response; it is not itself an inner/outer multilabel reason method. |
| [Angelopoulos et al., ICLR 2024, Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) | Generic risk-controlled outputs | Not required | Generic outcome | Can be | Expected monotone-loss control | No specific explanation target | A reason loss must satisfy its assumptions. Expected loss, micro reason precision and simultaneous containment differ. |
| [Alkhatib, Boström, Ennadir & Johansson, COPA 2023](https://proceedings.mlr.press/v204/alkhatib23a.html) | Intervals around predicted reference-explainer scores | Not the intended task | No; reference score approximated | Numeric intervals | Conformal regression | No | **Explanation output as a conformal target already exists**. Our unknown arises from unrevealed covariates, not expensive reference computation. |
| [Alkhatib, Boström & Johansson, COPA 2024](https://proceedings.mlr.press/v230/alkhatib24a.html) | Approximation-quality intervals for the whole attribution | Not central | No | Quality interval | Conformal | No | Goes beyond isolated feature-wise approximation guarantees; does not predict restored semantic memberships. |
| [Alkhatib & Lowry, COPA 2026](https://proceedings.mlr.press/v329/alkhatib26a.html) | Sufficient image regions from feature attributions | Excluded features perturbed | Not actual later verification | Feature subset | Conformal sufficiency | No | Preserves predictor output; our later attribution set can change despite preserved prediction. |
| [Löfström et al., ESWA 246:123154, 2024](https://doi.org/10.1016/j.eswa.2024.123154) | Feature-weight/probability intervals and rules | Perturbation, not our missing endpoint | No | Intervals/rules | Venn-Abers calibration | No | Uncertainty-aware explanation is known. CE's weights cannot be relabelled as verified TreeSHAP membership. |
| [Löfström, Hjort & Löfström, COPA 2026](https://proceedings.mlr.press/v329/lofstrom26a.html) | Supported candidate rule conditions | Candidate perturbations | No | Filtered rule set | Conformal-style guard; no finite-sample error guarantee for its guard threshold | No | Perturbation plausibility differs from reason survival after restoration; filtering unsupported components is known. |
| [Paes, Wei & Calmon, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html) | Selectively refined amortized attributions | Not central | More computation, not customer facts | Selective explanation | Quality selection, not this conformal bracket | No | Numerical approximation uncertainty differs from uncertainty about hidden inputs. |
| [Lou et al., GeoXCP, online 2025](https://doi.org/10.1080/13658816.2025.2574900) | Spatial feature-explanation intervals | Not central | No | Numeric intervals | Geographically weighted conformal construction | No | Also predicts explanation vectors and calibrates residuals. Spatial assumptions do not transfer automatically. |
| [Chandy et al., Conformal Shapley Intervals, 2026 preprint](https://arxiv.org/abs/2602.00171) | Modality-importance intervals and selected modalities | Modality subsets | No actual hidden-cell verification | Intervals/subsets | Conformal construction | No | Section 3 uses loss-reduction Shapley utilities and subset-trained predictors, not fixed-model logit attributions. |
| [Zaffran et al., ICML 2023](https://proceedings.mlr.press/v202/zaffran23a.html) | Response intervals with incomplete covariates | **Yes** | Response, not explanation | Intervals | Marginal/mask-conditional analysis | No | Missingness-aware CP is established; pooled coverage need not hold for each pattern. |
| [Fan et al., December 2025 preprint](https://arxiv.org/abs/2512.14221) | Response intervals after distributional imputation | **Yes** | Response, not explanation | Intervals | Reweighted mask-conditional framework | No | Completion plus conformal correction is studied; its assumptions and approximate validity cannot be imported uncritically. |
| [Vo et al., Explainability of Machine Learning Models under Missing Data, v3](https://arxiv.org/abs/2407.00411v3) | SHAP under missing/imputation workflows | **Yes** | Complete-data comparison | Point attributions/errors | Not inner/outer CP | No calibrated reason set | Prediction error and attribution error can diverge; that general distinction is not first observed here. |
| [Golchian & Wright, December 2025 preprint](https://arxiv.org/abs/2512.17689) | Learner-SHAP/PFI/PDP uncertainty intervals | **Yes** | Complete-data reference across repetitions | Intervals | Multiple-imputation/variance inference, not CP | Not an individual frozen-model reason set | Section 3.1 covers expected learner explanations across fits, not one record's restored membership. |
| [Laberge et al., JMLR 24(364), 2023](https://jmlr.org/papers/v24/23-0149.html) | Partial orders/consensus attribution statements | Not central | No | Partial explanation | Consensus over a Rashomon set | No | Partial statements are known; here the model stays fixed while information changes. |
| [Fokkema, de Heide & van Erven, JMLR 24(360), 2023](https://jmlr.org/papers/v24/23-0042.html) | Recourse-sensitive attributions; sets of attributions discussed | Not central | Recourse changes, not verification | Set-valued workaround | No CP bracket | No | Set-valued outputs are already motivated; their impossibility result concerns a different task. |
| [Murphy & Shrestha, Reveal-IG, June 2026 preprint](https://arxiv.org/abs/2606.03885) | Attribution along information-revelation distributions | Feature-wise uncertainty | Probe path, not later measured truth | Point attribution | No CP bracket | No | Information revelation is an existing attribution concept; this changes the explainer, which we freeze. |
| [Galwaduge & Samarabandu, EDFA, September 2026 preprint](https://arxiv.org/abs/2609.12179) | Recourse under acquired partial information | **Yes** | **Full-information recourse validation** | Recourse alternatives | Calibrated stopping/validity construction | Not attribution membership | Close to the broader motivation. Do not claim no earlier explanation work checks validity after fuller information. |

### Citation and status notes

The primary records above supply stable citations without invented proceedings DOIs.

- Cauchois, M., Gupta, S., & Duchi, J. C. (2021). *Knowing what You Know:
  valid and validated confidence sets in multiclass and multilabel prediction*.
  JMLR, 22(81), 1–42. **Eq. 2, Algorithms 2–3 and Corollary 11** are the direct
  comparison. [Primary PDF](https://jmlr.org/papers/volume22/20-753/20-753.pdf).
- Alkhatib et al. (2023), *Approximating Score-based Explanation Techniques
  Using Conformal Regression*, PMLR 204, 450–469; Alkhatib et al. (2024),
  *Estimating Quality of Approximated Shapley Values Using Conformal Prediction*,
  PMLR 230, 158–174.
- Alkhatib & Lowry (2026), *Turning Feature Attributions into Sufficient
  Explanations Using Conformal Prediction*, PMLR 329, 382–401; Guarded
  Explanations is in the same volume, 422–438. These are September 2026
  proceedings, not merely future submissions.
- GeoXCP DOI **10.1080/13658816.2025.2574900**; authors Xiayin Lou, Peng Luo,
  Ziqi Li, Song Gao and Liqiu Meng. Online year 2025; publisher issue year 2026.
  Section 3.2 and Appendix A predict/calibrate explanation vectors.
- Vo's repository-cited journal version is *Applied Soft Computing*,
  DOI **10.1016/j.asoc.2026.115105**. Its publisher endpoint returned 403 here;
  the scientific comparison uses accessible v3, not a fresh full-text journal review.
- Chandy, Fan, Golchian, Murphy and Galwaduge entries are treated as preprints
  unless a separately verified proceedings record is stated. An arXiv posting
  is not evidence of journal acceptance.

## 4. The proposed construction reduces to a known problem

This is our application-specific reduction, **not a new theorem**.
Let \(W=(X_{\mathrm{obs}},M)\). Condition on training data and freeze predictor
\(f\), attribution map \(A\), background, groups and thresholds. Restoration
determines \(Y=h_{f,A}(X_{\mathrm{restored}},M)\in\{0,1\}^{G}\).
Then \((W,Y)\) is ordinary multilabel prediction with coarsened inputs.
Observing \(Y\) only after verification does not create a new calibration algorithm.

The JMLR paper already gives inner/outer containment at marginal probability
at least \(1-\alpha\), with per-label scores and label-dependence methods. Its
direct inner/outer algorithm is a closer baseline than an uncalibrated confidence
threshold. [Section 3](https://jmlr.org/papers/volume22/20-753/20-753.pdf).

~~~mermaid
flowchart LR
    A["Incomplete values + mask"] --> B["Any fixed multilabel score"]
    V["Restored reason membership: calibration only"] --> C["Existing conformal calibration"]
    B --> C
    C --> D["Region of possible binary label vectors"]
    D --> E["Intersection = inner set L"]
    D --> F["Union = outer set U"]
~~~

For a nonempty configuration region \(\mathcal C(W)\), define

\[
 L(W)=\bigcap_{y\in\mathcal C(W)}\{g:y_g=1\},\qquad
 U(W)=\bigcup_{y\in\mathcal C(W)}\{g:y_g=1\}.
\]

Whenever \(Y\in\mathcal C(W)\), both containments hold. This coordinate hull
can admit combinations absent from \(\mathcal C\), losing label dependence.
Taiwan has eight groups and Polish seven: only 256/128 binary configurations.
Exponential label-space size is not a convincing computational novelty argument
for these experiments.

### A generic baseline already realizes the interface

Take fixed group scores \(\widehat p_g(W)\in[0,1]\): independent logistic
membership predictors fitted on development labels, or K8 empirical membership
frequencies. These are standard scoring alternatives, not calibrated posterior
probabilities. Use a joint nonconformity score

\[
 s(W,y)=\max_{g\in\mathcal G(W)} |y_g-\widehat p_g(W)|.
\]

Here \(\mathcal G(W)\) is the available group universe; the empty maximum is zero.
On \(n\) exchangeable held-out calibration customers evaluate scores at verified
labels. Let \(r=\lceil(n+1)(1-\alpha)\rceil\); take the \(r\)-th smallest score
as \(q\), with \(q=+\infty\) when \(r>n\). At inference set

\[
 B_g(W)=\{b\in\{0,1\}:|b-\widehat p_g(W)|\le q\}.
\]

For unavailable groups fix \(B_g=\{0\}\). If every \(B_g\) is nonempty, take
\(L=\{g:B_g=\{1\}\}\) and
\(U=\{g:1\in B_g\}\). Otherwise return the uninformative
\(L=\varnothing,U=\mathcal G(W)\), flagging that fallback.
This fallback only enlarges the valid-label configuration region.

The standard split-conformal rank argument gives **marginal joint** containment
under exchangeability conditional on fitting and score choices. No new neural
head or theorem is needed. This deliberately simple max-residual baseline is
not claimed optimal or identical to the more adaptive JMLR algorithms.
General split-conformal foundations are also described in
[Romano et al. (2020)](https://papers.nips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf).

A membership classifier on current attribution/mask statistics, and the same
calibration around rank/frequency scores, are strong retargeted baselines.
Historically weak **binary revision** selectors need not be weak
**multilabel verified-membership** selectors: the task changed. Their envelope
performance is NOT RUN.

## 5. Target-contract audit

The compatible primary target is the meaningful-positive v2 label, not a
silently revised historical top-k event:

\[
 O_0=\{j:\text{field }j\text{ is observed before artificial verification}\},
 \quad
 \Phi_g^*=\sum_{j\in g\cap O_0} A_{f,j}(X_{\mathrm{restored}}),
 \quad
 Y_g^*=\mathbf1\{g\cap O_0\ne\varnothing,\ \Phi_g^*>.01\}.
\]

\(O_0\) excludes both natural unknowns and artificially hidden fields. Restore
only artificial unknowns. Group sums retain the same **initially observed
members** at partial, completed and restored endpoints. Keep the raw-logit
reference/tolerance. Ranked sensitivity adds frozen top2 membership with
1e-6 rank-gap tolerance; it does not replace the primary target.

| Issue | Required interpretation |
|---|---|
| All restored fields versus observed-clue target | Adding newly revealed attribution summands changes the estimand and can remove LR invariance. Whole-group sensitivity stays separately labelled. |
| Existing candidate mask: current contribution >.01 | Suitable for retaining current claims, **not** the universe of an outer set predicting all available verified-positive groups. A currently negative observed group can become positive. |
| Existing verified function | Computes restored validity independently of current candidates. Candidate-masking its output would exclude newly emerging reasons by construction. |
| Ranked failure versus revision | A currently meaningful reason already outside top2 can fail ranked validity without a new revision. Revision requires paired membership comparisons. |
| Missing group | No initially observed members means unavailable under this target, not financial irrelevance. |
| Attribution scale | Signed sums, no absolute aggregation or rescaling for nicer results. Neural/tree magnitudes are not directly comparable even when both explain logits. |

These are future application requirements, not current-code modifications.
The inspected verified/rank functions are consistent with v2. Restricting a
future outer envelope to old candidates would be an **interface misuse**, not
a newly discovered historical bug.

There is a legitimate **new evaluation object within this repository**:
previous policies retain currently positive claims; an envelope also anticipates
observed-group claims that may emerge. This must be disclosed as a target extension.
It cannot be credited as improved old retained-reason coverage, and does not by
itself establish a new algorithm in the literature.

## 6. What the guarantee means

| Quantity | Meaning | Does not imply |
|---|---|---|
| \(P(L\subseteq Y^*\subseteq U)\ge1-\alpha\) | Both statements hold jointly for a random future case, averaging over calibration/test randomness | Conditional validity for every customer, missing pattern or deployment period |
| \(P(L\nsubseteq Y^*)\) | At least one inner claim fails | Fraction of all released reasons that fail |
| \(P(Y^*\nsubseteq U)\) | At least one verified reason is omitted | Average recall or useful outer size |
| \(\sum_i \lvert L_i\setminus Y_i^*\rvert/\sum_i\lvert L_i\rvert\) | Empirical micro reason-failure rate; undefined for zero releases | The same risk as joint conformal miscoverage |
| Mean \(\lvert U\setminus L\rvert\), mean \(\lvert L\rvert\), fraction \(\lvert L\rvert\ge1\) | Ambiguity and output size | Financial correctness or causality |

An analytical counterexample: a rule covers 90% of customers but releases
reasons only on the remaining failing 10%. Released-reason failure can be 100%.
Conversely, small micro reason failure need not mean most *entire* sets are
correct. CRC must match its declared loss; a selection-dependent ratio is not
automatically a monotone CRC loss.

The vacuous \(L=\varnothing,U=\mathcal G\) achieves containment without information.
Later evaluation would need joint containment, directional errors, inner customer
coverage/precision, outer recall, ambiguity and vacuous/empty-target frequencies.
Compare sizes and errors against generic CP, rank/frequency, strength/top1 and
release-all controls. “Stable” cannot mean individual 90% correctness.
Outside \(U\) means excluded under this convention, not causally irrelevant.

### Calibration and completion assumptions

- Freeze predictor, explainer, reference, target, score model, K and completion
  families before calibration. Fitting these on calibration labels can break
  the simple rank argument.
- Calibrate by customer, not completion or repeated mask. One prospectively
  sampled mask per customer gives a declared mixture task. A customer maximum
  over fixed environments instead targets their simultaneous coverage and can
  produce wider regions. Neither covers arbitrary new missing mechanisms.
- Pooled MCAR/MAR coverage is not environment-conditional coverage. The missing-
  covariate literature above addresses this distinction.
- Fold-specific predictors produce different target maps. Pooling their
  calibration rows does not justify the single-frozen-model split-conformal proof.
- Correct completions are **not required for generic marginal conformal validity**
  when scores and calibration/test pairs satisfy its assumptions. Misspecified
  completions can make sets wide or uninformative. Temporal/missingness shift
  breaks exchangeability; calibration does not cure arbitrary deployment shift.
- Taiwan/Polish reuse is exploratory. New masks or repartitioning inspected
  records do not create untouched confirmation. Polish exact-duplicate clusters
  do not establish company independence without company identifiers.
- Freddie temporal holdout would be a shift stress test, not automatically an
  exchangeable finite-sample validation. Official files remain absent.

## 7. Architecture controls and TabM

Common groups provide a common **semantic vocabulary**, not a shared reason
ground truth. Write \(Y^{*(m,A)}\): LR, XGB and GRU can assign different verified
memberships to the same restored record. Shared customers/masks give paired
conditions, not identical labels.

LR observed-member attribution is additive and invariant to restoration of
other fields. A tight LR envelope may be structurally easy, not economically
better. TreeSHAP and mask-conditional IG are different estimands. A common
permutation/KernelSHAP sensitivity could help, but must declare treatment of
mask/delta channels and feature coalitions. The existing temporal permutation
helper is not a ready-made adapter for all predictors.

For a future expressly applied benchmark, **TabM is the single selected modern
tabular DL representative**. Gorishniy, Kotelnikov & Babenko's
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)
studies parameter-efficient MLP ensembling, giving a different function class
without adding several attention architectures.
The [official code](https://github.com/yandex-research/tabm) uses
[Apache-2.0](https://raw.githubusercontent.com/yandex-research/tabm/main/LICENSE),
with license/notice obligations for redistribution. This supports baseline
selection, not a claim of Taiwan superiority.

No package was added, no TabM model trained, and no MPS performance or IG
completeness claimed. Engineering validation is NOT RUN because the method gate
failed. In a resumed application study, freeze and explain the same scalar
aggregation across TabM members: averaging logits is not averaging probabilities.
Existing LR/XGB/GRU, loss ablations and historical controls remain intact.

## 8. Answers to the motivating questions

| Question | Evidence-bounded answer |
|---|---|
| Scientifically distinct? | The exact financial future-verification target was not found in inspected sources. Inner/outer sets, explanation-target calibration and missing-covariate CP are known. A distinct algorithm is not specified. |
| Useful across families? | NOT RUN for envelopes. Historical MC/Stable-Core results do not establish this new utility. |
| Robust to architecture? | NOT ESTABLISHED. Architecture-independent software still targets model/explainer-specific labels. |
| Better than simpler baselines? | NOT RUN. Generic multilabel CP already returns the requested object; rank/frequency and strongest-reason controls remain decisive. |
| Strong method contribution? | Not on this specification/evidence. A bounded applied evaluation is plausible without a first-method claim or journal promise. |

The strongest reviewer objection is:

> This is existing multilabel conformal prediction with a generated explanation label.

The target distinction explains **why the application may matter**, not **what
new method was developed**. A new theorem is not mandatory for an applied-method
paper, but a consequential strategy beyond retargeted generic CP would be needed
to justify the proposed method branding. More model families cannot supply it.

The remaining narrow gap is evaluating **joint overstatement and omission of
model-specific reasons after verification**, with an interpretable ambiguity
budget. This is a candidate problem/evaluation contribution. The audit does not
claim another paper solved the exact end-to-end financial task, and it does not
manufacture a replacement architecture after a failed gate.

**Recommended next action:** prepare the evaluation-paper claim–evidence table,
including this reduction and the negative Stable-Core/acquisition findings,
before proposing another method. Any future benchmarking of existing conformal
methods should be labelled accordingly and prospectively specified.

## 9. Change and verification receipt

- Added this audit and a README pointer only.
- Preserved source, configurations, thresholds, groups and historical reports.
- Distinguished full-text evidence, official abstracts and bounded search inference.
- Generated no data, models, benchmark numbers or new test-pass counts.
  The historical 121-test receipt is not a new run.
- Documentation links and diff formatting are checked before publication.
  AI-assisted synthesis/source inspection is not independent peer review.
