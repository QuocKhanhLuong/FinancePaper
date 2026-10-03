# Novelty reassessment after the Stable-Core falsification

Research date: **2026-10-03**. Repository evidence: `140d4f4`. This is an
additive research decision, not a new experiment or an amendment to frozen
Taiwan, Polish or Freddie protocols. No financial dataset was run in this review.

**Verdict: the current evidence does not establish a new explanation algorithm.**
There is a defensible evaluation contribution, but “stable predictions, unstable
explanations,” completion uncertainty, stable subsets and calibration are not
individually new. A larger dataset would strengthen external validity, not create
algorithmic novelty. The one next direction worth a bounded feasibility study is
**budgeted verification of model-attribution claims**. Even this is a candidate
problem contribution, not an established new algorithm.

## 1. What the existing experiment permits us to say

The [measured Stable-Core results](STABLE_CORE_RESULTS.md) and
[decision](STABLE_CORE_DECISION.md) constrain any further proposal:

| Evidence | Implication for the paper |
|---|---|
| Meaningful-positive release-all failure is 2.56% in Taiwan and 3.90% in Polish, with greater coverage than Stable-Core | At the declared 10/15% aggregate budgets, the main coverage-superiority claim fails. A harder target cannot be selected afterward to rescue it. |
| At the 10% primary operating point every Stable-Core rule becomes `q10 > .01` | No evidence that combining sign/rank/magnitude thresholds supplies a distinct algorithmic mechanism. |
| Rank-only is strong on ranked reasons | Monte Carlo or Stable-Core cannot be evaluated only against weak prediction confidence. |
| Matched-size strength controls have 0.47/0.68 percentage points more meaningful failure | Reason identity matters, but this does not show superiority to rank/frequency controls. |
| Every conservative finite-grid policy is empty | Useful empirical coverage is not a deployment risk guarantee. |
| Taiwan/Polish assessment cohorts have been inspected; Freddie files are absent | New results on those cohorts would be exploratory. Large-scale confirmation is NOT RUN. |

The historical grouped top-2 restoration event remains distinct from the newer
reason-level meaningful-positive target. Neither estimates causal correctness.
Ranked release-all failure includes reasons already outside top-2 at the start;
it must not be used as the frequency of newly revised explanations.

## 2. Search and source verification

This is a targeted novelty audit, **not a PRISMA systematic review**. Search terms
included `explanation uncertainty acquisition`, `feature acquisition explanation
stability`, `explanation verification missing features`, `Shapley active feature
acquisition variance`, `explanation value of information`, `partial explanations
missing features`, and exact titles/authors discovered through citation chasing.
The search included current 2026 work and older foundational papers. Broad web
results were screened against original papers, proceedings, publisher pages and
author/institutional records; community summaries were discovery aids only.

Full primary PDFs/method sections were inspected for Guney, Kukar, EDFA, Vo's v3
preprint and AFABench. Other rows below use official abstracts/metadata unless
specified. OpenReview blocked direct retrieval of Beebe-Wang's PDF: its indexed
primary PDF excerpts and the ICML 2025 reference/method discussion corroborate
the limited claims below. No claim of having read that whole paper is made.
The Vo journal DOI endpoint failed to render, but its ScienceDirect and author
institutional records confirm the journal version. LTT's journal publication was
checked against the author's Stanford record and DOI resolution. No API-based
exhaustive citation or retraction audit was performed.

### Closest work and the claim each rules out

| Work and verified status | What it already covers | Remaining distinction, not proof of novelty |
|---|---|---|
| [Ghorbani, Abid & Zou, AAAI 2019, *Interpretation of Neural Networks Is Fragile*](https://ojs.aaai.org/index.php/AAAI/article/view/4252) | Input perturbations can substantially change explanations while preserving the predicted label. | Actual restoration of hidden financial inputs, probability-shift thresholds and a release decision are more specific than general fragility. |
| [Vo et al., *Applied Soft Computing* 196:115105, 2026](https://www.sciencedirect.com/science/article/pii/S1568494626005533); [v3 full preprint](https://arxiv.org/abs/2407.00411v3) | Missingness/imputation changes SHAP; predictive error and attribution error need not improve together. | Our endpoint concerns particular claims from one frozen predictor after verification, rather than comparing imputation workflows. |
| [Golchian & Wright, *Imputation Uncertainty in Interpretable Machine Learning Methods*](https://arxiv.org/abs/2512.17689), 2025 preprint; record reports IJCAI XAI workshop acceptance | Multiple-imputation uncertainty and interval coverage for SHAP, PFI and PDP. | Completion intervals alone cannot be the proposed contribution. |
| [Laberge et al., JMLR 24(364), 2023](https://jmlr.org/papers/v24/23-0149.html) | Partial, consensus attribution statements across a Rashomon set. | We vary missing input values for a fixed model; partial statements themselves are known. |
| [Paes, Wei & Calmon, *Selective Explanations*, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html) | Detect inaccurate amortized explanations and allocate more explanation computation. | Acquisition of true customer information is different from spending more compute to approximate an attribution. |
| [Löfström, Hjort & Löfström, *Guarded Explanations*, COPA 2026](https://proceedings.mlr.press/v329/lofstrom26a.html) | Filters unsupported candidate explanation conditions. Explicitly disclaims a finite-sample error guarantee for its guard. | Verification survival is a different target; suppressing unreliable explanation components is not new. |
| [Angelopoulos et al., *Learn then Test*, Annals of Applied Statistics, 2025](https://doi.org/10.1214/24-AOAS1998); [author publication record](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk) | Independent calibration and multiple testing for risks of set-valued outputs. | Applying established risk control is not a new calibration algorithm. |
| [Angelopoulos, *Conformal Risk Control for Non-Monotonic Losses*, 2026 preprint](https://arxiv.org/abs/2602.20151) | Risk control beyond a monotone scalar family, with stability-dependent bounds. | A more flexible threshold family or nonmonotonic loss is not an unoccupied research area. |
| [Ma et al., *EDDI*, ICML 2019](https://proceedings.mlr.press/v97/ma19c.html) | Conditional completion and expected information gain to acquire costly features for target variables. | Substituting reason-verification labels as targets is a strong baseline, not automatically a new algorithm. |
| [Beebe-Wang, Qiu & Lee, IMLH workshop at ICML 2023](https://openreview.net/pdf?id=1itfhff53V) | Conditional imputations, a fixed predictor and local explanations; acquires a missing feature with high SHAP variance to improve risk assessment. | Must compare against this close, simple acquisition rule. |
| [Guney et al., ICML 2025](https://proceedings.mlr.press/v267/guney25a.html) | Learns instance-specific acquisition orders from explanation rankings; evaluates prediction/cost efficiency. | “Explainability-driven feature acquisition” is already an established method direction. |
| [Kukar, *Active feature acquisition by prediction explanations*, Elektrotehniški vestnik 93(1–2):14–31, 2026](https://ev.fe.uni-lj.si/1-2-2026/Kukar.pdf) | Uses absolute SHAP importance to choose missing fields; includes the 30,000-record Taiwan credit dataset. | Adding SHAP acquisition on Taiwan has especially weak novelty by itself. |
| [Galwaduge & Samarabandu, *Explanations-Driven Active Feature Acquisition for Algorithmic Recourse*, 10 September 2026 preprint](https://arxiv.org/abs/2609.12179) | EDFA jointly addresses acquisition, recourse availability and full-information validity, with calibrated stopping. | The closest new conceptual threat: attribution-claim survival differs from recourse validity, but acquisition plus later verification is already present. |
| [Schütz et al., *AFABench*](https://arxiv.org/abs/2508.14734v3); [author's KDD 2026 record](https://rezarezvan.com/research/afabench/) | Common acquisition benchmarks, shared predictors, costs and myopic/non-myopic comparisons. | A generic acquisition benchmark is not new; a reason-verification target and controlled evaluation would need to add something substantive. |

### The especially important September 2026 collision

EDFA's algorithm uses Markov-blanket units, parent/child priority and conditional
information per cost. Section 4.2 stops using a calibrated prediction-confidence
threshold. Its Eq. 12 includes both prediction inconsistency and recourse
invalidation at full information. It uses XGBoost and a feature-tokenizer
transformer; the XGBoost setting fits models for feature subsets. These details
come from the full preprint, not its broad abstract alone.

Our possible distinction is narrower: preserve a single frozen predictor,
evaluate **attribution claims about initially observed information**, and choose
verification actions by their effect on those claims. This is a difference in
the decision target, not evidence of an algorithmic advance over EDFA. Do not
describe that paper as doing prediction only, or claim that no prior work checks
whether explanations survive fuller information.

## 3. One selected research direction

**Budgeted verification for reason release:** when a record cannot yet support
an explanation, choose which available-but-unverified field to check so that
more model-attribution claims can be released at a declared failure budget.

The research question is:

> At the same information-acquisition cost, does selecting fields to resolve
> reason-verification uncertainty produce more verified reasons than selecting
> fields for predictive accuracy, own-feature importance or own-feature variance?

This adds an action to the current release/withhold system without changing the
credit predictor. Stable-Core becomes one possible terminal communication rule,
not the claimed invention. The [prospective feasibility plan](BUDGETED_REASON_VERIFICATION_PLAN.md)
defines the target, one simple lookahead strategy, the strongest baselines and
stop criteria. It has **not** been implemented or evaluated on financial records.

### A checked mechanism, not a new empirical result

Consider the logit function

\[
f(a,b,c,u)=1+0.2a+(a-b)c+0.01u.
\]

Observe `a=b=1`; hide independent `c,u` taking values `-1,+1` equally often.
Use zero-reference Shapley attributions on the logit (equivalently here, an
independent zero-mean background), and point-impute missing values by zero.
At the partial record, the only positive reason is `a`, with
attribution `.2`. The exact full-record attributions are

\[
\phi_a=.2+c/2,\quad \phi_b=-c/2,\quad \phi_c=0,\quad \phi_u=.01u.
\]

The logit is always `1.2+.01u`. Acquiring `c` gives no predictive information in
this context and its own attribution has zero variance. Nevertheless, `c`
determines whether the originally positive reason `a` remains positive. A rule
that buys the missing field with greatest own SHAP variance prefers `u`, while a
reason-verification objective can prefer `c`. Enumerating every Shapley coalition
for all four complete states verified these identities numerically in this
review. The maximum partial/full probability shift was `0.001784`; the `a` claim
fails in two of four states at epsilon `.01`.

This is an **analytical fixture**, not Taiwan/Polish evidence, not a trained
model, and not proof that a new optimizer is necessary. The example depends on
the specified attribution reference. It does not show that every attribution
method or EDFA fails. A reason-targeted information-gain baseline can solve it.
Cancellation and interaction explanations are established phenomena; the example
only clarifies the mechanism the proposed experiment must test in real data.

The [plan's analytical appendix](BUDGETED_REASON_VERIFICATION_PLAN.md#analytical-appendix)
contains the self-contained coalition-enumeration check; it uses no financial data.

## 4. What could make a paper, and what would not

At most three candidate contributions should be pursued:

1. **Decision-target contribution:** an explicit, auditable distinction between
   verifying a prediction and verifying claims supporting it, with acquisition
   cost and reason failure measured separately.
2. **Mechanism and evaluation contribution:** demonstrate when a field of low
   predictive/own-attribution value has high value for verifying other observed
   reasons, then measure the practical gap under paired controls and temporal
   external validation. The analytical fixture alone is insufficient.
3. **Operational contribution, conditional on results:** a simple verification
   policy that improves the cost–coverage–failure frontier against reason-aware
   baselines, with a reproducible calibration and data-governance protocol.

Do **not** claim a new SHAP method, a new generic uncertainty algorithm, a new
active-acquisition framework, a new conformal technique, causal verification,
the first stable/partial explanations, or the first observation of prediction–
explanation decoupling. Do not add a neural head to manufacture a method claim.

The strongest counterargument is: **“This is EDDI with a different target and
an existing calibrated selector.”** That criticism is valid unless the research
demonstrates a consequential new decision problem and benefits beyond that
retargeted baseline. Beating only prediction-based acquisition would support the
problem formulation, not algorithmic novelty.

## 5. Decision and falsification gates

**GO for one bounded feasibility study; NO-GO for a method-novelty claim now.**
First measure headroom using development data only. Because release-all already
meets some budgets, it may turn out that no costly verification is useful. The
zero-cost policy must remain on every frontier. Keep the old targets/budgets and
do not invent a sub-5% budget because the new method needs one to win.

Stop the method branch if an optimistic one-query oracle cannot improve useful
coverage, simple rank/variance or reason-targeted information gain matches it,
completion families disagree materially, or the temporal confirmation fails.
Freddie is still inaccessible until official files are supplied; no performance
or licence assumption has changed. Any acquisition study would require a
separate prospective protocol, not a silent edit to its frozen Stable-Core run.

**Paper assessment: promising but incomplete as an evaluation/decision-policy
paper; insufficient evidence for a novel-algorithm paper.** Journal suitability
depends on the demonstrated decision benefit, sound calibration, domain meaning
and independent confirmation, not a count of components. ESWA acceptance or
sufficient novelty cannot be promised from this review.

## 6. Adversarial checks and limitations

- Scope check: the user needs publication-worthy originality; adding architecture
  is not equivalent to adding knowledge. Resolved by keeping the predictor fixed.
- Literature check: EDFA already includes full-information recourse validation.
  Resolved by narrowing the target and marking novelty as unestablished.
- Utility check: asking for more information when release-all already suffices
  could create cost without benefit. **Open and blocking for a full experiment.**
- Statistical check: adaptive acquisitions change the state distribution; masks
  drawn independently during old calibration do not validate a sequential policy.
- External-validity check: simulated masks and unit acquisition costs do not
  establish a lender's real verification workflow or customer benefit.

The search cannot prove absence of equivalent work; some full texts were
inaccessible, and no human credit-analyst validation has been performed. Sources
and their publication status should be checked again before submission. AI-assisted
search, synthesis and algebra checking were used; the proposed claims still need
author review. All historical negative results remain part of the evidence.
