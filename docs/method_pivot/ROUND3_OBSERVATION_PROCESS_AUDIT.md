# Third-round continuation: observation-process coupling

Review date: **2026-10-05**. Starting code/report commit:
`bc1346827ff9993da3c9348c811aca9fd794443d`, branch
`research/method-pivot`. This continues the search after the round-two numerical
candidate was rejected. It is a formulation audit, **not a new experiment**.

## Concrete construction considered before code

Suppose a complete-data reference population has discrete feature states `x`
with probabilities `p_x`, and a target deployment provides observable records
`z = (mask, observed values)` with probabilities `nu_z`. Assume the full feature
law and `eta(x) = P(Y=1 | X=x)` transport, and the observation mechanism is
conditionally independent of the outcome given the full features. These are
substantive assumptions, not facts established by the existing Taiwan pilots.

Introduce joint masses `w_xz` satisfying

```text
w_xz >= 0
w_xz = 0                         if x conflicts with z
sum_z w_xz = p_x                for every full state x
sum_x w_xz = nu_z               for every observable state z
```

One proposed benefit was to prevent separately chosen, adverse completion laws
for different missing patterns from reusing more population mass than exists.
For `nu_z > 0`, sharp bounds on the conditional outcome probability under this
declared model are the minimum/maximum of

```text
sum_x eta(x) w_xz / nu_z
```

over the coupled feasible set. A robust probabilistic predictor could minimize
its worst-case expected proper loss over the same set. No hidden value from the
query customer is needed at inference. If the reference population shifts too,
or missingness also depends on residual outcome information, these constraints
need not contain the deployment truth.

## Primary-source check and reduction

| Work | Status; inspected material | Exact overlap | Important scope difference |
|---|---|---|---|
| van Ommen, Koolen, Feenstra & Grünwald, *Robust probability updating* | IJAR 74 (2016), 30–57; author-hosted published PDF, §§1–3 | Unknown coarsening strategy, compatible full-state/message pairs, fixed full-state marginal, minimax probability updating | The proposed target-message marginal supplies an additional constraint; it does not itself supply a new optimizer |
| Zhou, Balakrishnan & Lipton, *Domain Adaptation under Missingness Shift* | AISTATS 2023; official proceedings/abstract | Changing observation process under a stable underlying population | UCAR and available-indicator cases must not be generalized to unrestricted MNAR |
| Rockenschaub et al., *Robust prediction under missingness shifts* | 2024 preprint v1; §§3–5 and assumptions | Distinguishes ignorable and nonignorable shifts; Bayes prediction can change under nonignorable shifts | Does not identify an arbitrary target mechanism from its mask frequency alone |
| Xu & Jiang, *Robust Contextual Optimization with Missing Covariates* | ICML 2026; official PDF, §§2–3 | Latent distributions constrained by their induced observed-data distribution; robust decision optimization | Main construction assumes MAR and complete-case positivity, and conditions its deployment decision on a full context. It is **not** an arbitrary-MNAR theorem for partial deployment inputs |
| Stokes et al., *Domain Adaptation Under MNAR Missingness* | 2025 preprint v1; §§2.2–3 | Uses imputation and domain adaptation under specified MNAR structures | Its imputation/identification assumptions cannot be inferred from observed masks alone |

Official sources:

- [van Ommen et al., DOI](https://doi.org/10.1016/j.ijar.2016.03.001),
  [author-hosted published text](https://homepages.cwi.nl/~pdg/ftp/robprob.pdf).
- [Zhou et al., proceedings](https://proceedings.mlr.press/v206/zhou23b.html).
- [Rockenschaub et al., version inspected](https://arxiv.org/html/2406.16484v1).
- [Xu & Jiang, proceedings](https://proceedings.mlr.press/v306/xu26v.html),
  [official proceedings PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/xu26v/xu26v.pdf).
- [Stokes et al., version inspected](https://arxiv.org/html/2504.00322v1).

**Strongest rejection.** Substituting `w_xz = p_x q(z|x)` into the construction
gives a coarsening game with known marginal constraints. The displayed bounds
are transportation linear programs. Adding a measured observable marginal is
useful information, but the complete proposed inference rule remains standard
constrained minimax updating/partial identification. No new statistical rate,
scalable solver, identifying assumption, or efficient structural reduction has
been established here. A favorable comparison only against independent pattern
bounds would be insufficient: the coupled transport baseline is required.

**Defensible difference still needed.** A demonstrated computational reduction
for high-dimensional financial masks, or a genuinely different estimable
mechanism under a defensible audit design, could justify further work. Simply
feeding this feasible set to an LP solver and naming the output an evidence
envelope would not meet that requirement.

**Decision: NO-GO for this construction as a new method.** No implementation or
financial experiment was started. This does not reject all observation-process
adaptation, and does not claim the cited papers solve the entire FinancePaper
application.

### Why the construction is useful despite failing the novelty gate

An **illustrative finite example, not financial data**: three full states have
`p=(1/4, 1/2, 1/4)` and outcome probabilities `eta=(0,1,0)`. Message `a` is
compatible with states `{1,2}`, and message `b` with `{2,3}`. If both messages
have marginal probability `1/2`, the constraints force state 2 to contribute
mass `1/4` to each message; both conditional outcome probabilities are `1/2`.
Looking only at each message's support would permit the whole interval `[0,1]`.
The coupling therefore removes real ambiguity. But this benefit follows from
the known coarsening/marginal constraints themselves; it is not evidence of a
new learning algorithm. A fair comparator must receive the same marginals.

## Additional nearest-work warning: prediction stability has dedicated methods

The public OpenReview submission *Missing Value Uncertainty: Could Collecting
Missing Values Change the Prediction?* defines a distribution of predictions
after revelation, a prediction-stability confidence, and a direct estimator.
It must be considered before claiming a novel response-distribution learner or
general superiority over prediction-based reliability methods.

- [TMLR submission PDF](https://openreview.net/pdf?id=BRWTS5e03Z).
- [Earlier ICLR submission PDF](https://openreview.net/pdf?id=CZzBapXqAO).

Evidence status: indexed primary-PDF passages inspected; direct OpenReview
retrieval later returned a browser-verification challenge. The retrieved PDF
retains an old **under review** label. A subsequent independent check, verified
during integration, found the official
[TMLR accepted-paper listing](https://jmlr.org/tmlr/papers/): David J. Burnett,
Surojit Ganguli, Lance M. Kaplan, Devesh Upadhyay and David I. Inouye,
**August 2026**. The listing links an
[author code release](https://github.com/inouye-lab/MissingValueUncertainty/releases/tag/tmlr).
Acceptance, authorship and the existence of that release are therefore verified;
the implementation itself has not been audited or run. A secondary index's
ICLR label is not used. No performance numbers from this work were reproduced.
Prediction stability and explanation stability remain different
targets; existence of this method does not establish that it detects reason
revision well.

Historical FinancePaper results remain comparisons against the specific
prediction-only controls actually run. They are not evidence that every
prediction-distribution method is weak.

## Independent alternative: invert task states instead of imputing coordinates

An independent research reviewer generated a distinct candidate before receiving
the coarsening construction above: compress the frozen model's risk/reason event
to a finite state `Z`, use complete audit records to estimate an observation
operator `M`, and infer a task functional from `r = M theta`, where
`theta = P(Z)` and `r` is the observable-record law. If `M' delta = a`, then
`a' theta = delta' r`; otherwise minimize/maximize `a' theta` over the compatible
simplex. This addresses a population functional, not an individually known
hidden value. Transporting `M` requires the observation process to be stable
within the chosen states or additional identifying information.

**Reduction gate: NO-GO.** The representer mechanism is a finite-state analogue
of the operator/null-space representer geometry studied by
Li, Miao & Tchetgen Tchetgen's *Non-parametric inference about mean functionals
of non-ignorable non-response data without identifying the joint distribution*
(JRSSB 85(3), 2023, 913–935,
[DOI](https://doi.org/10.1093/jrsssb/qkad047),
[institutional record](https://ir.pku.edu.cn/handle/20.500.11897/685440)).
It identifies functionals through an operator condition without requiring the
whole latent law. Its shadow-variable observation model differs from the audit
matrix above; this is a mathematical reduction, not literal model equivalence.
Substituting a frozen model output for the functional supplies no new inversion
principle.

Two further required controls are Cheng et al.'s
[ACCMV construction](https://arxiv.org/abs/2207.02289) (2022 preprint; limited
complete cases under an explicit identifying assumption, with IPW, regression
adjustment and multiply robust estimation) and Miao & Tchetgen Tchetgen's
[shadow-variable identification](https://doi.org/10.5705/ss.202016.0322)
(Statistica Sinica 28(4), 2018, 2049–2067). An auxiliary learned representation
does not automatically satisfy a shadow-variable condition. None was run here.
The independent review inspected the operator reduction; primary metadata and
the institutional representer description were checked during integration.

A **hand-derived synthetic falsifier, not a benchmark result**, is

```text
M = [[1, 1/2, 0],
     [0, 1/2, 1]]
theta_A = (1/3, 1/3, 1/3)
theta_B = (1/2, 0,   1/2)
a = (0, 1, 0)
```

Both yield `r=(1/2,1/2)`, but the probability of the **second state** is `1/3`
versus zero. In fact the identified interval for that probability is `[0,1]`.
The null direction `(1,-2,1)` explains the ambiguity. A point estimate claiming
to recover the state probability without additional information is invalid.
Random complete audits can supply additional information, but their direct
audit/IPW estimator then becomes an indispensable baseline using the same audit
design. This does not assert equal finite-sample efficiency: an inversion method
would still need to establish an efficiency or calibration advantage, rather
than claim a new source of identification.

The reviewer found no new identification mechanism in this construction.
This does not imply that every audit-based learning algorithm is useless.

## Independent alternative: audit-anchored adaptive acquisition

A second independent reviewer proposed learning a sequential verification policy
using a small randomized complete-audit sample, avoiding reliance on an imputer
to evaluate that policy. Let `W` be current observations and mask, `A` the audit
indicator with known propensity `rho(W)>0`, and `L_i(pi)` the outcome loss plus
acquisition cost obtained by replaying policy `pi` on an audited complete row.
For a cross-fitted conditional loss estimate `m_pi(W)`, the proposed value
estimator was

```text
V_hat(pi) = mean_i [m_pi(W_i) + A_i/rho(W_i) * (L_i(pi) - m_pi(W_i))].
```

The residual term is evaluated only for audited rows. A simultaneous upper
confidence bound over a finite preregistered policy class would be compared
against a no-acquisition policy. Outcome availability, audit positivity and
audit randomization conditional on `W` are required. Replaying acquisition also
requires that acquiring a value does not change its underlying value/outcome.

| Nearest work | Status and source | Overlap and reduction | Remaining difference needed |
|---|---|---|---|
| von Kleist et al., *Evaluation of Active Feature Acquisition Methods for Time-varying Feature Settings* | JMLR 26(60):1–84, 2025; [official paper](https://www.jmlr.org/papers/v26/23-1635.html) | Acquisition-policy evaluation under missing-data/offline-RL assumptions; direct, IPW and double-RL estimators | Random complete audits change the information design, not the AIPW identity |
| von Kleist et al., static-feature companion | 2023 preprint v2; [source](https://arxiv.org/abs/2312.03619v2) | Semi-offline policy evaluation for static values; MNAR extensions need additional missing-data assumptions | A new estimator or provable efficiency advantage beyond these controls |
| Valancius, Lennon & Oliva, *Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition* | ICML 2024; [official proceedings](https://proceedings.mlr.press/v235/valancius24a.html) | Nongreedy acquisition balancing information value and cost | Acquisition is not new merely because the terminal cost includes a frozen model diagnostic |

The reviewer inspected the policy-evaluation reduction; official abstracts and
metadata were independently verified during integration. Code/license was not
audited and none of these baselines was run. They are required comparisons, not
claimed reproductions.

**Strongest rejection.** For one fixed policy, the displayed estimator is the
standard augmented inverse-probability estimator of an audited mean. Sequential
policy replay is already central to AFAPE. Restricting the policy class and
requiring a confidence bound does not by itself supply a new learning mechanism.

**Defensible difference still needed.** A demonstrated improvement over matched
semi-offline/DR evaluation at the same audit and acquisition budget, or a new
identifying result under a strictly different defensible information design.
Neither has been established. **NO-GO for the current method claim**, before
implementation.

**Falsifier.** If a high-risk stratum has zero audit probability and no additional
identification assumption, its unobserved policy losses cannot be recovered by
the displayed estimator. Near-zero probability should produce high variance,
not confident selection. Any implementation reporting confident benefit there
without additional information would fail the design check.

This does not generalize the repository's failed one-field headroom experiment
to all sequential acquisition. That historical result and this algebraic
reduction are separate reasons concerning separate proposals.

## Evidence and execution boundary

- **VERIFIED:** primary mathematical formulations above were inspected;
  the displayed finite coupling reduces directly to the stated known programs.
- **REPORTED:** prior numerical and financial results retain their existing
  statuses and reports; this review does not remeasure them.
- **NOT RUN:** mechanism-coupling, task-inversion or acquisition pilots,
  dedicated MVU baseline, new financial
  development, and independent confirmation.
- **ASSUMED:** stable complete-data law, transported `eta`, outcome-independent
  selection conditional on `X`, and known population masses in the proposed
  finite formulation.

The independent adversarial review checked both finite examples and the
transport constraints. Its wording corrections on MVU publication status,
representer scope and audit-baseline efficiency are incorporated above.

**Round-three decision: no additional candidate passed the method gate.** No
code or new financial-data run follows from this review. This bounded search
cannot prove absence of all possible new methods. It establishes the specific
reductions that prevent the three current proposals from being presented as
successful novelty results. The user's requested supported method contribution
remains **unachieved**; the numerical round-two result is still the last new
experiment, and its negative control is retained.
