# Round 7 — decision after independent generation and exact falsification

**NO-GO for implementing any current Round 7 proposal as a new method.**
Exactly one candidate was selected for deeper scrutiny: verification-anchored
learning under completion misspecification. Its supplied fixed-teacher mechanism
is AIPW; its naive jointly trained extension introduces the wrong variance
penalty. The conditional-moment repair is a known baseline, not yet a distinct
contribution. [Formulation](ROUND7_LEARNING_FORMULATION.md),
[executed evidence](ROUND7_FALSIFICATION_RESULTS.md).

This decision is not based on requiring novel ingredients. An established
component can be part of a new method. Here the claimed distinctions either
reduce exactly to the matched method, use a different logical event, or rely on
an unsupported numerical/theoretical advantage. Neither worker's initial
positive recommendation survived coordinator verification and guided review.

## What genuinely remains open

There is a meaningful difference between (i) correcting a frozen model's
completion functional and (ii) efficiently learning a family of predictors
under limited verification with a changing teacher/representation. The latter
is **not proved solved by saying 'GMM'**. What is missing is a specified
finite-audit or finite-compute obstruction for the strongest conditional-moment
baseline, plus an algorithmic step overcoming it. No such step has been
derived here. This is an unresolved research question, not a promised novelty.

The most dangerous controls now are:

1. Same-audit AIPW regression for a fixed teacher.
2. Same-audit conditional-moment learning for joint predictors, with a matched
   critic/function class and explicit approximation error.
3. Robust active statistical inference if the contribution moves to measurement
   allocation. Uniform sampling alone is too weak a comparator.
4. Direct supervised ERM on masked inputs when Y is already available, plus
   generalized distillation with the same teacher data. Without a finite-budget
   advantage, learning a conditional expectation indirectly is unnecessary.

No new financial experiment follows this failed gate. XGB25, LR, existing GRUs,
reason definitions and all historical files remain unchanged. We did not run
TabM, Freddie, conformal envelopes, a neural revision head or a replacement
predictor. Taiwan/Polish remain reused research benchmarks; future confirmation
requires a frozen method and eligible, genuinely separate cohort.

## Evidence and claims

| Status | This round |
|---|---|
| VERIFIED | Exact finite means/variances, joint-gradient counterexample, normalized sensitivity LPs, allocation correction, XOR symmetry; 228 passing tests |
| REPORTED | Historical financial decoupling/detector findings read from repository; original research paper results not reproduced |
| NOT RUN | Joint neural training, three-seed financial pilot, independent confirmation, large-scale data |
| ASSUMED | Randomized verification and positivity in toy formulation; these have not been established for a real loan verification workflow |

Supported contribution remains evaluation/policy. This round adds a reproducible
falsification audit, **not a supported method contribution**, a venue-acceptance
forecast, or evidence that future learning approaches cannot work.

## One next action

**Establish a finite-verification-budget headroom bound for joint
conditional-moment learning against the strongest same-information baseline.**
Do this analytically/with a frozen finite toy before a new financial run. The
bound must separate critic approximation, audit variance and optimization cost;
an apparent benefit caused only by a weaker critic, extra audits, or suppressing
informative updates closes the direction. This is a narrower next gate, not an
authorization to invent another architecture or relabel the known dual objective.
