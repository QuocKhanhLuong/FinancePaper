# Prospective numerical feasibility gate

Drafted 2026-10-05 before code or numerical results. This is a local prospective
record, not an externally registered study. Implementation is conditional on the
independent reduction review recorded in ROUND2_DECISION.md. The primary candidate
is [conditional joint attribution moments](ROUND2_METHOD_SPEC.md), not a model
search. Historical reports and the first-round NO-GO remain unchanged.

## Fixed hypotheses and comparisons

H1 (correctness): the compiled conditional mean/covariance agree with exhaustive
interventional Shapley enumeration, including off-path dependencies, repeated
features and correlated completion support.

H2 (computational necessity): observation specialization saves work compared with
the identical unspecialized rectangle contraction. Exact inference has a useful
error/time operating point against direct completion TreeSHAP, without calling
an expensive precomputation free. WOODELF-HD plus enumeration/sampling is the
nearest strong prior baseline; superiority over stock SHAP alone is not a novelty
gate pass. Failure to run WOODELF-HD leaves this gate INCOMPLETE.

Primary endpoint: maximum absolute mean/covariance error against exact small-world
enumeration, tolerance 1e-8 for the independent double-precision oracle and 2e-5
against the stock float32-tree SHAP implementation. Computational endpoints are
wall time, compilation time, residual term count, maximum live matrix storage,
and Monte Carlo error versus exact moments. No weighted scalar performance score.

## Fixed worlds and protocol

1. Hand-defined two/three-feature numeric trees: additive stumps; an interaction;
   a constant prediction leaf with attribution dependence on an off-path feature;
   a repeated-feature interval. Background includes correlated rows. Exhaust
   every observation subset and finite completion state.
2. Two completion laws with equal first/second attribution moments and different
   sign probabilities. This tests the limitation of the output, not a desired
   release-policy result. Do not infer event probabilities from moments.
3. Synthetic engineering scale: 23 continuous independent standard-normal
   predictors, Bernoulli outcome with logit
   `-.8 + .9*x0 + .6*x1 + .4*x2`, plus `1.6*x0*x1` in the interaction world.
   Fit a fixed 200-tree depth-3 XGBoost classifier, learning rate .05, n_jobs=1,
   no tuning, 3,200 training rows; background first 64 training rows. Four new
   query rows are independently generated. Hide coordinates {1,3,5,7,9,11,13}.
   Product completion law has four fixed atoms {-1.5,-.5,.5,1.5}, equal weights
   per hidden coordinate (16,384 support points), with observed coordinates fixed.
   It is a declared numerical integration law, not the true Gaussian conditional.
   A second joint finite-support fixture tests correlations explicitly.

Seed 11 first; seeds 22 and 33 only if correctness passes. Reuse no historical
test rows. All model fits are synthetic numerical workload generators, not
evidence of financial predictive improvements. Do not change generators to make
the method faster. Fit/compile/integration timings are separate. CPU only.

Monte Carlo K={8,32,128,1024}; five fixed repetitions after warm-up, with identical
samples supplied to compatible point explainers. Report mean/SD of timing and
error. The exact method has zero sampling error under its supplied law but can
still be slower. Finite law direct enumeration and product rectangle integration
must agree on the small worlds. Never pretend K=8 estimates an event with the
same accuracy as exhaustive integration.

## Budget and stop rules

- 10 minutes total runner budget; 60 seconds per model/query stage.
- Maximum 2,000 residual rectangles; maximum dense joint-mass matrix 32 MiB.
  Exceeding the cap is BUDGET_EXCEEDED, not an implicit MC fallback.
- No extra hyperparameter/model variants, financial training, or external data.
- Any invariant mismatch stops the scientific run until documented and corrected.
- Correctness without strong-baseline advantage permits a reusable oracle only;
  it does not justify a new method paper.
- Incorrect/superficial novelty reduction closes this pilot before implementation.

## Required tests and receipts if implementation proceeds

Explicit missing mask; observed-only specialization; immutable background; exact
point-mass reduction; additive and repeated-feature correctness; correlated joint
law; cross-tree covariance; covariance symmetry/PSD tolerance; grouping linearity;
seed determinism; budget failure; provenance/source/config hashes. Test totals are
software evidence. Numerical fixtures are not independent financial validation.

Financial development is a separate, later protocol and is NOT RUN here. Since
the predictor does not change, an inference compiler cannot claim higher AP,
AUROC or better outcome calibration by construction. Its claim must concern the
accuracy and cost of the declared response-distribution computation.
