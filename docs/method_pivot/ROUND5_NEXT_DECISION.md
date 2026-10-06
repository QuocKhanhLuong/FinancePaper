# Round 5 — close the generic direct-rank construction

Decision date: **2026-10-06**. **NO-GO for a method claim based only on contrast
projection followed by event-probability integration.** Do not train another
revision head or implement a generic probability solver under a new name.

## What changed from round four

Round four found incremental information in explanation statistics beyond the
tested prediction-distribution controls. It did not establish a faster inference
algorithm. Round five asks the separate question of computing the relevant event
directly. The [primary-source audit](ROUND5_RANK_INFERENCE_AUDIT.md) distinguishes
fixed-input numerical SHAP ranking uncertainty from hidden-input verification
uncertainty, but target distinctness alone does not establish algorithmic novelty.

The [executed falsifier](ROUND5_RANK_INFERENCE_RESULTS.md) supplies one concrete
boundary: even exact attribution mean/covariance can conceal .50 versus .20
revision probability. This supports studying event distributions, **not** a claim
that our direct weighted-counting control is new or that rank instability fails.
On this toy, rank instability exactly rescales the event probability.

## Reduction test and dangerous baseline

For finite completions, the exact answer is `sum_l w_l E(z_l)`. For an encodable
density and piecewise compiled attribution map, it is weighted model integration
of a Boolean/numeric event. Contrasts are ordinary linear projection; common-term
cancellation can be exposed to an existing generic verifier. Region splitting
with lower/upper probability accumulation is also established. See
[Morettin et al.](https://arxiv.org/html/2402.04892v2) and
[Boetius et al., ICML 2025](https://proceedings.mlr.press/v267/boetius25a.html).

The dangerous comparator is **the same projected expression**, with the same
completion law and numeric/error budget, passed to generic weighted counting/
integration or evaluated by compiled point queries. Comparing only against full
stock SHAP recomputation would hide existing compilation and cancellation gains.
TopShap addresses another randomness source, so it is relevant prior art but not
an interchangeable benchmark with a future-verification probability query.

No source audit proves a generic solver wins every workload. No finite toy proves
the new construction cannot scale. The decision is narrower: **no new necessary
algorithmic step or cost/error advantage has yet been established**. Neither a
failed literature gate nor an unrun real benchmark is an empirical failure.

## One next action

Conduct **one bounded structural-feasibility audit of frozen XGB25 contrast
formulas on development inputs**, before constructing any new solver. Measure
whether conditioning and common-term cancellation create substantial reusable
structure beyond the same simplifications applied to a generic weighted-counting
baseline. Charge compilation and reuse costs; exclude historical test customers
from selection and do not use restored outcomes to select formulas.
Compare error/confidence requirements explicitly: a fixed K8 estimator is not
automatically an equal-accuracy alternative to a deterministic probability bound.

This is a headroom check, not a candidate novelty claim. If no exploitable
structure remains beyond the generic representation, close this inference branch
instead of tuning event thresholds or inventing another acronym. Any subsequent
new method still needs an exact mechanism, matched comparator and preregistered
error/cost test. No algorithm may pass merely by returning moments, constant
predictions, empty explanations or unbounded intervals.

The supported paper remains an evaluation/policy contribution. Round five adds a
reproducible limitation and a baseline oracle, not a defensible new-method paper.
The [AGY review receipt](ROUND5_ORCHESTRATION_REVIEW_RECEIPT.md) records the actual
worker settlement and the mathematical overclaims rejected by the coordinator.
