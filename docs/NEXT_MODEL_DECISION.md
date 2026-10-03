# Next-model decision — 2026-10-03, before outer assessment

**GO with one strategy: frozen XGBoost 25-view plus a separate, generic calibrated
revision selector.** Reposition as a verification-target/evaluation/policy study.
**NO claim of a novel recurrent architecture or a novel reliability head.**
No predictive architecture or loss is changed. The former 4,500 test customers
and 4,500 reserved risk-calibration customers remain excluded.

The [metric audit](REVISION_METRIC_AUDIT.md) and development Experiment B pass
the predeclared feasibility gate:

1. XGB revision remains 8.55%/6.66% under .01 rank tie gaps at MCAR30/MAR30;
   semantic grouping also retains events. It is not solely floating-point ties.
2. At |delta p|≤.02, MCAR30 XGB has 53 revised cases among 440 eligible stable
   cases; mask/delta GRU 79/544. More than 20 such cases occur before assessment.
3. Generic prediction-only boosted selector has mean fold AUROC .5741 at MCAR30
   and .5871 at MAR30. Adding current attribution summaries gives .7527/.7881.
   Prediction uncertainty is far from a perfect revision detector here.
4. [The literature review](SOTA_COMPARISON_2026.md) establishes close precedents,
   including learned explanation-quality selection. It does not establish this
   exact verification event/release evaluation as solved. This is not proof of novelty.
5. The best generic mask/prediction/attribution selector reaches development
   AUROC .7731/.7957, making a practical release policy plausible.

The strongest neural architectural falsifier is already the generic selector.
Do not build a neural head merely to reproduce it. Choose between fixed logistic
and shallow boosted recipes using **each fold's own diagnostic AP**, then freeze
all calibration/policies before any outer assessment. The strongest method-level
falsifier is the train-only completion Monte Carlo revision estimator. Retain
entropy, max probability, variance, missing fraction/groups, attribution variance,
rank/sign instability and prediction-only learned selectors.

## What the GRU diagnosis actually supports

With identical values, switching mask/delta GRU to all-observed context changes
MCAR30 reasons in 13.06% of valid eligible diagnostic cases and shifts probability
by .0240 on average; zeroing delta alone gives 2.51% and .00605. Vanilla ignores
both by construction. Value-only restoration at original context gives 8.64%
revision versus 15.31% under full restoration. This supports **availability-context
dependence** as a contributor. These nonlinear interventions do not add up into
a causal decomposition, and deliberately inconsistent masks are stress inputs.

The six fixed-pair absolute second differences average .00814 for mask/delta,
.01056 for vanilla and .01004 for XGB (additive numerical zero). This does **not**
support the claim that mask/delta simply has stronger interactions on these pairs.
It does not rule out other interactions. Mean-reference, expected-IG and a common
permutation explainer retain the qualitative vanilla/mask-delta contrast.

Directed diagnostic AP, mean across three folds:

| Input/encoder control | Complete | MCAR30 | MAR30 |
|---|---:|---:|---:|
| Vanilla: neither mask nor delta | .5137 | .4853 | .4652 |
| Mask only | .5302 | .4961 | .4900 |
| Delta only | .5114 | .4839 | .4714 |
| Mask + delta | .5160 | .4829 | .4746 |
| Static only | .3177 | .2823 | .3117 |
| Temporal only | .5238 | .4926 | .4812 |
| XGB25 | .5286 | .4863 | .4825 |

Repayment histories clearly contain predictive information relative to the five
static fields. These controls do **not** establish that recurrence/order is better
than flattened history; parameter matching and order-shuffled/flat controls would
be needed. Delta is not established as necessary; mask-only is stronger in these
development results. We do not promote it using the old test or change the next
strategy based on these predictor ranks. Keep both GRUs as controls, not a central
method. Artificial MCAR missingness is not naturally informative default behavior.

## Conditions for narrowing or abandoning

Stop architectural claims if the generic selector matches a neural head; that
decision is already taken. Stop method superiority if Monte Carlo/generic baselines
match coverage at the same revision budget. If conservative release permits very
little coverage, report insufficient calibration evidence rather than relax the
test threshold. If external known-cell restoration fails, limit to Taiwan.
If human-meaningful tie/group definitions remove the phenomenon, abandon the
operational framing. ESWA suitability remains contingent on external validation,
strong baseline comparisons and domain-grounded release costs—not model complexity.

Gate evidence: `audit_manifest.json`, `baseline_manifest.json` and
`revision_baseline_metrics.csv` under `outputs/revision_study/`; both manifests
state `outer_opened: false`. The global research decision uses pooled diagnostic
evidence on an already explored benchmark. Subsequent outer results are bounded
internal assessment, not an independent confirmatory nested-CV estimate of this
entire adaptive research process. Per-fold selector fitting/selection/calibration
never accesses its own outer customers.
