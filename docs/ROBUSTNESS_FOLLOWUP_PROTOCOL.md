# Bounded robustness follow-up — frozen 2026-10-03

This prospective plan follows the already inspected temporal pilot. It preserves
the research question, same-predictor restoration event and original baselines.
The request to find a model that wins every metric is a search objective, not a
result we can guarantee. Strict improvement over a zero revision rate is
mathematically impossible. Report wins, ties and losses, including failed
candidates; do not change denominators or thresholds after seeing test results.

## Literature decision

The [prior temporal review](TEMPORAL_MODEL_RESEARCH.md) still applies. Further
primary sources inspected on 2026-10-03 support a bounded experiment:

| Prior work | Relevant finding | Consequence for this round |
|---|---|---|
| [Robust Attribution Regularization, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/172ef5a94b4dd0aa120c6878fc29f70c-Abstract.html) | Training for robust IG is established | An attribution loss would be an adaptation, not a novelty claim |
| [ExpO, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/770f8e448d07586afbf77bb59f698587-Paper.pdf) | Differentiable regularization studies the accuracy/explanation-quality tradeoff | Do not assume stability and prediction improve together |
| [Neural Additive Models, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html) | Nonlinear additive predictors are already known | An additive anchor is an established control |
| [SCARF, ICLR 2022](https://openreview.net/pdf?id=CuV_qYkmKb3) | Feature corruption is established tabular representation learning | More corruption exposure is not new; distinguish marginal replacement from our missing-value masking |
| [SHAP, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | Additive feature attribution framework | Explain the actual blended logit using a common reference |
| [Adaptive analysis and holdout reuse, NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html) | Adaptive reuse can invalidate ordinary holdout claims | This follow-up is exploratory, even after freezing its own selection |

**Decision:** first close the tree augmentation-exposure gap and test shrinking
interactions toward an additive predictor. No new dependencies, no new recurrent
architecture, and no novelty claim. This isolates a plausible source of reason
changes with exact interventional TreeSHAP linearity. Gradient/IG consistency
training is deferred: endpoint-gradient matching does not guarantee IG stability,
and a second-order surrogate adds cost before the simpler interaction control is
measured. This is an engineering and methodological choice, not evidence that
the deferred approach cannot work.

For an additive logit `a(x)=b+sum_j a_j(x_j)` and a fixed interventional reference,
an observed financial field's attribution cannot change when other fields are
restored. This is a structural consequence, not proof of correct reasons. Trees
of depth one provide a nonlinear additive control. A blended logit
`g(x)=(1-w)*a(x)+w*t(x)`, with a fixed global `w`, has attribution
`phi_g=(1-w)*phi_a+w*phi_t` under the same background. We do not average
probabilities and then falsely use this logit-attribution formula. No hidden
true values or restored explanations are inputs at inference.

## Fixed experimental contract

- Keep the seed-42 five partitions and original evaluation masks/background64.
  Do not move previously seen test rows into training. The reason-risk calibration
  partition stays unused. The test is already exposed from the pilot; results
  are exploratory, not an independent confirmation or external validation.
- Restart seeds `[42,43,44,45,46]` change initialization and training corruption,
  not the customer split or test masks. Mean/sample SD describe training
  variability on one fixed cohort, not five independent datasets.
- Reuse the train-only temporal preprocessor, monthly mapping and flattened
  values/masks/deltas. No new financial features or imputation method.
- Preserve complete/one-view LR and XGBoost controls, augmented vanilla GRU and
  augmented mask/delta GRU with BCE. Their original hyperparameters remain fixed.
- Multi-view trees see 25 independently masked training views at the original
  `[0,.1,.2,.3]` schedule. View `e` uses exactly the neural callback's mask seed
  `restart_seed+700000+e`. Weight each copied row by `1/25`, so each customer's
  total training weight is one and regularization is not weakened by replication.
  This covers all possible 25 neural epochs, including views after early stopping;
  report actual neural epochs. It is an exposure-covered control, not identical
  optimization effort or identical effective exposure after checkpoint selection.
- Four additional tree components: the original depth3/200-tree settings with
  25 views; and regularized depth1/depth2/depth3 trees with 600 trees, learning
  rate .05, min_child_weight20, reg_lambda10, full row/column sampling, one CPU
  thread. The depth1 component is the additive anchor.
- The multi-view depth3/200 control and the regularized depth3/600 interaction
  component are mandatory standalone test baselines, alongside the original six
  controls. This separates the benefit of blending from greater mask exposure
  and from the component's regularization/tree count.
- Candidate pool: those four components plus blends of the additive anchor with
  each of the three nonadditive components at `w=.25,.5,.75` (13 candidates).
  Fit components once; blends add no training. No test-driven expansion.

## Selection, frozen before any new test evaluation

Use only restart42 for the bounded candidate screen. AP checkpoint/candidate
selection uses complete/MCAR10/MCAR30 development records, never MAR. Each
candidate gets its own positive-slope Platt fit on the reserved probability
calibration records pooled over these three conditions. F1 thresholds use pooled
development data as before. Probability calibration data never fit model weights.

Keep two explicitly different choices:

1. `predictive_choice`: highest mean development AP, tie lower calibrated log loss.
2. `joint_choice`: among candidates within .005 AP and .005 ROC-AUC of the best
   candidate mean, and within .002 calibrated Brier and .005 calibrated log loss
   of the predictive choice, minimize mean development MCAR10/MCAR30 revision at
   common 50% coverage; tie higher AP then name. If common coverage cannot reach
   50%, the joint gate fails and returns the predictive choice with failure logged.

These are predeclared screening tolerances, not significance/noninferiority
margins established for lending. Recall/F1/ECE are always reported but not used
to manufacture a multiobjective search score. Fix selected component recipes
and blend weights globally for all five restarts; retain the additive anchor as
a diagnostic control. Record the full development leaderboard and joint gates.

Fit all five restarts and their calibration/thresholds before the test phase.
Write a frozen selection with every selected recipe, training ID hashes,
attribution rules and source hashes. No test metrics are computed in the search
or fit phases. A resumed evaluation must verify the frozen source/artifact hashes.

## Explanation and reporting contract

Use 256 fixed development and 400 fixed test explanation IDs shared with the
temporal pilot. Tree/LR explanations use joint interventional raw-logit SHAP;
recurrent explanations retain conditional IG with fixed availability per path.
The latter distinction remains a cross-family interpretation limitation.
Aggregate signed encoded contributions to the same 23 original fields.

Use fixed `k=3`, minimum positive raw-logit attribution `.01`, original event
epsilon `1e-6` and rank tolerance0 for every model, selected before this run.
This replaces the previous model-specific development magnitude screen only in
this explicitly separate experiment. Report raw eligibility, numerical failures
and coverage so an empty or compressed explanation cannot masquerade as stable.
IG completeness uses the existing 32/64/128-node checks. Blended attribution must
pass completeness for the actual blended logit, with all metadata contributions
included in that check and excluded from released financial reasons.

Primary restoration restores values and availability. Complete controls must
have zero shift/revision. Store original/full attributions, per-record reasons,
scores, signs/ranks, eligibility, probability shift, original hidden masks and
revision events. Report top-k overlap, observed sign agreement and rank correlation.
The follow-up focuses on full restoration; value-only diagnostics remain in the
original pilot and are not silently relabelled as this run's measurements.

Report prediction ROC-AUC/AP/Recall/F1/Brier/log loss/ECE, both raw and calibrated,
classwise precision/recall/F1/support and calibration bins, for all 4500 test
records in each condition. Report mean/sample SD across restarts. For explanations,
report revised/eligible counts and matched coverage (.25/.50/1.0) on common
eligible IDs per restart/condition, always dividing coverage by all400 IDs.
Selection is by current third-reason strength, never restoration outcome.

Export a per-metric win/tie/loss table for each selected candidate versus each
baseline: calibrated prediction metrics in all four conditions, mean absolute
calibrated restoration shift and matched-50% revision under the three missing
conditions, plus eligible coverage/overlap/sign/rank diagnostics. Numerical tie
tolerance is `1e-8`; no practical or statistical dominance claim from a tiny win.
Undefined or unavailable metrics fail the dominance check rather than being
silently dropped. Strict-all-wins and weak Pareto dominance are reported separately.

No calibrated explanation-release guarantee, correctness claim, novel architecture
claim or four-way reliability labels. Stop this bounded search after its frozen
comparison and retain negative results. Further research requires a new declared
  protocol rather than silently extending the grid until test metrics turn positive.

Implementation correction before the first follow-up test evaluation: an initial
fit-only attempt at `outputs/robustness_followup/` was interrupted during restart44
because the standalone multi-view tree controls were missing from the evaluation
list. No test evaluation ran. Retain that incomplete directory; the corrected
complete run uses `outputs/robustness_followup_controlled/`. The candidate pool,
selection rule and selected blend weights were not changed by this correction.

## Orchestration status

Requested Orca coordination run `run_e85605ccca99`, task `task_0984a86463b0`:
two Codex starts failed at `agent_readiness: timeout`; the Claude attempt showed
`zsh: parse error near ')'` and no observed agent turn. It was explicitly stopped.
All three worker terminals were released; no worker produced research or code.
The root continues directly. This round has no independent Orca worker review;
software/artifact checks must not be described as an independent scientific audit.
