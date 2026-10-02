# Temporal model: literature decision

Decision frozen 2026-10-03, before temporal implementation or test evaluation.
This is a targeted literature audit through that date, not an exhaustive systematic
review. Source access and search logs are recorded in [research notes](research_notes/).
The existing protocol takes precedence; this branch does not replace LR/XGBoost,
change the restoration event, or establish causal/correct explanations.

## Decision A — justified as an experiment

**GO for a bounded temporal comparison; NO-GO for a new architecture claim.**
The official [UCI documentation](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)
verifies six monthly histories, April–September 2005, and no natural missingness
in 30,000 records. A six-step sequence can test recency and ordered trajectories;
it does not establish long-memory benefits. Wide XGBoost already sees all months.
The random customer split cannot establish generalization to a future calendar
period or to naturally missing banking records.

## Nearest verified work

“Not reported” means absent from the inspected source, not proof of absence from
all versions. Abstract-only records are marked A; other entries were inspected
in full text or an author manuscript. Results in these papers are not our results.

| Work | Model | Temporal? | Missing-aware? | Imbalance handling | Explanation | Reliability output | Gap relevant to us |
|---|---|---|---|---|---|---|---|
| [Che et al. 2018, GRU-D](https://doi.org/10.1038/s41598-018-24271-9) | GRU with learned input/hidden decay; also GRU-Simple | Yes | Observed mask, elapsed delta; decay toward training mean | Not credit-specific | None | None | Establishes our mask/delta inputs; no restoration reason target |
| [Cao et al. 2018, BRITS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html) | Bidirectional recurrent imputation | Yes | Mask/delta, differentiable imputations, consistency | Not credit-specific | None | None | Joint imputation/prediction already known |
| [Yoon et al. 2019, M-RNN](https://doi.org/10.1109/TBME.2018.2874712) | Multidirectional RNN | Yes | Within/across-stream imputation | Not credit-specific | None | Multiple-imputation capability | Does not predict verification-induced explanation changes |
| [Du et al. 2023, SAITS](https://doi.org/10.1016/j.eswa.2023.119619) | Masked self-attention imputation | Yes | Masks, observed/masked reconstruction | Not credit-specific | None | None | Strong alternative imputer; unnecessary first-pilot complexity at T=6 |
| [Kim et al. 2023](https://proceedings.mlr.press/v202/kim23m.html) | Probabilistic imputer + classifier | Yes | Completion distribution | Not credit-specific | None | Prediction/imputation uncertainty | Uncertainty is not our reason-revision event |
| [Wang et al. 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f88264fcc54775ee1706116e90fe351a-Abstract-Conference.html) | Task-oriented imputation evaluation | Yes | Multiple imputers | Not central | None | Downstream task evaluation | Reconstruction quality alone does not justify our branch |
| [Taiwan BiLSTM, 2021](https://doi.org/10.1186/s40537-021-00461-7) | Static branch + six-month BiLSTM | Yes, same dataset | Not a mask/delta robustness study | Predictive/calibration evaluation | Model interpretation | Calibration | Taiwan temporal/static separation is already published |
| [Thor & Postek 2024](https://doi.org/10.1002/for.3057) (A) | Corporate-default GRU; NewConnect annual histories | Yes | Not reported in abstract | Not established from abstract | Not reported | Not reported | Recurrent default prediction is established; different data |
| [Zandi et al. 2025](https://doi.org/10.1016/j.ejor.2024.09.025) | Dynamic graph + LSTM/GRU; Freddie Mac | Yes | Median fill, no missingness stress study | BCE; AUC/F1 | SHAP; attention diagnostics | No reason reliability | Credit recurrence + attribution already known |
| [Yang et al. 2026, KAN-GRU/LSTM](https://doi.org/10.1016/j.asoc.2026.115213) | Recurrent early-default models; Freddie Mac | Yes | Padding masks, not feature masks | Undersampled train/test | None reported | None reported | Test rebalancing cannot establish population calibration |
| [Yang et al. 2026, ResE-BiLSTM](https://doi.org/10.3390/info17010005) | Recurrent mortgage model | Yes | Deletes missing selected fields | Resampling comparisons | SHAP | No revision target | Explainable credit RNN itself is not new |
| [Han et al. 2025, SMART](https://doi.org/10.1038/s41598-025-99997-4) | rSVD + GAIN; Taiwan | No | Simulated MCAR imputation | Downstream AUC | None | None | Missing-data Taiwan evaluation already known |
| [Chang et al., 23 Sep 2026 preprint](https://www.preprints.org/manuscript/202609.2039) | Structured DBN + GRU/MLP + Bayesian head; Taiwan | Yes | Training masking, reconstruction; MCAR/MAR/MNAR stress | BCE | No feature-reason revision evaluation | Calibration, predictive uncertainty, selective prediction | Very close prior: temporal + masking + uncertainty is not a new combination; **not peer reviewed** |
| [Vo et al. 2026](https://doi.org/10.1016/j.asoc.2026.115105) | Missing-data prediction/SHAP study | Not required | Imputation comparison | Not central | SHAP | Explanation sensitivity | Prediction stability does not imply explanation stability |
| [Golchian & Wright 2025 preprint](https://arxiv.org/abs/2512.17689) | Multiple-imputation explanation uncertainty | Not required | Yes | Not central | SHAP/PFI/PD | Attribution intervals | MI and attribution uncertainty are already known |
| [Paes et al. 2024, Selective Explanations](https://proceedings.neurips.cc/paper_files/paper/2024/hash/647af5f6b2538524f6c047c1d9170fd9-Abstract-Conference.html) | Amortized explainer + selection | Not required | Not this target | Not central | Learned explanation | Explainer approximation uncertainty | Generic learned explanation reliability/selection is not novel |
| [Löfström et al. 2024](https://doi.org/10.1016/j.eswa.2024.123154) | Calibrated explanations | Not required | Not this target | Not central | Feature weights/counterfactuals | Calibrated uncertainty | Different uncertainty/event from restoring hidden truth |
| [Chen et al. 2024](https://doi.org/10.1016/j.ejor.2023.06.036); [Ballegeer et al. 2025](https://doi.org/10.1016/j.ejor.2025.05.039) | Imbalanced/cost-sensitive credit models | Not required | Not primary | Explicit imbalance/cost study | SHAP/LIME | Explanation stability | Imbalance can affect explanations; not our restoration event |

Further alternatives and access limitations: [missing series](research_notes/missing_series.md),
[credit studies, including 2024–2026](research_notes/credit_temporal.md),
[losses and explanation reliability](research_notes/reliability_losses.md).
Bidirectional processing of six already historical months is valid at October
prediction; only a separate online intermediate-month deployment would prohibit
using later months. We defer BRITS for scope/cost, not because this task forbids it.

## Decision B — one primary architecture

Choose **GRU-Simple with values, observed masks and elapsed deltas**, plus a
separate static MLP. Call it `mask_delta_gru`; it is not GRU-D. This input design
already appears as a baseline in the GRU-D paper. Vanilla GRU remains its control.

Our inference from the evidence is that full GRU-D decay is premature here:
regular monthly timing and short missing runs provide limited evidence for a
learned decay curve; repayment status is categorical; monetary mean reversion is
not established. GRU-D's input decay toward a train mean and hidden decay toward
zero are meaningful additional priors, not free missingness handling. A later
same-budget decay ablation can falsify this engineering choice. SAITS/BRITS/CSDI
remain possible imputation comparators, not necessary first implementations.

Under MCAR, masks are independent of the target by design. Under our MAR stress
test they depend only on always-observed anchors. Neither simulates evidence
that real reporting behavior predicts default. The branch tests robustness to a
declared simulator, not naturally informative missingness.

## Loss, attribution and auxiliary tasks

**Unweighted BCE is the default.** Weighted BCE uses training negatives/positives;
focal loss uses separately configured alpha/gamma. Focal loss, modified focal
loss, class weights, balanced sampling, cost-sensitive XGBoost and threshold
optimization are known. A focal-aware credit scorer already exists
([10.1016/j.eswa.2022.118158](https://doi.org/10.1016/j.eswa.2022.118158)).
We do not combine these treatments. Focal loss is not a strictly proper posterior
loss ([theory](https://openaccess.thecvf.com/content/CVPR2021/html/Charoenphakdee_On_Focal_Loss_for_Class-Posterior_Probability_Estimation_A_Theoretical_Perspective_CVPR_2021_paper.html));
empirical calibration gains are setting-dependent
([Mukhoti et al.](https://papers.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)).
Resampling can harm risk calibration
([van den Goorbergh et al.](https://doi.org/10.1093/jamia/ocac093)).
Report AP, ROC-AUC, classwise precision/recall/F1, raw/calibrated Brier and log loss,
calibration bins and ECE at natural test prevalence.

Use **Integrated Gradients on the raw logit**
([Sundararajan et al.](https://proceedings.mlr.press/v70/sundararajan17a.html)).
It permits direct field/month aggregation and a completeness check with ordinary
PyTorch gradients. Use continuous encoded values, including one-hot category
coordinates, never integer-ID interpolation. This path can pass through soft
categories, an acknowledged off-manifold limitation. GradientSHAP introduces
sampling/baseline variance; DeepSHAP support for the exact recurrent operations
would need additional validation. Neither is the primary method.

Masks/deltas remain fixed on each IG path. Thus financial-value attributions are
conditional on input availability; their reference logit can change after
verification. Log that reference and separately isolate value-only restoration
from full restoration. IG and interventional TreeSHAP are different attribution
estimands; compare model-specific revision and coverage descriptively, not their
raw magnitude as a common unit of explanation correctness.

**Stage F is deferred.** Reconstruction is known (BRITS, SAITS and the recent
credit preprint). If later tested, use status CE and amount Huber separately,
train-only normalization, and report extra time/parameters. Retain it only after
a same-budget robustness/calibration/revision ablation. A future frozen-predictor
revision head must train on honest held-out or out-of-fold restoration labels;
it may use only current representations/masks/attribution diagnostics at inference.
Calibrate its release rule independently. Compare missing fraction, prediction
variance, attribution variance, Monte-Carlo revision and a strong learned selector.
Those selectors are **not implemented by the current static pilot**; their absence
must not be presented as an experimental comparison.

## Decision C — what could be new?

**Known:** credit RNNs, Taiwan six-month/static branches, mask/delta and decay,
training corruption, reconstruction, focal/weighted losses, calibration, SHAP/IG,
prediction uncertainty, learned explanation uncertainty and selective explanations.

**Adaptation:** a compact shared-preprocessing comparison under matched simulated
missingness, conditional IG mapped to original fields, and explicit value/context
restoration diagnostics. These are useful engineering, not an architecture claim.

**Candidate contribution:** evaluate and eventually predict the event that
currently released observed reasons require revision when hidden truth is
verified for the **same frozen predictor**. The bounded search did not find this
exact target/protocol already solved. That is not proof of priority. A learned
head is only useful if it adds calibrated coverage over the strongest simple and
learned baselines. The paper remains primarily an explanation-reliability study.

## Frozen falsifiers and next decision

The one-seed local pilot is a feasibility/negative-result screen, not confirmatory
evidence. No test-driven model replacement or architecture tuning is allowed.
For a later five-seed paired study, predeclare these practical improvement gates:

- Against a shared-input augmented XGBoost control with a matched corruption
  budget (at least four rate-specific training views in the later study): +0.01 AP
  at MCAR30 and MAR30, without calibrated Brier worsening >0.005 or log loss >0.01.
- Against augmented vanilla GRU: improvements must survive no-mask/no-delta
  controls. Static-only, temporal-only, flat-neural and retrained fixed-order
  permutation controls must establish what temporal ordering contributes.
- At matched observed-reason coverage, target at least 5 percentage points lower
  revision rate without a predictive calibration penalty. IG/SHAP reference
  sensitivity must be analyzed before any cross-model reliability claim.
- Paired record bootstrap and seed variability must support a gain; no win from
  withholding more explanations or giving one model more input information.

The first screen uses one augmented view per XGBoost/LR training row, whereas
neural models see new masks each epoch. This bounded control is not the strongest
augmentation baseline and cannot by itself satisfy the later-study gate. Current
IG-versus-joint-TreeSHAP revision comparisons are descriptive; the five-point
criterion requires attribution/reference sensitivity analysis first.

Failure to meet these gates falsifies the proposed improvement at these declared
margins; it does not prove that every recurrent model is inferior. Stable
prediction with unstable reasons is a relevant finding even when the GRU loses.
A single Apple Silicon Mac is a reasonable engineering target for this bounded
experiment. Actual MPS correctness/runtime must be measured, not inferred from
CPU success. Full study and selector experiments remain separate future work.

**Final research decision: GO for the specified experimental branch; NO-GO for
promoting the recurrent model or a reliability head to the paper's main contribution
without the above evidence.** Exact implementation contract follows in
[TEMPORAL_MODEL_SPEC.md](TEMPORAL_MODEL_SPEC.md).
