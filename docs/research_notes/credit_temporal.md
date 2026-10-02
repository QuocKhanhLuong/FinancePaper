# Temporal and missing-aware credit-model research note

**Search date:** 2026-10-03.  This note is a bounded literature audit for the
Taiwan credit-card pilot.  It does not claim a systematic review of every paper
published by the search date.  Primary publisher pages, official proceedings,
official repositories, and author manuscripts were preferred.  Claims below are
limited to what could be verified from those sources; an abstract-only source is
identified as such.

## Repository alignment

The repository's methodological target is reason-revision risk: a fixed model
produces reasons from an incomplete record, the hidden values are restored, and
we measure whether the top reasons change.  A recurrent model is therefore a
comparator or robustness instrument, not automatically the paper's contribution.
The existing question does not require an RNN to win on AUC, and a temporal model
must be retained only if it changes the robustness or explanation-reliability
conclusions under matched controls.

## The official Taiwan chronology

The official [UCI Default of Credit Card Clients record](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)
states that the data contain 30,000 rows and 23 explanatory features, have no
original missing values, and track monthly payment records from April through
September 2005.  It gives the following meanings: `PAY_0` is September,
`PAY_2` is August, through `PAY_6` in April; `BILL_AMT1` is the September bill
through `BILL_AMT6` in April; and `PAY_AMT1` is the amount of the previous
payment in September through `PAY_AMT6` in April.  The target is next-month
default (`default payment`, Yes = 1).

The only defensible oldest-to-newest temporal tensor is therefore:

| step | month in the source | repayment status | statement bill | amount of previous payment |
|---:|---|---|---|---|
| 0 | April 2005 | `PAY_6` | `BILL_AMT6` | `PAY_AMT6` |
| 1 | May 2005 | `PAY_5` | `BILL_AMT5` | `PAY_AMT5` |
| 2 | June 2005 | `PAY_4` | `BILL_AMT4` | `PAY_AMT4` |
| 3 | July 2005 | `PAY_3` | `BILL_AMT3` | `PAY_AMT3` |
| 4 | August 2005 | `PAY_2` | `BILL_AMT2` | `PAY_AMT2` |
| 5 | September 2005 | `PAY_0` | `BILL_AMT1` | `PAY_AMT1` |

This ordering matters because the source column names run in reverse time and
`PAY_0` is a special name for the latest month.  The official wording is
“amount of previous payment”; it should not be silently reinterpreted as a
precisely timed settlement of the named bill.  The six rows are monthly fields
in one credit record, not six independent customer observations.  The [UCI
dataset DOI is 10.24432/C55S3H](https://doi.org/10.24432/C55S3H); the introductory
Yeh and Lien paper is [10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020).

Six monthly snapshots are enough to test whether recency, changes, and ordered
repayment states add information beyond a wide table.  They are not enough to
justify claims about long-range memory.  Every recurrent result needs a wide
XGBoost control, a static MLP control if a neural baseline is used, and temporal
order controls (permuted and reversed months).  If chronology permutations do
not change the result, the model is using a nonlinear tabular representation,
not temporal structure.

## Closest methods and credit papers

| Work | Model and data | Temporal? | Missing-aware? | Imbalance and evaluation | Explanation / reliability output | Gap relevant to this repository |
|---|---|---|---|---|---|---|
| Che et al., **GRU-D**, *Scientific Reports* 2018, [10.1038/s41598-018-24271-9](https://doi.org/10.1038/s41598-018-24271-9), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216/), [arXiv](https://arxiv.org/abs/1606.01865) | GRU variant for clinical time series (MIMIC-III, PhysioNet, synthetic tasks). | Yes; irregular clinical observations. | Yes. A per-variable mask and elapsed time since last observation enter trainable input and hidden-state decays. | Clinical classification, primarily AUC/cross-validation; not credit imbalance. | No attribution-reliability or calibrated explanation output. | Establishes mask + delta + decay as known. The authors explicitly motivate informative missingness in healthcare; under simulated MCAR in Taiwan, the mask must not be treated as naturally target-informative. |
| Cao et al., **BRITS**, NeurIPS 2018, [official proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf), [arXiv HTML](https://arxiv.org/html/1805.10572), arXiv DOI [10.48550/arXiv.1805.10572](https://doi.org/10.48550/arXiv.1805.10572) | Bidirectional recurrent imputation and downstream classification/regression on air quality, healthcare, and activity data. | Yes; timestamps may be irregular. | Yes. Mask and delta are inputs; missing values remain differentiable variables; forward/backward imputation consistency is trained jointly. | Imputation and task metrics; no credit-specific imbalance/calibration study. | No prediction uncertainty or explanation reliability. | Directly establishes that joint recurrent imputation + prediction is known. Taiwan has six historical months before the next-month label, so bidirectionality would be valid for retrospective record scoring; it is deferred because the first pilot tests a smaller, interpretable mask/delta control rather than because later historical months are unavailable at inference. |
| Du, Côté & Liu, **SAITS**, *Expert Systems with Applications* 2023, [10.1016/j.eswa.2023.119619](https://doi.org/10.1016/j.eswa.2023.119619), [arXiv full text](https://arxiv.org/html/2202.08516), [official code](https://github.com/WenjieDu/SAITS) | Self-attention imputation on multivariate time series; four public datasets. | Yes. | Yes, but primarily as an imputation method: mask, masked-imputation task, observed reconstruction, and two diagonally masked self-attention blocks. | Imputation MAE and downstream pattern-recognition results; no credit calibration/reliability. | No credit explanation output. | Stronger imputation alternative, not a compact credit classifier. For `T=6`, SAITS is a future sensitivity baseline rather than the first architecture to add. |
| Thor & Postek, **Gated recurrent unit network: A promising approach to corporate default prediction**, *Journal of Forecasting* 2024, [10.1002/for.3057](https://doi.org/10.1002/for.3057), [publisher abstract](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3057) | GRU on sequences of annual financial statements, announcements, and prices for NewConnect companies; bankruptcy/default context. | Yes; genuine annual corporate histories. | No mask/delta or missingness experiment reported in the accessible publisher record. | Compares Cox, boosted Cox, XGBoost, random survival forest and RNN; publisher abstract says out-of-sample forecasts, but details are not fully accessible here. | No explanation reliability or uncertainty output reported in the abstract. | GRU for default is already established; the dataset and annual corporate setting do not establish value for six monthly Taiwan fields. |
| Zandi et al., **Attention-based dynamic multilayer graph neural networks for loan default prediction**, *European Journal of Operational Research* 2025, [10.1016/j.ejor.2024.09.025](https://doi.org/10.1016/j.ejor.2024.09.025), [publisher page](https://www.sciencedirect.com/science/article/pii/S0377221724007288), [author full text](https://arxiv.org/html/2402.00299v2) | Freddie Mac mortgage loans; rolling six-month behavioural windows, GNN snapshot embeddings, then LSTM/GRU; default within the following year. | Yes; genuine monthly windows. | Nulls are median-imputed. No mask/delta/MCAR/MAR experiment. | BCE; reports AUC and F1 with bootstrap intervals. The dynamic GAT-LSTM-attention model is only modestly above XGB in the reported tables; no Brier/log loss/ECE. | SHAP for node features and learned temporal attention scores. Attention is a diagnostic in this paper, not a validated explanation-reliability measure. | Strong evidence that recurrent credit models and six-month windows are known. It does not test hidden-value restoration, explanation revision, or missing robustness. |
| Qiu & Wang, **Credit Default Prediction Using Time Series-Based Machine Learning Models**, *Artificial Intelligence and Applications* 2025, [10.47852/bonviewAIA52023655](https://doi.org/10.47852/bonviewAIA52023655), [open PDF](https://ojs.bonviewpress.com/index.php/AIA/article/download/3655/1322/28208) | American Express Default Prediction; 2–13 statements per customer and 191 features; CNN–LSTM–attention compared with XGB, LightGBM, NN, and LR. | Yes; genuine statement sequences. | Mode imputation for categorical and median for numerical features; no mask/delta or simulated missingness. | Reports accuracy, precision, recall, F1, AP and ROC-AUC, but calls accuracy the principal metric. The accessible paper does not describe a reproducible temporal split in sufficient detail and does not report calibration. | Calls attention “interpretability”; no SHAP or explanation-reliability test. | Shows that generic CNN/LSTM/attention credit claims already exist, but its split and near-perfect metrics require caution. It cannot support a temporal novelty claim. |
| Kwon, An & Kim, **Temporal Pattern-Based Credit Default Prediction**, *Journal of KIISE* 2025, [10.5626/JOK.2025.52.8.660](https://doi.org/10.5626/JOK.2025.52.8.660), [official full text](https://jok.kiise.or.kr/full-text/1873) | American Express monthly statements; zero-padding for short sequences and LSTM with T-SMOTE. | Yes; 13-month customer sequences, with variable length/padding. | Padding mask only; no observed-value mask, elapsed-time decay, MCAR/MAR, or true-value restoration. | T-SMOTE and other resampling; leaderboard-style normalized Gini/top-4% metric rather than calibrated probabilities. | No attribution or reliability output. | Confirms that short credit sequences and temporal imbalance augmentation are active topics; T-SMOTE and LSTM are not novel here. |
| Yang et al., **Kolmogorov–Arnold Networks-based GRU and LSTM for Loan Default Early Prediction**, *Applied Soft Computing* 2026, [10.1016/j.asoc.2026.115213](https://doi.org/10.1016/j.asoc.2026.115213), [arXiv full text](https://arxiv.org/html/2507.13685v1), [publisher page](https://www.sciencedirect.com/science/article/pii/S1568494626006617) | Freddie Mac mortgage data; 12–27 month windows, 3–12 month early-prediction gaps, OOT year splits; GRU-KAN/LSTM-KAN vs GRU/LSTM/attention/Transformer. | Yes; genuine monthly repayment sequences. | The Keras masking layer only ignores padding in variable-length sequences; it is not missing-value modeling. | Random undersampling is applied to both train and test to equalize classes; reports accuracy, precision, recall, F1 and AUC at 0.5, not AP/Brier/log loss/calibration. | No SHAP or explanation-reliability output. | Confirms GRU/LSTM + early-warning credit work is current and non-novel. Balancing the test set makes its probabilities unsuitable as a calibration precedent. |
| Wang & Bellotti, **Long short-term memory network with adapted attention mechanism for credit risk modeling**, *International Journal of Forecasting* 2026, [10.1016/j.ijforecast.2026.07.002](https://doi.org/10.1016/j.ijforecast.2026.07.002), [publisher abstract](https://www.sciencedirect.com/science/article/pii/S0169207026000622), [replication code](https://github.com/harryGod1/Replication_IJF/) | Discrete-time survival model + LSTM/adapted attention and washout phase on US mortgage transaction histories. | Yes; genuine discrete transaction time series. | No mask/delta or incomplete-record experiment described in the accessible abstract. | Survival fit/forecast comparisons; full article was not accessible through the publisher during this search. | Adapted attention is used to uncover temporal features; no explanation-reliability target. | Latest evidence that LSTM-attention credit risk is an active, established direction. It does not address post-verification reason revision. |
| Yang et al., **Transforming Credit Risk Analysis: A Time-Series-Driven ResE-BiLSTM Framework**, *Information* 2026, [10.3390/info17010005](https://doi.org/10.3390/info17010005), [publisher page](https://www.mdpi.com/2078-2489/17/1/5), [arXiv](https://arxiv.org/abs/2508.00415) | Freddie Mac; 44 cohorts, sliding windows and a 2-month gap before a 3-month observation target; ResE-BiLSTM vs LSTM, BiLSTM, GRU, CNN and RNN. | Yes; genuine monthly repayment histories. | Borrowers with missing selected features are dropped; no missing-aware mask/delta. | 70/30 OOS split with same-borrower separation; evaluates multiple train-only resampling strategies and AUC/F1/etc. | SHAP analysis of feature/time importance. No uncertainty or reason-revision evaluation. | The closest recent combination of recurrent credit prediction and SHAP still treats missingness by deletion, leaving incomplete-record explanation reliability open. |
| Han, Jung & Yoo, **SMART: Structured Missingness Analysis and Reconstruction Technique for credit scoring**, *Scientific Reports* 2025, [10.1038/s41598-025-99997-4](https://doi.org/10.1038/s41598-025-99997-4), [publisher full text](https://www.nature.com/articles/s41598-025-99997-4) | UCI Default of Credit Card Clients (the same complete 30k-row benchmark); rSVD denoising + GAIN imputation, evaluated at 5–80% simulated MCAR. | No sequence model; treats the 23 fields as a static table. | Yes for static imputation. It explicitly states the source data are complete and generates artificial MCAR. | Compares imputation RMSE and downstream AUROC against GAIN variants, MICE, MissForest and simple methods; no temporal branch. | No SHAP, explanation reliability, or calibrated probability analysis found. | This is direct precedent for simulated missingness on the same Taiwan benchmark, but not for temporal modeling or restoration-based reason revision. |
| Chang et al., **Uncertainty-Aware Credit-Risk Prediction with Structured Bayesian Temporal Modeling**, *Preprints.org* 2026, [v1 preprint](https://www.preprints.org/manuscript/202609.2039) (submitted 22 September and posted 23 September 2026; explicitly not peer-reviewed; no DOI was listed on the record) | Same UCI Default of Credit Card Clients benchmark after cleaning (29,965 customers); five static fields and six ordered monthly triples. Structured borrower-conditioned DBN + a GRU and flattened MLP pathways, fused with a variational BNN. | Yes; the paper uses the same six oldest-to-newest monthly observations. | Training independently zero-masks temporal elements and reconstructs the original sequence; this is denoising augmentation, not an observed-value mask/delta/GRU-D mechanism. Tests 10–30% MCAR, latest-month removal, MAR, and MNAR. | Ten paired seeds (42–51), 64/16/20 stratified split, natural test prevalence, AUROC/AUPRC/F1/Brier/10-bin ECE; compares LR, GRU, static Bayesian, flattened/lag-summary XGBoost/LightGBM, and Transformer + MC dropout. | Bayesian weight samples, predictive entropy, mutual information, temperature scaling, and fixed-coverage error. It states attention is only temporal aggregation, and the record contains no SHAP, feature attribution, top-reason revision, or explanation-reliability endpoint. | This is a very recent same-benchmark precedent for temporal modeling + synthetic missingness robustness + uncertainty/selective prediction. It removes any broad novelty claim for that combination. The remaining candidate gap must be the hidden-value restoration test for same-model reasons, and it should use this preprint as a close robustness/uncertainty comparator where feasible. |
| Liu et al., **RMT-Net: Reject-aware Multi-Task Network for Modeling MNAR Data in Financial Credit Scoring**, [arXiv](https://arxiv.org/abs/2206.00568), arXiv DOI [10.48550/arXiv.2206.00568](https://doi.org/10.48550/arXiv.2206.00568) | Approved/rejected loan applications; missing labels for rejected applicants. | Not a monthly sequence. | Yes, but MNAR selection/reject inference rather than missing feature values. | Multi-task default/rejection learning; not directly comparable to MCAR field masking. | No reason-revision reliability output. | Important boundary: “missing credit information” can mean reject-selection labels, but this paper does not solve our hidden-feature restoration problem. |
| Teimoori, Ashraf & Shahbeyk, **Leveraging RNNs for Predicting Loan Default: A Dual Approach with and without Uncertainty Considerations**, ESI 2024 proceedings, [conference proceedings PDF](https://www.qu.edu.qa/en-us/conference/esi2024/Documents/ESI2024%20Conference%20Proceeding.pdf) | Abstract describes traditional and uncertainty-aware RNNs for loan default; dataset/method details were not verifiable beyond the proceedings abstract. | Claimed RNN default prediction; evidence is abstract-only. | Not established. | Not established. | Prediction uncertainty is the stated focus; no explanation revision. | Generic uncertainty-aware RNN is also already proposed. No DOI or reproducible full method was located, so it cannot support a novelty claim. |

### What the recurrent literature establishes

1. GRU/LSTM for financial default, repayment, bankruptcy, mortgage survival, and
early warning are established applications.  A vanilla GRU, an LSTM, an
attention layer, or a GRU/LSTM plus a reconstruction objective must be treated as
known components.
2. GRU-D establishes mask and elapsed-time decay; BRITS establishes joint
bidirectional imputation and prediction; SAITS establishes self-attention
imputation.  None of these is a credit explanation-reliability contribution.
3. Recent credit papers usually either impute once, drop incomplete borrowers,
or mask only padding.  The targeted search did not find a primary credit paper
that combines a GRU-D/BRITS-style observed-value mask and delta with exact hidden
value restoration, same-model top-k reason revision, and a reliability release
decision.  This is a **candidate gap**, not a priority claim of novelty: it still
requires a broader systematic search and a clear distinction from selective
explanations.
4. Metrics in many recent papers are accuracy/F1/AUC only, use resampled test
   sets, or call attention “interpretability.”  Those practices cannot establish
   calibrated risk or reliable reasons.

### New same-benchmark overlap found on 2026-10-03

The [Chang et al. v1 preprint](https://www.preprints.org/manuscript/202609.2039)
is unusually close to this branch: it uses the same Taiwan benchmark, the same
five static attributes and six ordered monthly triples, a compact GRU pathway,
training-time temporal masking, reconstruction, MCAR/MAR/MNAR and latest-month
stress tests, ten paired seeds, calibration metrics, and entropy-based selective
prediction.  Its abstract reports AUROC 0.7623 under 30% MCAR and a 3.51% error
at 10% coverage; the full preprint reports the same comparisons against LR,
GRU, XGBoost, and a Transformer.  This source is a **v1 preprint explicitly
marked not peer-reviewed**, not an established peer-reviewed result, but it is
still a direct warning against claiming that temporal masking, reconstruction,
Bayesian uncertainty, or selective triage on this six-month Taiwan benchmark is
new.

The preprint independently zero-masks temporal elements and trains a
reconstruction branch; it does not pass an observed-value mask and elapsed-time
delta through a GRU-D-style decay.  More importantly for this repository, the
paper does not compute feature attributions or compare incomplete versus
restored top-k reasons.  Its attention weights are explicitly described as
temporal aggregation rather than feature explanations.  No reason-revision
event, explanation-reliability predictor, or verification-aware release rule is
reported.  Its own limitations are one public dataset, synthetic missingness,
overlapping repeated holdouts, and no genuine temporal/institutional external
validation.  Therefore the proposed branch should be framed as a replication and
extension of a now-nearby robustness/uncertainty line, with the scientific test
centered on reason revision rather than on temporal prediction or uncertainty
alone.

## Explanation reliability and imbalance precedents

The repository's current gap is narrower than generic “explainable RNN”:

- [Vo et al., *Explainability of Machine Learning Models under Missing Data*,
  arXiv:2407.00411](https://arxiv.org/abs/2407.00411), published as *Applied
  Soft Computing* 2026, [10.1016/j.asoc.2026.115105](https://doi.org/10.1016/j.asoc.2026.115105),
  shows that imputation choice changes SHAP values and interactions and that
  lower prediction MSE does not imply lower SHAP MSE.  “Missingness changes
  SHAP” is therefore not a new claim.
- [Slack et al., *Reliable Post hoc Explanations: Modeling Uncertainty in
  Explainability*, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/4e246a381baf2ce038b3b0f82c7d6fb4-Paper.pdf)
  models uncertainty in local attributions with Bayesian LIME/KernelSHAP.
- [Löfström et al., *Calibrated explanations*, ESWA 2024,
  10.1016/j.eswa.2024.123154](https://doi.org/10.1016/j.eswa.2024.123154)
  provides feature importance uncertainty and counterfactual rules.
- [Paes, Wei & Calmon, *Selective Explanations*, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/647af5f6b2538524f6c047c1d9170fd9-Paper-Conference.pdf)
  predicts when an amortized explainer is inaccurate and selectively spends
  more computation.  Its target is approximation error to a reference explainer,
  not revision after hidden customer values are verified.  A revision-risk head
  must state this distinction and compare against selective-explanation and
  SHAP-variance baselines.
- [Sundararajan, Taly & Yan, Integrated Gradients, ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
  is a practical attribution choice for a differentiable recurrent model.  It
  can attribute every `(month, field)` input and then aggregate six monthly
  contributions back to the original PAY/BILL/PAY_AMT fields.  DeepSHAP or
  GradientSHAP can remain a sensitivity analysis, but runtime and recurrent/MPS
  compatibility should be checked.  Learned attention weights must be logged as
  diagnostics, not presented as explanations.

Class weighting, balanced sampling, and focal loss are standard imbalance
controls.  [Focal Loss](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper)
([10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)) downweights easy
examples; [Liu et al.'s focal-aware cost-sensitive credit scorer](https://doi.org/10.1016/j.eswa.2022.118158)
already applies focal loss to credit scoring.  The [class-imbalance calibration
study](https://doi.org/10.1093/jamia/ocac093) shows why reweighting/oversampling
can distort estimated probabilities.  For this paper, standard BCE on the
natural training prevalence should be the default risk objective, with weighted
BCE and focal loss as explicit ablations.  If weighting is selected for recall,
calibration must be fitted on an untouched validation set and all Brier/log-loss
results must be reported on the natural-prevalence test set.  Never rebalance the
test set for the main probability or calibration result.

## Architecture decision

### Decision A — Is a temporal model justified?

**Conditional GO for an experimental branch; NO-GO as an automatic new paper
contribution.**  The UCI fields have a real monthly order and the last six
months precede the next-month default label, so testing an ordered representation
is scientifically defensible.  The sequence is only six steps, and XGBoost
already receives all six months as a wide vector.  The temporal branch is worth
implementing only to test whether ordered history changes missing-data robustness
and reason revision, not because RNNs are expected to outperform.

### Decision B — Primary architecture

Use a **compact GRU-Simple mask/delta-aware GRU** as the one primary temporal
model.  This means a standard GRU receives a normalized value, an observed-value
mask, and a per-feature monthly delta.  It has no learned GRU-D decay in this
pilot; full GRU-D decay is a deferred ablation.  Call the implemented model
“mask/delta-aware GRU” or “GRU-Simple,” never a novel GRU-D.  Compare it to a
vanilla GRU without mask/delta.

Reasons:

- GRU-D establishes why the supplied missingness information is useful, while
  the selected GRU-Simple control keeps the first six-step comparison small and
  auditable.  It has lower parameter/runtime cost than LSTM, which is appropriate
  for 30k rows and `T=6`.
- Monthly sampling is regular.  Set the base interval to one month; if a field is
  hidden for consecutive months, delta is the number of elapsed months since its
  previous observed value.  Document the first-step convention (for example,
  delta 0 at the sequence origin, then one month per subsequent step).
- BRITS would add bidirectional differentiable imputation and consistency losses
  that are unnecessary for the first falsifiable comparison.  All six Taiwan
  months are historical at next-month inference, but retaining the observed
  versus hidden distinction is central to the restoration experiment.
- SAITS is a strong future imputation comparator but is likely over-capacity for
  six steps and is not a direct explanation-reliability model.

### Decision C — Candidate contribution and its limits

**Known components:** recurrent credit scoring; GRU/LSTM temporal encoders;
GRU-D mask/delta/decay; BRITS joint imputation; SAITS self-attention imputation;
class weights/focal loss; masked reconstruction; SHAP/Integrated Gradients;
uncertainty-aware and selective explanations.

**Adaptation:** putting the Taiwan monthly fields into an oldest-to-newest six-step
tensor; separating static demographics/limit from monthly history; evaluating
MCAR/MAR masking; aggregating neural attributions to original fields/months; and
comparing restored-versus-incomplete reasons under a fixed model.

**Candidate contribution:** an empirical, evidence-bounded test of whether a
mask/delta-aware temporal encoder improves *reason-revision risk* or selective
reason release over static LR/XGBoost and close temporal/missingness baselines
under matched simulated missingness.  The 2026 same-benchmark preprint means
that temporal robustness, denoising/reconstruction, Bayesian uncertainty, and
selective prediction alone cannot be presented as the gap.  This remains a
candidate contribution only if the literature search is broadened, the controls
below pass, and the result is replicated.  It must be abandoned as a central
contribution if the temporal-order controls fail or the restoration-based
reason-revision endpoint is no better than existing baselines.

## Controls required to establish temporal value

Use the same leakage-safe split, preprocessing fit, masks, seeds, and natural test
prevalence for every model.  The minimum comparison is:

1. LR and XGBoost on the complete wide schema;
2. a static MLP, if a neural-capacity control is needed;
3. vanilla GRU on six ordered monthly triples plus static branch;
4. mask/delta-aware GRU on the same inputs;
5. each recurrent model trained on complete records versus missingness augmentation.

Add the following falsification controls before interpreting a win:

- **Temporal order:** random month permutation and reverse chronology at test time;
  a true temporal benefit should be measurably sensitive to order.
- **Feature summaries:** XGBoost with latest-month values, maxima, slopes/differences,
  bill/limit and payment/bill ratios.  This tests whether the GRU only learns
  simple engineered trends.
- **Branch ablations:** temporal-only, static-only, and fused models; no-mask and
  no-delta versions; mask-only and delta-only diagnostics.
- **Missingness mechanism:** MCAR 10/30% first; MAR 30% and contiguous/latest-month
  blocks as stress tests.  Because the source file has no missing values, a mask
  must not be interpreted as a naturally informative customer behavior.  Under
  MCAR, a mask-only classifier should perform at chance after split uncertainty.
- **Restoration test:** train/freeze the model, generate an incomplete explanation,
  restore only the hidden true fields, regenerate the explanation with the same
  model, and report top-k overlap, rank correlation, sign agreement, probability
  shift, and the revision event.  A reliability head must never see the hidden
  values at inference.
- **Calibration and imbalance:** ROC-AUC, AP, class-wise precision/recall/F1,
  Brier, log loss, calibration curve/ECE, and risk-coverage for release.  Choose
  thresholds and calibrators on validation only.
- **Uncertainty and explanation baselines:** compare the revision predictor with
  missing fraction, probability variance (MC dropout or mask ensemble), SHAP/IG
  variance, and the existing selector baseline.  Attention scores are not a
  substitute for attribution.
- **Uncertainty of the estimate:** use repeated seeds and record-level bootstrap
  intervals.  Keep complete-record identity when resampling; do not treat the six
  monthly fields as independent rows.

## Final research recommendation

1. **Should the paper use an RNN?** Yes as a controlled experimental branch if the
   objective is to test temporal robustness; no as an assumed winning replacement
   for XGBoost.
2. **Vanilla GRU or GRU-D?** Use a compact mask/delta-aware GRU as the primary
   candidate and vanilla GRU as its required ablation.  GRU-D itself is prior art.
3. **Is six months enough?** Enough for a falsifiable recency/trend test because
   the UCI chronology is real; insufficient for claims about long-range memory.
4. **Default imbalance loss?** Standard BCE for calibrated risk probabilities;
   weighted BCE and focal loss are ablations.  If one is selected for recall,
   recalibrate and evaluate at natural prevalence.
5. **Potential contribution beyond credit GRU/LSTM?** Only the restoration-based
   reason-revision experiment and a validated release-risk head, if they improve
   risk-coverage/revision prediction under matched controls.  The architecture
   itself is not a contribution.
6. **Already known:** recurrent credit scoring, six-month repayment windows,
   attention, mask/delta decays, imputation/reconstruction, class-weight/focal
   losses, SHAP/IG, uncertainty-aware and selective explanations.
7. **Falsifier:** if chronological order permutation/reversal has no effect, or
   mask/delta GRU does not improve probability degradation, explanation revision,
   or revision-risk AUROC/AP over XGBoost/feature-summary controls with acceptable
   calibration, the temporal improvement is not supported.
8. **Apple Silicon:** a one-layer hidden-size 32–128 GRU with six steps and 30k
   rows should fit a single Mac using PyTorch MPS or CPU fallback.  Avoid claiming
   MPS success until the exact attribution and decay operations are exercised.
9. **Debug outputs:** log raw and calibrated risk, missing fraction/per-month
   masks/deltas, hidden states in optional debug mode, static/temporal/fused
   embeddings, attribution by `(month, field)` and aggregated original feature,
   probability variance, top-k reasons before/after restoration, revision event,
   revision-risk probability, release decision, seed, mask ID, and model/data
   hashes.
10. **Paper emphasis:** keep the paper primarily an explanation-reliability and
    selective-release paper.  Make the missing-aware temporal model central only
    after it demonstrates a reproducible improvement in the reason-revision
    endpoint that survives the controls above.

## Search log and access notes

Representative searches run on 2026-10-03:

- `GRU-D Recurrent Neural Networks for Multivariate Time Series with Missing Values official paper DOI publisher`
- `BRITS bidirectional recurrent imputation for time series official paper DOI publisher`
- `SAITS self-attention imputation time series official paper DOI publisher`
- `credit default prediction GRU LSTM repayment history 2024 2025 2026 DOI`
- `credit default GRU LSTM temporal payment history 2025 journal DOI`
- `loan default prediction recurrent neural network monthly repayment 2024 DOI`
- `credit default missing data GRU-D BRITS recurrent credit scoring mask delta`
- `loan default missing-aware recurrent neural network mask time gap credit scoring`
- `class imbalance credit scoring weighted binary cross entropy focal loss calibration DOI`
- `explanation reliability prediction model selective explanations 2024 NeurIPS official`
- `missing data SHAP explanations imputation uncertainty 2024 2025 DOI`
- `"UCI Default of Credit Card Clients" GRU LSTM temporal`, `PAY_6 GRU credit default`
- `Uncertainty-Aware Credit-Risk Prediction Structured Bayesian Temporal Modeling 2026 preprint`

Full text was available for GRU-D (PMC/Nature), BRITS (arXiv/NeurIPS), SAITS
(arXiv), Zandi (arXiv author manuscript), Qiu (open PDF), Kwon (official full
text), Yang KAN (arXiv), SMART (Nature), and ResE-BiLSTM (arXiv/publisher
abstract).  The Thor and Wang/Bellotti records were verified through publisher
abstract pages; the latter publisher full text returned an access error during
this search, although its DOI and public replication repository were visible.
The Teimoori uncertainty-aware RNN was located only in a conference proceeding
abstract and has no verified DOI.  Those access limits are why this note does
not use their reported accuracy as evidence for the Taiwan pilot.
The Chang et al. record is openly readable as a v1 Preprints.org manuscript but
is explicitly marked not peer-reviewed; it has no DOI listed on the record, so
the URL is the stable citation used here.  Its close same-benchmark results were
treated as a prior-art warning, not as independently verified peer-reviewed
evidence.
