# Missing multivariate series: recurrent-model research note

**Date checked:** 2026-10-03
**Scope:** GRU-D, BRITS and related recurrent methods; modern time-series missing-data methods; masked reconstruction; and the implications for the Taiwan six-month credit pilot. This note is research evidence for `docs/TEMPORAL_MODEL_RESEARCH.md`; it does not implement a model or claim an experiment has been run.

## Decision in one paragraph

There is a scientific case for a small temporal branch, but there is no case for claiming a new recurrent architecture. A six-month chronological repayment tensor has a meaningful ordering, and an existing Taiwan study already shows that a recurrent model can use the same six-month payment, bill, and repayment-status fields. The recommended primary model for the pilot is a **minimal mask-and-delta GRU**: deterministic train-fitted fill values, an observed mask, and elapsed-months-since-observation concatenated to each recurrent input, plus a separate static branch. This is a transparent adaptation of missing-aware recurrent modeling, not a new method. Full GRU-D input/hidden decays should be an explicitly controlled ablation. With only six regular monthly steps, simulated MCAR masks, and mixed categorical/monetary fields, the learned decay-to-mean assumptions of GRU-D are not established as appropriate and may add capacity and a mean-reversion bias without evidence of benefit. If the ablation shows a reproducible gain under the planned missingness settings, the paper can report GRU-D as a tested variant; it should not be selected in advance as a novelty claim.

## Dataset and temporal constraints

The UCI repository identifies the Taiwan dataset as 30,000 records with 23 input variables and no original missing values. Its metadata maps `X6`--`X11` to repayment status from September 2005 back to April 2005, `X12`--`X17` to bill amounts in the same newest-to-oldest order, and `X18`--`X23` to payment amounts in the same order. The chronological tensor is therefore:

```text
t=1 April:     PAY_6, BILL_AMT6, PAY_AMT6
t=2 May:       PAY_5, BILL_AMT5, PAY_AMT5
t=3 June:      PAY_4, BILL_AMT4, PAY_AMT4
t=4 July:      PAY_3, BILL_AMT3, PAY_AMT3
t=5 August:    PAY_2, BILL_AMT2, PAY_AMT2
t=6 September: PAY_0, BILL_AMT1, PAY_AMT1
```

The representation is a six-step regular monthly sequence, not an irregular event stream. Artificial MCAR (and later MAR) masks create the missing values. Under MCAR, the mask is independent of the default label by construction. A model may need the mask to distinguish a filled value from an observed value, but a performance gain from a mask branch cannot be described as learning a naturally informative care-seeking or reporting process. Mask-only and delta-only ablations are necessary to detect finite-sample exploitation of the simulator.

The short horizon changes the modeling trade-off. A sequence length of six is enough to test recency and repayment trajectory, but gives very little data per person for learning a flexible decay curve, long-range recurrence, or a separate imputation process. Monetary fields are skewed and PAY fields are ordinal/categorical-coded; a single global mean decay target has no demonstrated semantic correctness for both types.

## Comparison of relevant methods

| Work | Model | Temporal? | Missing representation | Imputes or predicts? | Explanation/reliability output | Relevance to this pilot |
|---|---|---:|---|---|---|---|
| Che et al., *Scientific Reports* 2018, GRU-D | GRU-D | Yes | `m_t^d=1` observed, `0` missing; feature-wise elapsed time `delta_t^d`; input and hidden decay | Direct classification/regression; decay is used instead of a separate learned imputation output | None | Strong precedent for exposing mask and elapsed time. The decay-to-training-mean and hidden-to-zero terms are known components, not novelty. Original evidence is mainly irregular/informative clinical data, not six regular credit months. |
| Cao et al., *NeurIPS* 2018, BRITS | RITS-I/BRITS-I | Yes | Same observed-mask convention; exact per-variable accumulated delta; forward/backward consistency | Missing values are learned as variables in the recurrent graph; jointly trains imputation and prediction | None | A known, heavier bidirectional imputation/classification baseline. If all six historical months are available at the October decision/verification point, bidirectionality is permissible; a causal online-release variant would need an explicit direction constraint. It remains deferred here because of scope and because it does not provide a reason-reliability output. |
| Yoon et al., *IEEE TBME* 2019, M-RNN | Multi-directional RNN | Yes | Missingness across/within streams; interpolation and cross-stream imputation | Joint imputation; optional dropout for multiple imputations | None | Shows short measurement histories can be difficult for learned imputation; assumes MAR in the formulation. No explanation-revision target. |
| Luo et al., *NeurIPS* 2018, GRUI in a GAN | Modified GRU cell for imputation | Yes | Time lags/deltas and recurrent imputation | GAN-based imputation plus downstream prediction | None | Known heavier imputation family; no reason reliability, and unnecessary adversarial complexity for this pilot. |
| Luo et al., *IJCAI* 2019, E²GAN | End-to-end GAN with GRUI-based encoder/generator/discriminator | Yes | Incomplete-series input, recurrent time-lag handling, real-data forcing | One-stage imputation, then downstream classification/regression | None | A later, distinct E²GAN paper; it does not make the compact mask/delta risk model or explanation-revision target novel. |
| Ma et al., *IEEE TPAMI* 2022 (online 2020), AJ-RNN | Adversarial joint-learning RNN | Yes | Incomplete-series input with learned imputation/discriminator | Joint imputation and classification | None | Demonstrates another known recurrent imputation/classification route; evaluated on UCR/synthetic data rather than this credit/revision setting. |
| Du et al., *Expert Systems with Applications* 2023, SAITS | Two diagonal-masked self-attention blocks | Yes | Missing mask concatenated to attention inputs; diagonal masking prevents trivial self-copy | Masked imputation and observed reconstruction (MIT + ORT) | None | Strong modern imputation reference. Its gains and speed were shown on substantially longer/wider series; self-attention is not automatically preferable for six monthly steps. |
| Tashiro et al., *NeurIPS* 2021, CSDI | Conditional score-based diffusion | Yes | Condition on observed values and mask | Probabilistic imputation | No credit/explanation reliability output | Important probabilistic alternative, but sampling cost and model scope are disproportionate to a compact local pilot. |
| Kim et al., *ICML* 2023, Probabilistic Imputation for Time-Series Classification | Deep generative imputation + classifier | Yes | Multiple plausible completions conditioned on observed values | Joint imputation/classification with prediction and imputation uncertainty | Calibration metrics, not restoration-defined explanation revision | Useful evidence that uncertainty and calibration should be measured. It does not answer whether a current reason changes after hidden truth is restored. |
| Yao et al., 2024, end-to-end ITSC with missing values | GRU imputation + multi-scale CNN | Yes | Simulated missingness, including MCAR in the study | Joint reconstruction/classification | None | Recent direct classification precedent, but still no explanation reliability and no reason-revision event. |
| Wang et al., *NeurIPS* 2024 task-oriented imputation | Task-oriented imputation evaluation | Yes | Missingness evaluated through downstream task impact | Evaluates imputers by task outcome, not reconstruction alone | None | Supports measuring downstream risk and explanation behavior rather than reporting imputation error alone. |
| Du et al., TSI-Bench, 2024 | Benchmark of 28 imputation algorithms | Yes | Multiple masks/rates and datasets | Imputation; broad benchmark | None | Its central result is that no imputer wins universally. It argues for matched missingness and downstream ablations instead of importing a “best” model. |
| Miao et al., *IJCAI* 2025, MMNet | Missing-aware embedding + memory/mixture encoder | Yes | Explicit missing-aware representation | Imputation-focused | None | Recent alternative, but not a direct credit/explanation method and too broad for the first pilot. |
| Joshi & Hauskrecht, TMLR 2026, GRUwE | GRU with learnable exponential time basis/reset | Yes, irregular event time | Continuous-time/event timing | Forecasting/classification representations | None | Shows a small recurrent model can remain competitive, but the task is irregular-time prediction, not six regular MCAR credit steps. |

All rows above are prior art or evidence; none establishes a neural head for the project’s specific event: generate reasons with masked inputs, reveal the known hidden values, regenerate reasons using the **same predictor**, and label whether the top reasons changed.

## What the recurrent papers establish

### GRU-D: exact mask and delta convention

Che et al. define a per-feature mask with `1` for observed and `0` for missing, together with a time interval since the previous observation. For regularly sampled monthly data, the interval can be expressed in months. BRITS gives the same recurrence explicitly:

```text
delta_t^d = 0                                  if t is the first step
            (s_t - s_{t-1})                    if m_{t-1}^d = 1
            (s_t - s_{t-1}) + delta_{t-1}^d    if m_{t-1}^d = 0
```

The GRU-D input first uses a last-observation value when the current value is missing, then decays the result toward a training-set empirical mean. A trainable nonnegative rate controls that decay. The hidden state is also decayed toward zero, and the mask is included in the GRU update. These are established mechanisms. They are useful design references, but a decay toward the empirical mean is a modeling assumption, not a neutral way to encode MCAR.

### BRITS and learned imputation

BRITS treats missing values as variables in the recurrent computation and trains forward and backward recurrent imputers with consistency loss. It is an appropriate comparator when the scientific objective is high-quality imputation or when the complete historical window is available. For this project’s full six-month record, “backward” means using an earlier month while processing the fixed April-to-September history; it is not future leakage for an October decision or for verification of that record. If the paper later studies release after April, May, or another intermediate month, the comparison must become causal and BRITS-style backward context would be disallowed. BRITS remains deferred here because it adds parameters and an imputation objective without directly producing a reliability signal for the explanation.

### SAITS and modern alternatives

SAITS shows that masked self-attention plus masked/observed reconstruction can be a strong imputation method. CSDI and later probabilistic methods provide completion uncertainty. TSI-Bench and the 2024 task-oriented evaluation both caution against universal rankings: the missing mechanism, sequence shape, and downstream task matter. None of these papers makes a six-step Taiwan credit model with a restoration-defined explanation event the default choice. A transformer/diffusion branch would make the first local experiment harder to attribute and would not, by itself, answer the paper’s question.

## Challenge to full GRU-D for this setting

The full GRU-D equations are scientifically valid, but they are not automatically the primary architecture here.

1. **Regular cadence weakens the original irregular-time motivation.** Every observed opportunity is one month apart. `delta` mostly records a short missing run length (0/1/2/... months), rather than arbitrary elapsed time. A concatenated mask and delta can preserve this information without committing to a learned continuous decay.
2. **The sequence is too short to estimate a decay shape safely.** At six steps, consecutive missing runs are few under MCAR 10% and 30%. A feature-specific decay rate can fit a handful of patterns and still look plausible on one split. This is a capacity/variance concern, not proof that GRU-D cannot help.
3. **One decay target is questionable for mixed features.** GRU-D’s empirical-mean target is sensible as a generic numeric baseline, but PAY status is ordinal/categorical-coded and bill/payment amounts are skewed. Decaying a one-hot or integer status toward a mean creates a soft numerical code whose interpretation is not the status category; decaying raw money toward a global mean can erase customer-specific scale. Train-fitted median/mode fill plus a mask is easier to audit. A learned embedding could address status, but that is an additional design choice requiring an ablation.
4. **MCAR changes the interpretation of a mask.** In GRU-D’s motivating clinical setting, missingness can be informative. Here the simulator is explicitly target-independent under MCAR. The mask is still necessary to distinguish imputed from observed values, but the paper must not interpret a mask gain as a real-world missingness signal.
5. **Hidden-state decay may erase short-history context.** The hidden decay-to-zero term is a useful stale-state prior for irregular observations. With six fixed opportunities, it may be a stronger inductive bias than needed and can make the temporal/static comparison harder to interpret.

**Recommendation after this challenge:** make the primary recurrent experiment a **minimal mask-and-delta GRU** (imputed value + `m_t` + `delta_t`, with separate static branch). Add full GRU-D input/hidden decay as a named ablation, using identical masks, split, seeds, optimizer budget, and calibration. This is an evidence-bounded engineering choice: the literature justifies testing mask/delta and decay, but does not justify assuming full GRU-D is superior at `T=6`.

## Recommended architecture and controls

Primary pilot:

```text
temporal value x_t: train-fitted median/mode fill only
temporal mask m_t: 1 observed, 0 simulated hidden
temporal delta d_t: elapsed monthly steps since each feature was observed
temporal encoder: one-layer GRU, hidden size 32--64
static branch: scaled LIMIT_BAL/AGE plus encoded SEX/EDUCATION/MARRIAGE and static mask
fusion: concatenate final temporal state and static embedding
head: risk logit -> raw probability; calibrate separately on validation data
```

Planned controls (the full GRU-D branch may be deferred until the primary sanity checks pass):

```text
complete-data vanilla GRU + static branch
minimal mask-and-delta GRU + static branch
full GRU-D-style input/hidden decay + static branch (planned ablation; not required for the first run)
minimal GRU without mask
minimal GRU without delta
temporal-only and static-only variants
LR and XGBoost on the same splits and masks
```

Use the same missing masks for every model at each seed. Do not allow a reliability head to read hidden truth. A later reliability head can consume the fused embedding, mask summaries, prediction uncertainty, and attribution statistics, then be trained on the restoration event. That target is an adaptation to this paper’s question; it is not inherited from GRU-D, BRITS, SAITS, or the credit-RNN literature.

## Masked reconstruction: benefit, cost, and decision rule

Masked reconstruction is established in SAITS (MIT/ORT), Ti-MAE, TS-MAE, BRITS-style imputation, and several recent time-series representation methods. It cannot be claimed as the contribution. It may still be a useful auxiliary task because the input records are complete before simulation: hide observed cells during training, reconstruct their true values, and retain the default-risk loss.

For a controlled optional branch:

```text
L = L_default + lambda_rec * L_reconstruction
```

Use cross-entropy for PAY status and Huber or normalized MSE for scaled bill/payment amounts. Compute normalization and any reconstruction targets from training data only. Keep `lambda_rec` small and fixed before looking at test results. Record training time, parameter count, convergence, risk metrics, calibration, restoration probability shift, top-k overlap, and reason-revision rate.

Retain the auxiliary task only if it improves the pre-specified missingness-robustness and explanation-reliability endpoints on held-out masks without an unacceptable Brier/log-loss/calibration penalty. It is plausible that reconstruction improves representation under high missingness, but equally plausible that a six-step sequence makes it an unnecessary multi-task burden. No positive result should be described as a novel masked-autoencoding method.

## Credit-specific evidence and novelty boundary

An especially close precedent is the Journal of Big Data paper **“Modelling customers credit card behaviour using bidirectional LSTM neural networks”** (2021, DOI `10.1186/s40537-021-00461-7`). It uses the same UCI Taiwan dataset, separates static customer attributes, reshapes the six months into a temporal tensor with repayment status, bill amount, and payment amount, and evaluates a bidirectional LSTM with static context. It also reports probability calibration. Thus the following are already known and cannot be claimed as novel:

- using a GRU/LSTM on the six Taiwan payment months;
- separating static and temporal branches;
- attention over recurrent states;
- class weighting or focal loss;
- using SHAP/gradient attribution;
- mask/delta decay (GRU-D) or learned recurrent imputation (BRITS);
- masked reconstruction or probabilistic completion in general.

The defensible candidate contribution is narrower: evaluate whether a fixed credit predictor’s **current observed reasons** survive verification of hidden values, and test whether a pre-verification reliability score (including a possible learned revision-risk head) can select explanations with lower restoration-defined revision risk. A temporal encoder is an experimental instrument for that question. It becomes a central model contribution only if matched experiments show a robust advantage in prediction under missingness and in explanation reliability/coverage, not merely a higher AUC on complete records.

## Falsifiers and stopping criteria

Stop treating the temporal branch as an improvement if any of these hold after matched seeds and validation:

- vanilla or mask/delta GRU does not improve the planned missingness robustness endpoints over XGBoost/LR, or only improves complete-data AUC;
- full GRU-D’s gains disappear under a no-mask/no-delta/decay ablation or vary materially across seeds;
- weighted BCE/focal increases recall but produces materially worse Brier/log loss/calibration and no useful selective-reliability gain;
- prediction becomes stable while top-k reasons still revise at the same rate as XGBoost;
- a simple missing-fraction, SHAP-variance, or existing Monte-Carlo selector matches the learned revision-risk head;
- any reliability output uses hidden true values, restoration outcomes, or test masks at inference.

## Sources and verification notes

### Primary papers and official pages

- UCI Default of Credit Card Clients: [UCI dataset page](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- Che et al., “Recurrent Neural Networks for Multivariate Time Series with Missing Values,” *Scientific Reports* 8, 6085 (2018), DOI [10.1038/s41598-018-24271-9](https://doi.org/10.1038/s41598-018-24271-9), [arXiv full text](https://arxiv.org/abs/1606.01865), [official implementation](https://github.com/PeterChe1990/GRU-D).
- Cao et al., “BRITS: Bidirectional Recurrent Imputation for Time Series,” *NeurIPS* 2018, [official abstract](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html), [official PDF](https://papers.neurips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf), [code](https://github.com/NIPS-BRITS/BRITS).
- Yoon, Zame, and van der Schaar, “Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks,” *IEEE TBME* (2019), DOI [10.1109/TBME.2018.2874712](https://doi.org/10.1109/TBME.2018.2874712), [arXiv](https://arxiv.org/abs/1711.08742), [code](https://github.com/jsyoon0823/MRNN).
- Luo et al., “Multivariate Time Series Imputation with Generative Adversarial Networks,” *NeurIPS* 2018, [official PDF](https://papers.neurips.cc/paper_files/paper/2018/file/96b9bff013acedfb1d140579e2fbeb63-Paper.pdf), [official code](https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks).
- Luo et al., “E²GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation,” *IJCAI* 2019, pp. 3094–3100, DOI [10.24963/ijcai.2019/429](https://doi.org/10.24963/ijcai.2019/429), [official paper page](https://www.ijcai.org/Proceedings/2019/429), [PDF](https://www.ijcai.org/Proceedings/2019/0429.pdf).
- Ma, Li, and Cottrell, “Adversarial Joint-Learning Recurrent Neural Network for Incomplete Time Series Classification,” *IEEE TPAMI* (2022), DOI [10.1109/TPAMI.2020.3027975](https://doi.org/10.1109/TPAMI.2020.3027975), [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/32997624/), [code](https://github.com/qianlima-lab/AJ-RNN).
- Du, Côté, and Liu, “SAITS: Self-Attention-based Imputation for Time-Series,” *Expert Systems with Applications* 219, 119619 (2023), DOI [10.1016/j.eswa.2023.119619](https://doi.org/10.1016/j.eswa.2023.119619), [arXiv](https://arxiv.org/abs/2202.08516), [official code](https://github.com/WenjieDu/SAITS).
- Tashiro et al., “CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation,” *NeurIPS* 2021, [official abstract](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html), [official PDF](https://papers.nips.cc/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf).
- Kim et al., “Probabilistic Imputation for Time Series Classification,” *ICML* 2023, PMLR 202:16654–16667, [paper](https://proceedings.mlr.press/v202/kim23m.html), [PDF](https://proceedings.mlr.press/v202/kim23m/kim23m.pdf).
- Wang et al., “Evaluating Imputation Models for Time Series Classification: A Task-Oriented Approach,” *NeurIPS* 2024, [official abstract](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f88264fcc54775ee1706116e90fe351a-Abstract-Conference.html), DOI [10.52202/079017-4365](https://doi.org/10.52202/079017-4365).
- Du et al., “TSI-Bench: Benchmarking Time-Series Imputation,” 2024, [arXiv](https://arxiv.org/abs/2406.12747), [code/benchmark index](https://github.com/WenjieDu/AwesomeImputation).
- Yao et al., “An End-to-End Model for Time Series Classification In the Presence of Missing Values,” 2024, [arXiv](https://arxiv.org/abs/2408.05849).
- Miao et al., “MMNet: Missing-Aware and Memory-Enhanced Network for Multivariate Time Series Imputation,” *IJCAI* 2025, [official paper page](https://www.ijcai.org/proceedings/2025/357), DOI [10.24963/ijcai.2025/357](https://doi.org/10.24963/ijcai.2025/357).
- Joshi and Hauskrecht, “Still Competitive: Revisiting Recurrent Models for Irregular Time Series Prediction,” *Transactions on Machine Learning Research* (2026), [OpenReview PDF](https://openreview.net/pdf?id=YLoZA77QzR), [arXiv](https://arxiv.org/abs/2510.16161), [code](https://github.com/ankit204/GRUwE).
- “Modelling customers credit card behaviour using bidirectional LSTM neural networks,” *Journal of Big Data* (2021), DOI [10.1186/s40537-021-00461-7](https://doi.org/10.1186/s40537-021-00461-7), [full text](https://link.springer.com/article/10.1186/s40537-021-00461-7).
- Lin et al., “Focal Loss for Dense Object Detection,” *ICCV* 2017, [official paper](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html), [PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).
- Caplin, Martin, and Marx, “Calibrating for Class Weights by Modeling Machine Learning,” [arXiv](https://arxiv.org/abs/2205.04613).

### Search log

Searches run through 2026-10-03 included:

```text
GRU-D Recurrent Neural Networks for Multivariate Time Series with Missing Values Scientific Reports DOI
BRITS Bidirectional Recurrent Imputation for Time Series NeurIPS official
SAITS Self-Attention-based Imputation for Time Series DOI
GRU-I missing data recurrent imputation
M-RNN missing time series imputation DOI
modern missing time series classification 2024 2025 2026
GRU credit default prediction / recurrent credit scoring / temporal repayment history
class weighted BCE focal loss credit calibration
explanation reliability missing data SHAP selective explanations calibrated explanations
```

Full text or official PDFs were accessible for BRITS, GRU-D via arXiv, M-RNN via arXiv, SAITS via arXiv, CSDI, probabilistic imputation via PMLR, the Taiwan BiLSTM study, the UCI metadata, and the cited NeurIPS/IJCAI/TSI-Bench pages. The GRU-D publisher page redirected to authorization during checking; the arXiv copy and official code were used. Several ScienceDirect/Wiley pages were blocked or returned fetch errors; DOI metadata and publisher/search abstracts were used for those records. Newer 2026 papers were checked through the available arXiv/OpenReview/publisher records, but the absence of an exact prior work for the restoration-defined explanation event is a bounded search conclusion, not a proof that no unpublished or inaccessible work exists.

## Evidence versus inference

**Directly established by sources:** GRU-D uses observed masks, elapsed times, input/hidden decay; BRITS uses bidirectional learned imputation and consistency; SAITS uses masked self-attention and reconstruction tasks; probabilistic models can expose completion uncertainty; Taiwan has six chronological months with no natural missing values; recurrent credit scoring on this dataset is already published; class weighting/focal loss are standard and can affect calibration.

**Inference for this pilot:** a minimal mask/delta GRU is a better first primary model than full GRU-D for six regular monthly steps and mixed categorical/monetary fields; full GRU-D should remain an ablation; masked reconstruction may help only under an explicit robustness/reliability gate; the temporal model should remain subordinate to the explanation-reliability question unless matched experiments establish otherwise.
