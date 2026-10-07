# Evidence review — 2026-10-03

**Position: competitive benchmark prediction, not an established SOTA result.**
No inspected paper uses our exact split, masking scope, calibration pools and
explanation target. The historical test has already informed research; its
numbers are exploratory. This targeted, AI-assisted primary-source review is
not an exhaustive systematic review or proof of absence of prior art.
NR means not verified in the inspected source, not that a method lacks it.

## 1. Clean prediction: protocols matter

All rows below concern Taiwan next-month default unless explicitly stated.
AP and authors' AUPRC labels are preserved; their numerical integration may differ.

| Paper/model | Year | Split; seeds | Metric and reported clean performance | Missing data / calibration / code | Directly comparable? |
|---|---:|---|---|---|---|
| [Hlongwane, Ramaboa, Mongwe](https://doi.org/10.1371/journal.pone.0308718), LR / RF / XGB / LightGBM / CatBoost | 2024 | 80/20; repeat seeds NR; engineered/binned features | Table 5 AUROC .74891 / .75929 / .75766 / .75690 / .74793 | No simulated-missing comparison established; SHAP scorecards; calibration NR | No: different features/split |
| [Ala'raj, Abbod, Majdalawieh](https://doi.org/10.1186/s40537-021-00461-7), BiLSTM | 2021 | Five-fold CV; six monthly histories + static variables | Abstract accuracy 82.4%; numerical AUROC not transcribed here | Brier evaluation; no matched simulated-missing experiment established | No; accuracy cannot rank our AP |
| [Hu & Yeo](https://doi.org/10.3390/electronics15122656), transformer/BiGRU and LightGBM ensemble | 2026 | Taiwan 70/30, five-fold training CV; seed count NR | Taiwan AUROC Table 4 exists; cells not reliably extracted, therefore no invented number | Zero-fill input; code linked by authors; systematic Taiwan masking/calibration NR | No; AMEX competition scores are **not** Taiwan AUROC |
| [Dong et al.](https://doi.org/10.3390/e28080837), cross-attentional tabular transformer | 2026 | Taiwan supplementary 80/20; seeds 11/22/33/42/55 | AUROC .7697 ± .0075; AP .5358 ± .0135 | No matched missing/revision benchmark established | No; main 5,000-case **loan approval** experiment is a different target |
| [Chang et al.](https://www.preprints.org/manuscript/202609.2039), XGB-flat / XGB-lags / GRU / transformer-MC / proposed | 2026 preprint | 29,965 retained; 64/16/20; ten seeds 42–51 | AUROC .7825 / .7841 / .7830 / .7822 / .7828; AUPRC .5583 / .5576 / .5494 / .5505 / .5494 | Missing stress, calibration and predictive selection; code reproduction not run here | No: filtering, preprocessing, split and masking differ |
| Repository XGB25 / vanilla GRU / mask-delta GRU | 2026 exploratory | Fixed 4,500 inspected test customers; five training restarts | AP .5560 / .5494 / .5527 | MCAR/MAR, separate Platt calibration, repository code | Historical reference only |

Hlongwane's [2025 correction](https://doi.org/10.1371/journal.pone.0329901)
corrects Ramaboa's name, not the result table. These examples show why selecting
the largest reported number is not a defensible SOTA comparison. Strong trees
remain the necessary controls. A Taiwan transformer/GRU is already prior work.

## 2. Missing-data prediction

| Model/work | Mechanism; rate | Metric/performance | Calibration | Selective prediction |
|---|---|---|---|---|
| Chang et al. structured Bayesian temporal model | Temporal-only MCAR30; also MAR/MNAR/latest months | AUROC .7623 ± .0080 under MCAR30 | Validation temperature; clean Brier .1349, ECE .0132 | Entropy selection; 10%-coverage error 3.51% |
| Repository XGB25 | All 23 fields MCAR30 / anchor MAR30 | AUROC .7632 / .7646; AP .5237 / .5138 | Brier .1396 / .1427 | Historical explanation strength selection, not calibrated risk control |
| Repository mask/delta GRU | Same repository masks | AUROC .7580 / .7583; AP .5215 / .5172 | Brier .1400 / .1420 | Same limitation |
| [Han et al., SMART](https://doi.org/10.1038/s41598-025-99997-4), 2025 | Taiwan MCAR5/10/15/20 and heavier stress | rSVD + GAIN; imputation and downstream classification evaluation; matched numbers NR | NR | NR |
| [Zhao et al., MGAIN](https://doi.org/10.1016/j.asoc.2022.109273), 2022 | Incomplete credit assessment | Multiple GAN-based imputations and fusion | NR | NR |
| [Zheng et al., ESWA](https://doi.org/10.1016/j.eswa.2025.130268), 2026 | Fragmentary feature sets | Two-stage group/model combination without imputation; publisher abstract verified, full protocol unavailable | NR | NR |
| [Zheng et al., J. Big Data](https://doi.org/10.1186/s40537-025-01347-8), 2026 | Fragmentary data, simulations + application | Improved two-stage scoring; distinct from preceding ESWA paper | NR | NR |

Robust credit prediction under incomplete inputs is an established problem.
Our MCAR30 versus Chang's MCAR30 is **not head-to-head**: 23 versus 18 maskable
fields, different customers and preprocessing. Neither numerical proximity nor
a .001 advantage supports superiority.

### Exact 2026 competitor check

**Mingfan Chang, Ye Wang, Yunbo Liu, Hanxin Chen, Chenfeng Fan**,
*Uncertainty-Aware Credit-Risk Prediction with Structured Bayesian Temporal
Modeling*: submitted September 22, posted September 23, 2026, version 1,
**not peer reviewed** at the inspected [source](https://www.preprints.org/manuscript/202609.2039).
Static-conditioned latent transitions/filtering, GRU, MLP and variational
Bayesian head already combine temporal modeling, masking, reconstruction and
uncertainty. Monthly representation is three financial channels; 10% temporal
training masking, 100 MC prediction draws, validation temperature and F1 threshold.
No restored-truth reason-revision evaluation appears in the inspected methods
and experiments. Its clean tree controls remain competitive with its proposal.
Thus copying this combination is not our gap; publication status must remain explicit.

## 3. Explanation reliability: nearest targets

Y = yes; — = not the target established by inspected material. An absence here
is restricted to the material inspected, not a universal negative claim.

| Method | Missing-aware? | Local explanation? | Attribution variance/uncertainty? | Revision after verification? | Learns revision risk? | Selective explanation? |
|---|---|---|---|---|---|---|
| [Vo et al., ASC 2026](https://doi.org/10.1016/j.asoc.2026.115105) | Y | SHAP | Imputation effects/bias | Explanation error comparisons, not our release event | — | — |
| [Golchian & Wright](https://arxiv.org/abs/2512.17689), 2025 workshop/preprint | Y, multiple imputation | Shapley + global methods | Y, interval coverage | — | — | — |
| [Paes, Wei, Calmon, Selective Explanations](https://proceedings.neurips.cc/paper_files/paper/2024/file/647af5f6b2538524f6c047c1d9170fd9-Paper-Conference.pdf), NeurIPS 2024 | Not verification missingness | Amortized attribution | Learned/deep uncertainty | Approximation error, different target | Learns explanation quality, not this E | Y |
| [Löfström et al., Calibrated Explanations](https://doi.org/10.1016/j.eswa.2024.123154), 2024 | Not primary focus | Calibrated feature rules | Feature-weight intervals | — | — | Uncertainty available, different target |
| [Löfström, Hjort, Löfström, Guarded Explanations](https://proceedings.mlr.press/v329/lofstrom26a.html), COPA 2026 | Support of perturbations | Rules | CE backend | — | — | Filters unsupported rule representatives |
| [Lin & Wang, SHAP Stability in Credit Risk Management](https://doi.org/10.3390/risks13120238), 2025 | Not restoration protocol | Taiwan XGB SHAP | Across 100 seeds | — | — | — |
| [Jiang et al.](https://doi.org/10.3389/fdata.2024.1392662), 2024 | Graph measurement/structure uncertainty | GNN explanation | Learned parameter/data uncertainty | — | Different explanation uncertainty target | — |
| [Zhu et al.](https://arxiv.org/abs/2507.12913), 2025 preprint | Not this verification target | Feature/counterfactual | Predictive uncertainty decomposition | — | — | Reject unreliable explanations using epistemic uncertainty |
| [Vaghela, Swaminarayan, Tank, TrustXAI-Derm](https://doi.org/10.53365/nrfhh.1519), September 2026 | Dermoscopy, not missing credit | Grad-CAM++ | MC dropout/meta-model | Mask-overlap failure, not restoration | **Learns explanation failure** | Deferral; publisher abstract verified |
| Proposed evaluation/strategy | Simulated credit verification | Fixed observed reasons | Current + completion statistics | **Same frozen predictor before/after true restoration** | Generic supervised selector | Calibrated release at fixed revision budget |

Vo already establishes that good prediction need not mean good explanation
under imputation. Golchian/Wright establish imputation uncertainty is relevant;
single imputation understates uncertainty. Selective Explanations already learns
explanation quality; TrustXAI-Derm is additional, abstract-level evidence against
claiming a new generic reliability head. Jiang and Zhu further rule out claiming
prediction/explanation uncertainty as an entirely new distinction.

Guarded Explanations explicitly calls its score an empirical conformal-style
rank, **not** a standard conformal p-value or finite-sample guarantee for its
constructed perturbations. Do not transfer a conformal guarantee to our
conditional released-revision ratio without a valid argument.

## Known temporal and uncertainty components

| Family | What it does | Relevance / limit for this project |
|---|---|---|
| [GRU-D, Che et al. 2018](https://doi.org/10.1038/s41598-018-24271-9) | Masks, elapsed times, learned input/hidden decay; direct prediction | Our mask/delta concatenation is GRU-Simple-like, **not GRU-D**. Monthly six-step grid has limited irregular-time structure. |
| [BRITS, Cao et al. 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html) | Bidirectional recurrent imputation optimized with downstream objectives | Stronger imputation precedent, not revision supervision. |
| [SAITS, Du et al. 2023](https://doi.org/10.1016/j.eswa.2023.119619) | Self-attention imputation | Modern comparator; unnecessary compute for the present diagnostic question. |
| [Kim et al., ICML 2023](https://proceedings.mlr.press/v202/kim23m.html) | Probabilistic imputation for time-series classification | Multiple possible completions and classification uncertainty already known. |
| [Joshi & Hauskrecht, Still Competitive](https://arxiv.org/abs/2510.16161) | Revisits recurrent irregular-series prediction | Compact recurrent controls remain sensible; preprint/version status, not credit novelty. |
| [MC dropout](https://proceedings.mlr.press/v48/gal16.html), [deep ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Predictive uncertainty | Entropy/variance are controls. Completion spread is not automatically epistemic uncertainty. |
| [Learn then Test](https://arxiv.org/abs/2110.01052), [Conformal Risk Control](https://openreview.net/pdf?id=33XGfHLtZg) | Calibrate decisions/risks under stated assumptions | Use independent customers and explicit finite threshold family; pooled guarantees do not imply each missing mechanism is controlled. |
| [Robust attribution regularization](https://proceedings.neurips.cc/paper_files/paper/2019/hash/172ef5a94b4dd0aa120c6878fc29f70c-Abstract.html), [ExpO](https://proceedings.neurips.cc/paper/2020/file/770f8e448d07586afbf77bb59f698587-Paper.pdf) | Train for explanation robustness/fidelity | Known objectives; suppressing legitimate restoration-induced changes would be the wrong target. |

Weighted BCE, focal loss, sampling, threshold tuning and class-weighted boosting
are standard controls, not contributions. Keep BCE/default unweighted tree loss;
the previous weighted/focal calibration damage is already recorded. The next
stage freezes the credit predictor rather than introducing another loss search.

## One external recommendation, with a narrower claim

Home Credit is attractive in scale and natural missingness, but official Kaggle
data/rules pages did not expose usable terms in this retrieval. Its application
target also hides exact delinquency thresholds. Give Me Some Credit has a clearer
90-day delinquency/two-year target, but original competition download terms were
not verified; similarly named community copies do not establish permission.
Neither is silently downloaded or relabeled as freely licensed.

Recommend **UCI Polish Companies Bankruptcy, `5year.arff` only**, for an
independent **corporate-bankruptcy transfer test**, not consumer-default
replication. [Official UCI record](https://www.archive.ics.uci.edu/dataset/365/polish%2Bcompanies%2Bbankruptcy%2Bdata),
DOI [10.24432/C5F600](https://doi.org/10.24432/C5F600): CC BY 4.0,
5,910 financial statements, 410 bankruptcies, 64 financial ratios, natural missing
entries, one-year bankruptcy horizon; file 2.8 MB. The confusing filename denotes
the fifth observation-year group, **not a five-year outcome horizon**.
Do not pool all horizon files: the archive does not provide a dependable company
identifier for excluding repeated firms across horizons. Ratios are static,
not six monthly repayment records. Audit duplicate statements and split groups
before fitting. Only artificially hide originally known cells; natural unknown
values have no verification ground truth. License/provenance are usable, but
external training and restoration experiments are **NOT RUN** here.

## Actual remaining gap and limits

Candidate contribution is a **verification-supervised operational target and
evaluation**: predict whether specific currently observed positive reasons will
need material revision when hidden financial facts are verified, and compare
explanation coverage at a declared revision budget against strong predictive,
missingness and attribution selectors. The searched sources do not establish
this exact combination as solved; that is a bounded search conclusion, not a
priority claim.

Cannot claim novelty for GRU/GRU-D, temporal Taiwan encoding, masking augmentation,
reconstruction, Bayesian heads, imputation, SHAP uncertainty, selective prediction,
learned explanation quality, calibration, or the generic two-output layout.
The simplest useful method may be a frozen XGBoost and an ordinary revision
classifier. Its failure against completion Monte Carlo or a generic selector
would rule out architectural claims while leaving a carefully validated
problem/evaluation paper possible.
