# Large-dataset decision — frozen before data access

**SELECT: Freddie Mac Single-Family Loan-Level Dataset, Standard annual samples,
Release 47, released 2026-07-29. ACCEPT WITH RESTRICTIONS.** No other new dataset
is selected. Research stage: provenance verified / protocol specified / real data
NOT DOWNLOADED / mortgage experiments NOT RUN.

## Why this one

The [provenance comparison](DATASET_PROVENANCE_AUDIT.md) audits Freddie, Fannie,
both Home Credit competitions, AMEX, Give Me Some Credit and historical LendingClub,
and rechecks Taiwan/Polish. Freddie combines an explicit noncommercial academic
results exception with documented monthly mortgage performance, stable official
release documents and scale. Fannie's permission for our public-paper scope was
not established; Model Stability is competition-only; other candidates failed
verification of their own rights. Popularity was not a selection criterion.

Scientific ranking among eligible new candidates has only one entry. Freddie adds
real longitudinal serious-delinquency prediction and a temporal holdout. It does
not replicate credit-card default exactly. Natural unknowns remain unknown.
Engineering ranking favors official 50,000-loan annual samples over downloading
billions of monthly rows. These are provider-sampled, not outcome-selected by us.

## Exact planned release/cohort

Use **only Standard sample files from Release 47** and July 2026 schema, performance
cutoff 2026-03-31. The provider's [release notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf)
report about 49.2 million Standard origination records and 2.91 billion monthly
records. The approximately 56m landing-page figure includes Non-Standard data,
which are excluded here. The [guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf)
describes 50,000 loans per full annual sample.

| Stage/role | Source vintages | Planned source loans | Results access |
|---|---|---:|---|
| Stage 1 engineering pilot | 2000, 2001 | 100,000 | Loader/target/memory validation; subset of final training pool |
| Stage 2 train | 2000–2008 inclusive | 450,000 | Fit LR/XGBoost/preprocessing/completion models |
| Development | 2011 | 50,000 | Early stopping and fixed diagnostics |
| Default-probability calibration | 2014 | 50,000 | Frozen Platt procedure; retain raw scores |
| Explanation release calibration | 2017 | 50,000 | Whole/partial policies; no default fitting |
| Final temporal assessment | 2020, 2021, 2022 | 150,000 | Open once after fit/policy freeze |
| Stage 2 total | 15 annual samples | **750,000 unique source loans** | Exact eligible counts pending |

Pilot rows are not additional independent loans. No pilot results may choose new
features, cohorts, endpoint, completion hyperparameters or tolerances. Technical
bugs stop execution and get an incident report before correction. No outcome-based
sample expansion. If fewer than 500,000 eligible labelled loans remain, report the
shortfall and do not call this >=500k validation or append more years. If fewer than
100 events are available in train or final assessment, report inadequate endpoint
support, without relabelling REO or changing the horizon.

Exact filenames are `sample_YYYY.zip`, with `sample_orig_YYYY.txt` and
`sample_perf_YYYY.txt`. The local manifest will record actual file/member SHA256,
bytes, authorized download date, terms acknowledgement and documentation release.
No checksum or download date exists yet. A filename alone does not prove Release 47.
The offline command requires a researcher-supplied acquisition receipt and rejects
another release/schema. A later provider revision requires a new documented freeze.

## Fifteen decision answers

1–2. All seven requested large candidates and their official sources appear in the audit.
3–5. Freddie-specific terms effective November 2025 expressly cover our noncommercial
academic analysis and bounded public derived results; see the precise permission matrix.
6. Raw/transformed rows, loan-level outputs, donor banks and reference examples stay
local. Model artifact publication is restricted and not enabled.
7. No verified dataset DOI; cite the dated provider guide/release and stable page.
8–9. Our constructed endpoint is **observed mortgage 90+ days past due during the
next 12 monthly reporting periods before loan termination**, not generic default.
10–11. Monthly performance is temporal; official unknown/sentinel values establish
natural missingness, whose cohort prevalence remains unmeasured.
12. All six alternative candidates are REJECT for this project's current gate;
reasons distinguish a verified restriction from unavailable evidence.
13–14. Freddie alone is selected for explicit publication permission, interpretable
financial fields, temporal realism and manageable official samples.
15. Release 47 / 15 vintages / 750k source loans above; no actual data release has
yet been used. See [citation package](DATASET_CITATION.md), [target](FREDDIE_TARGET_DEFINITION.md)
and [leakage audit](FREDDIE_LEAKAGE_AUDIT.md).

## Stable-core decision

**GO for a preregistered feasibility comparison; NO efficacy or novelty claim yet.**
Use the same frozen predictor and current group candidates, releasing subsets only
after independent reason-level calibration. Compare donor and conditional-imputer
evidence separately, then a predefined intersection sensitivity. The
[specification and nearest prior work](STABLE_CORE_EXPLANATION_SPEC.md) distinguish
verification survival from attribution uncertainty and causal correctness.

The primary scientific risk is **completion-model misspecification under temporal
shift**: two completion families may share blind spots. Empty stable sets, loss of
Region B or no advantage over rank instability are valid negative outcomes.
