# Dataset provenance audit — 2026-10-03

**Selected: Freddie Mac Single-Family Loan-Level Dataset (SFLLD), ACCEPT WITH
RESTRICTIONS.** This is a documented research-use decision, not a general legal
opinion or permission to redistribute loan records. No new dataset was downloaded.
The researcher confirmed that official Freddie files are not yet available locally.

## Scope and search method

Audited the seven requested candidates and reverified both existing UCI datasets.
Searched provider/competition names with terms, license, academic, research,
publication, derived products, redistribution, target and data dictionary; followed
official landing-page links to terms and release documentation. Only original or
authorized-provider evidence establishes permission. Search date: 2026-10-03.
No competition was joined, account created, terms accepted on the researcher's
behalf, or mirror downloaded. Some Kaggle pages returned empty dynamic documents;
browser access was unavailable. Those are verification failures, not proof that
their actual licenses prohibit research. Under R2–R4 they are nevertheless REJECT
for this project now. A third-party paper's assertion of permission is insufficient.

| Dataset | Original provider | Size | Target | Temporal | Natural missing | License/terms | Academic use | Publish results | Raw redistribution | Stable citation | Decision |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| Taiwan | I-Cheng Yeh / authorized UCI repository | 30,000 | Next-month credit-card default payment, supplied binary label | Six prior months, one cross-section | No | CC BY 4.0 | Yes | Yes, attribution | Yes under license; repo policy NO | DOI 10.24432/C55S3H | ACCEPT |
| Polish, `5year.arff` | Sebastian Tomczak / UCI; source described as EMIS | 5,910 statements | Bankruptcy within one year | Financial ratios; not a monthly panel | Yes | CC BY 4.0 | Yes | Yes, attribution | Yes under license; repo policy NO | DOI 10.24432/C5F600 | ACCEPT |
| Freddie SFLLD | Freddie Mac | Release 47 Standard: about 49.2m loans / 2.91bn performance records | Constructed 12-month 90+ DPD endpoint, below | Origination + monthly performance | Documented unknown codes/blanks | SFLLD terms effective November 2025, plus website terms | Explicit | Explicit noncommercial research-results exception, with non-reconstruction/privacy conditions | NO public raw records | Official dated guide/release; no dataset DOI verified | ACCEPT WITH RESTRICTIONS |
| Fannie SF loan performance | Fannie Mae | Official current primary summary: 57,864,281 loans | Monthly delinquency/liquidation fields; no chosen target | Yes | Yes | FAQ quotes license §3.2, internal purposes and third-party restrictions | Intended external academic scope not established | No explicit public-paper exception verified | Prior written consent required | Official guide/page; release not frozen | REJECT |
| Home Credit Default Risk | Home Credit, original Kaggle competition | Exact count not reverified from accessible official metadata | Payment-difficulty label; exact X-day/Y-installment window not verified | Relational history | Not reverified | Original rules inaccessible in this audit | Unverified | Unverified | Unverified; NO use/share | Original competition URL, but exact licensed release unverified | REJECT |
| Home Credit Model Stability | Home Credit N.V., original Kaggle competition | Exact count not independently reverified | Sponsor-defined default after an unspecified period | Credit-case histories/time index | Documented missing history | Competition-specific rules + §B.7 | Competition use only; no separate research grant located | External paper not authorized by verified scope | Nonparticipants prohibited | Official competition page, no DOI verified | REJECT |
| AMEX Default Prediction | American Express, original Kaggle competition | Exact customer/row counts not reverified | Sponsor describes 18-month performance window and nonpayment in 120 days | Monthly statements | Not reverified | Original rules inaccessible in this audit | Unverified | Unverified | Unverified; NO use/share | Original competition page, no exact release verified | REJECT |
| Give Me Some Credit | Credit Fusion competition host, Kaggle | Exact count not reverified from original files | Two-year financial-distress task; exact 90-day dictionary not retrieved | Historical summary fields | Not reverified | Full original governing terms not verified | Host says papers allowed, but incomplete terms | Host comment supports papers, insufficient complete grant | Unverified | Competition page exists; source-owner/release chain incomplete | REJECT |
| LendingClub historical loans | Historical LendingClub provider | Depends on unknown historical release | Loan status; no defensible fixed horizon established here | Some historical files have servicing fields | Not reverified | Applicable historical download license unavailable | Unverified | Unverified | Unverified; NO mirror use | Historical link does not identify a licensed version | REJECT |

The Fannie summary count and Freddie Standard count describe different populations;
neither is the proposed experiment size. Freddie's landing page also says about
56m Standard **plus Non-Standard** loans; the Release 47 Standard-only figures are
the applicable ones. Monthly rows are not independent loans.

## Freddie: permission evidence and ten explicit answers

Authority: [SFLLD landing page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset),
[dataset-specific terms](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf)
(effective November 2025; updated 2025-11-03), and
[general website terms](https://www.freddiemac.com/terms).

Page 1 of the dataset terms expressly permits academic/research use and public
noncommercial research results/related derived products, provided they neither
contain nor permit reconstruction of dataset parts or identification of individuals.
Page 2 has a general third-party distribution prohibition. Our reading treats the
specific research exception as limited permission, not a blanket open-data license;
the dataset terms explicitly take precedence over conflicting general website terms.

| Question | Audit answer and repository implementation |
|---|---|
| Academic research? | YES, explicit dataset-specific exception. Study must concern mortgage credit performance. |
| Noncommercial research? | YES; chosen scope is noncommercial academic analysis. |
| Publish scientific results? | YES within that exception; no raw-data reconstruction or identification. |
| Statistics/figures/derived models? | Aggregate research products conditionally allowed; model weights are not individually cleared merely by being called a model. |
| Redistribute raw? | NO public raw redistribution under our selected permission. Separate agreement otherwise needed. |
| Redistribute processed rows? | NO under this repository policy; transformation does not remove reconstruction risk. |
| Share trained models? | RESTRICTED. Do not publish weights, tree thresholds, donor banks, SHAP backgrounds or imputer artifacts here. Code/recipes are the reproducibility route. |
| Record-level outputs? | NO here: predictions, attributions, IDs and case tables remain private local artifacts. |
| Registration? | YES, Clarity registration/sign-in; researcher personally accepts applicable terms. |
| Geographic/institutional restrictions? | Public SFLLD terms reviewed do not enumerate an academic country/institution whitelist. This is NOT a finding of unrestricted eligibility; account/click-through conditions and applicable law still govern. Verify on authorized registration. |

General website terms prohibit unauthorized automated collection and require
privacy safeguards. The preparer is offline: no scraper, login automation or
downloader. Dataset files and any applicable click-through terms must be supplied
by the authorized researcher. If those terms differ materially, STOP before use.
Do not upload loan rows to external AI services. No borrower linkage, public-record
join, external re-identification or loan-level publication is part of this study.
Publication by a commercial journal is not permission to commercialize a derived
data/model product: publish the noncommercial research analysis only, and resolve
any publisher demand for restricted artifacts before signing its data declarations.

## Other candidates: controlling evidence and rejection reason

**Fannie:** [official data page](https://capitalmarkets.fanniemae.com/credit-risk-transfer/fannie-mae-single-family-loan-performance-data)
(revised 2026-07-31), [FAQ dated 2026-03-10, questions 3–8 and §3.2 excerpts](https://capitalmarkets.fanniemae.com/resources/file/credit-risk/pdf/sf-loan-performance-dataset-faqs.pdf),
[statistical summary](https://capitalmarkets.fanniemae.com/media/document/pdf/FNMA_SF_Loan_Performance_Stat_Summary_Primary.pdf).
Registration and assent are required. The FAQ limits analytics to internal business
purposes, prohibits identification, and requires consent for third-party data/derived
product distribution. No public academic-results exception was verified. R3/R4:
reject for this paper without a separate grant; do not import Freddie's exception.
Geographic eligibility, model sharing and external row-output rights remain unverified.

**Home Credit Model Stability:** [official rules](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/rules)
explicitly identify Competition Use Only; §B.7.A limits use to the competition and
Kaggle forums; §B.7.B prevents nonparticipant access. §B.2 and §A.3 impose account,
age and sanctions restrictions. [Official data description](https://www.kaggle.com/c/home-credit-credit-risk-model-stability/data)
does not disclose a precise prediction horizon. R3/R4/R7: NOT ELIGIBLE FOR THIS PAPER
without a separate owner research license. Winner-to-sponsor model rights are not
a grant of publication rights to this project.

**Home Credit Default Risk:** [its own rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
and [data page](https://www.kaggle.com/competitions/home-credit-default-risk/data)
were attempted, including the `/c/` canonical variant. No governing text was
retrieved. Secondary descriptions suggest this older competition may have a
different research clause; that is not verified permission and **we do not assert
it has Model Stability's restriction**. R2/R3/R4, and unresolved exact endpoint.

**AMEX:** [rules](https://www.kaggle.com/competitions/amex-default-prediction/rules)
were not readable through available tools. [Sponsor data description](https://www.kaggle.com/competitions/amex-default-prediction/data?select=train_labels.csv)
supports the target wording in the table, not a research license. Academic use,
publication, derivative/weight/row redistribution and exact release remain unverified.
R2/R3/R4; do not infer restrictions or permissions from another competition.

**Give Me Some Credit:** [original rules](https://www.kaggle.com/competitions/GiveMeSomeCredit/rules)
could not be verified. An [original host-labelled discussion response](https://www.kaggle.com/competitions/GiveMeSomeCredit/discussion/870)
says researchers can write papers; it does not establish all data-access,
redistribution, model-sharing or version terms. Exact 90-day target semantics were
not recovered from an official dictionary. R2/R5 and incomplete source chain.

**LendingClub:** [historical download endpoint](https://www.lendingclub.com/info/download-data.action)
was unavailable. [Current terms endpoint](https://www.lendingclub.com/legal/terms-of-use)
now redirects to Happen Bank and cannot retroactively establish a particular old
loan-file license. R2/R3/R4/R9; no Kaggle/GitHub copy is an authority. Do not conflate
charged-off, late and paid-off status or retain still-censored loans as negatives.

## Existing datasets: citation and permissions reverified

[Taiwan UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
lists 30,000 records, 23 predictors, no missing values, CC BY 4.0 and citation:
Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H. Donation date 2016-01-25 differs from
the recommended citation year. The supplied workbook target is `default payment
next month`; predictors cover April–September 2005. No new DPD definition is inferred.
Associated paper: Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card
clients. *Expert Systems with Applications, 36*(2), 2473–2480.
[DOI 10.1016/j.eswa.2007.12.020, verified publisher page](https://www.sciencedirect.com/science/article/pii/S0957417407006719).
The original paper describes a 25,000-observation study; the deposited UCI artifact
has 30,000 records. These are distinct counts; the repository uses the UCI release.

[Polish UCI](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
describes EMIS provenance and licenses its deposited artifact CC BY 4.0. Citation:
Tomczak, S. (2016). *Polish Companies Bankruptcy* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5F600. The selected fifth-year file has 5,910
statements, 410 bankruptcies within one year and 5,500 negatives. Its 64 ratios,
4,666 missing cells and 2,879 naturally incomplete rows are verified from the
previous pinned local artifact, not the landing page's mixed-horizon headline.
This does not authorize extraction of additional EMIS records. Corporate bankruptcy
is kept separate from consumer default. Original source reference is Ziȩba,
Tomczak & Tomczak (2016), *Ensemble boosted trees with synthetic features generation
in application to bankruptcy prediction*, listed by UCI.

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) permits sharing/adaptation,
including commercial use, with attribution, license link and change indication;
it does not waive privacy or imply endorsement. No registration required for the
UCI artifacts. Repository policy remains stricter: no raw/processed rows or models
in Git, even where the dataset license would allow attributed sharing.

## Gate outcome

Exactly one new dataset passes for the bounded scope: **Freddie SFLLD**. Others
are not backup recommendations. Rejection for unverified rights can be reconsidered
only with authoritative evidence, never popularity or a mirror's license badge.
Exact Freddie raw checksums/download date remain PENDING, not invented. The next
gate is local provenance/schema intake, followed by synthetic-tested preprocessing
and the frozen [data protocol](FREDDIE_DATA_PROTOCOL.md).
