# Domain reason groups — frozen validation definition

Declared 2026-10-03 before computing the new grouped/K/policy/external results.
Historical reference: `04bf054`. The previous diagnostic already tested an
observed-member top-3 grouping; this is a prospectively specified extension of
that known sensitivity, not a claim that all Taiwan evidence is untouched.

## Taiwan

| Group | Original fields | Interpretation |
|---|---|---|
| Repayment behavior | PAY_0, PAY_2–PAY_6 | Recorded monthly repayment status |
| Outstanding balance | BILL_AMT1–BILL_AMT6 | Billed balance history |
| Previous payment amount | PAY_AMT1–PAY_AMT6 | Amounts of previous payments |
| Credit capacity | LIMIT_BAL | Granted credit limit |
| Age | AGE | Age, kept separate |
| Sex | SEX | Recorded sex category, kept separate |
| Education | EDUCATION | Recorded education category, kept separate |
| Marital status | MARRIAGE | Recorded marital category, kept separate |

Definitions follow the [official Taiwan dictionary](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Combining demographic attributes would conflate distinct statements and could
cancel their contributions without a coherent financial interpretation. Keeping
them separate is not an endorsement of their use in lending decisions. Monthly
amounts/status are grouped by economic quantity, not by observed revision rate.
This is a dictionary-grounded analyst interpretation, **not human-subject or
credit-expert validation**. Grouping loses timing detail and can conceal meaningful
month-specific changes; it is a falsification sensitivity, not an improved truth.

## Polish fifth-year cohort

Use the [official UCI variable definitions](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data),
not Taiwan concepts. Xj below corresponds to ARFF `attrj`. Every one of the 64
ratios belongs to exactly one group; no outcome-driven reassignment is allowed.

| Group | X indices | Rationale |
|---|---|---|
| Profitability and operating margin | 1,6,7,11,13,14,18,19,22,23,24,31,35,39,42,45,48,49,56,58 | Earnings/retained earnings/margins or costs relative to resources/revenue |
| Liquidity and working capital | 3,4,5,28,37,40,46,50,55,57 | Liquid resources, working capital and short-term funding coverage |
| Capital structure and leverage | 2,8,10,17,25,38,51,53,54,59 | Equity, liabilities and long-term capital structure |
| Debt and operating coverage | 12,15,16,26,27,30,33,34,41,63 | Earnings/cash-flow proxies or operations relative to debt/financing costs |
| Working-capital cycle | 20,32,43,44,47,52,62 | Inventory, receivable and payable days |
| Turnover and growth | 9,21,36,60,61,64 | Sales relative to assets/components and sales growth |
| Firm size | 29 | Log total assets |

Some ratios span concepts (e.g. sales/debt); the fixed assignment above uses the
denominator's financing role for debt coverage. These are economically motivated
coarse buckets, not validated latent factors. The source's unusual EBITDA wording
for X48/X49 is retained as documentation uncertainty; raw values are not recoded.

## Attribution and operational event

Preserve signed sums in **raw-logit units**, without normalization or absolute-value
aggregation. Primary grouped reason is the observed portion of a financial group:
`phi_G,O = sum(phi_j for j in G intersect originally_observed)` at **both** endpoints.
A group is unavailable if none of its members was originally observed. The fixed
originally observed set excludes natural and artificial missing fields on Polish.
This preserves the paper's observed-clue estimand: newly verified fields cannot
enter the displayed reason set or its competitors retrospectively.

Also report a separately labelled `whole_group` sensitivity:
`phi_G = sum(phi_j for j in G)`, with the same group-availability rule. This includes
model contributions of imputed members and therefore must not be described as
only observed evidence. It is not substituted for the primary metric.

Feature top-3 and grouped top-1/top-2/top-3 are all reported. Each requires exactly
k currently available positive contributions greater than .01; otherwise withhold.
The frozen event is any displayed reason becoming <=1e-6 or having >=k originally
available competitors greater by >1e-6 after verification. Same predictor,
background and attribution method at both endpoints. Ties follow schema order.

Rank-gap sensitivity is 0/.005/.01/.02, alongside the frozen 1e-6 reference;
the sign boundary stays 1e-6 so widening a rank tie does not change the sign
definition. This follows the existing audit's separate sign/rank treatment.
These are logit attribution gaps, **not normalized attribution units**. All
thresholds are reported, with eligibility and a common-eligible comparison.
Probability-stability cutoffs are .01/.02/.05 in the frozen **raw probability**
scale; calibrated-probability sensitivity is additional, not a replacement.

Summing feature SHAP is not recomputing coalition/group SHAP. Neither is causal
truth, explanation correctness or an adverse-action reason validation.
