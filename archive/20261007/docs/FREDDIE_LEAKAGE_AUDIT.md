# Freddie leakage audit and required gates

Status: schema/design audit and synthetic tests; **real-file audit NOT RUN**.
Release 47 is a revised historical dataset, not a vintage-preserved real-time feed.
No strong point-in-time availability claim is justified even after period filtering.
References: [official guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf)
and [release changes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf).

## Input allowlist

Origination fields: classic FICO; original DTI, LTV, CLTV, UPB, rate, term, MI%;
number of units; first-time-homebuyer, occupancy, channel, property type, loan
purpose and one-versus-multiple borrowers. Collapse Number of Borrowers >=2 to
multiple because its disclosure granularity changed in 2018; 99 stays unknown.
Exclude new VantageScore4 to avoid historical reconstruction/availability ambiguity.

Six monthly values each: delinquency bucket, actual UPB and current interest rate,
in chronological order F,...,t. These are source fields, not future differences.
Performance context after t is accessible only to target construction. Naturally
missing values remain NaN; both true unknowns and simulation masks are retained.

| Field or transformation | Allowed role | Leakage control |
|---|---|---|
| Origination immutable predictors above | Model inputs | Exact named allowlist; fit encoders on train only |
| Loan ID, First Payment Date, filename vintage | Entity/time metadata | Never encoded as risk features or included in completion model |
| Period >t delinquency | Label only | Physically separate target file; no feature reducer sees these rows |
| Period >t UPB/rate | Neither input nor label | Ignore for feature construction |
| Zero Balance Code/Effective Date | Eligibility before t; censoring/competing event after t | Never a feature; chronological stop at exit |
| REO/foreclosure/disposition fields | Censoring audit only | RA is not silently redefined as 90+ |
| Future modification/deferral/assistance | Excluded | No ever-modified flag or all-history aggregation |
| Actual loss, recoveries, proceeds, expenses, bankruptcy cramdown, deferred/removed UPB | Excluded | May materialize after event; no model path |
| Origination Servicer / MI cancellation in old layouts | Excluded | Release 47 moved these to performance; old positional parser rejected |
| Seller, servicer, ZIP, MSA, state, external linked data | Excluded from chosen model | No identifying linkage or hidden-vintage proxy |
| SHAP background, imputers, donors, standardization | Training features only | No target columns, held-out rows or verification truth |
| Current attributions vs restored attributions | Serving vs audit | Separate APIs/artifacts; no restored score as release feature |

## Chronology and entity tests

Train 2000–2008, development 2011, probability calibration 2014, release calibration
2017, assessment 2020–2022. Years not listed are not extra training. Check
`max(t+12 in earlier role) < min(t in next role)` on actual dates; fail if false.
This isolates both customer partitions and outcome maturation. One ID occurs in
exactly one role; monthly rows do not cross partitions. Stage 1 is an explicitly
overlapping subset of training, not an additional evaluation.

Provider loan ID is the independence unit requested by the protocol; it does not
identify a person. Refinance loans can refer to a prior loan: exclude records with
a nonblank Pre-HARP link or Relief Refinance flag Y using origination metadata,
and never use the link as a feature. This reduces known linkage; unknown repeated
borrowers/property relationships remain a limitation. Do not claim household independence.

Changing any future performance value must leave the current feature vector
identical. Changing artificial hidden truth must leave completion/release outputs
identical until the separate verification evaluator is called. A naturally missing
cell must never enter the artificial mask or receive invented restoration truth.

## Remaining risks to report, not fix using assessment outcomes

- Release 47 may retroactively correct historical origination and performance
  fields. Restrict claims to retrospective temporal generalization.
- Termination/censoring and missing history can select an easier labelled cohort.
- As-of months combine reporting systems, especially before May 2019; timestamps
  do not certify same-day operational availability.
- Excluding known refinance links does not resolve unknown same-borrower loans.
- Completion distributions may miss new market regimes even without leakage.

Any contradictory schema, impossible temporal overlap or evidence of future-input
access stops the real experiment and requires a written incident before correction.
