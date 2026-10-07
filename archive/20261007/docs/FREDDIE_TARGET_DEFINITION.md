# Freddie target: mortgage serious delinquency, not generic default

Frozen 2026-10-03, before raw access. Provider facts come from the
[Release 47 General User Guide, July 2026, pp. 10–11 and 16](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf).
The endpoint and censoring choices below are OUR study definition, not a label
supplied or endorsed by Freddie Mac.

## Official field support

`Period` is the record's as-of month. Current Loan Delinquency Status uses numeric
monthly buckets: 00 is current/<30 days, 01 is 30–59, 02 is 60–89, 03 is 90–119,
and higher numeric values increase the delinquency bucket, capped at 99. `RA`
denotes REO acquisition; `XX` is unavailable. Termination is recorded separately
by Zero Balance Code/Effective Date. Accounting/reporting conventions change over
time and records can be corrected; a historical as-of field is not a timestamped
archive of what a public user actually knew then.

## Frozen study construction

One loan, one landmark. Let F be the original First Payment Date (calendar month).
Set **t = F + 5 months**. The observation window is `[F,t]` (six reporting months).
The performance window is **`t+1,...,t+12`**. Input values may come only from
origination and performance rows with Period <=t. Do not use mutable provider
Loan Age to define the window: modifications can reset it. Calendar arithmetic
must handle year boundaries; subtracting YYYYMM integers is invalid.

Primary task name:

> Mortgage serious-delinquency prediction: whether a loan has an observed numeric
> 90+ days-past-due status during the next 12 reporting months, before termination.

This is a retrospective reporting-time estimand, not guaranteed real-time public
data availability, a regulatory default definition, or lifetime bankruptcy risk.

## Eligibility at the landmark

- Standard fixed-rate loan with a valid original First Payment Date, within
  January of the filename vintage through March of the following year.
- Six consecutive performance rows `[F,t]`; no duplicate loan/month pairs.
- No known numeric status >=03, RA, or termination in that history.
- At t, numeric status is known and <03, with positive known UPB.
- Earlier history fields can contain genuine unknowns. Unknown is not current.
- Origination fields may be missing and are not grounds for complete-case removal.

Sample selection and source membership precede outcome inspection. Dates/identifiers
are metadata, not predictive features. No class balancing or future-outcome filter
is applied to source sampling.

## Label/censoring decision table

Process future periods in chronological order, stopping at the first terminal/RA
record; never infer outcomes after loss of follow-up.

| Evidence in horizon | Outcome |
|---|---|
| Numeric status 03–99 before or at first terminal record | `y=1`, observed 90+ event; later missing months do not erase known event |
| Known numeric <03 in all 12 consecutive months, no earlier event/exit | `y=0`, complete negative follow-up |
| Voluntary payoff/maturity (zero balance 01), no earlier event, complete known <03 trajectory through payoff month | `y=0`, absorbing competing payoff; flag separately |
| RA without a numeric 90+ observation before it | Censored, not automatically positive |
| Other terminal code (02/03/09/15/16/96), no prior observed 90+ | Censored, not a delinquency proxy |
| XX/blank or missing future month and no observed event before exit | Censored; never silently negative |
| Dataset cutoff before horizon ends, no known event/valid payoff | Censored |
| Malformed/unsupported status, conflicting/duplicate month, inconsistent termination date | Stop with data-quality error |

A gap before an observed positive still proves an **ever-observed** event. A gap
before payoff does not establish a negative. Unknown status in a terminal month
cannot certify a negative. Event in month t+12 counts; t+13 does not. Label=0 at
payoff means no event before this loan ended, not that a borrower could never have
become delinquent absent payoff. Censoring is potentially informative: report its
frequency/reasons by split and do not describe labelled-only metrics as unbiased
performance for all originated loans. Survival/IPCW modelling is outside this phase.

## Frozen data availability and prevalence gate

Release 47 performance ends 2026-03-31, which allows the planned vintages to mature.
Real eligibility, positive prevalence and censoring are **UNKNOWN / NOT RUN**.
Require >=100 observed positives in train and assessment for a useful predictive
comparison; otherwise report inadequate support and stop without changing target.
All reporting includes source loans, landmark-eligible loans, labelled loans,
censored loans, observed positives, payoff negatives and complete-follow-up negatives.

Synthetic tests cover horizon boundaries, year rollover, missing reports, RA,
termination, unknown codes, modified-age independence and future-input invariance.
They establish software behavior only, not financial validity or actual prevalence.
