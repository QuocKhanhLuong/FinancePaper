# Large-scale validation status — NOT RUN on Freddie data

Date: 2026-10-03. Historical Taiwan/Polish results are unchanged. This is an
execution receipt and explicit evidence boundary, not a mortgage results paper.
The researcher confirmed **no officially acquired Freddie files are available**.
No new raw dataset was downloaded or used. No credentials or registration terms
were supplied/accepted by this implementation.

## What is established

| Item | Status | Evidence |
|---|---|---|
| Original-provider provenance and publication scope | ACCEPT WITH RESTRICTIONS | [Nine-dataset audit](DATASET_PROVENANCE_AUDIT.md), including seven large candidates |
| Exact selected release and cohort | PREDECLARED | Release 47, 15 Standard annual samples, nominal 750,000 source loans |
| Endpoint, censoring, chronology and features | SPECIFIED | [Target](FREDDIE_TARGET_DEFINITION.md), [leakage audit](FREDDIE_LEAKAGE_AUDIT.md), [protocol](FREDDIE_DATA_PROTOCOL.md) |
| Offline local intake and current/future separation | IMPLEMENTED / SYNTHETIC-TESTED | `data/freddie.py`, `scripts/prepare_freddie.py` |
| Stable-core evidence/calibration/metrics | IMPLEMENTED / SYNTHETIC-TESTED | `reliability/stable_core.py`; per-reason risk is separate from any-reason risk |
| Second completion family | IMPLEMENTED / SYNTHETIC-TESTED | `reliability/conditional_completion.py`; training-only conditional leaf sampling |
| Official Release 47 file conformance | NOT TESTED | No ZIPs, source checksums, authenticated receipt or real missingness counts |
| Mortgage training/evaluation integration | NOT IMPLEMENTED END TO END | Existing generic LR/XGBoost/TreeSHAP components retained; frozen mortgage runner still required after intake |
| Predictor fit / calibration / attributions | NOT RUN | No mortgage models, explanation caches or fitted release policy |
| Region B / stable-core efficacy / runtime at scale | UNKNOWN | No measured large-scale scientific result |

## Planned versus measured scale

| Quantity | Planned | Actually processed this stage |
|---|---:|---:|
| Pilot source loans | 100,000 | 0 real loans |
| Full source loans | 750,000 | 0 real loans |
| Labelled eligible loans | At least 500,000 desired; no cohort growth if short | Unknown |
| Source monthly records | Determined by official annual files | Unknown |
| Prediction assessment | All labelled loans from 2020–2022 | 0 |
| Explanation assessment | Fixed maximum 30,000 loans across three test vintages | 0 |

No discrimination, calibration, revision, selective coverage, false-stable rate,
completion-family gain or latency estimate is fabricated. The large public
provider population is not the size of our completed experiment.

## Engineering verification

The suite passes **105 tests**, including **21 new synthetic tests** for this
branch. Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal pytest -q
```

Tests cover twelve-month endpoint boundaries, year rollover, known-positive
evidence despite gaps, unknown/RA/terminal censoring, competing payoff, duplicate
and layout rejection, frozen-artifact mutation, temporal/entity separation,
future-input invariance, natural-versus-artificial missingness, partial reason
release, undefined empty-set risk, failure of pooled calibration in an environment,
dependent-reason calibration, conditional categorical support, fixed observed
values and training-only fitting. No fixture is a real Freddie record.

An attempted pilot command with the intentionally unaccepted example receipt
exited with code 2 **before opening a raw path or creating an output directory**.
This is the required provenance gate, not a failed real-data experiment. Receipt
attestation and matching hashes cannot themselves prove authorized acquisition.

Three pre-existing SHAP/matplotlib deprecation warnings remain; no test failures.
No MPS/CUDA result, large-cohort memory measurement, or overnight runtime claim
is inferred from this small CPU software suite.

## Remaining execution gate

The next input is the exact official Release 47 sample files and a truthful local
acquisition receipt. First run pilot intake; stop and document any schema/target
contradiction. Then integrate the frozen CPU LR/XGBoost training, grouped TreeSHAP,
two completion families, and whole/stable-core calibration on non-test vintages.
The confirmation intake refuses to open test archives until fitted-artifact
hashes and the completed development intake are supplied. It cannot validate
their scientific contents by hash alone; review remains necessary.

This limitation is intentional and visible: the repository is **not yet an
end-to-end tested mortgage experiment**. Stable-core is GO for feasibility only.
Publish only aggregate, reviewed, non-reconstructive results after actual runs.
Do not infer efficacy or novelty from an implemented policy primitive.
