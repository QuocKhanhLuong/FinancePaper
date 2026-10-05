# Round 4 baseline sanity: measured results

Date: 2026-10-05. Protocol committed before implementation at `f7cb080`.
**This is an exact synthetic example and engineering check, not a new method or
financial experiment.** The hypothesis and generator were frozen in
[the sanity protocol](ROUND4_BASELINE_SANITY_PROTOCOL.md).

| Quantity | Executed value |
|---|---:|
| Probability for h=0 | 0.7310585786300049 |
| Probability for h=1 | 0.7310585786300049 |
| Prediction variance | 0 |
| Hard-vote confidence | 1 |
| Agreement with current action | 1 |
| Verified top-1 observed-reason revision probability under the declared law | 0.25 |

Attributions `(phi1,phi2,phih)` were `(0.125,0.875,0)` at h=0 and
`(0.625,0.375,0)` at h=1. Exhaustive coalition evaluation agrees with the analytical
formula. The winner changes with a strict gap, not a rank tie. Attribution sums
reconstruct the margin. The additive control returns `(1,1,0)` for both values.

The 25% is a consequence of the **constructed Bernoulli(1/4) law**. It is not an
estimate of financial prevalence, not the historical 37/148 finding, and not a
claim that a learned prediction-only detector cannot learn a population prior.
It demonstrates only that prediction/class stability alone does not logically
force attribution ranking stability. The example does not establish an algorithm.

## Execution and provenance

Verified hardware: Apple M4 Pro, 25,769,803,776 bytes RAM (24 GiB). CPU only.

```bash
.venv/bin/python scripts/run_prediction_distribution_sanity.py \
  --output outputs/method_pivot/round4/sanity_01
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

The runner rejects an existing output directory; choose a new run directory for
another invocation. No third-party MVU code or dataset was executed.

- New tests: **15 passed** in 0.09 s.
- Combined suite: **168 passed, 0 failed, 0 skipped**, 12.89 s.
- Three upstream SHAP/Matplotlib deprecation warnings; no suppressed failure.
- Log: `outputs/method_pivot/round4/tests.log`.
- Log SHA-256: `d024ce326a866a6c2b48c7c9e7ebe610fcb74cd5deaa7ed8cd016924392bf04a`.
- Results SHA-256: `fb0242c25c39d9379549dff4599087f85d328b9e0077f38dffba7deb69e4aa0e`.
- Manifest SHA-256: `2039042162f9de82d986b5438babbf70b9ce1f68fc9ca25f8cb3ce2115180394`.

The manifest records the truthful dirty implementation state after protocol
commit and hashes measured sources. A preliminary shell invocation used system
Python and failed to import financepaper before creating artifacts; rerunning
with the project virtual environment succeeded. No failed run was counted as a
scientific result. Raw receipts stay ignored.

**VERIFIED:** arithmetic, API validation, deterministic/order-invariant controls,
input exclusion signature and existing-suite compatibility.
**REPORTED:** historical Taiwan/Polish findings remain in historical reports.
**NOT RUN:** new financial development comparison, learned DMV, confirmation.
**ASSUMED:** the synthetic hidden law, point attribution reference and exact
predictor declared in the protocol.
