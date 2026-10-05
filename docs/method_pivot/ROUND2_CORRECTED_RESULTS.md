# Corrected numerical result and decisive strong-control failure

Date: 2026-10-05. **Current conclusion: NO-GO for the proposed method claim.**
Numerical correctness survived; a standard orthogonal-basis moment control is
substantially faster on the same predeclared workload. This distinguishes a
failed necessity claim from an incorrect implementation or an unrun method.

## Corrected run 02

The [pre-run amendment](ROUND2_AUDIT_AMENDMENT.md) corrects float32 execution
semantics and includes baseline HD-table construction and covariance aggregation
in timing. Models, seeds, queries, masks, targets and completion support are
unchanged. The initial [run-01 report](ROUND2_RESULTS.md) remains as history.

Measured source: clean commit `4f230f34c3cb31766e50f482e590b871c741ffb0`.
Receipt: `outputs/method_pivot/round2/run_02/manifest.json`. Runner COMPLETE,
141.7756 seconds, 355,237,888-byte whole-process peak RSS (338.78 MiB).
COMPLETE means execution completed; it does not mean every scientific gate passed.

| Method | K/support | Query ms, mean ± SD | Mean maximum-entry covariance error |
|---|---:|---:|---:|
| Specialized rectangle moments | implicit 16,384 | 67.022 ± 10.792 | 1.119e-8 |
| Cached WOODELF-HD exact enumeration | 16,384 | 129.523 ± 16.776 | reference |
| Cached WOODELF-HD + MC | 8 | 66.622 ± 2.788 | .08848 |
| Cached WOODELF-HD + MC | 32 | 58.786 ± 2.886 | .04459 |
| Cached WOODELF-HD + MC | 128 | 59.445 ± 3.007 | .02373 |
| Cached WOODELF-HD + MC | 1,024 | 65.063 ± 3.055 | .00735 |
| Stock TreeSHAP + MC | 8 | 4.454 ± .243 | .08848 |
| Stock TreeSHAP + MC | 32 | 17.167 ± .896 | .04459 |
| Stock TreeSHAP + MC | 128 | 67.903 ± 3.285 | .02373 |
| Stock TreeSHAP + MC | 1,024 | 540.227 ± 25.082 | .00735 |

Times now include attribution-to-moment aggregation. Support construction,
predictor fitting, explainer compilation and verification-only diagnostics remain
separate: these are kernel/query timings, not service end-to-end latency.
Compilation: .0595–.0659 s. WOODELF model initialization: .0092–.0186 s;
HD constructor: .000053–.000087 s; warm-up/table construction: .0978–.1079 s.

Twenty-four queries, five post-warm-up repeats per MC/specialized method.
Enumeration has one measurement per query. SD is descriptive, mixing query and
repeat variation; no independent-customer CI or significance claim is inferred.
Ratio of mean query times: enumeration/specialized = **1.93×**.

## Covariance and scope checks

All 24 query mean/covariance arrays are persisted, with raw-score efficiency
diagnostics. Maximum absolute mean-efficiency error: 5.384e-7; maximum absolute
variance-efficiency error: 3.121e-6. Symmetry error zero; minimum eigenvalue over
these matrices was 4.357e-11. All pass the declared 2e-5 float32-tree tolerance.
These identities validate model-relative arithmetic, not the completion law.

The fixed tiny joint-law fixture supplies two dependent coordinates. Exact
operators agree. With no observed coordinate, specialization takes .0916 ms
versus .0632 ms generic: bookkeeping costs more. With one observed coordinate,
three instead of six terms take .0597 versus .0601 ms. This does **not** establish
a universal specialization speedup. The full-scale generic contraction still
hits its original term cap on all 24 queries; that comparison is incomplete.

The supplied-law interface is now explicit: callers provide q(X_hidden|observed).
Clamping known fields does not condition weights from an unconditional donor
pool. Native NaN/categorical tree inference is outside this finite numeric
prototype; such support is not silently claimed.

## Fourier-basis falsification control

The [control protocol](ROUND2_FOURIER_CONTROL_PROTOCOL.md), commit `22b670c`, was
written before its implementation/results, **after** the first pilot. It is a
disclosed post-pilot strengthening of the baseline, not original preregistration.
No generator or law was modified to make either operator win.

This independently implemented control uses the same compiled attribution map,
but represents each interval indicator in the known orthonormal four-atom Walsh
basis. Mean is the constant coefficient; covariance is the Gram matrix of
nonconstant vector coefficients. The identities are standard orthogonal-series
calculus. It is an adaptation motivated by [FourierSHAP](https://papers.nips.cc/paper_files/paper/2025/hash/1b331c20064e37e204a5bcd12481bfac-Abstract-Conference.html),
**not** the authors' official implementation or a reproduction of their numbers.

Three seeds × two fixed generators × four queries × five repeats; both query
operators are timed contemporaneously with shared compilation. Local receipt:
`outputs/method_pivot/round2/fourier_control_01/manifest.json`, COMPLETE, 12.2053 s.

| Operator | Query ms, mean ± SD | Maximum mean/covariance disagreement |
|---|---:|---:|
| Proposed rectangle-intersection moments | 67.588 ± 11.355 | reference |
| Known Fourier-basis moment adaptation | 10.901 ± 1.612 | 2.665e-15 / 2.192e-15 |

The control is **6.20× faster** by ratio of means on this paired workload. It
retains 94–195 nonconstant frequency vectors instead of forming a dense
360–509-square intersection matrix. This is coefficient storage, not a measured
per-method peak-RAM comparison. It is not evidence for correlated completion
laws, arbitrary continuous inputs, or all tree depths.

| Seed/world | Rectangle ms | Fourier control ms |
|---|---:|---:|
| 11/additive | 65.249 | 11.015 |
| 11/interaction | 89.892 | 12.413 |
| 22/additive | 63.416 | 12.590 |
| 22/interaction | 65.609 | 8.850 |
| 33/additive | 52.326 | 8.732 |
| 33/interaction | 69.035 | 11.807 |

This is a direct failure of the proposed operator's practical necessity in the
registered numerical setting. Do not rename the faster baseline as our new
method. Do not search a new completion law just to reverse this result.

## Evidence labels and decision

- **VERIFIED:** numerical checks and the measured paired strong-control result.
- **REPORTED:** earlier financial decoupling, Stable-Core and acquisition studies;
  none was rerun or reinterpreted in this stage.
- **ASSUMED:** a fixed predictor/background and a supplied completion law are the
  response-distribution objects. They need not be correct descriptions of truth.
- **NOT RUN:** real financial benefit, new learned completion model, independent
  dataset confirmation, full-scale correlated-law timing, isolated method peak
  memory, official FourierSHAP execution and compatible-circuit software benchmark.

No new method contribution is supported. The result is useful as a checked
research oracle and a strong reduction control. Historical financial prediction
performance is unchanged. The [decision](ROUND2_NEXT_DECISION.md) closes this
construction as a method candidate; it does not declare every future method in
missing-data inference impossible.

Reproduce the added control, after the base environment is installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_fourier_moment_control.py \
  --output outputs/method_pivot/round2/new_fourier_control
```

The [audit receipt](ROUND2_FINAL_AUDIT.md) records final tests, source/artifact
integrity checks and publication status. Logs/arrays/models remain ignored.
