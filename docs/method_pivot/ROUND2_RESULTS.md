# Exact conditional attribution moments: measured feasibility, no method claim

2026-10-05. Prospective protocol commit: `7cb4c8d`. Research branch only.
This reports **newly run synthetic numerical evidence**, not financial validation.
The historical Taiwan/Polish reports, targets and negative results are unchanged.

**Run-01 timing is provisional:** the [post-review amendment](ROUND2_AUDIT_AMENDMENT.md)
identified omitted baseline initialization/aggregation time and an input-dtype
boundary bug outside the tested support. Run 01 is retained; corrected run 02
must supersede its timing conclusion. Correctness applies to its tested support.

## What ran

Apple M4 Pro, 24 GiB host RAM; CPU, one native thread. Three fixed seeds
11/22/33, two preregistered outcome generators, six fixed 200-tree depth-3 XGBoost
fits, four fresh queries per fit. No model search. The integration law has seven
hidden coordinates, four atoms each, hence 16,384 equally weighted completions.
It is a declared product law; it is **not** the true Gaussian conditional law of
the generator. Correlated joint laws were checked in small exhaustive fixtures,
not benchmarked at financial scale.

The frozen inference target is the mean and full covariance of raw-margin
interventional attribution under that supplied law. The empirical background is
kept joint; it is not replaced by independent background marginals.

Main run: `outputs/method_pivot/round2/run_01/`, manifest status COMPLETE,
144.7846 seconds; process high-water RSS 317,505,536 bytes (302.80 MiB).
This includes both comparators and temporary enumeration arrays, not isolated
method memory. No GPU or new external dataset was used.

## Accuracy and timing

The error below is the mean over measurements of the **maximum absolute entry
error in the covariance matrix**, versus exact finite-support WOODELF-HD
aggregation. It is not explanation failure probability, AP or calibration error.
All inputs/draws are paired between compatible point explainers.

| Method | K/support | Query time, ms | Covariance error |
|---|---:|---:|---:|
| Specialized exact moments | implicit 16,384 | 68.224 ± 13.145 | 1.119e-8 |
| WOODELF-HD exact enumeration | 16,384 | 133.148 ± 21.163 | 0 by reference definition |
| Cached WOODELF-HD + MC | 8 | 68.678 ± 3.116 | .08848 |
| Cached WOODELF-HD + MC | 32 | 60.103 ± 2.605 | .04459 |
| Cached WOODELF-HD + MC | 128 | 60.525 ± 2.536 | .02373 |
| Cached WOODELF-HD + MC | 1,024 | 66.194 ± 2.891 | .00735 |
| Stock TreeSHAP + MC | 8 | 4.689 ± .805 | .08848 |
| Stock TreeSHAP + MC | 32 | 17.526 ± .901 | .04459 |
| Stock TreeSHAP + MC | 128 | 69.247 ± 3.445 | .02373 |
| Stock TreeSHAP + MC | 1,024 | 552.582 ± 28.843 | .00735 |

For specialized/MC methods, these are mean ± sample SD across 24 queries × five
timing repetitions, after excluding a warm-up per query/method/K. The SD mixes
workload differences and repeat variability; it is **not a confidence interval**.
Enumeration has one measurement for each of the 24 queries: its SD is across
queries, not five repeats. No statistical significance claim is made.

The ratio of mean enumeration to specialized query time is **1.95×**, excluding
both methods' setup. Compilation took .0605–.0658 s per model; WOODELF model
loading took .0094–.0160 s and its warm-up/table construction .0997–.1093 s.
These setup costs must be paid or amortized. Do not quote the 1.95× ratio as an
end-to-end financial service speedup.

| Seed/world | Specialized ms | Exact enumeration ms |
|---|---:|---:|
| 11/additive | 66.372 | 123.535 |
| 11/interaction | 93.580 | 148.629 |
| 22/additive | 63.186 | 129.332 |
| 22/interaction | 64.629 | 134.862 |
| 33/additive | 52.351 | 124.960 |
| 33/interaction | 69.227 | 137.574 |

Maximum specialized mean error was 3.0961e-8; maximum covariance error 2.3702e-8,
below the preregistered 2e-5 float32-tree comparison tolerance. The independent
double-precision coalition fixtures passed their 1e-12 test tolerance.

Observation specialization reduced 3,237–3,731 compiled rectangles to 360–509
residual rectangles. Joint-matrix storage was 1,036,800–2,072,648 bytes. The generic
unspecialized contraction hit the **declared 2,000-term cap on all 24 queries**.
That is a budget result, not proof that generic moment inference is impossible
or a measured speedup over a completed generic run.

## Strong comparator and fairness

Official author implementation:
[WOODELF repository](https://github.com/ron-wettenstein/woodelf), MIT,
commit `e3a528e0309882042e885078913a3d7852203ddd`, package 0.4.8,
Treelite 4.7.0. The runner explicitly supplies `HighDepthWoodelfPathToSVectors`;
it does not silently use the default sparse path instead of HD.

A documented subclass memoizes background-specific HD tables using all their
dependencies as the key. This is an **adaptation for amortized timing**, not an
official paper benchmark. Formulae and outputs are unchanged; point outputs also
match stock interventional TreeSHAP. WOODELF has noticeable small-batch overhead:
stock SHAP is the faster comparator at K=8. Both remain in the report.

The first dependency preflight found Treelite's missing `libomp` runtime search
path. Setting `DYLD_LIBRARY_PATH` to the already installed Homebrew libomp resolved
it; no silent comparator replacement or CUDA dependency was used.

## What this does and does not establish

- **VERIFIED:** exact conditional moments on the tested discrete worlds; observed
  input restriction; nonzero attribution variance at zero prediction variance;
  useful numerical accuracy/cost point versus the measured baselines.
- **VERIFIED limitation:** equal attribution means/variances can have different
  sign probabilities. Moments cannot be substituted for the existing revision
  event or release policy.
- **REPORTED, not rerun here:** earlier financial decoupling and selector studies.
- **ASSUMED by this numerical experiment:** the supplied completion law and the
  fixed model/background are the inference objects of interest. Exact integration
  does not validate that law against true missing values.
- **NOT RUN:** financial development, deployment release decisions, real correlated
  completion benchmark, FourierSHAP execution, compatible-circuit integration
  benchmark, native-NaN/categorical tree support, independent confirmation.

No AP/AUROC/Brier improvement is possible from this compiler alone: it does not
change the predictor. No stronger reason-risk calibration is established.

The current solver has a concrete numerical advantage over finite enumeration,
but the [reduction audit](ROUND2_NEXT_DECISION.md) still prevents a new-method
claim. Correctness plus a 1.95× workload speedup is insufficient to assert a new
inference principle or superiority over all existing compiled-moment algorithms.

## Reproduction and receipts

The optional comparator is kept out of the project's default dependency set.
It was installed from the pinned official source, not copied into this repository.
After the normal repository environment setup:

```bash
uv pip install --python .venv/bin/python \
  'woodelf-explainer @ git+https://github.com/ron-wettenstein/woodelf.git@e3a528e0309882042e885078913a3d7852203ddd' \
  'treelite==4.7.0'
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_conditional_moments_pilot.py \
  --config configs/conditional_moments_pilot.yaml \
  --output outputs/method_pivot/round2/new_run
.venv/bin/python scripts/report_conditional_moments_pilot.py \
  --run outputs/method_pivot/round2/new_run
```

The Homebrew runtime path is specific to the measured Mac; use the installed
libomp location on another Mac. Output directories must be new; the runner
rejects overwriting a prior manifest. Missing comparator/provenance causes an
explicit failure. It never falls back to another method or historical data.

Local raw measurements, workload setup timings, source/config hashes, exact
versions, stdout log and manifest are retained under the run directory. The
report script verifies raw artifact hashes before producing summary tables and
`accuracy_cost.png/.pdf`. Six source/config/protocol hashes and two raw artifact
hashes were independently checked against the completed run.

Initial repository suite after implementation: **147 passed**, 0 failed,
0 skipped, three upstream plotting deprecation warnings, 12.86 s.
Ten added tests include 24 exhaustive world/observation-subset combinations;
those cases must not be inflated into a claim of 24 independent scientific tests.
See the final audit receipt for any post-review rerun or supported-scope changes.
