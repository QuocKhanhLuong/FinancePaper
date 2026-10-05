# Round 2 final engineering and evidence receipt

2026-10-05; branch `research/method-pivot`. No main merge.

## Tests actually run

Final combined implementation and tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q
```

**153 passed, 0 failed, 0 skipped, 3 warnings, 11.83 s.** Warnings are upstream
Matplotlib color-map deprecations emitted by SHAP, not ignored test failures.
The new test module has 16 pytest cases, including parameterized exhaustive
coalition checks. This count is engineering evidence, not scientific sample size.

Current log: `outputs/method_pivot/round2/final_tests.log`, SHA-256
`1d91def47ae97a87ea3e86e48c0e0274646dcb42fabffa0edc844a2e557b96d2`.
Historical `outputs/method_pivot/*tests.log` files are **not** the receipt for
this round. The reporting script was also executed successfully and its plots
visually inspected after the final test invocation.

## Provenance verified independently

| Run | Source verification | Raw artifact verification | Manifest SHA-256 |
|---|---|---|---|
| run_01 | 6 measured files against preserved pre-fix snapshots | 2/2 | `b56a67f8ae8d3d807c12d0e02dbc8fa2c213a754a4768d04f162ff309d81cff7` |
| run_02 | 9 files against clean commit `4f230f34c3cb31766e50f482e590b871c741ffb0` | 5/5 | `5ad513cc49f9496e1531961a433fe5e7ef93119977993f0098abb0e7f87d7930` |
| fourier_control_01 | 5 measured files against source hashes | 2/2 | `b71d98c3c37eb6e20250eb377c5311c9195ec07d5a708bba770b8d1dcd2078bc` |

The Fourier control ran from protocol commit `22b670c` with `git_dirty=true`
because its implementation was not committed yet. Its manifest truthfully records
that state and hashes all measured implementation/config/protocol files. Do not
claim that commit alone reproduces the control; use its source hashes and the
subsequent implementation commit. The original run 01 also records a dirty tree;
its source snapshot is retained. No manifest was rewritten to hide these facts.

Integrity receipt: `outputs/method_pivot/round2/integrity_audit.json`, SHA-256
`9099fa24929e21eb3f4d75ebf66764c38ea8508c862a4d12f8f0c44cce35b06f`.
An independent read-only reviewer also checked the run-02 committed sources and
all five raw artifacts, and found no remaining algebra/correctness blocker.

## Fixes and limits

- Float32 threshold-neighbor routing now matches XGBoost. Finite atom laws use
  the same execution dtype; generic hand-tree oracles can remain float64.
- Both sides of interval endpoints, repeated splits, point-mass reduction,
  same-law hidden-placeholder exclusion, correlated small laws, PSD, efficiency,
  signed grouping and the equal-moment/different-sign limitation are tested.
- Baseline initialization and moment aggregation are timed in run 02. Support
  materialization is outside reported query-kernel timing. Different per-query
  law/basis preparation is included in each operator as implemented; no common
  service-latency or isolated-memory claim is made.
- Full-scale generic contraction remains BUDGET_EXCEEDED. Correlated-law scale
  performance and isolated per-method peak RSS remain NOT RUN.
- The current mechanism is **NO-GO as a new method** after the measured stronger
  control. The faster baseline is not relabeled as a contribution.

## Repository policy

Only documentation, configs, source, runner/report scripts and tests are intended
for publication. `git check-ignore` confirms raw receipts, third-party source,
arrays and test logs under `outputs/` stay excluded. No data, model weights,
credentials, historical output rewrite, force push or main merge is part of this
change. Final branch/remote identity is verified separately at publication.
