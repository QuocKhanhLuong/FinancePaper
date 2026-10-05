# Post-review correction before run 02

2026-10-05. Run 01 and its measured source snapshot are retained. This amendment
is written **before corrected run 02**. It changes no model, mask, generator,
completion support, seed, target, or primary numerical tolerance.

## Identified issues

1. The generic numeric compiler accepted arbitrary float64 points, whereas
   XGBoost executes numeric comparisons after float32 conversion. A float64
   `nextafter` neighbor of a tree threshold can therefore route differently.
   The tested primary queries were float32 and completion atoms exactly
   representable; no known measured run-01 mismatch results, but the public
   interface was too broad. Add explicit input dtype and cast finite law support
   consistently for an XGBoost compiler. Preserve float64 for independent oracles.
2. Baseline HD-table constructor and mean/covariance aggregation were not timed.
   This favors the baseline, not the candidate, but prevents a complete cost
   comparison. Time all three separately/inclusively as appropriate and rerun.
   No end-to-end service latency claim is permitted.
3. Generic full-scale contraction hit the registered cap. COMPLETE meant runner
   completion, not all scientific gates passed. Record separate execution,
   comparator and method-claim status fields. Do not lift the cap after results.
4. Dense joint-matrix bytes are not peak method memory. Whole-process RSS remains
   labelled as such; isolated per-method peak memory is NOT RUN. The prototype's
   rectangle queries loop over all p coordinates. Its actual bound includes
   O(R² p A + R² G + R G²) for A atoms/coordinate, not a path-depth-only bound.
5. The law API clamps known coordinates in an **already supplied conditional
   hidden-value law**. It does not estimate or Bayes-condition donor weights.

## Fixed additional correctness/accounting receipts

- Explicit float32 rounding neighbors at both interval endpoints; parser
  repeated-feature splitting; unchanged-law hidden-placeholder mutation.
- Actual attribution laws with identical moments and different sign support.
- Fixed small-world timings of specialized and unspecialized contraction and
  a correlated empirical-law oracle. These are fixture checks, not a new scale
  workload chosen for speed. Scale correlated-law performance stays NOT RUN.
- Persist query mean/covariance and efficiency/PSD/symmetry diagnostics.
- Record Python/runtime/library-search-path and dirty-file/source provenance.
- Rerun the same six synthetic fits/24 queries, original 10-minute budget and
  five timing repetitions. Run-01 figures are superseded for timing conclusions.

These engineering corrections cannot change the independent novelty reduction:
the current mechanism remains a standard moment query on a compiled function.
