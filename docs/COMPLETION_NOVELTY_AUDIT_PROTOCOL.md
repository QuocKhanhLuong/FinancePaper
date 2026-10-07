# Frozen challenge protocol, 2026-10-08

This is an exploratory mathematical/algorithm audit, not a preregistered human
study or evidence of originality. The 19 development tests and agent sanity
checks were already seen before this freeze. No old financial endpoint is used.
Configuration: `configs/completion_novelty_audit.json`; source/config SHA256s are
frozen at the first run. Changes require a new run; failed attempts stay local.

## Objects and checks

1. C0: first/second moments of completion-level interventional SHAP for a fixed
   additive raw-output tree model, fixed product reference P and product Q.
   Observed values can be point masses in Q. All features remain SHAP players.
   Product-Q does not imply independence of leaves sharing a coordinate.
2. 100 seeded d=3 finite-reference/completion cases, two depth-3/2 trees with
   repeated features, tested against direct coalition/background/completion
   enumeration. Check mean, raw second moment, covariance, not only variance.
3. Eight fixed box-ensemble benchmarks (equivalent explicit trees with zero
   leaves). Targets 0 and 1, three timing repeats, one CPU thread. Compare
   bivariate quadrature, polynomial coefficient convolution, exact leaf-pair
   threshold-cell enumeration, and completion MC with 1,024/16,384 draws.
   Pair enumeration cap 65,536 states per pair; exceeding it is NOT RUN, not a
   measured timeout. All endpoints/timings are reported, including losses.
   MC seeds vary by benchmark, repeat and draw count. Error relative to C0 is
   only a numerical discrepancy unless an independent exact oracle completed.
   These are Python prototypes, not optimized implementations of prior papers.
4. C2: 12 arbitrary finite joint-law constructions: m=2,3,4 observed binary
   features; S=3,5 hidden states; two seeds. Full-support product P, same hidden
   law in P and Q, observed a=1 under Q. Verify each desired SHAP row using
   coalition/background enumeration; prediction is constant; moment covariance
   matches the supplied finite law. This is mathematical synthetic evidence.
5. Tail boundary: n=8,12,16 integer knapsack weights; compare exhaustive count,
   integer dynamic program and SHAP sign counts. This checks the reduction,
   not a novel complexity theorem. Moments do not determine tail probabilities.

Correctness tolerance 1e-10 absolute; failure ends the stage. Floating quadrature
is polynomial-exact in real arithmetic, not exact rational or bitwise arithmetic.
No clipping covariance eigenvalues or selecting successful seeds. Performance
has no preclaimed win gate. Full law realizability assumes unrestricted real
leaf values; no probability/financial/human utility interpretation is supplied.

## Execution and stop rules

Use a unique ignored `runs/decision_value_pilot/novelty_*` directory. JSONL
progress, tqdm/ETA, per-task atomic receipts, source/dependency/output hashes and
`--resume` are mandatory. Exercise a real `--stop-after-tasks 1` and resume. Resume
must reject changed code/config or completed output. No training or data download.
Publish aggregate rows, execution/validation receipts and one consolidated report;
raw logs and AI transcripts stay local. Literature equivalence overrides numerical
success: generic quadrature, circuit moments and prediction-invariant explanation
instability are prior art. A narrow theorem is not automatically a paper-level
novel contribution. Report the actual novelty verdict, including NO-GO.
