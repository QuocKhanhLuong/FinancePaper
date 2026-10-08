# Numerical repair addendum — reference-moment audit

2026-10-08, after the first run stopped. This is an explicit numerical execution
amendment, not a prospective untouched confirmation. Keep the original protocol,
config, source snapshot and partial run immutable. No model profile, hypothesis,
threshold, dimension, comparison, mathematical formula or 1e-9 acceptance
tolerance is changed.

Original run `reference_moments_20261008T115000Z` completed 272/336 tasks and
stopped at m=16, p=9/10, all b=0. The generic HiGHS default LP returned
0.9999999988978737 while the rational bound is 1, discrepancy
1.1021262791288677e-9. Its minimum beta coefficient is 1/24000000000000000.
The exact feasible point z=1,s_k=1/2 has zero event residual and satisfies every
LP bound; z<=1 proves global optimality. Thus this failed comparison is a
numerical optimization discrepancy, not evidence against the rational frontier.

A diagnostic re-solved the same LP with default presolve, without presolve,
and tighter dual-simplex tolerances: all returned the same value. Interior-point
HiGHS without presolve returned exactly 1. The likely mechanism is handling of
tiny coefficients; this is not asserted as a fully isolated solver-library bug.

An initial IPM-only implementation still failed the same regression when its
optimality tolerance was tightened. Changing solver mode alone is therefore
not a sufficient repair. V2 also multiplies the event-constraint row (whose
right-hand side is zero) by 1e6. This is an algebraically equivalent LP and
preserves more small coefficients above the solver's absolute entry threshold.

V2 changes the generic moment-LP backend to `highs-ipm`, disables presolve,
sets primal/dual feasibility tolerances to 1e-10 and IPM optimality tolerance to
1e-12, and applies that fixed row scaling. It retains the original 1e-9 scientific acceptance threshold and all
full-truth-table LP settings. New regression tests cover the failing boundary
and large-dimension profiles. A new output directory and source hashes are
mandatory. Do not resume the failed original run with changed source.

Every case, including the original failure, is run again. Report the first
failure alongside the rerun, label new tests as failure-driven, and preserve
the 64 originally unfinished tasks as NOT RUN in the first run. The rerun can
complete them under the amended numerical execution if all checks pass.
