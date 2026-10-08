# Additive numerical amendment before the completed uncertainty run

The first frozen runner stalled at task 00490_box after 490 completed tasks:
m=4, p=.1, b=(1,1,1), radius=0, event mass z=0. It was manually interrupted
(exit 130). The post-task elapsed-time guard cannot interrupt a stalled solver;
the original run must not be described as complete. Its 1,226 remaining tasks
are NOT RUN. Detailed artifacts and original source are retained.

A separate diagnostic imposed maxiter=1000: the original HiGHS IPM without
presolve hit its iteration limit on this zero-objective degenerate LP. With
presolve enabled, both IPM and dual simplex returned the exact residual 0;
at probe z=.5 all returned the exact residual .25. This identifies a numerical
degeneracy, not a failure of the rational lower-corner theorem.

V2 keeps the scientific config, LP, objective scaling and 1e-9 acceptance
tolerance unchanged. It enables HiGHS presolve and adds solver time_limit=10
seconds and maxiter=10000. A solver non-success still stops the run; the
post-task 180-second guard remains a secondary check. The original runner is
not edited. A new regression checks the stalled LP and an interior box.

The rerun is explicitly a numerical repair after observing a failure, not an
untouched numerical confirmation. Do not combine timings across runs or hide
the initial stall. Resume must use V2's own source hashes and a fresh directory.
