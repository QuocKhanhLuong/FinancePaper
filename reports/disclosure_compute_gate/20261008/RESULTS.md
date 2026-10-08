# Compute/coverage gate: cheap moments work; C6 has no incremental gain here

2026-10-08. Branch `research/disclosure-compute-gate-20261008`.
Base commit `9577a2045f2be11b47cd09d585afec7b969e13b0`.

## Decision

**RUN result:** a cheap moment certificate passes the prespecified local
cost/coverage gate. At risk budget 5%, a mean-only bound certifies 96/324
enumerated law/model cells (29.63%); its median paired cost advantage over
exhaustive enumeration at the largest enumerated dimension is 193.95x.

**NO-GO for incremental C6 utility when the reference mean is available.** A
simple reference-aware Markov/TV baseline always matches or improves C6, both
algebraically and in this run. C6 adds zero releases across all 486 cells and
three fixed risk budgets. Exact-variance Cantelli is also much stronger in this
benchmark. These results support cheap certification as a mechanism, not a new
C6 implementation advantage, general efficiency breakthrough or financial benefit.

The user authorized this next gate after the [C6 stage](../../law_robust_disclosure/20261008/RESULTS.md).
The [protocol](../../../docs/DISCLOSURE_COMPUTE_GATE.md), [config](../../../configs/disclosure_compute_gate.json),
model family, seeds, comparators, thresholds and stopping rule were fixed before
benchmark outcomes. The reference-aware dominance argument was derived before
execution and was explicitly included as a potential falsifier. No post-outcome
parameter changes or model selection were made.

## What the new algebra check establishes

For the C6 setting, `d=f(10,H)-f(01,H)`, `b=E_R d`, `G=(d+b)/2`, `t=E_Q G`.
All are properties of a fixed bounded predictor and fixed SHAP reference R.
Knowing b gives the upper range endpoint `G <= (1+b)/2`. Standard Markov and
the TV event inequality therefore give, for `b>-1`,

\[
B(t,b,\epsilon)=\operatorname{clip}_{[0,1]}
\min\left\{1-t,\;1-\frac{2t}{1+b},\;
\epsilon+\frac{1-b}{1+b}\right\}.
\]

For every feasible b, this is no larger than the C6 bound
`min(1-t,(1-t+t*epsilon)/(1+t))`. When `epsilon<1-t`, the two b-dependent terms
intersect at `b*=(2t+epsilon)/(2-epsilon)`, where their common value is C6.
The minimum cannot exceed that value at any b. The generic term handles the
other regime. See the full proof and endpoint conventions in the frozen protocol.

Thus C6 can be recovered by eliminating a nuisance reference mean from ordinary
inequality constraints. Its earlier sharp population statement remains valid
given only `(t,epsilon)`. Retaining b improves the information set, and b is
already computed in this benchmark. This clarification narrows the candidate
contribution to a sharp information-limited envelope; it is not a new Markov
or TV inequality. The previous stage's numerical results are not rewritten.

## Executed synthetic design

- 54 fixed model instances: k in {8,12,16,20,28,40}, three polynomial families,
  and three fixed model seeds. Each instance has nine reference/completion laws,
  for **486 cells** and 1,458 repeated cell/budget combinations.
- A single hidden SHAP player H encodes k internal bits. They are not separate
  SHAP players. There are exactly two observed players; this does not extend the
  C6 theorem to many separately represented missing variables.
- `d(H)` is a bounded normalized integer polynomial. Families are linear,
  sparse degree-1-to-3 interactions, and a shared gate. The query prediction is
  identically 1/2. No predictor was fitted.
- R is a biased independent-bit law; `Q=(1-epsilon)R+epsilon U`, with U uniform.
  The TV upper bound is known by construction. No estimated completion model or
  estimated TV budget is used. Exact rational first/second moments are available
  through standard finite character-product identities.
- Full optimized enumeration ran for k<=20: **324 cells**, sharing
  **10,066,176 evaluated hidden states** across 36 model instances. The other
  **162 cells** at k=28,40 are prespecified scaling probes without full enumeration.
- Structure-aware branch-and-bound visited **165,665 nodes** in 162 model/q
  tasks. It fully resolved 137; the remaining 25 retain explicit unresolved
  probability intervals after the 5,000-node budget.
- Direct MC used **1,119,744 sampled completions**, producing 972 upper bounds
  at fixed 256/2048-draw budgets. Its total failure allowance 0.05 was divided by
  all 972 bounds, including scaling probes. No sequential stopping or selection
  of a sample budget after observing its result.

Model/law cells and repeated budgets are not independent people or datasets.
No population confidence interval for the descriptive coverage percentages is
claimed. Randomized model/method order and repeated timing calls control local
measurement artifacts; timing repetitions are not independent scientific evidence.

## Coverage on the 324 fully enumerated cells

Each entry is the number of cells with a reported upper bound at or below the
specified failure budget. Failure is a nonpositive **signed fixed-reference SHAP
pair gap**, including ties; it is not business error or prediction error.

| Method | 1% budget | 5% budget | 10% budget |
|---|---:|---:|---:|
| Generic mean/range | 12 | 72 | 132 |
| Previous ordinary TV/mean transfer | 28 | 84 | 177 |
| C6 | 28 | 96 | 180 |
| Reference-aware Markov/TV | 28 | 96 | 192 |
| Exact-variance Cantelli | 90 | 252 | 288 |
| Minimum of reference-aware and Cantelli | 90 | 252 | 288 |
| Budgeted branch-and-bound upper endpoint | 252 | 288 | 318 |
| MC 256, binomial confidence bound | 0 | 241 | 284 |
| MC 2048, binomial confidence bound | 218 | 288 | 298 |
| Exact enumerated risk | 252 | 288 | 324 |

The MC rows are finite-sample confidence statements under the known sampling
law; the analytic rows use exact population moments and conditional deterministic
bounds. Their guarantee types differ. No MC bound missed the enumerated risk
in this run (648 checked bounds), which does not prove zero failure probability.
The remaining 324 MC bounds have no enumerated-risk check.

At 5%, the three families each contain 108 enumerated cells. C6 certifies
48 linear, 24 sparse-polynomial and 24 shared-gate cells. Cantelli certifies
106, 86 and 60 respectively; MC-2048 and the enumerated oracle certify
108, 108 and 72. Strong baselines therefore survive across the frozen families.

Across all 486 cells, reference-aware Markov/TV is strictly tighter than C6 in
324 cells, equal in the 162 matched-law cells. It adds 18 releases at the 10%
budget and loses none. At 5%, large non-enumerated cells yield 48/162 C6 releases,
136/162 Cantelli releases, 116/162 branch upper-bound releases and 144/162
MC-2048 releases. These are conditional certificates, not independently checked
large-domain true risks.

## Measured cost and the gate

At k=20, each of nine model instances represents 1,048,576 hidden states and
nine law cells. The table gives median **total seconds per nine-cell model task**,
including the shared first-moment/bound bundle wherever it is needed.

| Information/computation bundle | Median time |
|---|---:|
| Exact means + all mean-only bounds | 0.217 ms |
| Above + exact variance/Cantelli | 0.355 ms |
| Above + MC 256 | 1.624 ms |
| Above + MC 2048 | 3.766 ms |
| Above + budgeted branch-and-bound | 9.303 ms |
| Above + optimized full enumeration | 42.789 ms |

The prespecified gate required at least 25% coverage at alpha=.05 and at least
10x median paired enumeration/mean cost ratio at the largest enumerated size.
Observed values are **29.63% and 193.95x: PASS**. The paired ratio is not the
ratio of the two marginal medians in the table. Mean/variance timings use the
median of five calls; MC, branch traversal and enumeration run once per task.
Generation/import costs are excluded for all methods. Raw repeats and method
order remain in local task artifacts.

This is a descriptive speed/coverage tradeoff on compact synthetic models.
The result is not a complexity lower bound: ordinary moments, direct sampling
and structure-aware inference all scale better than exhaustive enumeration.
Specialized weighted-sum dynamic programming for linear controls and broader
exact-inference solvers were NOT RUN; superiority over them is not established.
The much stronger Cantelli certificate costs only about 0.355 ms per model
task here, further weakening a practical argument for using C6 alone.

## Verification and reproducibility

**REPRODUCED / RUN this stage:** the algebraic dominance check, finite moment
and SHAP-identity tests, all 54 benchmark tasks, full repository tests, real
pause/resume, complete resume, scratch config-drift and payload-corruption guards,
and hash-verified public artifact export.

- Exact integer thresholds prevent floating tie classification errors.
  Enumerated first/second moments agree with rational formulas to maximum
  absolute error **3.3306690738754696e-16**.
- All deterministic upper bounds and branch intervals contain enumerated risks
  within the frozen 1e-10 numerical tolerance. No benchmark warnings.
- **15 new targeted tests passed. Full suite: 222 passed, 1 skipped, 3 warnings
  in 23.31 s.** MPS hardware was unavailable; the three warnings were existing
  SHAP/Matplotlib color-map pending deprecations. The benchmark used CPU only.
- Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3, tqdm 4.70.1;
  macOS 26.2 arm64; one numeric thread. Actual scientific interval:
  **10:14:57.894–10:15:24.243 UTC**, including an intentional pause.
  Summed model-task elapsed time: **1.445963 s**. The completing resume loop
  took 1.475842 s. These small-problem timings are not financial-model benchmarks.
- Actual pause: one task; completing resume: 53 new / 1 reused; final resume:
  **0 new / 54 reused**. Scratch config drift and scratch payload corruption
  were rejected; completed scientific artifacts were preserved.
- No independent agent rerun or external human review occurred in this stage.

Public evidence: [summary](summary.json), [coverage](coverage.csv),
[cost summary](cost_summary.csv), [all cells](cells.csv), [model costs](costs.csv),
[manifest](manifest.json), [execution receipt](execution_receipt.json),
[validation receipt](validation_receipt.json), [export hashes](artifact_export.json).
The [implementation](../../../src/financepaper/evaluation/disclosure_compute.py),
[runner](../../../scripts/run_disclosure_compute_gate.py),
[validator](../../../scripts/validate_disclosure_compute_gate.py) and
[summarizer](../../../scripts/summarize_disclosure_compute_gate.py) are additive.

Logs, per-model polynomials/results, repeated timings, task receipts and tqdm/ETA
progress remain local in `runs/decision_value_pilot/disclosure_compute_20261008T102000Z`.
Validation logs are in `runs/decision_value_pilot/disclosure_compute_validation_20261008T102500Z`.
Directory tags are identifiers; receipts contain actual execution times. Public
receipt paths/hashes identify local evidence; full logs are not published.

Fresh reproduction uses new ignored output directories:

```bash
.venv/bin/python scripts/run_disclosure_compute_gate.py --output runs/decision_value_pilot/compute_NEW --stop-after-tasks 1
.venv/bin/python scripts/run_disclosure_compute_gate.py --output runs/decision_value_pilot/compute_NEW --resume
.venv/bin/python scripts/validate_disclosure_compute_gate.py --run runs/decision_value_pilot/compute_NEW --output runs/decision_value_pilot/compute_validation_NEW
.venv/bin/python scripts/summarize_disclosure_compute_gate.py --run runs/decision_value_pilot/compute_NEW --validation runs/decision_value_pilot/compute_validation_NEW --output reports/disclosure_compute_gate/NEW
```

Resume rejects source/config/dependency drift. The summarizer refuses to overwrite
an existing report directory. All previous source/config/result files remain
unchanged; main was fetched for verification and was not merged or modified.

## Sources, prior evidence, and next action

Standard character expansions/moments are computational infrastructure; see
O'Donnell, [Analysis of Boolean Functions](https://www.cambridge.org/core/books/analysis-of-boolean-functions/B05A66E4DCC778E02B84C16376F4D1FD),
2014. The direct sampling comparator uses the one-sided Clopper-Pearson binomial
upper bound, corresponding to SciPy's
[exact binomial interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
The installed SciPy version is recorded above; the online documentation is not
the execution receipt. Neither comparator is claimed as a new method.

**REPORTED only:** C3/C5/C6 earlier scientific audits and their prior-collision
reviews, linked in the [previous consolidated report](../../law_robust_disclosure/20261008/RESULTS.md).
They were not rerun as new scientific experiments here; passing existing unit
tests is a separate software check. No new exhaustive novelty search was done.

**One next action, PROPOSED / NOT RUN:** audit the general-m C5 certificate
against a baseline retaining all reference moments that its computation already
produces. Resolve whether the nontrivial cardinality result supplies information
beyond elementary range/moment constraints before investing in a broader
application experiment. Do not select a model using inspected Taiwan/Polish data.

**NOT RUN / not established:** financial evaluation or training; human judgments
or decision benefit; estimated reference/completion laws or estimated TV budget;
separately represented multiple missing SHAP players; full enumeration at k=28,40;
specialized dynamic programming; external peer review; a new practical algorithm
or confirmed publication novelty. This stage provides an auditable synthetic
cost/coverage result and a negative incremental-utility finding.
