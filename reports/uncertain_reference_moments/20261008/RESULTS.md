# Reference-moment information survives nonzero uncertainty, with a sampling caveat

2026-10-08. Branch `research/uncertain-reference-moments-20261008`.
Base `43e7c76bc5b3e8d694cdb843299a20adca959cb4`.

## Result and decision

**The vector advantage survives nonzero moment uncertainty in the configured
synthetic audit.** At a 5% rank-failure budget and deterministic moment radius
.05, the full-vector bound certifies 52/231 profiles; C5 certifies 25/231 and
the range-aware scalar Markov bound 33/231. These methods receive the same
coordinate lower bounds. No model was fitted or selected.

The sampling comparison also favors the vector method at that budget, even
against C5 with a direct scalar confidence bound that avoids the coordinate
union penalty. **However, every sampling case certified at 5% has true risk
zero and a deterministic contrast vector under this stage's attaining-law
generator.** The finite-sample gain therefore does not demonstrate performance
on nondegenerate stochastic interior cases. Repeated deterministic cases do
not provide 32 independent pieces of scientific confirmation.

The supported contribution candidate is an **information-loss characterization**:
compressing the cardinality moment vector to its weighted mean can prevent rank
certification, and this separation can persist over a nonzero uncertainty box.
This is a monotonic extension of the previous bound, not a new generic probability
inequality. Publication novelty and practical extraction value remain unestablished.

## What was derived and implemented

For C5's coefficients alpha>beta>0, increasing any reference moment b increases
the gap offset and decreases the nonnegative loss mean. Thus the feasible
bad-event mass set shrinks coordinatewise. The sharp worst-case bound over
`l <= b <= u` is exactly **R(l)**. The old attaining construction at l proves
sharpness; upper endpoints alone cannot improve it. The independent LP below
optimizes over unknown b, rather than silently fixing b at the claimed corner.

All prior assumptions remain: a bounded model, iid Bernoulli observed reference
variables, one finite hidden SHAP player, matched reference/completion laws,
signed pair gap, and ties counted as failure. The adversarial model and hidden
law can vary. This does not prove fixed-law sharpness or strict-reversal sharpness.

See the frozen [protocol](../../../docs/UNCERTAIN_REFERENCE_MOMENTS.md),
[implementation](../../../src/financepaper/evaluation/uncertain_reference_moments.py),
and [config](../../../configs/uncertain_reference_moments.json).

### A concrete robust separation

The already configured m=3, p=.5, center b=(0,1) example has uncertainty radius
epsilon and lower corner `(-epsilon,1-epsilon)`. For 0<=epsilon<=.1, substitution
in the previous piecewise-linear formula gives

    R = 5*epsilon/(8-6*epsilon).

At epsilon=.05, the exact vector bound is **5/154 = 3.2468%**. The scalar
range-aware bound is **13/97 = 13.4021%**; C5 is **18.8819%**. The lower corner
is strictly interior and has an exactly verified attaining law with positive
failure mass. Hence exact knowledge that a moment equals 1 is not necessary
for this deterministic-box separation.

The algebraic 5% radius threshold is epsilon=4/53, approximately .07547;
direct rational evaluation returned exactly 1/20 there. This closed-form
interpretation and threshold check were made **after** the audit, not used
to choose its radius grid or select a new model. They are not an additional
independent experiment.

## Actual execution: RUN / REPRODUCED

V2 completed **1,716 tasks**: 1,617 uncertainty boxes plus 99 sampling cells,
each with 32 replicates, producing 3,168 sampling rows.

| Check actually executed | Result |
|---|---:|
| Exact lower-corner witnesses and moment/event identities | 1,617 boxes |
| Independent fixed-event-mass LP solves over b and s | 2,985 |
| Largest LP residual discrepancy against rational expression | 7.633e-17 |
| Smallest positive residual at an above-bound probe | .0005578304 |
| Strict vector improvement over C5 across all radii | 1,203/1,617 |
| Simultaneous vector lower-bound coverage failures | 0/3,168 |
| Direct scalar lower-bound coverage failures | 0/3,168 |
| Observed risk underbounds, each of six methods | 0/3,168 |
| Repository tests in this turn | 272 passed, 1 skipped, 3 warnings |

LP residual acceptance remained **1e-9**. Displayed strict comparisons use a
1e-12 margin; release decisions use rational comparisons and the exact forward
C5 frontier. Zero observed underbounds is not itself proof of a 95% confidence
statement; validity follows from the stated iid/boundedness assumptions and
the concentration argument, conditional on valid lower bounds.

### Deterministic moment uncertainty

Each radius has the same 231 previously inspected synthetic profiles, including
duplicates. Counts are configuration counts, not prevalence or independent datasets.

| Radius | Full vector | C5 | Range-aware Markov | Generic range |
|---:|---:|---:|---:|---:|
| 0 | 74 | 33 | 35 | 33 |
| .001 | 74 | 33 | 34 | 25 |
| .005 | 74 | 33 | 34 | 25 |
| .01 | 71 | 33 | 34 | 25 |
| .02 | 68 | 33 | 33 | 24 |
| .05 | 52 | 25 | 33 | 24 |
| .10 | 0 | 0 | 0 | 0 |

All counts above use risk budget 5% and denominator 231. All nonzero-radius
lower corners lie strictly inside (-1,1). Nevertheless, the 111 profiles whose
**original centers** were strictly interior still yield zero 5% certifications
at every radius. The positive result is local robustness around the favorable
old profiles; it is not broad coverage over an interior population.

### Sampling with the same observations

For each of 33 configured laws, N iid hidden-state observations are represented
by sufficient binomial counts. We did not materialize 36,900,864 vectors:
there were 1,728 binomial-count replicates and 1,440 deterministic replicates,
representing that many observations in total. Sampling law probabilities use
NumPy floating arithmetic; all ensuing sample moments are rational.

Vector methods use simultaneous one-sided coordinate lower bounds. Direct C5
uses the same observed contrasts but only estimates their weighted scalar mean.
Each method separately uses confidence 95%; no minimum of separate certificates
is advertised as jointly 95% valid. Risk budget 5% and confidence error 5% are
different quantities; neither gives a joint 5% failure guarantee here.

| N | Full vector | Vector-derived C5 | Vector-derived Markov | Direct C5 | Direct generic |
|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 0 | 0 | 0 | 0 |
| 2,048 | 192 | 96 | 96 | 96 | 0 |
| 32,768 | 384 | 96 | 96 | 96 | 96 |

All entries are certifications at 5% out of **1,056 replicates per N**. Because
every certified vector case is deterministic, the counts correspond to 6/33
configured laws at N=2,048 and 12/33 at N=32,768, versus 3/33 for direct C5.
This is not a paired significance test or a population effect estimate.

The scalar control does win elsewhere: at N=2,048 and risk budget 10%, it alone
certifies three replicates. Across all N, its numerical bound is strictly
tighter in 1,027/3,168 replicates; the vector bound is tighter in 1,716. The
vector is not uniformly superior once estimation uncertainty differs.

All prespecified budgets (1%, 5%, 10%), interior subsets and pair counts are in
[coverage.csv](coverage.csv), [paired_sample.csv](paired_sample.csv), and
[sample_cells.csv](sample_cells.csv). Individual sample rows remain local in
the hashed run artifacts; [box.csv](box.csv) contains the deterministic audit.

## Numerical failure retained, then repaired

Original run `runs/decision_value_pilot/uncertain_moments_20261008T121900Z`
completed 490 tasks, then stalled on the degenerate m=4, p=.1, b=(1,1,1),
radius=0 LP at z=0. It was interrupted with exit 130; **1,226 tasks were NOT
RUN in that attempt**. A post-task timing guard could not interrupt the solver.

A capped diagnostic reproduced IPM's iteration-limit failure without presolve.
IPM and dual simplex with presolve returned the exact expected residuals.
V2 enables presolve and enforces solver time/iteration limits, preserving the
scientific config, objective scaling, and 1e-9 tolerance. It passed all checks.
This is a disclosed numerical repair, not an untouched confirmation.

The original source and partial artifacts remain intact. All 490 shared rows
have identical scientific fields in V2; only LP numerical residuals may differ.
See the [amendment](../../../docs/UNCERTAIN_REFERENCE_NUMERICS_V2.md),
[failure diagnostic](first_run_failure.json), [partial rows](first_run_partial.csv)
and [first manifest](first_run_manifest.json).

## Runtime and reproduction

CPU, one numeric thread, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, pandas
2.3.3, tqdm 4.70.1, macOS arm64. V2 ran during **12:21:30.777–12:21:59.459 UTC**,
including the intentional pause. Sum of task times: **15.8711 seconds**;
completing resume loop: **16.7514 seconds**. These are audit timings, not a
real-predictor moment-extraction or deployment benchmark.

Pause completed one task; completing resume did 1,715 new/1 reused; completed
resume did 0 new/1,716 reused. Scratch config drift and payload corruption were
rejected. Audit warnings: none. Tests took 24.00 seconds; unavailable MPS was
skipped, with three existing SHAP/Matplotlib pending-deprecation warnings.
No independent sub-agent review or human review occurred.

```bash
.venv/bin/python scripts/run_uncertain_reference_moments_v2.py --output runs/decision_value_pilot/uncertain_NEW --stop-after-tasks 1
.venv/bin/python scripts/run_uncertain_reference_moments_v2.py --output runs/decision_value_pilot/uncertain_NEW --resume
.venv/bin/python scripts/validate_uncertain_reference_moments.py --run runs/decision_value_pilot/uncertain_NEW --output runs/decision_value_pilot/uncertain_validation_NEW
```

Unique directory names are identifiers; receipts record actual timestamps.
Source/config/dependency hashes block mixed-version resume. Detailed logs and
source snapshots remain ignored locally. Public [execution](execution_receipt.json),
[validation](validation_receipt.json), [summary](summary.json),
[progress](progress_milestones.json), and [export hashes](artifact_export.json)
record provenance. All pre-existing tracked files are unchanged. Latest main
was fetched and was already an ancestor; main was not merged or modified.

## Evidence boundaries and one next action

**REPORTED only:** previous C3/C5/C6 and full SHAP-definition audits. Their
complete historical grids were not rerun. This turn independently checked the
new interval LP and executed the sampling protocol.

**Prior source checked:** Hoeffding (1963), Theorem 2, equation (2.6), page 16,
supports the bounded-independent-sample concentration step. Its application
coordinatewise with a union bound yields the simultaneous radius used here.
The primary scanned pages 15–16 were read after local rendering; this ingredient
is established prior art, not our contribution.
[Primary paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf),
[publisher record](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830).

**PROPOSED, one next action:** freeze a sampling audit of strictly interior,
nondegenerate laws with positive true rank-failure mass, retaining direct-mean
and vector controls at the same confidence level. This targets the specific
remaining limitation rather than increasing the number of deterministic repeats.

**NOT RUN / not established:** that nondegenerate sampling audit; new full
truth-table reconstruction; actual predictor moment extraction/cost; financial
training or real-data evaluation; human decisions, economic utility or clinical
claims; external peer review; novelty priority or publication readiness.
