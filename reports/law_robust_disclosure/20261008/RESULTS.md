# From completion ambiguity to law-robust reason disclosure

2026-10-08. Branch: `research/law-robust-disclosure-20261008`.
Base: `dcb59d5a449d2de3550ce49d5b1e3caee3e74c71`.

## Decision and answer to the extended research question

The work is stronger as a coherent mathematical argument. The user explicitly
authorized a small extension of the research question to connect minor results.
We therefore derived and audited C6, a sharp completion-law mismatch extension
of C5. This stage does not establish publication priority or practical decision
value. Counting several minor results does not establish major novelty.

**Proposed unified question:** What signed reason-ranking reliability can be
certified from incomplete records when the completion law may differ from the
SHAP reference law?

| Component | Role in the same argument | Evidence in this stage |
|---|---|---|
| C3: prediction-invariant attribution ambiguity | Explains why stable prediction alone cannot justify stable reasons | REPORTED from the earlier [novelty report](../../completion_novelty/20261008/RESULTS.md); not rerun here |
| C5: sharp mean-to-rank-risk certificate under matched laws | Central mathematical result; reduces the problem to cardinality cells | REPORTED from the earlier [rank report](../../completion_rank/20261008/RESULTS.md); its scientific audit not rerun here |
| C6: sharp degradation under TV-bounded law mismatch | Repairs the law-equality restriction for two observed players, using the same signed-pair target | REPRODUCED/RUN here: proof, exact construction, independent coalition computation, full-table LP checks |
| C0: exact completion moments | Supporting computation and comparison machinery | Historical supporting result; not headline algorithmic novelty |

The shared object is a fixed-reference, signed interventional-SHAP pair claim
under hidden completion. C3 concerns coordinate ambiguity, while C5/C6 concern
pair ordering; the transition is explicit rather than treating these as the
same endpoint. C4 hardness is excluded as a headline contribution because of
its known generalized-Khintchine collision.

**Verdict:** proceed with a narrow theory-note synthesis. Major novelty,
submission readiness, business utility and human usefulness remain unestablished.

## C6 result and scope

The [frozen theorem and proof](../../../docs/LAW_ROBUST_DISCLOSURE.md) assume two
observed binary players at query `(1,1)`, one finite hidden SHAP player, fixed
reference `Bernoulli(p)^2 × R`, `0<p<1`, and a fixed model bounded in `[0,1]`
over the full domain including both supports. Completion uses hidden law `Q`.
All three variables remain SHAP players. Zero reference masses are allowed.

Define

\[
G=(\phi_1^R-\phi_2^R)/(1-p),\quad t=E_QG\in[0,1],\quad
z=Q(G\le0),\quad \operatorname{TV}(R,Q)\le\epsilon.
\]

TV is half the L1 distance. Ties count as failure to support a strict ranking.
Then the sharp population certificate is

\[
z\le \min\left(1-t,\frac{1-t+t\epsilon}{1+t}\right).
\]

The equivalent sharp frontier is
`t <= (1-z)/(1+max(z-epsilon,0))`. A two-state bounded construction attains it
while predictions on the completion slice remain constant at any `c in [0,1]`.
For strict reversal, the corresponding value is a supremum for `0<z<1`;
zero-reference-mass boundary cases require perturbing the law as well as the
model. Full-support restrictions can also turn attainment into a supremum.

At zero mismatch C6 recovers the two-observed-player case of C5. At unrestricted
mismatch it reduces to the generic bounded-margin certificate. The TV event
inequality is standard; candidate novelty is only the sharp specialized frontier.
The reference remains R: this is not a theorem about changing to target-reference
SHAP `phi^Q`, absolute importance, causal truth or financial correctness.

An exact mean and a valid TV upper bound are inputs, not outputs of the theorem.
No sample-size guarantee or procedure for estimating the TV budget is supplied.

### Same-information comparison

For `t=1/2` and `epsilon=1/10`, the audited bounds on rank failure are:

| Bound | Value | Interpretation |
|---|---:|---|
| C6 | `11/30 = 36.67%` | Sharp for the stated model class and information |
| Generic bounded margin | `1/2 = 50%` | Valid |
| Ordinary matched-law bound plus TV event/mean transfer, combined with generic bound | `1/2 = 50%` | Valid comparator using the same mean and TV budget |
| Blind reuse of matched-law C5 | `1/3 = 33.33%` | Invalid under mismatch; negative control |

This is an improvement in a mathematical upper bound, not a measured reduction
in actual errors. C6 strictly improves the specified valid transfer comparator
in 14 of 30 fixed mean/budget cells and ties in the others. This is not a contest
against every possible use of TV: a sufficiently structure-aware derivation
recovers C6 itself. Exact enumeration or variance information supplies additional
information and can be stronger; no universal superiority is claimed.

As an algebraic disclosure interpretation, a declared pair can meet tolerance
`alpha` when its normalized mean is at least
`(1-alpha)/(1+max(alpha-epsilon,0))`. For `alpha=0.1, epsilon=0.05`, this requires
`t >= 6/7`. This stringent threshold illustrates why practical usefulness still
needs testing. It is a direct corollary, not an additional novelty claim or an
executed deployment experiment.

## Actually executed: REPRODUCED / RUN

Protocol was fixed in the [config](../../../configs/law_robust_disclosure_audit.json)
and proof document before the run. The [runner](../../../scripts/run_law_robust_disclosure_audit.py)
constructs its SHAP pair operator independently from coalition/background
enumeration of full truth-table basis functions. The LP optimizes the entire
bounded truth table, with separate bad-state inequalities and constant query
predictions; it does not optimize the proposed formula.

| Check | Executed result |
|---|---|
| Two-state attaining construction | 270 full-table LP solves: 3 observed reference probabilities × 5 failure masses × 6 TV budgets × 3 constant predictions |
| Independent coalition vs rational witness gaps | Maximum absolute error `2.220446049250313e-16` |
| Attaining LP vs sharp frontier | Maximum absolute error `1.1102230246251563e-16` |
| Three-state laws | 1,350 LP solves: all 15 × 15 ordered denominator-four simplex laws × 6 nonempty proper declared bad subsets |
| Three-state frontier violations | Maximum numerical excess `5.551115123125783e-17`, below prespecified `1e-10` tolerance |
| Blind matched-law negative control | Violated in 225/270 constructed rows; all are mismatched-law rows |
| Zero reference mass | 135 constructed rows, included in coalition/LP checks |
| Baseline comparison | 30 fixed mean/TV cells |
| New unit tests | 15 passed |
| Full repository regression suite | **207 passed, 1 skipped, 3 warnings in 23.52 s** |

Total: **106 tasks, 1,650 result rows, 1,620 LP solves**. The three-state checks
constrain declared bad states, possibly of zero Q mass. Additional states may
also have nonpositive gaps; these checks establish an upper-bound audit, not
exact failure-set attainment for every fixed law pair. Constructive risk follows
the exact rational gap construction, rather than classifying floating-point
near-ties. Finite audits support the proof; they are not its
replacement.

The scientific run used CPU and one numeric thread, Python 3.11.16, NumPy 2.4.6,
SciPy 1.17.1, pandas 2.3.3, tqdm 4.70.1. Actual run interval was
09:47:57.643–09:48:08.118 UTC, including an intentional pause. Summed task work
was 1.045610 s; the completing resume loop took 1.100948 s. These are local
small-problem audit timings, not model runtime benchmarks. No audit warnings.
The full-suite skip was unavailable MPS hardware. Its three warnings were
existing SHAP/Matplotlib color-map pending deprecations.

The two non-originator AI reviewers checked the coordinator's C6 derivation and
framing. Neither independently reran this audit; they are not human peer review.
One boundary clarification was retained: strict reversal at zero reference mass
requires a law perturbation, not just decreasing an already minimal model value.

### Resume, integrity and artifacts

The real run stopped after task 1, resumed with 105 new / 1 reused tasks, then
completed a second resume with **0 new / 106 reused**. On a separate scratch
copy, a byte-level config change was rejected; after restoring that scratch
config, corrupting its task payload was also rejected. The successful original
run was not corrupted. The [validator](../../../scripts/validate_law_robust_disclosure.py)
executed those checks and the full test suite.

- [Audit summary](audit_summary.json), [execution receipt](execution_receipt.json),
  [manifest](manifest.json), [validation receipt](validation_receipt.json).
- [Attaining cases](witness.csv), [three-state LP checks](three_state.csv),
  [baseline cells](baseline.csv).
- [Export hashes](artifact_export.json) verify byte-identical publication copies.
  Source/config/dependency hashes are in the manifest and execution receipt.
- Local ignored run: `runs/decision_value_pilot/law_robust_20261008T094800Z`.
  Local validation: `runs/decision_value_pilot/law_validation_20261008T095500Z`.
  Directory timestamps are identifiers; actual times are in receipts.
- Per-task receipts, JSONL progress, tqdm/ETA and command/worker logs stay local.
  Public receipts refer to their paths/hashes; they are not bundled transcripts.

For a fresh reproduction choose a new ignored output path:

```bash
.venv/bin/python scripts/run_law_robust_disclosure_audit.py --output runs/decision_value_pilot/law_robust_NEW --stop-after-tasks 1
.venv/bin/python scripts/run_law_robust_disclosure_audit.py --output runs/decision_value_pilot/law_robust_NEW --resume
.venv/bin/python scripts/validate_law_robust_disclosure.py --run runs/decision_value_pilot/law_robust_NEW --output runs/decision_value_pilot/law_validation_NEW
```

Resume requires the same source/config/dependency hashes. To change the protocol,
use another config and new output directory. This stage only adds files; old
model-fitting code, configs, results, cohorts and historical reports are intact.

## Literature read and novelty boundary

The bounded search/review did not identify this exact frontier, which is not
proof of priority. Relevant prior work already covers distributional SHAP,
ranking uncertainty and selective explanation release.

- Cifuentes et al., ECAI 2024, arXiv v4: SHAP ranges under an uncertainty region
  of population distributions. This rules out a broad claim to invent robust
  SHAP under uncertain laws. [Primary source](https://arxiv.org/abs/2401.12731v4).
- Zhu, August 4, 2026, preprint v1: target-background interventional SHAP under
  covariate shift, posterior propagation, overlap and selective action risk.
  This is a close conceptual collision, although its target-background estimand
  differs from our fixed-reference completion-tail target. The manuscript was
  read through a [full-text mirror](https://www.researchgate.net/publication/411198930_Bayesian_Target-Domain_SHAP_under_Covariate_Shift_Posterior_Propagation_Overlap_and_Selective_Action_Risk).
  The [DOI/primary host](https://doi.org/10.21203/rs.3.rs-10566829/v1) could not be
  fetched in this session; venue/peer-review status is not verified. This is
  reported as a preprint, not a peer-reviewed result, and its experiments were
  not reproduced.
- Earlier collision review for Imprecise SHAP, confident feature ranking,
  statistical significance of rankings, and selective explanations remains in
  the [C5 report](../../completion_rank/20261008/RESULTS.md). Those prior findings
  are REPORTED here, not a new systematic literature review.

Retain only: **a candidate sharp bounded-model certificate for a fixed-reference
signed SHAP pair under TV-bounded completion-law mismatch**. Avoid claims of the
first robust SHAP method, the first explanation risk certificate, a novel TV
inequality, or a new general attribution algorithm.

## Proposed paper structure and the next substantive gate

PROPOSED structure: (1) formal completion/disclosure target; (2) C3 ambiguity
construction; (3) C5 matched-law frontier and cardinality reduction; (4) C6 law
mismatch extension; (5) exact/LP falsification and baseline comparisons; (6)
limitations and practical evaluation. C0 computation belongs in methods/support.
The wider C5 theorem and narrower two-observed-player C6 proposition must retain
their different scopes rather than being advertised as one general theorem.

**Next priority:** establish a useful information/computation regime in which
the mean-based certificate is cheaper to obtain than exact completion-tail
enumeration, yet certifies a nontrivial number of pairs under a justified TV
budget. Fix the setting and comparators before inspecting results. Include exact
enumeration, generic bounds, ordinary TV transfer and variance-aware bounds with
their information costs explicitly accounted for. Multiple missing SHAP players
are a key scope challenge, not a proved extension. If no such regime survives,
retain a narrow mathematical note and drop the practical decision-value claim.

## NOT RUN / not established

- New Taiwan/Polish evaluation, predictor fitting or model selection: **NOT RUN**.
- Human judgments, user study, business/financial returns or causal validation:
  **NOT RUN**.
- Finite-sample estimation of the mean or TV budget: **NOT RUN**.
- General C6 theorem for more than two observed players, multiple separately
  represented missing players, dependent observed reference variables, or
  conditional SHAP: **NOT ESTABLISHED**.
- New practical cost/coverage benchmark or demonstrated human decision benefit:
  **NOT RUN**.
- C3/C5 scientific experiment reruns in this stage: **NOT RUN**; their earlier
  reported results remain separate from this stage's new C6 execution.
- Exhaustive priority search, external peer review, submission readiness:
  **NOT ESTABLISHED**. Main was not merged or modified.
