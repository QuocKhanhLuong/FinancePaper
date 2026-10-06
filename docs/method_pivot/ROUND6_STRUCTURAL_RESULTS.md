# Round 6 — structural audit of frozen XGB25 contrasts

Executed **2026-10-06**, `research/method-pivot`. Starting checkout
`5b077e1c7d0f2a5d55061ae32966017b72c69612`; protocol committed at `ad155bf`;
corrected harness committed and run with a clean working tree at
`3788189193dff0a1e73d40de743d6b413136b5d1`.

**Result: ordinary conditioning is useful, but this contrast construction adds
no rectangle-union reduction. Do not implement a new solver on this evidence.**
The original union-ratio gate also has a structural limitation: the retained
sign/contrast map is injective, making its proposed union contraction impossible
in exact arithmetic for this representation. This is a correction to the audit's
interpretation, not a changed threshold or evidence that every verifier fails.

## Execution boundary and incident

The [preregistered protocol](ROUND6_STRUCTURAL_PROTOCOL.md) uses 32 fixed
development customers, the same customers in four conditions, and their first
eight cached donor completions. There are 128 current episodes, **not 128
independent customers**. No predictor retraining, new donor draw, hidden truth,
verification file, outcome metric, outer/test cohort or dataset download.
The historical bundle contains training/calibration outcomes on deserialization;
the audit extracts only the model, fitted preprocessor, background and train IDs
and does not use those outcomes. This is development reuse, not confirmation.

The first preflight stopped on an incorrect new-audit assumption about deltas.
The [incident and amendment](ROUND6_PREFLIGHT_INCIDENT.md) were committed before
rerun. April's complete-state delta is zero; later months are one fifth, under
the frozen encoding. Failed artifacts were retained. No historical preprocessing,
predictor, explainer, event or numerical tolerance changed.

The fresh compiler has 200 trees, 134 encoded inputs, a fixed 64-row background,
and 3,013 rectangle terms. Only initially observed original fields contribute
to the eight frozen semantic reason groups. Metadata columns are fixed to
complete-state values for completions, but their attribution is not relabeled as
an original financial reason. Group scores need not reconstruct the whole risk
logit because these metadata contributions are excluded by the frozen convention.

## Measured structure

All term summaries below are medians over eligible current episodes in each
condition. Ineligible records remain in the denominator and are not replaced.

| Condition | Eligible / selected | Group terms before conditioning | Group terms after conditioning | Contrast union terms | Contrast/group union ratio | Distinct K8 signatures |
|---|---:|---:|---:|---:|---:|---:|
| MCAR10 | 26 / 32 | 2,923.5 | 73 | 73 | 1.00 | 5 |
| MCAR30, primary | 24 / 32 | 2,885 | 271 | 271 | 1.00 | 7 |
| MAR30 | 24 / 32 | 2,886 | 273 | 273 | 1.00 | 6 |
| Group missing | 24 / 32 | 2,868 | 201 | 201 | 1.00 | 6 |

The ratio is exactly 1 in **all 98 eligible episodes**, not just at the median.
Primary eligibility passes 16; the <=.75 contraction screen does not pass.
Conditioning/identical-term merging reduces the primary group union to a median
paired fraction .0939 of its pre-conditioning size. That gain is already present
in the generic full-group control.

| Condition | Group coefficient nonzeros | Contrast coefficient nonzeros | Contrast channels | Hidden variables in term graph | Min-fill width upper bound |
|---|---:|---:|---:|---:|---:|
| MCAR10 | 132.5 | 489 | 16 | 3 | 1 |
| MCAR30 | 407 | 1,513 | 12 | 8 | 4.5 |
| MAR30 | 423 | 1,565 | 14 | 8 | 4 |
| Group missing | 297 | 1,381.5 | 14 | 6 | 4 |

These are medians, hence noninteger widths/counts are possible. Contrasts can
cancel within individual scalar channels while increasing the total coefficient
count. The reported graph connects fields that co-occur within one additive
rectangle term; it is **not** the rank-event primal graph. Thresholding a sum can
couple disconnected term components. No tractability or independence theorem
follows. Signature reuse is ordinary finite-support deduplication, not a new
algorithm, and no speedup from implementing that deduplication is measured here.

## Why the union gate could not succeed

For a fixed, already merged rectangle r, let c_rh be its coefficient for each
available group h. The contrast vector retains a candidate sign c_rg and every
competitor difference d_r,hg = c_rh - c_rg. Therefore

```
c_rh = d_r,hg + c_rg, for h != g;
c_rg is retained directly.
```

Unavailable groups have zero coefficients under observed-only aggregation. Thus
the projection is injective on the available coefficient subspace:

```
all retained sign/contrast coefficients are zero
    iff all available group coefficients are zero.
```

So it preserves the nonzero rectangle union in this fixed representation. The
same support union also preserves its term-support graph. This elementary
linear-algebra observation was recognized **after execution**; three added
regressions check the reconstruction identity. It is not a new theorem, a
preregistered discovery, or a statistical falsification of all possible event
algorithms. Retaining only contrasts could remove a common component, but the
frozen sign-revision event requires absolute candidate contributions; silently
dropping them would change the task. Event-specific pruning is a different,
unimplemented mechanism, also subject to existing verifier baselines.

Exact-zero coefficient counts differ slightly between operation orders in
13 episodes because floating-point addition is not associative; the largest
value discrepancy is 2.665e-15. No tolerance-based coefficient pruning was
introduced. Union sizes and every tested event agree.

## Numeric and runtime checks

- Fresh compiled/grouped values vs cached TreeSHAP reference: maximum absolute
  discrepancy **1.4892e-8**, across all 128 × 8 completed inputs.
- Two operation orders: maximum discrepancy **2.6646e-15**.
- All **784 eligible completion events** match the frozen evaluator exactly.
  This is numerical agreement with a frozen event, not verified explanation truth.
- Compilation: **62.99 ms**. Audit elapsed: **4.84 s** inside the runner,
  including its artifact handling; peak process RSS **390,152,192 bytes**.
  Coefficient array: **3,229,936 bytes**. Actual machine: Apple M4 Pro, 24 GiB,
  macOS 26.2, Python 3.11.16, NumPy 2.4.6, CPU one numerical thread.

Evaluation milliseconds per customer **batch of eight completions**:

| Condition | Timed eligible customers | Full encoded compiler | Specialized full groups | Project after specialization | Project before specialization |
|---|---:|---:|---:|---:|---:|
| MCAR10 | 8 | 7.428 ± .100 | .124 ± .007 | .124 ± .007 | .124 ± .004 |
| MCAR30 | 8 | 7.586 ± .073 | .382 ± .010 | .377 ± .006 | .374 ± .007 |
| MAR30 | 7 | 7.502 ± .072 | .378 ± .005 | .377 ± .004 | .377 ± .006 |
| Group missing | 8 | 7.428 ± .031 | .382 ± .006 | .387 ± .015 | .396 ± .006 |

Mean and sample SD of **five repetition-level cohort means**, with method order
rotated. First eight selected customers are used; the one ineligible MAR30 case
is not replaced. These are repeated numerical timings, not five financial seeds.
Raw timings include the matrix projection needed at evaluation, but exclude
preparation, compilation, data I/O and completion generation.

Primary MCAR30 preparation was measured once per timed customer (no repeated
preparation uncertainty estimate). Charge its cohort mean explicitly:

| Path | Query preparation, ms | Mean K8 evaluation, ms | Preparation + one K8 evaluation, ms |
|---|---:|---:|---:|
| Full compiled encoded attributions | 0 | 7.586 | 7.586 |
| Generic specialized full groups | 2.003 | .382 | 2.385 |
| Project after specialization | 2.039 | .377 | 2.416 |
| Project before specialization | 2.108 | .374 | 2.482 |

The common one-time compilation is charged separately above for **every** path;
it must be amortized for repeated queries. Generic specialization recoups its
preparation within one K8 query in this timed cohort, but contrasts have no
one-query advantage over that control. Using observed mean differences alone,
their extra preparation amortizes after roughly 7/14 repeated evaluations of
the same prepared query; those tiny differences are comparable to timing noise
and are **not a reliable speed advantage**. No stock TreeSHAP latency was rerun,
no WMI solver benchmark was run, and these figures are not end-to-end deployment
latencies or equal-error continuous-law integration results.

## Reproduction and artifact provenance

```
uv sync --frozen --extra temporal
uv run --frozen --extra temporal pytest -q tests/test_contrast_structure.py
uv run --frozen --extra temporal python scripts/run_contrast_structure_audit.py \
  --config configs/contrast_structure_audit.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal pytest -q
```

The runner requires the exact local historical model/current caches named and
hashed in the config; it never downloads or synthesizes replacements. It refuses
to overwrite an existing result directory. Public reports are aggregates; raw
customer IDs/completions, weights and output caches remain ignored. Fresh clones
without these artifacts cannot claim to have reproduced the development audit.

Canonical successful directory: `outputs/method_pivot/round6/structural_amended/`.
Source/config/protocol/amendment hashes, execution commit and clean-tree receipt
are in `manifest.json`. Raw parent log: `outputs/method_pivot/round6/runner_amended.log`.

| Artifact | SHA-256 |
|---|---|
| results.json | `07423be29749c23e11f0644b6cedd9ed0ee2b1c85e1a5f872ba2d1c1e8b90382` |
| manifest.json | `31bc86fb8c082fb8ce6937b44f81031d1cd5542b38726404cb2ff75d9b9c1a6b` |
| rows.jsonl | `c55de34c3267d6647d3f49429c4d082ac231dd75d1234b16e42517d85f92f500` |
| timing.csv | `026e575bcf0114eeb82561ab8f79490160a86572874fd763f5a9fe991c4648a2` |

Final combined suite: **218 passed, 0 failed, 0 skipped**, three upstream SHAP
Matplotlib deprecation warnings, **11.30 s**. The new audit file contributes 17
tests, including three post-run injectivity checks. Final raw receipts:
`outputs/method_pivot/round6/pytest_final.log` and `pytest_final.xml`.
The earlier unrestricted-thread run stopped progressing after 204 progress dots;
root sampled the owned process and observed a PyTorch CPU `sqrt`/`libomp` barrier
wait, then terminated that process and reran with the numerical thread limits
shown above. This evidence suggests a runtime threading interaction; it is not
a proven diagnosis of one library's bug. No tests were removed or skipped and
no scientific output was rerun to obtain better results. The interrupted log,
native sample and `pytest_initial_stop.json` remain in the same ignored folder.
Engineering passes are not evidence of a new method's scientific benefit.

**VERIFIED:** the amended run, artifact hashes, identity/event checks and machine
above. **REPORTED:** prior financial performance and prediction–explanation
decoupling in historical reports, not rerun here. **NOT RUN:** a new solver,
continuous-law certificate, WMI package comparison, predictive training, TabM,
Freddie, external confirmation. **ASSUMED:** the frozen donor support remains a
useful completion model; this audit neither tests its truth coverage nor fixes
completion misspecification. See [the decision](ROUND6_NEXT_DECISION.md).
