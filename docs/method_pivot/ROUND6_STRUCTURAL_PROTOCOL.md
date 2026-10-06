# Round 6 — frozen XGB25 structural feasibility audit

Preregistered 2026-10-06, starting HEAD `5b077e1c7d0f2a5d55061ae32966017b72c69612`,
branch `research/method-pivot`. This executes the one next action in ROUND5_NEXT_DECISION.
No new method, predictor, probability solver, target, threshold or completion law.

## Question and comparator

Does the frozen attribution-rank query expose useful structure after ordinary
conditioning, identical-rectangle merging and contrast cancellation? Separate
**generic engineering headroom** from **an advantage unique to a new method**.
No latter claim follows from smaller expressions alone.

Use the existing compiled form `phi(z)=sum_r c_r I_r(z)`, fixed 64-row background,
raw margins and original-feature aggregation. For each current record, map only
initially observed financial fields into the eight frozen Taiwan groups. Select
the frozen positive top-2 (.01; rank/sign tolerance 1e-6). Competitors include
every initially available group, including non-displayed groups. Ineligible cases
are reported, never replaced by more favorable customers.

Compare:

1. Full compiled encoded attribution evaluation, then original/group aggregation.
2. Generic **specialize then project**: clamp fixed completion inputs, merge
   identical remaining rectangles, aggregate groups, form candidate sign and
   competitor-minus-candidate channels, remove exact zero coefficient rows.
3. Generic **project then specialize**: the same channels formed before clamping
   and merging. This is an alternate order of standard linear operations, **not
   a proposed new algorithm**. Verify agreement before interpreting sizes/cost.
4. Stock TreeSHAP/cache values as numeric reference, not the sole speed comparator.

All scalar channels have the same frozen event as round five. Compile costs and
per-query preparation are charged. No WMI package or certified bound implementation
will be called a benchmark when it has not been run. A fixed K8 estimate and an
exact continuous-law certificate do not have equal error contracts.

## Data boundary frozen before structural measurements

Use **fold 0 only**, the existing `revision_calibration` current-only caches under
`outputs/decisive_validation/taiwan/fold_0/`. This is development reuse, not
independent confirmation. Model and partition hashes are locked in the config.
600 IDs in this role; choose the first 32 of a default_rng(20261006) permutation
of sorted IDs, identical customers in MCAR10, MCAR30, MAR30 and group_missing.
Use the first eight existing completions; no new donor selection or data download.

Validate exact partition membership, training-only donor IDs, natural/artificial
masks, erased hidden placeholders, preserved observed cells and cache receipts.
Exclude all outer/test/reserve records. Do not open `verification_only.npz`,
restored outputs or outcome files. The legacy predictor bundle contains training/
early-stop/calibration objects; extract only XGB25, preprocessing, background and
training IDs, and never use its outcome arrays. Metadata/schema inspection before
this freeze established 200 trees, 134 encoded inputs and a (64,134) background;
no structure or timing aggregate was used to set this protocol.

### Encoding constraint

The 134 inputs include one-hot/value columns plus 23 masks and 18 deltas. On a
complete donor completion, all mask columns are 1 and delta columns are 1/5,
including those associated with originally hidden fields. **Fix those completion
metadata values**, not the current incomplete metadata. Fix all value columns
whose original fields were initially observed. Only encoded values of hidden
fields remain variable. Never sample one-hot columns, masks or deltas independently.
Use only the original eight legal donor completions for numeric comparisons.

Structural graphs collapse encoded coordinates to original financial fields.
They describe co-occurrence **within additive indicator terms**, not conditional
independence of the completed rank event. Thresholding a sum can couple otherwise
disjoint terms; a small term-support graph does not prove tractable event integration.
The empirical donor law is joint, not a product of encoded marginals.

## Endpoints and falsifiers

Record per customer/condition:

- Global terms, surviving terms, merged terms, exact-zero terms, coefficient
  nonzeros; full-group versus sign/contrast union size and channels.
- Scalar contrast variable supports and zero/constant/duplicate channels.
- Term-support components, largest component, deterministic min-fill induced
  width **upper bound** (not exact treewidth or an event complexity theorem).
- Number of distinct indicator signatures among the same K8 atoms; identical
  signatures can be merged by any generic finite-law implementation.
- Numeric discrepancies from full compiled attributions and cached TreeSHAP;
  frozen event agreement, with no truth/restoration involved.
- Compilation/preparation/evaluation time, coefficient-array footprint and peak
  process RSS. Five timing repetitions on the first eight selected customers per
  condition (ineligible rows retained for accounting but not event timing).
  Runtime excludes file I/O/model loading and is not end-to-end deployment latency.

Numeric guard: compiled versus cached original-group attributions within
`atol=5e-6, rtol=1e-6`; alternative linear-expression paths within `atol=1e-10,
rtol=1e-10`. Frozen revision events must agree exactly. Stop and document any
mismatch; do not change tolerances/target to manufacture agreement.

Predeclared **engineering headroom screen**, primary MCAR30: at least 16 eligible
customers and median contrast/full-group rectangle-union ratio <=.75. This is a
budget/practicality screen, not a significance threshold or novelty evidence.
Report all other conditions, whether favorable or not. Runtime is secondary and
must include preparation/break-even, not just a precomputed matrix multiply.
If generic operations explain every reduction, no new-method claim passes,
even if this engineering screen passes. No financial matrix or three-seed method
experiment is authorized without a separate new-mechanism gate.

## Bounded execution and provenance

CPU, one thread, no GPU/rented service. Total runner wall budget 600 seconds,
4 GiB peak RSS ceiling, 100,000 compiled terms, one small preflight record before
the full 128 current episodes. Time/size violations are reported as budget stops,
not silently handled by smaller backgrounds/ensembles. No full covariance matrix.
Timing repeats reuse an unchanged tree; seeds only choose rows, not new fits.
Source/config/protocol/artifact SHA-256, starting/execute commit, dirty status,
versions, row IDs and run logs go to ignored `outputs/method_pivot/round6/`.
Public reports contain aggregates only. Synthetic tests cover cancellation,
clamping, encoding guards, sparse/constant cases, event ties and baseline equality.

Decision: implement the **audit/control harness only**. Keep a scoped NO-GO for
the generic method claim unless a distinct necessary mechanism is subsequently
derived and preregistered; this audit itself cannot establish one by renaming
known simplification operations.
