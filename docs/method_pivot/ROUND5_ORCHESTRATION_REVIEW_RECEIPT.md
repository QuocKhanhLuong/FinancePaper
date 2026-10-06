# Round 5 — actual AGY worker and coordinator review

Date: **2026-10-06**. The user authorized AGY research assistance; root retained
responsibility for guidance, source checks, mathematical review and execution.
This was a real Orca/AGY terminal, not a collaboration subagent labeled AGY.

## Task and outcome

- Orca Run: `run_011cf78920a8`.
- Task: `task_658b94033735`; Dispatch: `ctx_afc9c30762fe`.
- Dedicated operator-created AGY terminal: `term_8a0388cd-9bca-4f65-b599-c08c3a01fb2f`.
- Worker scope: independent primary-source/reduction review only; write under
  ignored `outputs/method_pivot/round5/`. No tracked edits, training, data access,
  commit/push or subworkers.
- Accepted lifecycle settlement: `worker_done`, outcome `succeeded`, message
  `msg_a9dc2c1148d5`, **2026-10-06 09:30:50 UTC**, matching both expected IDs.
- Scientific acceptance: **partial, with material coordinator corrections below**.
  A worker's completion status is not scientific endorsement.

The worker produced `agy_reduction_review.md` (SHA-256
`eb99a03623ecd90ed9fe2bb915d132feb59699eb349f39c8746145ca44de68ba`)
and `source_notes.md` (`cccc7988727d11eb639dd6ca74cca841b948fdd949f7e271fbd967285f83675a`).
They remain raw ignored review notes, **not authoritative reports**. Root's
[source audit](ROUND5_RANK_INFERENCE_AUDIT.md) records its own inspection levels;
worker assertions of access do not promote metadata inspection to full-text reading.

## Accepted after independent review

1. Fixed-input explainer sampling error is different from variability under a
   supplied missing-input completion law.
2. The compiled finite rank-event query can be encoded as weighted counting/
   integration. This establishes representability, not hardness or speed.
3. Linear contrast projection and generic partition-based probability bounds do
   not, on their own, supply a new algorithmic mechanism.

## Rejected/corrected in the public decision

| Worker claim/problem | Coordinator correction |
|---|---|
| Reducing our query **to** general WMI proves our query #P-hard | Wrong reduction direction. No reverse reduction was supplied; no hardness theorem for our specific query is claimed |
| Any outsider overtaking any initial reason is the frozen top-k event | Not equivalent at ties/tolerance. Keep the actual observed-only, positive-eligible candidates, cardinality `>=k`, and fixed sign/rank tolerances |
| Auxiliary attribution reals constrained by equalities may be integrated as ordinary additional coordinates | Eliminate deterministic definitions or specify input-only integration; otherwise equalities have zero Lebesgue mass. Free auxiliary Boolean normalization can also overcount |
| Feasibility exists *only* for <=2 trees / <=3 missing variables | Arbitrary restrictions, not established theory or measurements; removed |
| Specific bound ranges and MC8 <10 ms used as evidence | Unmeasured numerical illustrations, not evidence; removed |
| A 1000×1000 grid is an exact oracle for arbitrary tree cuts | False without alignment/error proof; use exact cell integration or a controlled quadrature error |
| Fixed K8 estimates versus deterministic certificates are automatically matched computational comparators | Match output accuracy/confidence requirements and charge compilation; the tasks have different error contracts |
| All completion-law variation is intrinsic/irreducible aleatoric uncertainty | Completion-model estimation and misspecification also matter. Distinguish supplied-law variation from truth and from numerical sampling error |
| Absolute claims about every verifier and automatic large-tree intractability | Not supported by the inspected sources; no empirical failure/scalability claim adopted |

A new regression test demonstrates the event discrepancy: current candidates
are groups 1/2; restored contributions `(1,1,2)` leave only one group strictly
above either candidate, so the frozen k=2 event is false despite an outsider
overtaking both. This preserves historical tolerant membership; it does not
silently redefine a tie as revision.

The correction message raced with worker settlement and was rejected with
`dispatch_inactive`; root did **not** claim the worker read or applied it. Root
made the corrections in the public audit/results/decision, leaving raw worker
notes intact for accountability. The numerical falsifier and tests were executed
by root; the worker did not independently reproduce them.

## Cleanup

The delivered completion was inspected before acknowledgment. `worker-release`
returned `retained / external_terminal / processAction:none`, because this was
an operator-created terminal. Root then explicitly closed that exact task-owned
terminal after settlement; the close receipt confirmed `ptyKilled:true`.
No user terminal was closed. The Run's reclaimable-worker list was empty.
No unfinished worker or scientific success was inferred from a heartbeat/timeout.

The final decision is root's bounded judgment: the proposed generic mechanism
lacks a demonstrated new step, **not** a proof against all future specialized
methods. The [result report](ROUND5_RANK_INFERENCE_RESULTS.md) distinguishes
VERIFIED, historical REPORTED, NOT RUN and assumptions.
