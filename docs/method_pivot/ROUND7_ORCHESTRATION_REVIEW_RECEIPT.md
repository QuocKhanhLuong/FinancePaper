# Round 7 — actual Orca AGY work and coordinator corrections

2026-10-07, `research/method-pivot`, starting SHA `a9de607`.
Actual Orca runtime `0933d728-2467-43c3-bdae-32ae82931d8f`, version1.4.204;
run `run_b04716b915be`. Used the installed orchestration skill and actual
`/Users/alvinluong/.local/bin/agy --mode accept-edits` terminals. No substitute
Codex collaboration workers. Provider default displayed Gemini3.8Flash/high;
coordinator did not override model/effort. No credentials are included here.

| Task / Dispatch | Work | Settlement and acceptance |
|---|---|---|
| `task_e0686276b8df` / `ctx_749c090558f1` | Independent statistical learning/observation design; owns only ignored report/source files | worker_done succeeded; report produced, scientific claims corrected/rejected |
| `task_6abceff2fb27` / `ctx_64f77f597ad5` | Independent inference/sensitivity/information-theory candidates | worker_done succeeded; report produced, claimed new joint advantage rejected |
| `task_baf7634b9935` / `ctx_d2f10c1282ac` | Learning follow-up through same-terminal worker-start | failed at agent_readiness/timeout; no completed review attributed to this attempt |
| `task_2bf1dfb7b27a` / `ctx_e982aa9036d9` | Inference follow-up through same-terminal worker-start | same readiness failure; retained as failed attempt |
| `task_61fcc087dbf0` / `ctx_3c6e581e22ba` | Bounded cross-review, recovered through explicit low-level dispatch + same task-owned AGY terminal | worker_done succeeded; agreed on exact reductions/counterexamples; coordinator still corrected proof/scope issues |

Two independent generation reports preceded cross-reading; coordinator also
wrote independent ideas first. Cross-review was **guided by coordinator
objections**, not a blind independent replication. The final numerical audit
was written and run by the coordinator. Workers did not run financial training.
Do not turn worker agreement into scientific validation.

Local ignored files: `outputs/method_pivot/round7/{agy_learning,learning_sources,
agy_inference,inference_sources,root_independent_ideas,agy_inference_review}.md`.
Original reports were preserved, not silently rewritten. Their full-text badges
and citations were not accepted without coordinator verification. The tracked
[source ledger](ROUND7_CANDIDATE_DIRECTIONS.md) states coordinator reading depth.

## Corrections required before accepting claims

- Learning report incorrectly described historical findings as explanation
  stability tracking prediction stability. FinancePaper's decoupling evidence
  survives; a new worker summary cannot reverse it.
- Fixed-teacher correction is exactly AIPW. Toy uncertainty/3.8x variance gain
  had no run receipt and was contradicted by exact enumeration. Orthogonality
  does not alone imply a root-n rate for arbitrary neural parameters. MNAR does
  not make every conceivable imputer necessarily misspecified.
- The 37.7% allocation gain is hand-computable for a **known** allocation rule,
  not financial prediction performance. Its proposed regret bound was unproved.
- Shared-adversary top-1 advantage confused ANY and ALL competitors. The review
  agreed, but its written converse proof repeated the same inequality direction;
  coordinator supplied the correct two-sided proof in the results document.
- The review's claim that clipping violates a <=budget constraint was too
  strong: it leaves unused budget and loses optimality. This is corrected.
- The review's statement 'deterministic given W implies mutual information zero'
  is false. The valid counterexample is a **globally constant** target.
- Squared theta-dependent pseudo-target error is diagnosed by its extra
  conditional-variance term; a nonzero cross term alone is not a general proof
  of bias for every possible objective.
- Worker citation errors: Dorn–Guo DOI ends **2069572**, not2063194;
  Yadlowsky et al. is **Annals of Statistics 2022**, DOI10.1214/22-AOS2195,
  not the claimed JMLR entry. Unneeded legal assertions and unverified recent
  worker citations were excluded from the accepted rationale.

## Lifecycle and publication boundaries

worker-start cannot reliably recognize this custom AGY readiness on reuse.
The explicit failed receipts were inspected; a permitted low-level recovery
created authoritative new Task/Dispatch context before terminal injection.
That recovered lane was **unsupervised in Orca resource-ownership terminology**;
it still returned a valid worker_done. No duplicate editor was launched.

All accepted completions were processed before inbox acknowledgment. Releases
reported `external_terminal` or `no_owned_resource`, so they did not kill
operator-created processes. Coordinator then explicitly closed only the two
terminals created for this task; both receipts confirmed `ptyKilled=true`.
Final `worker-list --run run_b04716b915be --terminal-state reclaimable` returned
an empty worker list. Failed readiness attempts remain visible, not called
successful reviews. No merge or main-branch publication is authorized.
