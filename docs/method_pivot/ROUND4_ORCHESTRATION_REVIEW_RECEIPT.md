# Round 4 supervised AGY research and coordinator acceptance

Date: 2026-10-05. Run `run_6c1d0d8de61e`, dedicated coordinator
`term_5f3d2810-9322-48dd-a248-42d3cb5e35a5`. Scope: FinancePaper research branch;
no main merge, no external messaging, no test-set selection.

The requested **Orca AGY workflow actually ran**. It was not substituted with an
in-process collaborator while described as AGY. Two workers reached completed
dispatches in the exact FinancePaper workspace:

| Responsibility | Dispatch | What worker supplied | Coordinator acceptance |
|---|---|---|---|
| Cached development protocol and leakage/source audit | `ctx_c43b6e375896` | Draft protocol, cache/schema/partition locations and local receipts | Useful discovery accepted; source-only checks and unmeasured timings were not treated as execution evidence. Coordinator froze the final protocol, implemented guards and ran the audit |
| Independent mechanism/prior-art rejection test | `ctx_319a6d08174d` | Two candidate constructions and strongest-rejection draft | Initial mathematical and citation errors rejected. Coordinator reread primary sources and replaced the draft with the corrected review |

The coordinator independently handled the MVU accepted-paper/source audit.
That task's attempted worker launches did not execute successfully; it is not
listed as a successful third AGY report. Independent finite-completion code and
all new measured tests/audits were implemented and run by the coordinator.

## Substantive corrections, not rubber-stamp review

- Expected NLL on stochastic targets need not destroy entropy; its optimum is
  the target law under appropriate capacity/data assumptions.
- Efficient full attributions reconstruct the frozen margin; their information
  relation with the prediction is not generally "neither contains the other".
- Smooth level-set dimension requires regularity and is not a tree derivative.
- FastSHAP year/authors and Explanation Bottleneck Models title were corrected.
- Slack's adversarial model wrapper is not the same fixed-model missing-input
  search. Veritas generic expressibility does not prove equal efficient cost.
- Removed invented timing and overbroad claims that predictive uncertainty can
  never detect revision or that all future method development is impossible.

Unaccepted worker drafts are preserved locally under ignored
`outputs/method_pivot/round4/review/`. Their statements are not research results.
Final accepted work is in [the independent review](ROUND4_INDEPENDENT_MECHANISM_REVIEW.md),
[protocol](ROUND4_DEVELOPMENT_PROTOCOL_DRAFT.md) and
[executed comparison](ROUND4_PREDICTION_DISTRIBUTION_RESULTS.md).

## Infrastructure failures and cleanup

Initial launches encountered an unconfigured `agy` agent name, provider readiness
timeouts, and an unavailable Claude executable. The ambiguous initial workspace
binding was replaced with the explicit FinancePaper workspace before any worker
task ran. Failed launches are not counted as research. The actual AGY executable
was then used in task-owned terminals. One transient connection error resumed
the same dispatch; completion came through durable worker reports.

A correction dispatch `ctx_2bbe9c666ebe` failed at readiness before receiving the
task. Therefore the corrected review is **coordinator work**, not a claimed worker
rerun. Settled reports were acknowledged. Owned failed dispatch resources were
released; the two operator-created AGY terminals were closed after settlement.
Orca classifies their receipts as retained/external even after terminal closure;
this is not an active-worker assertion. No unrelated terminal/worktree was
reset, closed or committed. Runtime logs stay local and are not copied into the
public repository with session details or credentials.
