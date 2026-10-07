# Stable-Core v2: final decision

**Option 2: operational strategy, not a demonstrated algorithmic contribution.**
The [measured results](STABLE_CORE_RESULTS.md) reject broad coverage superiority
against strong simple baselines. Keep the implementation and negative evidence;
do not tune another threshold family on these assessment results.

1. **Better than whole abstention?** Sometimes compared with an exactly-top-2
   policy, but not against the strongest whole/top-1/rank controls. Much of the
   aggregate benefit is newly allowing one-reason candidates. On common >=2
   eligibility, meaningful-target coverage is slightly lower than whole top-2.
2. **Useful coverage?** Yes empirically: both-family sets explain 90.82% of Taiwan
   and 92.64% of Polish customer-condition cases; reason coverage 93.64%/92.75%.
3. **Multiple completion models?** Yes in these bounded experiments: agreement
   lowers observed false stability with modest coverage loss. Both can still be
   misspecified; this is not a mathematical worst-case guarantee.
4. **Large external validation?** UNKNOWN / NOT RUN. No official Freddie files
   exist locally. No mortgage efficacy, runtime, transfer or risk-control claim.
5. **Beats rank instability?** NO for the declared fixed-budget coverage objective.
   Rank-only provides greater coverage under the ranked target. Stable-Core
   exchanges coverage for lower failure; no universal winner is established.
6. **More than fewer reasons?** YES for reason identity: at identical per-customer
   set sizes, both-family selection reduces meaningful reason failure versus
   strength matching by 0.47/0.68 percentage points (paired CIs exclude zero).
7. **Defensible algorithmic novelty?** NOT ESTABLISHED. Stable subsets, uncertainty
   filtering and calibrated set-valued decisions have close prior work. Do not
   claim a new generic uncertainty or conformal algorithm. At the 10% primary
   operating point the fitted rule reduces to `q10 > .01`; sign/rank thresholds
   are both zero. The compound rule does not demonstrate a distinct mechanism.
8. **Applied-policy value?** YES, with limits: a tested variable-size output,
   verification target, paired baselines and explicit cost/risk tradeoffs strengthen
   the evaluation paper. Primary meaningful failure is already low enough that
   release-all satisfies the 10/15% aggregate budgets; report that plainly.
9. **Inference output?** Risk raw/calibrated; stable and withheld candidate groups;
   per-family sign/top-2 frequencies and magnitude quantiles; NONE/PARTIAL/
   ALL_CANDIDATES; target, declared budget, calibration scope and thresholds. Never
   show verification truth or an individual-customer safety guarantee.
10. **Replace whole release?** NO. Retain it and top-1/rank/frequency controls.
    Offer stable subsets as a conservative operating option, with the reason-level
    target and lack of guarantee visible to the downstream user.

## Critical limitation

Every conservative finite-grid bound produced empty sets. Useful coverage here
comes from empirical calibration, and Polish MCAR30 has a 5% budget violation
for several simple policies. These results do not support a deployment risk
guarantee. Ranked release-all counts also include candidates already below top-2;
do not misreport them as pure post-verification revision.

## Next action

Acquire the exact officially licensed Freddie Release 47 files and run the
already declared temporal confirmation with the v2 policy and strong baselines
frozen. No further Taiwan/Polish method search is justified by this result.
