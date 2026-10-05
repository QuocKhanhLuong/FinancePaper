# Round 2 decision after the numerical pilot

**Keep a research oracle; NO-GO as a new-method claim on this formulation.**
Date: 2026-10-05. This is a technical reduction decision, not a novelty score
or a claim that the entire missing-information research problem is solved.

The [pilot](ROUND2_RESULTS.md) passed correctness and found a useful accuracy/cost
point. It did **not** fail numerically. The failure is that the currently proposed
mechanism has not survived the closest-prior reduction:

1. Fixed-background attribution is a vector-valued finite step function. The
   rectangle representation follows from [Filom et al.](https://arxiv.org/html/2302.08434v4),
   path-game linearity, and related compiled point-attribution methods.
2. After fixing observed coordinates, pruning and merging rectangles restricts
   that same function to the remaining variables. This is ordinary evidence
   restriction/symbolic simplification, not a new statistical identification step.
3. With indicator vector r, coefficient matrix C, p=E[r], J=E[rr'], the output is
   `C' p` and `C' (J-p p') C`: standard moments of a multi-output step function.
   Full covariance correctly retains cross-tree terms, but doing so is necessary
   correctness rather than novelty.
4. For empirical q this reduces to exact weighted evaluation; for product q it
   reduces to sum-product rectangle integration. For compatible probabilistic
   and regression circuits, [Khosravi et al.](https://proceedings.neurips.cc/paper_files/paper/2019/file/fccc64972a9468a11f125cadb090e89e-Paper.pdf)
   already give tractable regression moments, including missing-input queries.

This rejection is **not** “all components existed, so combinations cannot be
new.” The entire current inference operator can be written as a standard moment
query on a known compiled function, with no additional objective, identification
result or asymptotically better structural algorithm established. A future
algorithm could overcome this reduction; the present one has not.

[WOODELF-HD](https://arxiv.org/abs/2604.10569) was actually run and the candidate
was faster than its finite enumeration in this workload. That is valuable
engineering evidence. It does not remove the reduction or establish superiority
over [FourierSHAP](https://papers.nips.cc/paper_files/paper/2025/hash/1b331c20064e37e204a5bcd12481bfac-Abstract-Conference.html)
and compatible-circuit moment solvers, which were reviewed but not executed.
The O(R²) bound is an implementation cost, not a new complexity theorem.

## What was gained

- A checked exact oracle for completion-relative attribution moments, enabling
  numerical errors to be separated from completion-model errors on tractable laws.
- An explicit counterexample rejecting original-prediction-leaf quotienting.
- A measured cost frontier with a current strong official comparator, preserving
  the fact that small-K stock TreeSHAP is much cheaper.
- A sharper necessity requirement for further work: improving inference must
  target a problem not already solved by compiling the response and integrating
  it, or establish a real new structural cost bound for that operation.

The frozen predictor, historical reason definitions, and financial conclusions
are unchanged. No new release threshold, revision head, architecture, or renamed
uncertainty score is introduced. Financial development and confirmation do not
start on the strength of this numerical result alone.

## One next research action

Derive and adversarially audit an observation-dependent structural complexity
bound for joint response inference against generic circuit integration **before
any further implementation**. Ordinary ANOVA, control variates, sparse covariance
or a different basis do not qualify without a precise non-reducible advantage.
This is a required open theoretical gate, not a promised successful next method.
If it closes by direct reduction again, do not spend a financial benchmark to
manufacture a method claim.
