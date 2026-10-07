# Round 7 — method card for the candidate actually falsified

Status: **NO-GO for method implementation in this form**. This card defines what
was tested, not an approved new model. [Protocol](ROUND7_PREREGISTERED_FALSIFICATION.md),
[nearest work](ROUND7_CANDIDATE_DIRECTIONS.md), [results](ROUND7_FALSIFICATION_RESULTS.md).

**A — Question.** Can verification data correct learning from a wrong completion
law without adding spurious pressure to suppress legitimate risk updates?

**B — Failure.** An imputer can be internally coherent but wrong about hidden
values. The actual target is an expectation under the data law P, not the
imputer law Q. Our new finite example tests this directly; it is not a measured
failure rate for financial customers or the 2026 SSL paper.

**C — Inputs.** W=(observed covariates, mask); H=additional measured covariates;
A=verification indicator. A=1 reveals H during training only. Deployment sees
W. The true financial event Y, when available, must separately anchor prediction.

**D — Outputs.** f_theta(W) is a partial-information prediction; g_phi(W,H) a
refined prediction. Neither is a causal explanation. Residual coherence is a
diagnostic, not evidence of outcome correctness. A constant model can be coherent.

**E — Assumptions.** Independent records, A independent of H conditional on W,
known randomized pi(W)>0, finite second moments. These are assumptions about
verification, not a claim that initial natural missingness is MCAR. Audit
nonresponse depending on unobserved H breaks identification without further
assumptions. Unavailable natural missing values cannot be restored.

**F — Formulation.** For a fixed teacher, define m(W)=E_P[g|W],

    T_q = q(W) + A/pi(W) * (g - q(W)).
    E[T_q|W] = m(W).
    Var(T_q|W) = Var(g|W)/pi + (1/pi - 1)*(m-q)^2.

Minimizing E[(f_theta-T_q)^2] over the student minimizes
E[(f_theta-m)^2] plus a student-independent constant. This is valid even if q
is wrong, with known randomized pi. It is **exactly the AIPW baseline**. It is
not uniformly variance-improving. Parametric rates would additionally require
identification, regularity, function-class conditions and nuisance-rate controls;
the worker's unqualified root-n assertion is withdrawn.

For jointly changing predictors, let m_theta=E[g_theta|W]. Then

    J(theta) = E[(f_theta-m_theta)^2]
    E[(f_theta-T_q,theta)^2]
      = J(theta) + E[Var(T_q,theta|W)].

The added term generally depends on theta. Its derivative is not part of the
desired coherence gradient. An unbiased target therefore does not justify this
squared objective. Stopping its gradient would define another update, not prove
the original claim. Independent draws from a wrong Q remove finite-MC squaring
bias relative to Q; they do not turn Q into P.

A matched conditional-moment baseline uses

    r_theta = f_theta(W)-g_theta(W,H)
    J(theta) = sup_h E[2*h(W)*r_theta - h(W)^2]
             = sup_h E[2*h(W)*A/pi(W)*r_theta - h(W)^2].

The equality uses an unrestricted square-integrable critic and actual-law audit
assumptions. Restricting h to a finite family/RKHS changes the criterion; one
must not claim exact equality or perfect enforcement for an arbitrary neural
critic. An augmented residual q_r + A/pi*(r-q_r) may reduce variance; that is
again an established correction, not an automatically new mechanism.

**G — Exact new step.** None survives in the supplied construction. A distinct
finite-budget estimator, optimizer, or statistical result remains to be derived.
The conditional-moment dual is an application of known methods, not our theorem.

**H — Reduction.** Freeze teacher -> ordinary AIPW student regression.
Remove augmentation -> IPW conditional moments. Use the unrestricted critic ->
the same population coherence loss. New labels for these operations do not
create an algorithmic difference.

**I — Identification.** Audits with known inclusion probabilities identify the
specified teacher functional on covered W. This does not identify hidden truth
for each individual, rescue an incorrect teacher, or resolve arbitrary MNAR.
For inference about actual Y, teacher correctness/transport needs its own check.

When training outcomes Y are already observed, ordinary supervised learning on
W with a proper loss directly targets E[Y|W] under the declared observation
law. An expressive population-optimal family is already coherent. Thus audits
and distillation cannot claim an automatic identification advantage in Taiwan's
simulated-missingness setting. Any benefit must be finite-sample, computational,
or under a separately justified observation-process limitation. Direct masked
supervised ERM and generalized distillation belong in the baseline set. Extra
complete-record teacher training must be charged to the information budget.

**J — Train / inference.** Audited H may supervise training; hidden truth,
restored predictions and future outcomes must not be inference features. A real
pilot would split original customers before teacher/nuisance/student fitting,
cross-fit learned nuisances, and reserve separate calibration. None was run.

**K — Falsifier.** Exact AIPW equivalence defeats the fixed-teacher novelty claim.
An extra theta-dependent variance penalty defeats naive joint squared training.
An existing conditional-moment baseline removes that population artifact. All
three hold in this round; no financial sweep follows.

**L — Compute.** A future student update needs a coarse call and, on audited
records, a refined call; completion estimates add their own K calls and a critic
adds training cost. Parameters and actual training latency are **NOT RUN**, not
guessed from a named architecture. The present exact finite audit has no trained
parameters or GPU requirement and a preregistered <60s/<512MiB execution budget.
