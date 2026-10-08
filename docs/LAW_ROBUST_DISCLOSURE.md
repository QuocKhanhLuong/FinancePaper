# Linking the research question: disclosure under completion-law uncertainty

2026-10-08. User explicitly asked to extend the research question slightly so
minor contributions form a coherent result. This document freezes a new C6
claim and its audit before running it; older protocols/results remain unchanged.

## Proposed paper question

**What signed reason-ranking reliability can be certified from incomplete
records when the completion law may differ from the SHAP reference law?**

The logical chain is: C3 quantifies prediction-invariant attribution ambiguity;
C5 establishes a sharp mean-to-ranking-risk certificate under a matched law;
C6 below quantifies how that certificate degrades when the laws differ.
C0 moment computation is supporting machinery with known prior collisions,
not a separate novelty claim. Human utility is a different empirical question.

This gives a coherent theory-note direction rather than an accumulation of
unrelated results. Publication-level novelty is not inferred from the number
of propositions. The strongest unresolved substantive gap remains scope:
multiple missing players and practical completion laws are not covered by C6.

## C6 statement: two observed players, two different hidden laws

There are two observed binary players A1,A2 at query (1,1), and one hidden
SHAP player H. Keep the fixed raw model f in [0,1] on the full finite domain
(including the union of the reference/completion supports). Reference is
P=Bernoulli(p)^2 x R, 0<p<1. Completion sets A=(1,1) and H~Q. Both R and Q
are probability laws; zero masses are allowed. Use interventional SHAP, with
all three players retained. Standard TV means half the L1 distance.

Let G(h)=(phi_1^R(1,1,h)-phi_2^R(1,1,h))/(1-p), t=E_Q G in [0,1],
and z=Q(G<=0). Ties count as failure to support a strict signed ranking.
For TV(R,Q)<=epsilon, with 0<=epsilon<=1,

  t <= F_epsilon(z) = (1-z)/(1+max(z-epsilon,0)),

or equivalently

  z <= min(1-t, (1-t+t*epsilon)/(1+t)).

This bound is sharp over bounded models and finite R,Q, even with f(1,1,h)=c
for any c in [0,1]. It is a deterministic population statement with an exact
mean and a valid TV upper bound. It neither estimates epsilon nor supplies
a statistical confidence level. When both laws must have full support, cases
requiring R(bad)=0 are supremum-only. For strict reversal instead of ties,
the same supremum can be approached for 0<z<1; at the zero-reference-mass
boundary this requires perturbing the law as well as the model.

## Proof and attaining construction

Define d(h)=f(1,0,h)-f(0,1,h) in [-1,1] and b=E_R d. The three-player
Shapley definition gives G(h)=(d(h)+b)/2. The equal-pair rows cancel.
Let B={d<=-b}, r=R(B), z=Q(B). The ordinary TV event inequality gives
r>=r0=max(z-epsilon,0). Since d<=-b on B and d<=1 elsewhere,

  b <= -r*b+(1-r), hence b <= (1-r)/(1+r).
  a=E_Q d <= -z*b+(1-z).
  t=(a+b)/2 <= (1-z)*(1+b)/2 <= (1-z)/(1+r0).

To attain equality, use bad/good hidden states with R=(r0,1-r0), Q=(z,1-z),
b0=(1-r0)/(1+r0), d_bad=-b0 and d_good=1. Then E_Rd=b0 and G_bad=0.
Set f(1,0,h)=(1+d(h))/2, f(0,1,h)=(1-d(h))/2, and equal-pair rows to c.
All outputs are bounded and the prediction slice is constant. Inverting the
two monotone pieces yields the displayed risk bound. For t<0 only the trivial
bound z<=1 is provided; the implementation rejects out-of-range inputs.

At epsilon=0 this exactly recovers the m=2 C5 frontier. At epsilon=1 only
the generic oscillation bound remains. Thus C6 repairs the concrete reference/
completion-law counterexample from C5 without silently changing its estimand.

## Baselines and contribution limits

Use the same available information (exact t, epsilon, bounded model) for:

1. Generic oscillation certificate: z<=1-t.
2. Ordinary event-TV transfer plus a mean-shift bound: |E_R G-E_Q G|<=epsilon,
   then z<=min(1,epsilon+(1-max(t-epsilon,0))/(1+max(t-epsilon,0))).
   Combine with baseline 1 by taking the smaller bound.
3. Naively reusing the matched-law formula: (1-t)/(1+t). It is an invalid
   negative control under mismatch, not a valid method to beat.
4. Exact completion enumeration when available. It is stronger information,
   so no universal superiority over it is claimed. Exact variance/Cantelli
   also uses additional information and can be tight on a two-point witness.

TV event transfer, bounded-mean inequalities, and release/abstain decisions are
standard. The only candidate novelty is the sharp specialized SHAP frontier
and its role in a unified ambiguity-to-certificate argument. Prior collision
review and correctness review are separate. General distributional SHAP and
selective-risk ideas already exist; no broad first-of-its-kind claim is allowed.

## Prespecified executable audit

Use configs/law_robust_disclosure_audit.json. For constructive cases, independently
enumerate SHAP coalitions/backgrounds, fix prediction c, solve the full model
truth-table LP, and compare to the sharp expression. Include epsilon=0 and 1,
epsilon<z, epsilon=z, epsilon>z, and zero reference masses. Separately examine
all ordered pairs of three-state laws on the denominator-four simplex and
every nonempty proper declared bad-state subset: the universal frontier must
upper-bound the full LP, but need not be attained for a fixed R,Q.

Audit baselines on the fixed grid. Freeze source/config/dependency hashes, CPU
device, per-task receipts, progress JSONL, tqdm/ETA, pause/resume and drift/
corruption rejection. No fitting, finance evaluation, human data, or retrospective
choice of predictor. One consolidated stage report records actual execution,
previous reported results, proposed paper framing, and NOT RUN.
