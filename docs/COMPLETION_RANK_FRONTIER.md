# C5: a sharp mean-margin / completion-rank-failure frontier

Coordinator-originated, 2026-10-08, following the C3/C4 audit. This is a new
research question, not a replay of historical finance endpoints. Initial status:
**PROPOSED**; correctness and novelty decisions belong in the stage report.

## Target and assumptions

There are m>=2 observed binary players A, query A=1, and one finite hidden
player H. Reference P=Bernoulli(p)^m x q, 0<p<1; completion Q fixes A=1 and
uses the same strictly positive q. All m+1 features remain SHAP players.
The fixed raw-output model is bounded in [0,1] on the full reference support.
It may additionally satisfy f(1,h)=c for every h. Use interventional SHAP.

For a fixed pair i,j, let Delta(h)=phi_i(1,h)-phi_j(1,h), mu=E_q Delta,
and z=q(Delta<=0). Ties count as failure to support the strict ranking i>j.
This is a **signed pair-ranking** target under a declared completion law. It
is not an absolute-attribution ranking, causal relevance, financial correctness,
human utility, or a finite-sample confidence statement. Assume the mean is exact.

## Reduction to a positive mixture

Put r=m-2 and u(t)=p+(1-p)t. For k=0,...,r define

  alpha_k = C(r,k) integral_0^1 u(t)^k (1-u(t))^(r-k) dt,
  beta_k  = C(r,k) integral_0^1 t u(t)^k (1-u(t))^(r-k) dt,
  eta_k   = beta_k / alpha_k.

The alpha masses sum to 1 and the beta masses to 1/2. Equivalently, T is
uniform[0,1], K|T is Binomial(r,u(T)), and eta_k=E[T|K=k]. All eta are in
(0,1); they increase with k (monotone likelihood ratio of the binomial family).

Let d_s(h)=f(A_i=1,A_j=0,s,h)-f(A_i=0,A_j=1,s,h) in [-1,1]. Average d_s
over other-observed states s with the same cardinality k, obtaining d_k(h).
The ordinary Shapley difference identity, combining coalitions differing in
i or j, and the beta integral for their weights give exactly

  Delta(h)/(1-p) = sum_k alpha_k [eta_k d_k(h)+(1-eta_k) E_q d_k].
  mu/(1-p) = sum_k alpha_k E_q d_k.

Both the A_i=A_j=0 and A_i=A_j=1 truth-table rows cancel. Consequently fixing
the prediction slice does not tighten this particular pair-ranking frontier.
The derivation is specific; a generic arbitrary bounded random margin need
not admit this representation.

## Exact frontier

For 0<z<1 define a standard fractional knapsack value

  L_m,p(z) = min sum_k alpha_k l_k
             subject to sum_k alpha_k [z+(1-z)eta_k] l_k >= 1/2,
                        0<=l_k<=1.
  F_m,p(z) = 1-2z L_m,p(z).

Then mu/(1-p) <= F_m,p(z), and this upper bound is attained by a bounded
model with two hidden states of probabilities (z,1-z), constant prediction
f(1,h)=c, and exactly z mass at Delta=0. For z=1 the frontier is 0; its
continuous z=0 extension is 1. The strict-reversal version has the same
supremum for 0<z<1, approached by perturbing an attaining table; ties are
included in the stated attainable theorem.

Proof of the optimization reduction: average d_k within B={Delta<=0} and
outside B to x_k,y_k. Write c_k=z+(1-z)eta_k. The averaged bad-margin
constraint is sum alpha_k[c_k x_k+(1-z)(1-eta_k)y_k]<=0. Its objective is
sum alpha_k[z x_k+(1-z)y_k]. Starting at x=y=-1, this is a continuous
knapsack over increases. The objective/constraint ratio for every y variable
is 1/(1-eta_k)>1; for every x variable it is z/c_k<1. Thus all y variables
are filled first. At y=1,x=-1 the constraint is -z<=0, so doing so is
feasible. Set x=1-2l to obtain the displayed L problem. Fill l in descending
eta order, with at most one fractional cell. This uses m-1 cells, not 2^m
truth-table variables. Fractional knapsack itself is established mathematics.

Attainment: use x_k=1-2l_k in the bad hidden state and y_k=1 otherwise.
For the (A_i,A_j)=(1,0) row set f=(1+d_k)/2, and for (0,1) set f=(1-d_k)/2.
Set equal-pair rows to c. These values lie in [0,1]; the prediction slice is
c. The bad margin is exactly zero, while the good margin is positive when
z<1. A finite decision tree always represents the table, but no O(m)-size
tree claim is made for these symmetric count-dependent witnesses.

F is continuous and strictly decreasing from 1 to 0. For a fixed filled
prefix with A=sum alpha and B=sum beta, and fractional-cell eta=e, the cost
is [1/2-(B-eA)(1-z)]/[e+(1-e)z]. Put c0=B-eA>=0. The derivative of zL
has numerator (1/2-c0)e+2c0 e z+c0(1-e)z^2>0; c0<1/2. At breakpoints the
formulas agree. Thus exact mean mu/(1-p)>=F(alpha) certifies z<=alpha.
For an arbitrary fixed finite q, the bound is valid but may be conservative
because only its subset masses are attainable. Sharpness above varies q.

## Closed forms and dimension-free envelope

For m=2, eta=1/2 and

  F_2,p(z)=(1-z)/(1+z),  z <= (1-mu/(1-p))/(1+mu/(1-p)).

For all m and p,

  F_m,p(z) <= F_infinity(z)=1-2z/(1+sqrt(z)).

To prove this, write l=l(K), L=E l. Since E[T l(K)]=E[eta_K l(K)] and
0<=E[l(K)|T]<=1, the elementary upper-tail rearrangement bound gives
E[T l(K)]<=L-L^2/2. Feasibility therefore implies
L-(1-z)L^2/2>=1/2, or L>=1/(1+sqrt(z)). The envelope is sharp as m grows
for any fixed p in (0,1): K/(m-2) consistently reveals u(T), so threshold
rules on K approximate the upper-tail rule on T, including its cost and
constraint. Finite audits do not prove this limiting claim; the argument does.

Writing a=1-mu/(1-p), a dimension-free upper bound on z is
[(a+sqrt(a*a+8*a))/4]^2, for 0<=mu/(1-p)<=1. The basic oscillation-only
bound is z<=1-mu/(1-p). These bounds use different structural information;
no superiority over exact completion enumeration is claimed.

## Pre-run audit and decision gates

Freeze configs/completion_rank_audit.json before executing the audit. Build
the entire SHAP linear operator independently by exact coalition/background
enumeration using the prior definition oracle. Compare its pair coefficients
with the new mixture, solve full truth-table LPs with all slice values fixed,
and check exact rational witness margins for each configured p,m,z,c.
Include multi-state q/event subsets, strict-reversal perturbations, a deliberately
mismatched completion law falsifier, and the generic oscillation-only baseline.
The primary correctness stop is any coefficient, witness, or LP discrepancy.
No inspected financial data, training, human judgments, or outcome selection.

The runner must freeze source/config/dependency hashes, write task receipts
and JSONL progress with device/time/warnings/ETA, show tqdm, and resume only
verified outputs. Test pause/resume and corruption/drift rejection on separate
scratch runs. Export one consolidated report with RUN, REPORTED, PROPOSED,
NOT RUN and novelty limitations. Do not revise older configs/results.

## Novelty gate

The candidate contribution is the SHAP-specific sharp rank-tail frontier and
its m-1-cell reduction, with prediction-invariant attaining models. Shapley
beta integrals, fractional knapsack, rearrangement, risk thresholds, and union
bounds are not new. Compare against Lin/Covert/Lee removal-based robustness,
Goldwasser/Hooker ranking inference, Neuhof/Benjamini confident ranking,
stochastic SHAP, selective explanations, and distributional SHAP uncertainty.
Absence from a bounded search cannot establish priority or publication value.
