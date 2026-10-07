# A sharp bound under prediction-invariant completion

Coordinator-generated C3, 2026-10-08, after C0/C1/C2 prior-art collisions.
Status before executable audit: **PROPOSED theorem; originality not established**.
The earlier arbitrary-law construction motivates this bounded question but is
not evidence for the theorem or its novelty. This document is frozen by the C3
run manifest; final adjudication belongs in the consolidated stage report.

## Statement and exact scope

There are m observed binary SHAP players A_1,...,A_m and one hidden player H
with finite support. Reference P is the product of iid Bernoulli(p) observed
coordinates and q(H), with 0<p<1 and all q masses positive. Completion Q sets
all observed A=1 and uses the same q(H). H is one player, even if categorical.
The fixed raw model has 0<=f<=1 everywhere on this full finite reference support
and f(1,h)=c for every h. Compute ordinary interventional SHAP at each completed
input (1,h), keeping the hidden coordinate as a player.

For any observed i, define

\[
w_m(p)=(1-p)\int_0^1 t[p+(1-p)t]^{m-1}\,dt,
\qquad \omega_m(p)=(1-p)-w_m(p).
\]

Then osc_Q(phi_i)<=omega_m(p) and Var_Q(phi_i)<=omega_m(p)^2/4.
Both bounds are sharp over this model/distribution class for every m,p,c:
two equiprobable hidden states and a single bounded tree with 2m+2 leaves and
depth m+1 attain them. For m=1,p=1/2, width=1/4 and variance=1/64; for
m=2,p=1/2, width=7/24 and variance=49/2304. These are population bounds
under the stated artificial law, not confidence intervals or calibration claims.
For output range [L,U], multiply width by U-L and variance by (U-L)^2.

## Proof

Use the standard beta-integral representation of Shapley permutation weights.
Terms with H excluded do not depend on h. Thus phi_i(h)=b_i+F_i(h), where

\[
F_i(h)=\sum_{a\in\{0,1\}^m}K_i(a)f(a,h),
\quad
K_i(a)=(1-p)(2a_i-1)\int_0^1 t
\prod_{j\ne i}\left\{\begin{array}{ll}
p+(1-p)t,&a_j=1,\\
(1-p)(1-t),&a_j=0
\end{array}\right. dt.
\]

The leading t is the probability that H precedes i in the Bernoulli-coalition
integral. Summing positive coefficients gives (1-p)/2, and summing negative
coefficients gives -(1-p)/2. The positive all-one coefficient is w_m(p).
Since its model value c is constant in h, it contributes no oscillation.
Every remaining f(a,h) lies in [0,1], so

\[
|F_i(h)-F_i(h')|\le\sum_{a\ne\mathbf1}|K_i(a)|
=(1-p)-w_m(p).
\]

The variance bound is the standard bounded-range variance inequality: for
X in [a,b], E[(X-a)(b-X)]>=0 gives Var(X)<= (E X-a)(b-E X)
<= (b-a)^2/4. This variance inequality is not new.

To attain the bound, take fair H in {0,1}. At nonslice states let
f(a,0)=1 if a_i=1 and 0 otherwise; let f(a,1)=1-f(a,0).
At a=1 override both outputs with c. All free positive and negative coefficient
differences align; hence the two SHAP values differ by exactly omega. A tree
splits on H, then A_i, then tests the remaining A coordinates until a zero is
seen (or reaches the all-one override). It has 2m+2 leaves, depth m+1, and
bounded leaf values 0,1,c. This establishes sharpness with O(m) explicit model
size, not just an exponentially large lookup table.

For a direction v over observed attributions, define K with the all-one column
removed. The same proof yields the sharp bound

\[
\operatorname{osc}(v^\top\phi_A)\le\|K^\top v\|_1,
\qquad\operatorname{Var}(v^\top\phi_A)\le\|K^\top v\|_1^2/4.
\]

Choose endpoint outputs according to the sign of (K^T v)_a to attain it with
fair H. This general directional witness may need a truth table; the O(m)
tree claim applies to scalar coordinate directions only. It does not imply
all directional maxima can be attained by one common model simultaneously.

In fact the exact set of possible two-completion contrast vectors is the
symmetric zonotope K[-1,1]^(2^m-1): every difference of bounded output tables
belongs to this cube; conversely, for any z in that cube set the two nonslice
tables to (1+z)/2 and (1-z)/2 and keep their all-one values c. This is a set of
**contrasts**, not an unconditional claim about the set of absolute SHAP vectors:
their h-independent offset depends on the model and reference expectations.

For a **fixed** finite hidden law q, the exact supremum of directional variance is

\[
\sup_f\operatorname{Var}_q(v^\top\phi_A)
=\beta(q)\|K^\top v\|_1^2,\qquad
\beta(q)=\max_{B\subseteq\operatorname{supp}(q)}q(B)(1-q(B)).
\]

Proof: the variance is a convex quadratic function of its finite vector of
values in an interval of width ||K^T v||_1, so some maximum occurs at an endpoint
assignment. Such assignments have variance q(B)(1-q(B)) times width squared.
The endpoint model tables can be chosen independently for every hidden state,
so every partition B is attainable. The coefficient 1/4 is attainable exactly
when a subset has mass 1/2. Computing beta here uses exhaustive small-support
enumeration; no efficient partition algorithm is claimed. Arbitrary-partition
realizers may require more than the binary scalar witness's O(m) leaves.

## Actual-audit plan frozen before results

`configs/bounded_completion_audit.json` fixes 45 cases: m=1,...,5, p=1/10,
1/2,9/10, c=0,1/2,1. For each m,p, build the SHAP linear operator by direct
coalition/background enumeration using exact Fraction arithmetic, independently
of the integral formula. Verify every coefficient of the two-completion
difference equals [-K,K]. Use scipy linear programming on all bounded truth-table
values, fixing the two all-observed-one outputs at c, to check all coordinate,
all-ones, and first-minus-last directions. Check scalar witness SHAP and variance
against the rational bound. LP solutions are mathematical extremizers, not fitted
predictors or choices from financial test results. Exercise pause/resume and
source/config/output hash rejection. Unique ignored run; CPU, progress/ETA,
warnings and aggregate receipts; no human data or training.

Additional prespecified fixed-q checks use the four laws in the config. Compare
the attaining partition with direct coalition/background SHAP at each hidden
state. For m<=2 and hidden support<=3, enumerate **all** endpoint-valued free
truth tables (at c=1/2) and check the largest variance independently against
beta(q)*omega^2. Publish counts; these small finite exact checks do not establish
the theorem for all m. The algebraic proof supplies that general statement.

## Novelty boundary

Generic prediction-preserving attribution manipulation and unbounded impossibility
are prior art (Bilodeau et al., PNAS 2024; Slack et al., AIES 2020). Generic
robustness/operator-norm bounds are also prior art (Lin, Covert & Lee, NeurIPS
2023). Beta-integral SHAP and bounded-range variance are established tools.
The candidate delta is the explicit **sharp completion-slice correction
w_m(p), together with a bounded O(m)-leaf attaining tree and joint directional
envelope**, for this restricted finite product-reference model class. Whether
this narrow theorem is a worthwhile original contribution needs comparison
with the exact assumptions and bounds of those papers. Do not claim first
robust SHAP, a new explainer, novelty by failed search, or human decision value.
