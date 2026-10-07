# C4: scalar versus directional sharp completion bounds

Coordinator derivation, 2026-10-08. Companion to C3, not a hardness claim about
the implemented point-SHAP or fixed-tree moment oracle. Initial status: proposed
proof pending exact reduction audit and prior-work adjudication.

## Computational problem

Fix the bounded C3 class with m observed Bernoulli(1/2) players, one fair binary
hidden player, and constant completed prediction c=1/2. Input is (m,v), where
v is an integer observed-feature direction in binary encoding. The output is
the exact rational sharp within-model contrast bound

\[
D_m(v)=\sup_{0\le f\le1,\ f(\mathbf1,h)=1/2}
|v^\top\phi_A(f,\mathbf1,1)-v^\top\phi_A(f,\mathbf1,0)|
=\|\bar K^\top v\|_1.
\]

The supremum ranges over all bounded finite-domain models. No explicit model f
is input. General attaining trees may have exponential size; the scalar C3
witness has linear size. Exact D_m is **#P-hard under polynomial-time Turing
reductions**, even restricted to directions with sum zero. This is a claim about
the best universal joint bound, not about certifying an explicitly supplied tree,
not about approximation hardness, and not a claim of #P-completeness.

## Reduction

For a subset a of r observed ones, 1<=r<=m-1, the C3 kernel coefficient is
A_r>0 on included coordinates and -B_r<0 on excluded ones. Let

\[
\alpha_r=A_r+B_r
=2^{-(m-1)}\int_0^1t(1+t)^{r-1}(1-t)^{m-r-1}\,dt>0.
\]

If sum_i v_i=0, then (K^T v)_a=alpha_r sum_{i in a}v_i. The empty/all-one
terms vanish, so omitting the all-one column does not change this identity.

Given positive integer #SUBSET-SUM weights w_1,...,w_n and target C, put W=sum w,
m=n+2, and v(C)=(w_1,...,w_n,-C,-(W-C)). Split subsets according to membership
of the two new anchor coordinates. Neither/both-anchor terms are independent
of C. Complementing the original n-coordinate subset in the second-anchor term
gives

\[
D_m(v(C))=\text{constant}+\sum_{S\subseteq[n]}
[\alpha_{|S|+1}+\alpha_{n-|S|+1}]\,|w(S)-C|.
\]

For integer x, |x-(C+1)|-2|x-C|+|x-(C-1)| equals 2 if x=C and zero otherwise.
Thus three exact D-oracle calls give

\[
\Delta^2D(C)=2\sum_{S:w(S)=C}[\alpha_{|S|+1}+\alpha_{n-|S|+1}].
\]

To remove the cardinality weights, for each k=0,...,n lift each w_j to M+w_j,
with M=W+C+2 (C>=0), and target C_k=kM+C. Any equality then has cardinality k,
since |w(S)-C|<M. Divide the lifted second difference by the positive rational
2(alpha_{k+1}+alpha_{n-k+1}) to recover the number of size-k solutions. Sum these
counts. This uses 3(n+1) oracle calls and polynomial-size integers.

Each alpha is computable in polynomial time by binomial expansion and rational
integration; its bit length is polynomial in m (denominators divide a power of
two times lcm(2,...,m)). The D-oracle output also has polynomial bit length for
binary integer v. This gives a counting-function Turing reduction from the
standard #P-complete exact subset-sum counting problem. Negative and zero anchor
coordinates are permitted. Negative target counts are trivially zero and need
not enter the reduction. Large binary weights are essential to the usual counting
hardness statement; pseudo-polynomial special cases are not excluded.

## Prospective exact audit and originality boundary

`configs/directional_hardness_audit.json` fixes n=2,...,6, two seeds, and targets
zero, floor(W/2), W+1: 30 weight/target instances. The executable oracle actually
enumerates the generic rational C3 kernel support, independently of the reduced
weighted-absolute-sum expression. Audit every cardinality via three exact oracle
queries and compare with brute subset counts; retain zero-count and boundary
cases. A finite audit checks the reduction implementation, not a complexity proof.
JSONL progress/tqdm/ETA, pause/resume and hash guards are required.

Counting subset sums, absolute Rademacher sums/Khintchine constants, and linear
operator methods are established. This reduction must be compared with those
sources. The possible contribution is the specific **tractability separation
for sharp prediction-invariant completion bounds**: scalar directions have the
C3 closed form and O(m)-leaf witnesses, while arbitrary signed joint directions
require a #P-hard exact universal bound. Do not claim novel counting hardness
itself, approximation hardness, or that fixed-tree SHAP is hard under product P.
