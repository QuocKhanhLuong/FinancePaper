# Round 7 — executed analytical falsification results

Measured **2026-10-07**. Preregistration commit:
`3218761d42831f53ba9b6354df3e1e7c51d3e03f`. This round executes finite
counterexamples and LPs, **not trained financial models**. No Taiwan/Polish
outcome cache or new dataset was evaluated. Historical reported results were
read, not independently reproduced.

## 1. Frozen-teacher correction works, but is the matched baseline

The frozen toy has g=.5 or 1 with probabilities 2/3 and 1/3, q=.9 and
randomized verification pi=.1. The correct mean is 2/3. Values below come from
exact enumeration, not Monte Carlo training or fitted confidence intervals.

| Estimator | Expected mean | Analytic SE of mean | Interpretation |
|---|---:|---:|---|
| Misspecified imputer alone | .900000 | Not applicable | Bias +.233333 |
| Proposed corrected target / ordinary AIPW, n=1000 | .666667 | .032335 | Exact same estimator under either name |
| Oracle-q AIPW, n=1000 | .666667 | .023570 | Oracle reference; q not estimated |
| Verification-only mean, fixed 100 random audits | .666667 | .023570 | Same expected audit budget; fixed-count vs Bernoulli design differs |

For the misspecified correction, per-record target variance is 1.045556;
variance of its mean is .001045556. Fixed-100 verification mean variance is
.000555556. Ratio **1.882**: the worker's asserted 3.8x variance reduction and
`.667 +/- .008` are not supported. This is a homogeneous toy; it does not show
that AIPW generally loses or that covariate adjustment is useless. With a fixed
audit count and constant q, the corresponding normalized estimator can reduce
exactly to the audited mean. Either way there is no claimed toy improvement.

## 2. Joint learning exposes a real objective error, not a new theorem

W constant, H uniform {-1,+1}, refined representation g_theta=theta*H, f=0.
The actual conditional mean is zero for every theta, hence desired coherence
loss and its gradient are zero. AIPW target has mean zero but squared loss
theta^2/pi and gradient 2*theta/pi. All 12 preregistered theta/pi combinations
match these values; the scalar conditional-moment critic returns the desired
zero criterion.

At theta=1, pi=.1 the wrong gradient is **20**, despite correct gradient **0**.
Add the illustrative supervised representation loss (theta-1)^2 and lambda=1:
the correct joint objective chooses theta=1, outcome MSE=0; the naive squared
target chooses theta=1/11, outcome MSE=100/121=**.826446**. This demonstrates
unwanted shrinkage of a useful signal. It is not financial default Brier score,
nor a reproduction/failure claim about the cited SSL implementation.

Conditional-moment duality already supplies a population-level control. A
finite-class critic may have its own approximation/optimization errors; those
were not evaluated. The double-sampling issue is known in other fields and
does not itself justify a method-paper novelty claim.

## 3. Shared-adversary proposal has no claimed top-1 advantage

For compact box-constrained, normalized weights and continuous linear gaps,

    inf_w min_j gap_j(w) = min_j inf_w gap_j(w).

Proof: every gap is at least b=min_j inf_w gap_j, so the left side is >=b;
the pointwise minimum is no greater than any selected gap, giving <=b.
No union bound is involved. Here minima are attained; strict-positive stability
must treat the zero boundary explicitly.

| Gamma | Pairwise / correct ANY-competitor minimum | DIFFERENT ALL-competitors criterion |
|---:|---:|---:|
| 1.0 | 4.000000 | 4.000000 |
| 1.8 | .800000 | 3.111111 |
| 2.0 | 0, within 7e-16 | 3.000000 |
| 2.2 | -.363636 | 2.909091 |

Both correct formulations lose strict stability at Gamma=2. The apparent gain
requires replacing min over competitors with max, thereby changing the event.
These are worst-case **expected attribution gaps**, not probabilities that
verified top-k reasons survive. They cannot replace the historical target.

## 4. Budget allocation: real arithmetic, established method

At cost budget5, the two-stratum allocation changes the variance functional
from .162500 (uniform) to .101250 (known optimal allocation), a **37.6923%**
reduction. This is an analytic population-variance result for the specified
functional, not a measured predictive gain or new allocation algorithm.

At budget8, clipping the unconstrained rule spends only **5.888889**. It remains
feasible for a <=budget constraint, but is no longer optimal: re-solving the
multiplier uses the full budget with probabilities (1,.6) and reduces the
functional from .087031 to .082083. The worker's proposed regret rate and claim
of minimal arbitrary classifier excess risk have no supporting proof.

## 5. Information-certificate checks

A reason that is constant for every customer has zero entropy/information but
can be released with coverage1 and error0. The proposed vocabulary-size bound
would return0; its unstated assumptions are fatal. This refutes that proposed
bound, not valid Fano inequalities. Being deterministic **given** observed
information alone would not imply zero mutual information; our counterexample
is globally constant.

For independent uniform two-bit XOR, exact interventional Shapley enumeration
gives phi1=phi2=-.25 on (0,0)/(1,1), +.25 on (0,1)/(1,0). The contrast is
always0, contradicting the worker's suggested one-bit contrast signal. The
different, already executed round-four decoupling witness remains unchanged.

## Reproduction and receipts

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal python scripts/audit_round7_claims.py \
  --output outputs/method_pivot/round7-repeat/audit
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  uv run --frozen --extra temporal pytest -q
```

Actual hardware rechecked: Apple M4 Pro, 25,769,803,776 bytes RAM (24GiB),
macOS26.2/arm64. Python3.11.16, NumPy2.4.6, SciPy1.17.1. No GPU, downloads or
additional dependencies. Exact audit core runtime .016104s, process peak RSS
76.078MiB. The timing excludes imports and provenance serialization, is one
execution, and is **not a deployment-latency benchmark**.

- Focused checks: **10 passed** in .76s; `outputs/method_pivot/round7/checks.xml`.
- Final combined suite after the last code edit: **228 passed, 0 failed,
  0 skipped**, 3 SHAP pending-deprecation warnings, 18.08s.
- Raw test log/JUnit: `outputs/method_pivot/round7/full_pytest.{log,xml}`.
- Exact results/provenance: `outputs/method_pivot/round7/audit/{results,manifest}.json`.
- Result SHA256: `dbb24c540424d88f51956788e396c0a51fc7de1a6f614adcc39e603cb762094a`.
- Script SHA256: `191d8fe3836b1d3adc4ad3545aed51b2c26924bf8f9ee9f0b87dc0397ced72d1`.
- Protocol SHA256: `80f57f3a5cf245750a210575f59602c8c685812d24449058e92614c32343c6be`.

Run manifest records the preregistration HEAD plus then-untracked audit code;
source hashes identify the executed code subsequently committed. All generated
artifacts remain ignored. New runs refuse existing output directories.
Test passes establish software/algebra checks, not scientific generalization.

**Decision:** no proposed new method passed. See [next decision](ROUND7_NEXT_DECISION.md).
