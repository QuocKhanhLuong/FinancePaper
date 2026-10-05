# Additional strong reduction control, frozen before its execution

2026-10-05, after runs 01/02, before any result of this control. This is a disclosed
post-pilot falsification control, **not** an originally preregistered comparison.
It changes no selected mechanism, predictor, data, seed or completion law.

Reason: [FourierSHAP](https://papers.nips.cc/paper_files/paper/2025/hash/1b331c20064e37e204a5bcd12481bfac-Abstract-Conference.html)
was reviewed but not run. The present four-atom independent law has an especially
simple known orthogonal product basis. Testing only repeated point explanations
could conceal a much stronger direct-moment baseline.

## Baseline, not new method

Re-express the **same compiled attribution map** in tensor products of the
four-point Walsh basis, orthonormal under uniform atom weights. For interval
indicator I_j, its four basis coefficients are H' I_j / 4, with H a 4x4 Hadamard
matrix whose first column is one. Expand the path-product terms, merge identical
frequency keys, and retain vector coefficients a_s. Then

```
E[phi] = a_empty
Cov(phi) = sum_{s != empty} a_s a_s'.
```

These are the standard orthogonal expansion/Parseval identities. This is a
**Fourier-basis moment adaptation**, not the authors' official FourierSHAP
implementation or an attributed reproduction of their reported benchmark.
It receives the same compiled function as the candidate, sharing compilation
cost. Its purpose is to test whether a standard basis change solves this query
better than the proposed quadratic rectangle intersections.

## Frozen experiment

Use all six existing synthetic model specifications, seeds 11/22/33, four query
rows each, the same seven hidden fields and four equiprobable atoms. Recreate
models deterministically from the frozen configuration (no new fit selection).
Verify regenerated point attributions against stock TreeSHAP and moments against
the specialized exact operator; cross-reference the run-02 oracle receipts.
Independent small coalition fixtures validate the basis separately.

One warm-up + five repetitions; time both query operators in the same process.
Record expansion count, retained frequency count, source/config hashes, whole
process memory and compilation separately. Ten-minute total/60-second stage
budgets remain; at most 200,000 frequency keys. No financial dataset is opened.

If the generic Fourier-basis control matches accuracy and is materially cheaper,
the candidate's practical necessity claim fails on this workload. Do not rescue
it by changing the law, increasing tree depth, or searching for a favorable case.
If slower, this does not establish novelty or general superiority: the control
applies only to the predeclared independent four-atom law. Correlated q and
arbitrary finite laws remain outside this baseline implementation.
