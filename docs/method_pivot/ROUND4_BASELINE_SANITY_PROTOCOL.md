# Round 4: baseline sanity protocol (before execution)

2026-10-05. This is a synthetic correctness/falsification control, not a new
learning method and not financial validation. It precedes any new financial
outcome read in round 4. The separate development protocol requires review.

## Purpose

Implement independently the finite-completion prediction controls defined in
ROUND4_MVU_SOURCE_AUDIT.md. Test that full predictive certainty need not determine
an observed-feature attribution ranking. No official MVU code is copied or run.

## Fixed analytical example

Use margin f(x1,x2,h)=x1*h+x2*(1-h), risk sigmoid(f), current observed x1=x2=1,
H~Bernoulli(1/4), modal imputation h=0, fixed point-reference b=(0,0,1/4).
Explain the **margin**, using exhaustive three-feature interventional Shapley
coalitions at this point reference. No trained model, donor, fitted distribution,
or financial claim. Features x1 and x2 are observed and eligible; h is hidden.
A top-1 positive reason is the diagnostic, with epsilon .01 (margin units only).
This example must not be compared numerically with normalized repository rates.

Analytical attributions are

```text
phi1 = x1*(h+1/4)/2
phi2 = x2*(2-h-1/4)/2
phih = (h-1/4)*(x1-x2)/2
```

At x1=x2=1 the margin is always 1 and phih=0. Restoring h=1 switches the largest
observed positive attribution with a nonzero margin, not a numerical tie.
Compute exact probabilities by enumerating h in {0,1} with weights {3/4,1/4}.
An additive control f_add=x1+x2 ignores h and must have invariant observed
attributions. A second simple probability list {.51,.99} checks that class
confidence one is weaker than probability stability.

## Implementation contracts

The current-information statistics API accepts only current probabilities and a
finite completion-probability array [N,K]; it cannot accept restored input,
restored probabilities, explanations or outcomes. Reject empty K, mismatched
shape, nonfinite or out-of-range probabilities. No clipping invalid input.
Use >=.5 for probability ties and label 1 on an exactly tied completion vote.
Return hard confidence, soft confidence, current-action agreement, population
variance, standard deviation and fixed .1/.5/.9 quantiles. Identify MC frequency
as a statistic, not a calibrated posterior. Determinism and completion-order
invariance are mandatory. K=1 is valid but cannot estimate dispersion reliably.

## Stop criteria and evidence

Stop on any exhaustive-coalition disagreement, failure of attribution efficiency,
incorrect sign/ranking, use of hidden truth in the inference signature, or invalid
range. No seed search is needed for exact enumeration. Save actual outputs, code
hashes and test logs under ignored outputs/method_pivot/round4. Report measured
values separately from the analytical expectations above. Success validates the
control and a counterexample, **not novelty, empirical prevalence or a method win**.
