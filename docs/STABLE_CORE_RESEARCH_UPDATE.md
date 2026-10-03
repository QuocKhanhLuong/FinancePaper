# Stable-Core: novelty check before the operational experiment

Search updated 2026-10-03. This supplements the nearest-work table in
[the initial specification](STABLE_CORE_EXPLANATION_SPEC.md). Primary-source
queries covered calibrated partial explanations, selective attribution, conformal
feature selection, SHAP false discovery, verification calibration and uncertainty
sets. No architecture search is justified by these sources.

The strongest generic precedent is **Learn then Test** (Angelopoulos, Bates,
Candès, Jordan and Lei): independent calibration and hypothesis testing of
set-valued outputs, including false-discovery control, are established techniques.
[Author-hosted paper](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/ltt.pdf),
[original preprint record](https://arxiv.org/abs/2110.01052).
Our micro failure ratio is not their mean per-instance false-discovery proportion.
We must define and calibrate the actual requested estimand, not inherit a theorem
by calling the implementation conformal.

[Laberge et al., JMLR 24(364), 2023](https://jmlr.org/papers/v24/23-0149.html)
already extract partial, mutually consistent attribution statements across a
Rashomon set. Our completions vary missing fields of one model, rather than the
model itself; stable-subset presentation is not new.

[Löfström, Hjort and Löfström, COPA 2026](https://proceedings.mlr.press/v329/lofstrom26a.html)
filter candidate explanatory rule conditions for distributional support. They
explicitly distinguish empirical guard scores from finite-sample guarantees.
This is a close operational comparator for suppressing unsupported statements,
although the event tested here is survival after actual hidden-value restoration.

An additional 2026 study, [Explanation audits reveal silent failures of machine
learning models under distribution shift](https://link.springer.com/article/10.1007/s44163-026-02012-6),
distinguishes pointwise explanation volatility, aggregate attribution drift and
predictive error. It finds confidence useful for its prediction-error target and
warns that stable explanations can consistently rely on inappropriate evidence.
This does not establish detection of our reason-verification endpoint. It does
invalidate broad rhetoric that explanation uncertainty is universally superior.

**Pre-experiment decision: Option 2 is the defensible default.** Stable-Core is an
operational strategy within the verification framework. Thresholding sign/rank/
magnitude evidence, returning subsets and calibrating set risks are known ideas.
The potential contribution is the concrete reason-level verification target,
paired missing-information benchmark and calibrated communication policy, if the
experiments support them. No exact equivalent was established by this targeted
search; absence from the search is not proof of novelty.

The experiment must beat rank/strength/frequency baselines and controls matched
per customer on set size. Otherwise report the simpler strategy or negative result.
Freddie transfer remains required for a large-scale generality claim.
