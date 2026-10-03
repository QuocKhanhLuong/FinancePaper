"""Small common model-agnostic attribution game for explainer sensitivity."""
from dataclasses import replace
import numpy as np

from financepaper.data.schema import FEATURE_NAMES


def permutation_attributions(batch, references, predict, *, permutations=32, seed=42):
    """Original-field Shapley sampling, endpoint metadata held fixed.

    Whole one-hot groups move together. References and permutations are identical
    between partial/restored calls. Telescoping ensures completeness, not low
    Monte Carlo variance. This is not joint value/mask TreeSHAP.
    """
    if permutations < 1 or len(references) < 1:
        raise ValueError("positive permutation/reference budget required")
    rng = np.random.default_rng(seed)
    n, d = len(batch), len(FEATURE_NAMES)
    origins_t = np.array(batch.temporal_origins).reshape(batch.temporal.shape[1:])
    origins_s = np.array(batch.static_origins)
    phi = np.zeros((n, d))
    baseline_logits = np.zeros(n)
    for rep in range(permutations):
        ref = rep % len(references)
        t = np.broadcast_to(references.temporal[ref], batch.temporal.shape).copy()
        s = np.broadcast_to(references.static[ref], batch.static.shape).copy()
        current = replace(batch, temporal=t.copy(), static=s.copy())
        prev = predict(current)
        baseline_logits += prev / permutations
        order = rng.permutation(d)
        # Antithetic permutation next time is optional; fixed RNG is shared
        # between endpoints and models, never seeded from a restoration label.
        for j in order:
            t[:, origins_t == FEATURE_NAMES[j]] = batch.temporal[:, origins_t == FEATURE_NAMES[j]]
            s[:, origins_s == FEATURE_NAMES[j]] = batch.static[:, origins_s == FEATURE_NAMES[j]]
            nxt = predict(replace(batch, temporal=t.copy(), static=s.copy()))
            phi[:, j] += (nxt-prev)/permutations
            prev = nxt
    logits = predict(batch)
    residual = phi.sum(1)-(logits-baseline_logits)
    return dict(original=phi, baseline_logit=baseline_logits, input_logit=logits,
                residual=residual, valid=np.abs(residual) < .002,
                steps_used=np.full(n, permutations))
