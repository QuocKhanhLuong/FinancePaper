"""Stable-Core v2: variable-size reasons, calibrated against separate verification.

Historical stable_core.py and whole-explanation event definitions are untouched.
"""
from dataclasses import dataclass
from itertools import product
import hashlib

import numpy as np

SUPPORT_GRID = (0., .5, .625, .75, .875, 1.)
LOWER_GRID = (.01, .02, .05)
METHODS = ("release_all", "whole_legacy2", "whole_mc1", "whole_mc2", "whole_mc_all",
           "rank", "variance", "frequency", "strength", "lower_quantile", "entropy",
           "stable_donor", "stable_conditional", "stable_both")


def rank_membership(phi, hidden, k=2):
    phi, hidden = np.asarray(phi), np.asarray(hidden, bool)
    if phi.ndim == 2:
        phi = phi[:, None, :]
    above = (phi[..., None, :] > phi[..., :, None] + 1e-6) & ~hidden[:, None, None, :]
    return (above.sum(-1) < k) & ~hidden[:, None, :]


def evidence(current, donor, conditional, hidden, probability):
    """Serving evidence accepts no restored values or reason-failure labels."""
    current, donor, conditional = map(lambda x: np.asarray(x, float), (current, donor, conditional))
    hidden, probability = np.asarray(hidden), np.asarray(probability, float)
    if (current.ndim != 2 or donor.ndim != 3 or conditional.shape != donor.shape
            or donor.shape[::2] != current.shape or hidden.shape != current.shape
            or hidden.dtype != bool or probability.shape != (len(current),)):
        raise ValueError("Current/completion/mask shapes disagree")
    if not all(np.isfinite(x).all() for x in (current, donor, conditional, probability)):
        raise ValueError("Nonfinite current evidence")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid prediction probabilities")
    result = {"phi": current, "hidden": hidden, "candidate": (current > .01) & ~hidden}
    p = np.clip(probability, 1e-9, 1-1e-9)
    result["entropy"] = -(p*np.log(p) + (1-p)*np.log1p(-p))/np.log(2)
    for name, draws in (("donor", donor), ("conditional", conditional)):
        rank = rank_membership(draws, hidden)
        meaningful = (draws > .01) & ~hidden[:, None, :]
        result[name + "_sign"] = (draws > 1e-6).mean(1)
        result[name + "_rank"] = rank.mean(1)
        result[name + "_meaningful"] = meaningful.mean(1)
        result[name + "_ranked"] = (meaningful & rank).mean(1)
        result[name + "_quantiles"] = np.quantile(draws, [.1, .5, .9], axis=1).transpose(1, 0, 2)
        result[name + "_variance"] = draws.var(1)
        if name == "donor":
            result["donor_survive_meaningful"] = meaningful
            result["donor_survive_ranked"] = meaningful & rank
            result["donor_survive_legacy"] = (draws > 1e-6) & rank
    return result


def concatenate(blocks):
    return {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}


def verified(restored, hidden, target):
    restored = np.asarray(restored, float)
    if restored.shape != np.shape(hidden) or not np.isfinite(restored).all():
        raise ValueError("Invalid verification attribution")
    if target not in ("meaningful", "ranked"):
        raise ValueError("Unknown verification definition")
    survives = (restored > .01) & ~np.asarray(hidden, bool)
    return survives if target == "meaningful" else survives & rank_membership(restored, hidden)[:, 0]


def strongest(ev, count, *, require_exact=False):
    cand, phi = ev["candidate"], ev["phi"]
    wanted = np.full(len(phi), count) if np.isscalar(count) else np.asarray(count)
    order = np.argsort(-np.where(cand, phi, -np.inf), axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable")
    selected = cand & (ranks < wanted[:, None])
    if require_exact:
        selected &= (cand.sum(1) >= wanted)[:, None]
    return selected


def matched_random(ev, counts, ids, *, seed=20261005):
    result = np.zeros_like(ev["candidate"])
    for i, identity in enumerate(ids):
        value = int.from_bytes(hashlib.sha256(f"{seed}/{identity}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(value)
        pool = np.flatnonzero(ev["candidate"][i])
        result[i, rng.choice(pool, int(counts[i]), replace=False)] = True
    return result


def selected_set(ev, method, parameters, target):
    cand = ev["candidate"]
    if parameters is None:
        return np.zeros_like(cand)
    if method == "release_all":
        return cand.copy()
    if method.startswith("stable_"):
        family = method[7:]
        families = ("donor", "conditional") if family == "both" else (family,)
        sign = np.minimum.reduce([ev[f + "_sign"] for f in families])
        rank = np.minimum.reduce([ev[f + "_rank"] for f in families])
        lower = np.minimum.reduce([ev[f + "_quantiles"][:, 0] for f in families])
        s, r, q = parameters
        return cand & (sign >= s) & (rank >= r) & (lower > q)
    threshold = parameters[0]
    if method.startswith("whole_"):
        chosen = cand if method == "whole_mc_all" else strongest(ev, 1 if method == "whole_mc1" else 2, require_exact=True)
        survived = ev["donor_survive_legacy" if method == "whole_legacy2" else "donor_survive_" + target]
        score = (chosen[:, None, :] & ~survived).any(2).mean(1)
        return chosen & (score <= threshold)[:, None]
    if method == "entropy":
        return cand & (ev["entropy"] <= threshold)[:, None]
    scores = {"rank": ev["donor_rank"], "variance": ev["donor_variance"],
              "frequency": ev["donor_" + target], "strength": ev["phi"],
              "lower_quantile": ev["donor_quantiles"][:, 0]}
    return cand & ((scores[method] <= threshold) if method == "variance" else (scores[method] >= threshold))


def make_grids(current_design):
    """Unlabelled, disjoint revision-calibration pool fixes continuous score grids."""
    grids = {m: [(float(v),) for v in np.linspace(0, 1, 9)] for m in METHODS}
    grids["release_all"] = [()]
    for m in ("stable_donor", "stable_conditional", "stable_both"):
        grids[m] = list(product(SUPPORT_GRID, SUPPORT_GRID, LOWER_GRID))
    continuous = {"variance": current_design["donor_variance"], "strength": current_design["phi"],
                  "lower_quantile": current_design["donor_quantiles"][:, 0], "entropy": current_design["entropy"]}
    for method, scores in continuous.items():
        values = scores[current_design["candidate"]] if scores.ndim == 2 else scores
        cuts = np.unique(np.r_[np.quantile(values, np.linspace(0, 1, 21)) if len(values) else 0, -np.inf, np.inf])
        grids[method] = [(float(c),) for c in cuts]
    return grids


@dataclass(frozen=True)
class ReasonSetPolicy:
    method: str
    target: str
    alpha: float
    conservative: bool
    parameters: tuple | None

    def release(self, current_evidence):
        return selected_set(current_evidence, self.method, self.parameters, self.target)


def fit_policy(ev, survived, environment, cluster_ids, *, method, grid, target, alpha, conservative=False):
    survived, environment, clusters = np.asarray(survived), np.asarray(environment), np.asarray(cluster_ids)
    if survived.dtype != bool or survived.shape != ev["candidate"].shape:
        raise ValueError("Verification labels must be separate aligned boolean arrays")
    if environment.shape != (len(survived),) or clusters.shape != environment.shape or not 0 < alpha < 1:
        raise ValueError("Invalid calibration metadata/risk target")
    groups = []
    for env in np.unique(environment):
        block = environment == env
        _, inv = np.unique(clusters[block], return_inverse=True)
        groups.append((block, inv, np.bincount(inv)))
    if not groups:
        raise ValueError("No calibration environments")
    best = ReasonSetPolicy(method, target, alpha, conservative, None)
    objective = (-1, -1)
    for parameters in grid:
        released = selected_set(ev, method, parameters, target)
        counts, failures = released.sum(1), (released & ~survived).sum(1)
        valid = True
        for block, inv, sizes in groups:
            if counts[block].sum() == 0:
                valid = False
                break
            if conservative:
                d = (failures[block] - alpha*counts[block]) / survived.shape[1]
                cluster_d = np.bincount(inv, weights=d) / sizes
                ok = cluster_d.mean() + np.sqrt(np.log(len(grid)*len(groups)/.05)/(2*len(sizes))) <= 0
            else:
                ok = failures[block].sum() / counts[block].sum() <= alpha
            if not ok:
                valid = False
                break
        key = (int(counts.sum()), int((counts > 0).sum()))
        if valid and key > objective:
            best, objective = ReasonSetPolicy(method, target, alpha, conservative, parameters), key
    return best


def inference_output(ev, policy, group_names, raw_probability, calibrated_probability):
    """Public serving schema, containing no verification-only information."""
    release = policy.release(ev)
    outputs = []
    for i, selected in enumerate(release):
        cand = ev["candidate"][i]
        outputs.append({"risk": {"prob_raw": float(raw_probability[i]), "prob_calibrated": float(calibrated_probability[i])},
            "stable_reasons": [group_names[j] for j in np.flatnonzero(selected)],
            "uncertain_reasons": [group_names[j] for j in np.flatnonzero(cand & ~selected)],
            "status": "NONE" if not selected.any() else "ALL_CANDIDATES" if np.array_equal(selected, cand) else "PARTIAL",
            "policy": {"method": policy.method, "target": policy.target, "risk_budget": policy.alpha,
                       "calibration": "conservative" if policy.conservative else "empirical",
                       "parameters": policy.parameters, "individual_guarantee": False},
            "evidence": {f: {group_names[j]: {"sign_support": float(ev[f + "_sign"][i,j]),
                "top2_support": float(ev[f + "_rank"][i,j]),
                "magnitude_quantiles": ev[f + "_quantiles"][i,:,j].tolist()} for j in np.flatnonzero(cand)}
                for f in ("donor", "conditional")}})
    return outputs
