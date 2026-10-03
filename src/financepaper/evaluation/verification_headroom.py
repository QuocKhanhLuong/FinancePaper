"""Evaluator-only one-query bounds. Never import this as an inference policy."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


def actual_reveal_states(full, artificial):
    """Enumerate STOP and single actual reveals; naturally unknown cells stay NaN."""
    full, artificial = np.asarray(full, float), np.asarray(artificial)
    if artificial.dtype != bool or artificial.shape != full.shape or full.ndim != 2:
        raise ValueError("Aligned 2-D values and boolean verification mask required")
    if np.any(artificial & ~np.isfinite(full)):
        raise ValueError("Cannot reveal a naturally unknown cell")
    partial = np.where(artificial, np.nan, full)
    owners, fields = [np.arange(len(full))], [np.full(len(full), -1)]
    for j in range(full.shape[1]):
        idx = np.flatnonzero(artificial[:, j])
        owners.append(idx)
        fields.append(np.full(len(idx), j))
    owner, field = np.concatenate(owners), np.concatenate(fields)
    values, remaining = partial[owner].copy(), artificial[owner].copy()
    queried = np.flatnonzero(field >= 0)
    values[queried, field[queried]] = full[owner[queried], field[queried]]
    remaining[queried, field[queried]] = False
    return owner, field, values, remaining


def optimize_oracle(sets, survived, owners, fields, *, alpha, budget, objective):
    """Exact finite-family, empirical-risk oracle, using evaluation truth explicitly.

    Includes implicit abstention. Returns selected state index, -1 for abstention.
    No individual reason can be edited out of a terminal rule's output.
    """
    sets, survived = np.asarray(sets), np.asarray(survived)
    owners, fields = np.asarray(owners), np.asarray(fields)
    if (sets.dtype != bool or survived.dtype != bool or sets.ndim != 2
            or survived.ndim != 2 or sets.shape[1] != survived.shape[1]
            or owners.shape != (len(sets),) or fields.shape != owners.shape
            or np.any(owners < 0) or np.any(owners >= len(survived))
            or budget not in (0, 1) or not 0 < alpha < 1
            or objective not in ("reasons", "customers")):
        raise ValueError("Invalid oracle inputs")
    n, g = survived.shape
    # Equivalent sets have identical risk/coverage; prefer STOP then first field.
    unique = {}
    for i in np.lexsort((fields, owners)):
        if not sets[i].any() or (budget == 0 and fields[i] >= 0):
            continue
        unique.setdefault((int(owners[i]), sets[i].tobytes()), i)
    idx = np.asarray(list(unique.values()), dtype=int)
    result = np.full(n, -1, int)
    if len(idx) == 0:
        return result, {"optimal": True, "variables": 0, "objective": objective}
    count = sets[idx].sum(1)
    failed = (sets[idx] & ~survived[owners[idx]]).sum(1)
    query = fields[idx] >= 0
    # Integer-scaled lexicographic objective; lower priorities cannot outweigh
    # a single unit of the preceding priority over the entire cohort.
    if objective == "reasons":
        utility = count * (n + 1)**2 + (n + 1) - query
    else:
        utility = np.full(len(idx), (n*g + 1)*(n + 1)) + count*(n + 1) - query
    assign = coo_matrix((np.ones(len(idx)), (owners[idx], np.arange(len(idx)))),
                        shape=(n, len(idx))).tocsr()
    risk = coo_matrix((failed-alpha*count)[None, :]).tocsr()
    constraints = LinearConstraint(vstack([assign, risk]),
                                   np.full(n+1, -np.inf), np.r_[np.ones(n), 0.])
    fitted = milp(-utility.astype(float), integrality=np.ones(len(idx)),
                  bounds=Bounds(0, 1), constraints=constraints,
                  options={"mip_rel_gap": 0., "time_limit": 120.})
    if not fitted.success or fitted.status != 0:
        raise RuntimeError(f"Oracle did not prove optimality: {fitted.message}")
    chosen = idx[fitted.x > .5]
    result[owners[chosen]] = chosen
    c = sets[chosen].sum()
    f = (sets[chosen] & ~survived[owners[chosen]]).sum()
    if len(np.unique(owners[chosen])) != len(chosen) or f > alpha*c + 1e-8:
        raise RuntimeError("Invalid oracle solution")
    return result, {"optimal": True, "variables": len(idx), "objective": objective,
                    "released": int(c), "failed": int(f), "gap": float(fitted.mip_gap)}


def selected_outputs(sets, actions):
    out = np.zeros((len(actions), sets.shape[1]), bool)
    emitted = actions >= 0
    out[emitted] = sets[actions[emitted]]
    return out


def paired_intervals(selected, survived, initial, query, weights):
    """Customer bootstrap ratios; caller shares weights across conditions/methods."""
    count, failed = selected.sum(1), (selected & ~survived).sum(1)
    inputs = {"risk": (failed, count), "reason_coverage": (count, initial.sum(1)),
              "customer_ge1": (count > 0, np.ones(len(count))),
              "customer_ge2": (count >= 2, np.ones(len(count))),
              "mean_reasons": (count, np.ones(len(count))),
              "query_fraction": (query, np.ones(len(count)))}
    row, samples = {"reasons": int(count.sum()), "failures": int(failed.sum())}, {}
    for key, (num, den) in inputs.items():
        num, den = np.asarray(num, float), np.asarray(den, float)
        d = weights @ den
        sample = np.divide(weights@num, d, out=np.full(len(weights), np.nan), where=d > 0)
        finite = sample[np.isfinite(sample)]
        lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        row.update({key: float(num.sum()/den.sum()) if den.sum() else np.nan,
                    key+"_lo": float(lo), key+"_hi": float(hi)})
        samples[key] = sample
    return row, samples
