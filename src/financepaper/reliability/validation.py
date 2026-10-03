"""Additive validation utilities; frozen historical metric code is untouched."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import beta
from financepaper.data.schema import FEATURE_NAMES

TAIWAN_GROUPS = {
    "repayment": ["PAY_0", *[f"PAY_{j}" for j in range(2, 7)]],
    "balance": [f"BILL_AMT{j}" for j in range(1, 7)],
    "payment": [f"PAY_AMT{j}" for j in range(1, 7)],
    **{name: [name] for name in ("LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE")},
}
POLISH_NAMES = tuple(f"Attr{j}" for j in range(1, 65))
POLISH_GROUPS = {name: [f"Attr{j}" for j in ids] for name, ids in {
    "profitability": [1,6,7,11,13,14,18,19,22,23,24,31,35,39,42,45,48,49,56,58],
    "liquidity": [3,4,5,28,37,40,46,50,55,57],
    "capital_leverage": [2,8,10,17,25,38,51,53,54,59],
    "debt_coverage": [12,15,16,26,27,30,33,34,41,63],
    "working_capital_cycle": [20,32,43,44,47,52,62],
    "turnover_growth": [9,21,36,60,61,64], "size": [29],
}.items()}


def aggregate_groups(phi, hidden, names, groups, *, observed_only=True):
    """Sum signed contributions with the same preverification mask at all draws."""
    phi, hidden = np.asarray(phi, float), np.asarray(hidden, bool)
    fields = sum(groups.values(), [])
    if sorted(fields) != sorted(names) or len(set(fields)) != len(names):
        raise ValueError("groups must partition all original fields exactly once")
    if phi.shape[0] != len(hidden) or phi.shape[-1] != len(names):
        raise ValueError("attribution dimensions disagree")
    values, masks = [], []
    for members in groups.values():
        idx = [names.index(f) for f in members]
        available = ~hidden[:, idx]
        if phi.ndim == 3:
            available = available[:, None, :]
        values.append((phi[..., idx] * available if observed_only else phi[..., idx]).sum(-1))
        masks.append(hidden[:, idx].all(-1))
    return np.stack(values, -1), np.stack(masks, -1)


def revision_arrays(before, after, hidden, *, k=3, magnitude=.01, rank_epsilon=1e-6):
    """Vectorized frozen event; after is [N,D] or [N,K,D]. No truth needed for MC."""
    before, after, hidden = np.asarray(before), np.asarray(after), np.asarray(hidden, bool)
    if before.ndim != 2 or hidden.shape != before.shape or not 1 <= k <= before.shape[1]:
        raise ValueError("invalid reason dimensions/k")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("nonfinite attributions")
    if magnitude < 0 or rank_epsilon < 0:
        raise ValueError("negative reason tolerances")
    squeeze = after.ndim == 2
    if squeeze:
        after = after[:, None, :]
    if after.ndim != 3 or after.shape[0] != len(before) or after.shape[2] != before.shape[1]:
        raise ValueError("restored/completion arrays disagree")
    eligible = ((before > magnitude) & ~hidden).sum(1) >= k
    order = np.argsort(-np.where((before > magnitude) & ~hidden, before, -np.inf),
                       axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(after, order[:, None, :], axis=2)
    rank = ((after[..., None] > selected[:, :, None, :] + rank_epsilon)
            & ~hidden[:, None, :, None]).sum(2)
    sign, exits = selected <= 1e-6, rank >= k
    event = (sign | exits).any(2)
    result = dict(eligible=eligible, reasons=order, event=event, sign=sign.mean(2),
                  rank=exits.mean(2), num_revised=(sign | exits).sum(2))
    if squeeze:
        for key in ("event", "sign", "rank", "num_revised"):
            result[key] = result[key][:, 0]
    return result


def variants(current, names, groups, *, sensitivity=True):
    """Current-only transformed evidence. Verification is deliberately not input."""
    p, h, cp = current["phi"], current["hidden"], current["completion_phi"]
    yield "feature3", p, h, cp, 3, None
    modes = ("group", "whole_group") if sensitivity else ("group",)
    for mode in modes:
        a, mask = aggregate_groups(p, h, names, groups, observed_only=mode == "group")
        ca, _ = aggregate_groups(cp, h, names, groups, observed_only=mode == "group")
        for k in (1, 2, 3):
            yield f"{mode}{k}", a, mask, ca, k, mode


class ValidationDonors:
    """The historical donor algorithm generalized to named complete references.

    Natural unknowns may remain NaN: fill_mask denotes only artificial verification
    cells. This class never accepts restored values or target labels.
    """
    def __init__(self, reference, categorical=(), *, seed=42, neighbours=32, max_reference=2048):
        if reference.isna().any().any() or not np.isfinite(reference.to_numpy()).all():
            raise ValueError("complete finite training-only donors required")
        if len(reference) < neighbours or not reference.index.is_unique:
            raise ValueError("insufficient/duplicate donor IDs")
        self.names = tuple(reference.columns)
        pos = np.sort(np.random.default_rng(seed).choice(len(reference), min(len(reference), max_reference), replace=False))
        self.reference = reference.iloc[pos].copy()
        self.scale = np.maximum(np.diff(reference.quantile([.25, .75]).to_numpy(), axis=0)[0], 1.)
        self.categorical = np.array([f in categorical for f in self.names])
        self.neighbours = min(neighbours, len(pos))

    def sample(self, partial, fill_mask, *, k=16, seed=42):
        if tuple(partial.columns) != self.names or k not in (1, 2, 4, 8, 16):
            raise ValueError("invalid completion schema/budget")
        x, fill = partial.to_numpy(float), np.asarray(fill_mask)
        if fill.dtype != bool or fill.shape != x.shape or np.isfinite(x[fill]).any():
            raise ValueError("artificial hidden values must be erased before completion")
        observed = np.isfinite(x)
        ref = self.reference.to_numpy(float)
        result = np.repeat(x[:, None, :], k, axis=1)
        selected = np.empty((len(x), k), int)
        rngs = [np.random.default_rng(seed), np.random.default_rng(seed + 100000)]
        for i, row in enumerate(x):
            distance = np.minimum(np.abs(ref - np.nan_to_num(row)) / self.scale, 5.)
            distance[:, self.categorical] = ref[:, self.categorical] != row[self.categorical]
            distance = np.where(observed[i], distance, 0).sum(1)
            nearest = np.argsort(distance, kind="stable")[:self.neighbours]
            chosen = np.concatenate([rng.choice(nearest, 8, replace=True) for rng in rngs[:1 if k <= 8 else 2]])[:k]
            selected[i] = chosen
            result[i] = np.where(fill[i], ref[chosen], row)
        return result, self.reference.index.to_numpy()[selected]


def observable_strata(hidden):
    return np.digitize(np.asarray(hidden, bool).mean(1), [.15, .30], right=True)


def assigned_environment(ids, environments, seed):
    unique = np.sort(np.unique(ids))
    chosen = np.random.default_rng(seed).choice(environments, len(unique))
    return dict(zip(unique.tolist(), chosen.tolist()))


@dataclass(frozen=True)
class ValidationPolicy:
    family: str
    thresholds: tuple
    conservative: bool
    risk_target: float

    def release(self, probability, eligible, strata):
        probability, eligible, strata = np.asarray(probability), np.asarray(eligible, bool), np.asarray(strata)
        if probability.shape != eligible.shape or strata.shape != probability.shape or not np.isfinite(probability).all():
            raise ValueError("invalid current-only release inputs")
        out = np.zeros(len(probability), bool)
        for j, threshold in enumerate(self.thresholds):
            if threshold is not None:
                out |= eligible & (probability <= threshold) & ((strata == j) if self.family == "stratified" else True)
        return out


def fit_validation_policy(frame, *, family, risk_target, conservative, environments, delta=.05):
    """frame is release calibration ONLY; environment absent from serving API."""
    if family not in ("pooled", "stratified", "robust"):
        raise ValueError("unknown policy")
    family_size = 3 if family == "stratified" else len(environments) if family == "robust" else 1
    subsets = range(3) if family == "stratified" else [None]
    thresholds = []
    for stratum in subsets:
        f = frame if family == "robust" else frame[frame.assigned]
        if stratum is not None:
            f = f[f.stratum == stratum]
        if len(f) < (50 if stratum is not None else 1):
            thresholds.append(None)
            continue
        best, best_n = None, 0
        for t in np.linspace(0, 1, 101):
            ok = True
            blocks = [f[f.condition == c] for c in environments] if family == "robust" else [f]
            for block in blocks:
                selected = block.eligible.to_numpy(bool) & (block.probability.to_numpy() <= t)
                n = int(selected.sum())
                events = int(block.event.fillna(1).to_numpy()[selected].sum())
                # Empty environment has no observed risk; fail closed for certification.
                risk = 1. if n == 0 or events == n else (float(beta.ppf(1-delta/(101*family_size), events+1, n-events)) if conservative else events/n)
                if risk > risk_target:
                    ok = False
                    break
            released = int((f.eligible & (f.probability <= t)).sum())
            if ok and released > best_n:
                best, best_n = float(t), released
        thresholds.append(best)
    return ValidationPolicy(family, tuple(thresholds), conservative, risk_target)
