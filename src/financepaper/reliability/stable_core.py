"""Current-only stable-reason evidence and separate verification calibration.

Empirical completion frequencies are not posterior probabilities. Calibration
targets per-reason failure; whole-explanation failure is a different estimand.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

THRESHOLD_GRID = (0., .5, .625, .75, .875, 1.)


def _attributions(values, ndim):
    result = np.asarray(values, dtype=float)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError("Attributions must be finite aligned arrays")
    return result


def _available(hidden, shape):
    hidden = np.asarray(hidden)
    if hidden.shape != shape or hidden.dtype != bool:
        raise ValueError("Group availability must be a boolean mask")
    return ~hidden


def _survival(phi, available, k, epsilon):
    # Axis: customer, completion, candidate, competing group.
    above = (phi[..., None, :] > phi[..., :, None] + epsilon) & available[:, None, None, :]
    top = above.sum(-1) < k
    positive = phi > epsilon
    return positive & available[:, None, :], top & available[:, None, :]


@dataclass(frozen=True)
class StableEvidence:
    candidates: np.ndarray
    sign_frequency: np.ndarray
    rank_frequency: np.ndarray
    joint_frequency: np.ndarray
    attribution_quantiles: np.ndarray

    def __post_init__(self):
        shape = self.candidates.shape
        if len(shape) != 2 or self.candidates.dtype != bool:
            raise ValueError("Candidates must be a boolean customer/group array")
        for values in (self.sign_frequency, self.rank_frequency, self.joint_frequency):
            if values.shape != shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Completion frequencies must be finite and between zero and one")
        if self.attribution_quantiles.shape != (shape[0], 3, shape[1]) or not np.isfinite(self.attribution_quantiles).all():
            raise ValueError("Invalid attribution quantiles")


def completion_evidence(current, completions, group_hidden, *, k=2, minimum=.01, epsilon=1e-6):
    """No restored data accepted. Inputs must share raw-logit attribution reference."""
    current, completions = _attributions(current, 2), _attributions(completions, 3)
    if completions.shape[0] != len(current) or completions.shape[2] != current.shape[1] or not completions.shape[1]:
        raise ValueError("Completion dimensions do not match current attribution")
    if not 1 <= k <= current.shape[1] or minimum < 0 or epsilon < 0:
        raise ValueError("Invalid reason selection definition")
    available = _available(group_hidden, current.shape)
    candidates = np.zeros(current.shape, bool)
    order = np.argsort(-current, axis=1, kind="stable")
    for i, ranking in enumerate(order):
        chosen = ranking[(available[i] & (current[i] > minimum))[ranking]][:k]
        candidates[i, chosen] = True
    sign, rank = _survival(completions, available, k, epsilon)
    return StableEvidence(candidates, sign.mean(1), rank.mean(1), (sign & rank).mean(1),
                          np.quantile(completions, [.1, .5, .9], axis=1).transpose(1, 0, 2))


def verified_survival(restored, group_hidden, *, k=2, epsilon=1e-6):
    """Verification-only evaluator; never used to create serving evidence."""
    restored = _attributions(restored, 2)
    sign, rank = _survival(restored[:, None, :], _available(group_hidden, restored.shape), k, epsilon)
    return (sign & rank)[:, 0, :]


@dataclass(frozen=True)
class StableCorePolicy:
    tau_sign: float | None
    tau_rank: float | None
    alpha: float
    family: str

    def release(self, evidence: StableEvidence) -> np.ndarray:
        if self.tau_sign is None or self.tau_rank is None:
            return np.zeros(evidence.candidates.shape, bool)
        return evidence.candidates & (evidence.sign_frequency >= self.tau_sign) & (evidence.rank_frequency >= self.tau_rank)


def fit_stable_policy(evidence, survived, customer_ids, environments, *, alpha, family="robust", conservative=True):
    """Use independent release calibration only; one row/customer/environment.

    A bounded customer-level Hoeffding criterion avoids treating two reasons as
    independent observations. No distribution-shift guarantee is implied.
    """
    if family not in ("robust", "pooled") or not 0 < alpha < 1:
        raise ValueError("Invalid calibration policy")
    survived = np.asarray(survived)
    ids, environments = np.asarray(customer_ids), np.asarray(environments)
    shape = evidence.candidates.shape
    if survived.shape != shape or survived.dtype != bool or ids.shape != (shape[0],) or environments.shape != ids.shape:
        raise ValueError("Calibration labels and customer/environment identities must align")
    if not len(ids) or evidence.candidates.sum(1).max(initial=0) > 2:
        raise ValueError("Frozen calibration bound supports up to two current reasons")
    envs = np.unique(environments)
    for env in envs:
        block_ids = ids[environments == env]
        if len(np.unique(block_ids)) != len(block_ids):
            raise ValueError("Repeated customer inside environment; cluster before calibration")
    # Pooled empirical data can repeat customers across environments. Conservative
    # pooled calibration would need a different cluster bound, so is disallowed.
    if family == "pooled" and conservative:
        raise ValueError("Pooled mode is empirical only; use robust mode for the bound")
    blocks = [environments == e for e in envs] if family == "robust" else [np.ones(len(ids), bool)]
    best, best_key = StableCorePolicy(None, None, alpha, family), (-1, -1, -1., -1.)
    for s, r in product(THRESHOLD_GRID, repeat=2):
        policy = StableCorePolicy(s, r, alpha, family)
        selected = policy.release(evidence)
        counts, failures = selected.sum(1), (selected & ~survived).sum(1)
        valid = True
        for block in blocks:
            if counts[block].sum() == 0:
                valid = False
                break
            if conservative:
                d = (failures[block] - alpha * counts[block]) / 2
                risk_ok = d.mean() + np.sqrt(np.log(36 * len(blocks) / .05) / (2 * block.sum())) <= 0
            else:
                risk_ok = failures[block].sum() / counts[block].sum() <= alpha
            if not risk_ok:
                valid = False
                break
        key = (int((counts > 0).sum()), int(counts.sum()), s, r)
        if valid and key > best_key:
            best, best_key = policy, key
    return best


def stable_core_metrics(selected, survived, candidates):
    selected, survived, candidates = (np.asarray(x, bool) for x in (selected, survived, candidates))
    if selected.shape != survived.shape or candidates.shape != selected.shape or selected.ndim != 2:
        raise ValueError("Reason arrays must align")
    if np.any(selected & ~candidates):
        raise ValueError("Cannot release a reason outside the original candidates")
    count = selected.sum(1)
    failure = (selected & ~survived).sum(1)
    true_released = int((selected & survived).sum())
    total = int(count.sum())
    denominator = int((candidates & survived).sum())
    return {"reason_precision": true_released / total if total else None,
            "false_stable_rate": int(failure.sum()) / total if total else None,
            "candidate_survivor_recall": true_released / denominator if denominator else None,
            "mean_reasons": float(count.mean()) if len(count) else None,
            "customer_coverage": float((count > 0).mean()) if len(count) else None,
            "customer_any_failure_risk": float((failure[count > 0] > 0).mean()) if (count > 0).any() else None,
            "customers_by_reason_count": {str(k): int((count == k).sum()) for k in range(3)}}
