"""Sensitivity diagnostics; the historical binary target is not changed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from financepaper.data.schema import FEATURE_NAMES
from financepaper.explanations.reasons import extract_reasons, validate_attributions


def event_components(reasons, restored, hidden, names=FEATURE_NAMES, *, k=3,
                     sign_epsilon=1e-6, rank_epsilon=1e-6, rank_tolerance=0):
    """Separate numerical sign and rank tolerances (in raw logit units)."""
    phi, hidden = validate_attributions(restored, hidden, names)
    if k < 1 or rank_tolerance < 0 or min(sign_epsilon, rank_epsilon) < 0:
        raise ValueError("invalid event tolerances")
    if not reasons:
        return None
    if len(reasons) != k or len(set(reasons)) != k:
        raise ValueError("expected exactly k distinct reasons")
    lookup = {name: i for i, name in enumerate(names)}
    if any(name not in lookup or hidden[lookup[name]] for name in reasons):
        raise ValueError("reason must be originally observed")
    signs, exits = [], []
    for name in reasons:
        value = phi[lookup[name]]
        signs.append(value <= sign_epsilon)
        exits.append(np.count_nonzero(phi[~hidden] > value + rank_epsilon) >= k + rank_tolerance)
    failed = np.logical_or(signs, exits)
    return dict(event=bool(failed.any()), sign_change=bool(any(signs)),
                top_k_exit=bool(any(exits)), num_revised=int(failed.sum()))


def normalized_shift(before, full, hidden):
    observed = ~np.asarray(hidden, dtype=bool)
    a, b = np.asarray(before)*observed, np.asarray(full)*observed
    return np.abs(a-b).sum(-1) / np.maximum(np.abs(a).sum(-1)+np.abs(b).sum(-1), 1e-12)


def observed_groups(phi, hidden):
    """Only sum fields observed BEFORE verification, at both endpoints."""
    names = ("repayment_history", "bill_history", "payment_history",
             "LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE")
    groups = [tuple(f for f in FEATURE_NAMES if f.startswith("PAY_") and not f.startswith("PAY_AMT")),
              tuple(f for f in FEATURE_NAMES if f.startswith("BILL_AMT")),
              tuple(f for f in FEATURE_NAMES if f.startswith("PAY_AMT"))]
    groups += [(f,) for f in names[3:]]
    values, masks = [], []
    for fields in groups:
        idx = [FEATURE_NAMES.index(f) for f in fields]
        values.append((phi[:, idx] * ~hidden[:, idx]).sum(1))
        masks.append(hidden[:, idx].all(1))
    return np.array(values).T, np.array(masks).T, names


def audit_grid(before, full, hidden, valid):
    """One-factor sensitivities, with an explicit common primary-eligible set."""
    specs = [("primary", 3, .01, 1e-6, 0, False)]
    specs += [(f"k{k}", k, .01, 1e-6, 0, False) for k in (2, 5)]
    specs += [(f"magnitude_{m}", 3, m, 1e-6, 0, False) for m in (.001, .05)]
    specs += [(f"rank_gap_{g}", 3, .01, g, 0, False) for g in (.005, .01)]
    specs += [("rank_tolerance1", 3, .01, 1e-6, 1, False),
              ("semantic_groups", 3, .01, 1e-6, 0, True)]
    rows, primary = [], None
    for label, k, magnitude, gap, tolerance, grouped in specs:
        a, b, h, names = before, full, hidden, FEATURE_NAMES
        if grouped:
            a, h, names = observed_groups(before, hidden)
            b = observed_groups(full, hidden)[0]
        eligible, events, signs = np.zeros(len(a), bool), np.zeros(len(a), bool), np.zeros(len(a), bool)
        for i in np.flatnonzero(valid):
            reasons = extract_reasons(a[i], h[i], names, k, magnitude)
            comp = event_components(reasons, b[i], h[i], names, k=k, rank_epsilon=gap, rank_tolerance=tolerance)
            if comp is not None:
                eligible[i], events[i], signs[i] = True, comp["event"], comp["sign_change"]
        if primary is None:
            primary = eligible.copy()
        common = primary & eligible
        rows.append(dict(variant=label, n=len(a), pair_valid=int(valid.sum()),
            eligible=int(eligible.sum()), revised=int(events.sum()), sign_events=int(signs.sum()),
            coverage=float(eligible.mean()), revision_rate=float(events[eligible].mean()) if eligible.any() else None,
            common_primary_n=int(common.sum()),
            common_primary_revision=float(events[common].mean()) if common.any() else None,
            normalized_shift=float(normalized_shift(a, b, h)[valid].mean()) if valid.any() else None))
    return pd.DataFrame(rows)


def paired_customer_interval(frame, metric, *, draws=1000, seed=42):
    """Resample complete customer clusters, retaining all masks/restarts."""
    ids = frame.record_id.unique()
    rng = np.random.default_rng(seed)
    groups = [g for _, g in frame.groupby("record_id", sort=False)]
    values = []
    for _ in range(draws):
        sample = pd.concat([groups[i] for i in rng.integers(len(ids), size=len(ids))], ignore_index=True)
        value = metric(sample)
        if np.isfinite(value):
            values.append(value)
    return dict(estimate=float(metric(frame)), low=float(np.quantile(values, .025)) if values else None,
                high=float(np.quantile(values, .975)) if values else None, customers=len(ids), draws=len(values))
