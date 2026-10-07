"""Assumption-only crossed-design simulation; never participant evidence."""
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, roots_hermitenorm
from scipy.stats import norm


def scenarios(config):
    return [dict(scenario=i, **design, profile=profile["name"], effect=effect,
                 parameters=profile)
            for i, (design, profile, effect) in enumerate(product(
                config["designs"], config["profiles"], config["marginal_effects"]))]


def validate_config(config):
    if config["replicates"] < 1 or config["bootstrap_draws"] < 20 or config["chunk_size"] < 1:
        raise ValueError("Positive replicates/chunks and at least 20 bootstrap draws required")
    if len(config["stratum_logits"]) != 4 or not 0 < config["alpha"] < 1:
        raise ValueError("Exactly four strata and a valid alpha required")
    if not 0 < config["minimum_finite_bootstrap_fraction"] <= 1:
        raise ValueError("Invalid finite bootstrap threshold")
    names = [p["name"] for p in config["profiles"]]
    if len(names) != len(set(names)) or 0.0 not in config["marginal_effects"]:
        raise ValueError("Unique profile names and a null scenario required")
    if len(config["marginal_effects"]) != len(set(config["marginal_effects"])):
        raise ValueError("Duplicate effects")
    for s in scenarios(config):
        if s["reviewers"] < 4 or s["reviewers"] % 4 or s["cases"] < 16 or s["cases"] % 16:
            raise ValueError("Reviewer count must be divisible by 4, case count by 16")
        p = s["parameters"]
        if not (0 < p["baseline"] < 1 and 0 < p["baseline"] + s["effect"] < 1):
            raise ValueError("Marginal probabilities must be inside (0, 1)")
        for key in ("reviewer_sd", "case_sd", "reviewer_slope_sd", "case_slope_sd"):
            if not np.isfinite(p[key]) or p[key] < 0:
                raise ValueError("Invalid random effect SD")
        if not all(0 <= p[key] < 1 for key in ("reviewer_dropout", "item_missing")):
            raise ValueError("Invalid MCAR probability")


def marginal_mean(intercept, sd, offsets, nodes=64):
    x, w = roots_hermitenorm(nodes)
    values = expit(intercept + sd * x[:, None] + np.asarray(offsets)[None, :])
    return float(np.sum(w * values.mean(axis=1)) / np.sqrt(2 * np.pi))


def calibrate(scenario, offsets):
    p = scenario["parameters"]
    var3 = p["reviewer_sd"] ** 2 + p["case_sd"] ** 2
    var4 = var3 + p["reviewer_slope_sd"] ** 2 + p["case_slope_sd"] ** 2
    result = {}
    for arm, variance, target in ((3, var3, p["baseline"]),
                                  (4, var4, p["baseline"] + scenario["effect"])):
        intercept = brentq(lambda x: marginal_mean(x, np.sqrt(variance), offsets) - target,
                          -30, 30, xtol=1e-13)
        checked = marginal_mean(intercept, np.sqrt(variance), offsets, nodes=128)
        if abs(checked - target) > 1e-9:
            raise ValueError("64/128-node quadrature disagreement")
        result[f"arm{arm}_intercept"] = intercept
        result[f"arm{arm}_target"] = target
        result[f"arm{arm}_128node_error"] = checked - target
    return result


def balanced_assignment(reviewers, cases, rng):
    if reviewers < 4 or reviewers % 4 or cases < 16 or cases % 16:
        raise ValueError("Unbalanced design dimensions")
    offsets = rng.permutation(np.tile(np.arange(4), reviewers // 4))
    strata = np.repeat(np.arange(4), cases // 4)
    phases = np.concatenate([rng.permutation(np.tile(np.arange(4), cases // 16))
                             for _ in range(4)])
    arms = 1 + (offsets[:, None] + phases[None, :]) % 4
    return arms, strata


def simulate_trial(scenario, calibration, offsets, rng):
    p = scenario["parameters"]
    r, c = scenario["reviewers"], scenario["cases"]
    arms, strata = balanced_assignment(r, c, rng)
    intercepts = rng.normal(0, p["reviewer_sd"], (r, 1)) + rng.normal(0, p["case_sd"], (1, c))
    slopes = rng.normal(0, p["reviewer_slope_sd"], (r, 1)) + rng.normal(0, p["case_slope_sd"], (1, c))
    logits = (intercepts + np.asarray(offsets)[strata][None, :]
              + np.where(arms == 4, calibration["arm4_intercept"] + slopes,
                         calibration["arm3_intercept"]))
    outcomes = rng.binomial(1, expit(logits)).astype(float)
    kept = rng.random(r) >= p["reviewer_dropout"]
    observed = kept[:, None] & (rng.random((r, c)) >= p["item_missing"])
    return outcomes, observed, arms, strata, int(kept.sum())


def bootstrap_weights(reviewers, strata, draws, rng):
    rows = rng.multinomial(reviewers, np.full(reviewers, 1 / reviewers), size=draws).astype(float)
    cols = np.empty((draws, len(strata)), dtype=float)
    for s in range(4):
        ix = np.flatnonzero(strata == s)
        cols[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=draws)
    return rows, cols


def weighted_contrasts(outcomes, observed, arms, strata, rows, cols):
    """Product-weight estimates; NaN whenever any arm/stratum denominator is zero."""
    values = np.zeros(len(rows))
    for s in range(4):
        ix = strata == s
        for arm, sign in ((3, -1), (4, 1)):
            mask = (observed[:, ix] & (arms[:, ix] == arm)).astype(float)
            den = ((rows @ mask) * cols[:, ix]).sum(axis=1)
            num = ((rows @ (mask * outcomes[:, ix])) * cols[:, ix]).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                values += sign * num / den / 4
    return values


def analyze_trial(outcomes, observed, arms, strata, config, rng):
    r, c = outcomes.shape
    point = weighted_contrasts(outcomes, observed, arms, strata,
                               np.ones((1, r)), np.ones((1, c)))[0]
    wr, wc = bootstrap_weights(r, strata, config["bootstrap_draws"], rng)
    boots = weighted_contrasts(outcomes, observed, arms, strata, wr, wc)
    finite = np.isfinite(boots)
    valid = bool(np.isfinite(point) and finite.mean() >= config["minimum_finite_bootstrap_fraction"])
    low, high = (np.quantile(boots[finite], [config["alpha"] / 2, 1 - config["alpha"] / 2])
                 if valid else (np.nan, np.nan))
    variance = 0.0
    for s in range(4):
        for arm in (3, 4):
            sample = outcomes[observed & (arms == arm) & (strata[None, :] == s)]
            if not len(sample):
                variance = np.nan
                continue
            rate = sample.mean()
            variance += rate * (1 - rate) / len(sample) / 16
    radius = norm.ppf(1 - config["alpha"] / 2) * np.sqrt(variance)
    return [dict(method="two_way_percentile", valid=valid, estimate=point, lower=low, upper=high,
                 finite_bootstrap_fraction=float(finite.mean())),
            dict(method="naive_wald", valid=bool(np.isfinite(point + radius)), estimate=point,
                 lower=point - radius, upper=point + radius, finite_bootstrap_fraction=np.nan)]


def replicate(config, scenario, calibration, index):
    """Index-addressed streams make chunk size/order/resume irrelevant to outcomes."""
    data_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], scenario["scenario"], index, 0]))
    boot_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], scenario["scenario"], index, 1]))
    outcomes, observed, arms, strata, kept = simulate_trial(scenario, calibration, config["stratum_logits"], data_rng)
    rows = analyze_trial(outcomes, observed, arms, strata, config, boot_rng)
    for row in rows:
        row.update(scenario=scenario["scenario"], replicate=index, reviewers_kept=kept,
                   primary_observed=int((observed & (arms >= 3)).sum()),
                   reject=bool(row["valid"] and (row["lower"] > 0 or row["upper"] < 0)),
                   cover=bool(row["valid"] and row["lower"] <= scenario["effect"] <= row["upper"]),
                   width=row["upper"] - row["lower"])
    return rows


def wilson(successes, total, alpha=0.05):
    if not total:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    # Wilson endpoints include the MLE; force exact boundary containment when
    # subtraction roundoff makes the nominal zero endpoint slightly positive.
    return float(min(p, max(0, middle - radius))), float(max(p, min(1, middle + radius)))


def summarize(records, config):
    rows = []
    for s in scenarios(config):
        for method in ("two_way_percentile", "naive_wald"):
            part = records[(records.scenario == s["scenario"]) & (records.method == method)]
            if len(part) != config["replicates"] or set(part.replicate) != set(range(config["replicates"])):
                raise ValueError("Missing/duplicate replicate statistics")
            valid = part[part.valid]
            lower, upper = wilson(int(part.reject.sum()), len(part))
            row = {key: s[key] for key in ("scenario", "reviewers", "cases", "profile", "effect")}
            row.update(method=method, replicates=len(part), valid=len(valid), invalid_fraction=1-len(valid)/len(part),
                       rejection_rate=float(part.reject.mean()), rejection_wilson_lower=lower, rejection_wilson_upper=upper,
                       rejection_given_valid=float(valid.reject.mean()), coverage_given_valid=float(valid.cover.mean()),
                       coverage_all=float(part.cover.mean()), mean_width=float(valid.width.mean()),
                       mean_estimate=float(part.estimate.mean()), bias=float(part.estimate.mean()-s["effect"]),
                       mean_primary_observed=float(part.primary_observed.mean()),
                       min_finite_bootstrap_fraction=float(part.finite_bootstrap_fraction.min()))
            rows.append(row)
    summary = pd.DataFrame(rows)
    gates = summary[summary.effect == 0].copy()
    g = config["diagnostic_gates"]
    gates["null_screen_pass"] = ((gates.rejection_wilson_upper <= g["null_rejection_wilson_upper_max"])
                                 & (gates.invalid_fraction <= g["invalid_fraction_max"]))
    gates["conservative_flag"] = gates.rejection_rate < g["conservative_null_rejection_below"]
    gate_columns = ["reviewers", "cases", "profile", "method", "null_screen_pass", "conservative_flag"]
    summary = summary.merge(gates[gate_columns], on=gate_columns[:4], validate="many_to_one")
    return summary, gates
