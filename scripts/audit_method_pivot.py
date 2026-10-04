"""Exhaustive mathematical witnesses and historical aggregate rechecks.

This is NOT a new predictor or a training pipeline. No financial records, models,
hidden customer values, external code or data are loaded. No RNG is required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations, product
import json
import math
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/method_pivot/PREREGISTERED_PILOT.md")
CONFIG = Path("configs/method_pivot_audit.json")
HISTORICAL = {
    "region_b": "outputs/decisive_validation/analysis/region_B_intervals.csv",
    "detectors": "outputs/decisive_validation/analysis/B_detector_metrics.csv",
    "paired_ap": "outputs/decisive_validation/analysis/paired_fold_mean_differences.csv",
    "stable_core": "outputs/stable_core_study/analysis/metrics.csv",
    "headroom": "outputs/verification_headroom/analysis/elementary_bounds.csv",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    z = math.exp(value)
    return z / (1 + z)


def validate_world(states, weights, values):
    if not states or len(states) != len(weights) or len(states) != len(values):
        raise ValueError("Support, weights and values must have equal nonzero lengths")
    width = len(states[0])
    if width == 0 or any(len(s) != width for s in states):
        raise ValueError("Rectangular nonempty support required")
    if len(set(states)) != len(states):
        raise ValueError("Support states must be unique")
    if not all(math.isfinite(w) and w >= 0 for w in weights):
        raise ValueError("Invalid probability weights")
    if not math.isclose(sum(weights), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Weights must sum to one; do not silently renormalize")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Values must be finite")


def conditional_mean(states, weights, values, observed):
    """Exact oracle under a declared law, queried only with observed (index,value).

    The support/value table is the known synthetic population, not query truth.
    A probability of zero is an unsupported query and fails explicitly.
    """
    validate_world(states, weights, values)
    if any(type(j) is not int or j < 0 or j >= len(states[0]) for j in observed):
        raise ValueError("Invalid observed coordinate")
    indices = [i for i, s in enumerate(states)
               if all(s[j] == value for j, value in observed.items())]
    mass = sum(weights[i] for i in indices)
    if mass <= 0:
        raise ValueError("Observed state is outside the declared support")
    return sum(weights[i] * values[i] for i in indices) / mass


def predictions(states, weights, values, subset):
    return [conditional_mean(states, weights, values, {j: s[j] for j in subset})
            for s in states]


def weighted_mean(weights, values):
    return sum(w * v for w, v in zip(weights, values, strict=True))


def expected_losses(weights, truth, prediction):
    if not (len(weights) == len(truth) == len(prediction)):
        raise ValueError("Loss arrays must align")
    if any(not 0 < p < 1 for p in prediction) or any(not 0 <= t <= 1 for t in truth):
        raise ValueError("Expected Bernoulli losses need valid probabilities")
    return {
        "brier": weighted_mean(weights, [t * (1-t) + (p-t)**2
                                        for t, p in zip(truth, prediction)]),
        "log_loss": weighted_mean(weights, [-t*math.log(p)-(1-t)*math.log1p(-p)
                                           for t, p in zip(truth, prediction)]),
    }


def subsets(dimension):
    return [tuple(c) for r in range(dimension+1)
            for c in combinations(range(dimension), r)]


def tower_error(states, weights, terminal):
    parts = subsets(len(states[0]))
    by_subset = {s: predictions(states, weights, terminal, s) for s in parts}
    errors = []
    for small in parts:
        for large in parts:
            if set(small) <= set(large):
                refined = predictions(states, weights, by_subset[large], small)
                errors.extend(abs(a-b) for a, b in zip(by_subset[small], refined))
    return max(errors), len(errors)


def refinement_losses(states, weights, terminal, coarse, subset=(0,)):
    """Sum every independent conditional pair; no noisy Monte Carlo estimate."""
    single, two = 0., 0.
    for i, s in enumerate(states):
        compatible = [j for j, t in enumerate(states)
                      if all(s[k] == t[k] for k in subset)]
        mass = sum(weights[j] for j in compatible)
        single += weights[i] * sum(weights[j]/mass * (coarse[i]-terminal[j])**2
                                   for j in compatible)
        two += weights[i] * sum(weights[j]*weights[k]/mass**2
                                * (coarse[i]-terminal[j])*(coarse[i]-terminal[k])
                                for j in compatible for k in compatible)
    conditional = predictions(states, weights, terminal, subset)
    bias = weighted_mean(weights, [(a-b)**2 for a, b in zip(coarse, conditional)])
    variance = weighted_mean(weights, [(a-b)**2 for a, b in zip(terminal, conditional)])
    return dict(single_refinement=single, independent_pair=two,
                conditional_squared_bias=bias, legitimate_update_variance=variance)


def run_finite_audit(config):
    if config["mode"] != "finite_state_audit_not_method_training":
        raise ValueError("Only the finite-state audit is implemented")
    coefficients = config["main_effects"]
    if len(coefficients) != 3:
        raise ValueError("Preregistered audit has exactly three coordinates")
    states = list(product((-1, 1), repeat=3))
    if len(states) > config["max_support_states"]:
        raise ValueError("Support exceeds execution budget")
    tilt = config["q_second_feature_positive"]
    if not 0 < tilt < 1:
        raise ValueError("Completion tilt must preserve both states")
    p = [1/len(states)] * len(states)
    q = [.25 * (tilt if s[1] == 1 else 1-tilt) for s in states]
    intercept = config["intercept"]
    tol = config["identity_tolerance"]
    if not 0 < tol <= 1e-10:
        raise ValueError("Identity tolerance must remain stringent")
    results, checks = {}, {}
    for name, interaction in (("additive", 0.), ("interaction", config["interaction"])):
        logits = [intercept + sum(b*x for b, x in zip(coefficients, s))
                  + interaction*s[0]*s[1] for s in states]
        truth = list(map(sigmoid, logits))
        oracle = predictions(states, p, truth, (0,))
        wrong = predictions(states, q, truth, (0,))
        prevalence = weighted_mean(p, truth)
        constant = [prevalence] * len(states)
        mean_fill = [sigmoid(intercept+coefficients[0]*s[0]) for s in states]
        tower_p, comparisons = tower_error(states, p, truth)
        tower_q, _ = tower_error(states, q, truth)
        constant_error, _ = tower_error(states, p, constant)
        refining = refinement_losses(states, p, truth, oracle)
        # Also use biased coarse values, so unbiasedness is not tested only at zero.
        refining_biased = refinement_losses(states, p, truth, wrong)
        shifted_logits = [intercept + (coefficients[0]+config["allocation_shift"])*s[0]
                          + coefficients[1]*s[1]+coefficients[2]*s[2]
                          + interaction*s[0]*s[1]-config["allocation_shift"]*s[0]
                          for s in states]
        reconstructed = max(abs(a-b) for a, b in zip(logits, shifted_logits))
        ranges = []
        for observed in (-1, 1):
            possible = [truth[i] for i, s in enumerate(states) if s[0] == observed]
            ranges.append(dict(x1=observed, minimum=min(possible), maximum=max(possible),
                               worst_case_lower_bound=(max(possible)-min(possible))/2))
        laws = []
        for key in ("source_hide_probabilities", "target_hide_probabilities"):
            probabilities = config[key]
            if len(probabilities) != 2 or any(not 0 < v < 1 for v in probabilities):
                raise ValueError("Mechanism probabilities must be in (0,1)")
            mass = [w*probabilities[int(s[1] == 1)] for w, s in zip(p, states)]
            total = sum(mass)
            laws.append([m/total for m in mass])
        source_risk = predictions(states, laws[0], truth, (0, 2))
        target_risk = predictions(states, laws[1], truth, (0, 2))
        losses = {key: expected_losses(p, truth, values) for key, values in
                  (("conditional_oracle", oracle), ("constant", constant),
                   ("mean_fill", mean_fill), ("wrong_q", wrong))}
        results[name] = dict(
            support_states=len(states), nested_state_comparisons=comparisons,
            tower_max_error=tower_p, wrong_q_self_tower_max_error=tower_q,
            constant_tower_max_error=constant_error, expected_outcome_losses=losses,
            wrong_q_actual_law_coherence_mse=weighted_mean(p, [(a-b)**2 for a, b in zip(wrong, oracle)]),
            refinement_losses=refining, biased_refinement_losses=refining_biased,
            probability_vs_mean_logit_max_gap=max(abs(a-b) for a, b in zip(oracle, mean_fill)),
            allocation_reconstruction_max_error=reconstructed,
            allocation_shift_magnitude=abs(config["allocation_shift"]),
            hidden_response_ranges=ranges,
            source_hidden_risk_by_x1_x3={str((s[0], s[2])): v for s, v in zip(states, source_risk)},
            target_hidden_risk_by_x1_x3={str((s[0], s[2])): v for s, v in zip(states, target_risk)},
            mechanism_shift_mean_abs_risk_change=weighted_mean(laws[1],
                [abs(a-b) for a, b in zip(source_risk, target_risk)]))
        checks[name] = {
            "tower_identity": tower_p <= tol,
            "wrong_law_can_be_self_coherent": tower_q <= tol,
            "constant_is_coherent_but_less_informative": constant_error <= tol
                and losses["constant"]["brier"] > losses["conditional_oracle"]["brier"],
            "single_sample_penalizes_update_variance": abs(refining["single_refinement"]
                - refining["conditional_squared_bias"]-refining["legitimate_update_variance"]) <= tol,
            "two_sample_identity_zero_and_nonzero_bias": all(abs(r["independent_pair"]
                - r["conditional_squared_bias"]) <= tol for r in (refining, refining_biased)),
            "fidelity_does_not_identify_allocation": reconstructed <= tol,
            "hidden_uncertainty_is_nonzero": all(r["worst_case_lower_bound"] > 0 for r in ranges),
        }
    if not all(v for world in checks.values() for v in world.values()):
        raise AssertionError(f"Mathematical check failed; do not retune: {checks}")
    return dict(status="EXHAUSTIVE_SYNTHETIC_CHECK_NOT_MODEL_VALIDATION",
                config=config, worlds=results, identity_checks=checks,
                identities_passed=sum(sum(v.values()) for v in checks.values()))


def historical_audit(root):
    result = {}
    for key, relative in HISTORICAL.items():
        path = root / relative
        if not path.is_file():
            result[key] = dict(path=relative, status="UNAVAILABLE")
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames
        entry = dict(path=relative, status="EXISTING_AGGREGATE_CHECKED_NOT_RERUN",
                     sha256=digest(path), rows=len(rows), columns=columns)
        if key == "region_b":
            selected = [r for r in rows if r["variant"] == "group2" and r["condition"] == "mcar30"
                        and float(r["cutoff"]) == .02 and r["probability_scale"] == "prob_shift"]
            if len(selected) != 2:
                raise AssertionError("Region B rows absent/duplicated")
            for r in selected:
                expected = {"taiwan": (42, 727), "polish": (85, 552)}[r["dataset"]]
                actual = (int(r["B"]), int(r["stable_n"]))
                if actual != expected or not math.isclose(float(r["B_given_stable"]), actual[0]/actual[1]):
                    raise AssertionError("Historical Region B does not match declared report")
            entry["checked_rows"] = selected
        elif key == "detectors":
            selected = [r for r in rows if r["variant"] == "group2" and r["condition"] == "mcar30"
                        and r["probability"] == "calibrated_score"
                        and r["method"] in ("prediction_only", "mc8", "rank_instability")]
            means = {}
            for dataset in ("taiwan", "polish"):
                means[dataset] = {}
                for method in ("prediction_only", "mc8", "rank_instability"):
                    values = [float(r["average_precision"]) for r in selected
                              if r["dataset"] == dataset and r["method"] == method]
                    if len(values) != (3 if dataset == "taiwan" else 1):
                        raise AssertionError("Unexpected detector fold count")
                    means[dataset][method] = sum(values)/len(values)
            entry["group2_mcar30_fold_mean_ap"] = means
        elif key == "paired_ap":
            entry["checked_rows"] = [r for r in rows if r["variant"] == "group2"
                and r["condition"] == "mcar30" and r["metric"] == "AP"
                and r["first"] == "mc8" and r["second"] in ("prediction_only", "rank_instability")]
        elif key == "stable_core":
            selected = [r for r in rows if r["target"] == "meaningful" and float(r["alpha"]) == .1
                        and r["conservative"].lower() == "false" and r["condition"] == "overall"
                        and r["method"] in ("release_all", "stable_both")]
            if len(selected) != 4:
                raise AssertionError("Expected four aggregate Stable-Core rows")
            for r in selected:
                ratio = int(r["failed_reasons"])/int(r["released_reasons"])
                if not math.isclose(ratio, float(r["reason_risk"]), abs_tol=1e-12):
                    raise AssertionError("Reason denominator mismatch")
            entry["checked_rows"] = selected
        elif key == "headroom":
            if len(rows) != 12 or any(float(r["maximum_possible_gain"]) != 0 for r in rows):
                raise AssertionError("Historical retained-candidate ceiling mismatch")
            entry["checked_rows"] = rows
        result[key] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/CONFIG)
    parser.add_argument("--historical", action="store_true", help="Recheck existing aggregate CSVs only")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/method_pivot/audit.json")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.output.exists():
        parser.error("Output exists; choose a new --output path, never overwrite evidence")
    config = json.loads(args.config.read_text())
    budget = config["execution_budget_seconds"]
    if not 0 < budget <= 60:
        raise ValueError("Audit budget must be in (0,60] seconds")
    report = run_finite_audit(config)
    report["historical"] = historical_audit(ROOT) if args.historical else {"status": "NOT_REQUESTED"}
    elapsed = time.perf_counter()-started
    if elapsed > budget:
        raise RuntimeError("Audit exceeded budget; stop before expanding any experiment")
    report["provenance"] = dict(
        utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        platform=platform.platform(), device="cpu", elapsed_seconds=elapsed,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        protocol_sha256=digest(ROOT/PROTOCOL), config_sha256=digest(args.config),
        source_sha256=digest(__file__), seeds="not_applicable_exact_enumeration",
        trained_parameters=0, attribution_calls=0, external_downloads=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards a concurrent writer after the initial check.
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(args.output), status=report["status"],
                          identities_passed=report["identities_passed"], elapsed_seconds=elapsed)))


if __name__ == "__main__":
    main()
