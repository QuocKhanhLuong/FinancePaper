"""Synthetic F1 software fixture, never human evidence or bank validation.

The reviewer exporter accepts only CurrentCase. Full records, strata and answers
are constructed and saved through a separate private channel.
"""
from dataclasses import dataclass, asdict
from itertools import product
import json

import numpy as np
import pandas as pd

from financepaper.evaluation.decision_value import write_json


@dataclass(frozen=True)
class CurrentCase:
    case_id: str
    resource_a: float
    context_b: float
    context_c: float
    requirement_h: float | None
    risk_probability: float
    contributions: tuple[float, float, float]
    completion_support_counts: tuple[int, int, int]
    completions: int = 8

    def __post_init__(self):
        if self.requirement_h not in (None, 0., 1.):
            raise ValueError("H must be absent or in the declared domain")
        values = (self.resource_a, self.context_b, self.context_c, self.risk_probability, *self.contributions)
        if not all(np.isfinite(x) for x in values) or not 0 <= self.risk_probability <= 1:
            raise ValueError("Invalid current numeric evidence")
        if len(self.contributions) != 3 or len(self.completion_support_counts) != 3 or self.completions != 8:
            raise ValueError("Unexpected disclosure dimensions")
        if any(type(n) is not int or not 0 <= n <= self.completions for n in self.completion_support_counts):
            raise ValueError("Invalid completion support count")


def independent_answer(resource_a, requirement_h):
    """Logical entailment of A>=H from current facts, never SHAP/score/restore."""
    maximum_h = 1.0 if requirement_h is None else requirement_h
    return "supported" if resource_a >= maximum_h else "review"


def observed_attributions(a, b, c, h):
    # Exact zero-reference Shapley terms for f=a*(1-h)+b*h+c;
    # hidden H's own term is excluded from these three displayed groups.
    return np.array([a * (1 - h / 2), b * h / 2, c])


def initial_top2(phi):
    order = np.argsort(-phi, kind="stable")[:2]
    return order[phi[order] > .01]


def changed_reason(before, after):
    chosen = initial_top2(before)
    return bool(any(after[j] <= .01 or (after > after[j] + 1e-6).sum() >= 2 for j in chosen))


def render_case(case, arm):
    """Strict allowlist: no dataclass __dict__ forwarding and no private argument."""
    if arm not in (1, 2, 3, 4):
        raise ValueError("Unknown presentation arm")
    result = {
        "case_id": case.case_id,
        "task": "Does the available record support the statement 'resource A covers requirement H (A >= H)'? Choose supported only if every value consistent with the record satisfies it; otherwise choose review.",
        "context": "Synthetic ledger task; amounts are fictional units. H is either 0 or 1. The score is a simulated risk indicator, not the answer to the evidence question.",
        "available_record": {"resource_A": case.resource_a, "context_B": case.context_b,
                             "context_C": case.context_c, "requirement_H": case.requirement_h},
        "risk_probability": round(case.risk_probability, 6),
        "explanation": [],
        "notice": "",
        "choices": ["supported", "review"],
    }
    if arm >= 2:
        result["explanation"] = [f"Group {name}: contribution {value:+.3f} score units."
                                 for name, value in zip(("A", "B", "C"), case.contributions)]
    if arm >= 3:
        result["notice"] = ("Requirement H is missing. Explanations can depend on missing information. "
                             "A positive contribution is not an observed financial fact." if case.requirement_h is None else
                             "Requirement H is recorded. A positive contribution is not an observed financial fact.")
    if arm == 4:
        result["completion_information"] = [
            f"Group {name}: contribution above 0.01 in {count}/{case.completions} hypothetical completions."
            for name, count in zip(("A", "B", "C"), case.completion_support_counts)]
        result["completion_caveat"] = "Completions may miss possible values; these counts are not probabilities that the statement is correct."
    return result


def synthetic_pool(seed):
    rng = np.random.default_rng(seed)
    pool = []
    for a, b, c in product((.25, .5, .75, 1., 1.25), (.25, .5, .75, 1., 1.5, 2., 2.5),
                          (.015, .02, .25, .4, .5, .75, 1.)):
        # One record per (A,B,C); a reviewer cannot encounter the same ledger
        # again under a different missingness/full-information version.
        full_h, missing = float(rng.integers(2)), bool(rng.random() < .7)
        current_h = None if missing else full_h
        imputed_h = 0. if missing else full_h
        before = observed_attributions(a, b, c, imputed_h)
        restored = observed_attributions(a, b, c, full_h)
        # Deliberately imperfect completion distribution supplies false negatives.
        draws = np.zeros(8) if rng.integers(2) == 0 else np.tile([0., 1.], 4)
        if not missing:
            draws[:] = full_h
        phis = np.array([observed_attributions(a, b, c, h) for h in draws])
        revised = changed_reason(before, restored)
        mc_flag = any(changed_reason(before, p) for p in phis)
        order = np.sort(before)[::-1]
        near_tie = bool(abs(order[1] - order[2]) <= .02)
        stratum = "mc_false_negative" if revised and not mc_flag else "near_tie" if near_tie else "revised" if revised else "stable"
        logit = a * (1 - imputed_h) + b * imputed_h + c
        case = CurrentCase("", a, b, c, current_h, float(1 / (1 + np.exp(-logit))),
                           tuple(before.tolist()), tuple((phis > .01).sum(0).tolist()))
        private = {"full_record": {"A": a, "B": b, "C": c, "H": full_h},
                   "answer": independent_answer(a, current_h), "stratum": stratum,
                   "computational_revision": revised, "near_tie": near_tie,
                   "mc_false_negative": revised and not mc_flag,
                   "restored_contributions": restored.tolist(),
                   "restored_probability": float(1 / (1 + np.exp(-(a * (1 - full_h) + b * full_h + c))))}
        pool.append((case, private))
    return pool


def export_synthetic_bundle(output, seed=20261007, per_stratum=4, reviewer_slots=16):
    if per_stratum % 4 or reviewer_slots % 4:
        raise ValueError("Balanced schedule requires multiples of four")
    rng = np.random.default_rng(seed)
    pool = synthetic_pool(seed)
    selected = []
    for stratum in ("revised", "stable", "near_tie", "mc_false_negative"):
        block = [item for item in pool if item[1]["stratum"] == stratum]
        if len(block) < per_stratum:
            raise ValueError(f"Insufficient fixture cases in {stratum}")
        for index in rng.choice(len(block), per_stratum, replace=False):
            current, private = block[index]
            case_id = rng.bytes(12).hex()  # no stratum/answer encoded in ID
            current = CurrentCase(**{**asdict(current), "case_id": case_id})
            private.update(case_id=case_id, inclusion_probability=per_stratum / len(block),
                           pool_stratum_size=len(block), origin="SYNTHETIC_SOFTWARE_FIXTURE")
            selected.append((current, private))
    private_path = output / "private"
    expert_path = output / "expert"
    private_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    expert_path.mkdir(parents=True, exist_ok=True)
    # These are allocation slots, not invented participants or responses.
    slot_offsets = np.tile(np.arange(4), reviewer_slots // 4)
    rng.shuffle(slot_offsets)
    schedule = []
    for slot, offset in enumerate(slot_offsets):
        records = []
        for position, (current, private) in enumerate(selected):
            arm = int((position + offset) % 4 + 1)
            record = render_case(current, arm)
            records.append(record)
            schedule.append({"slot": f"slot_{slot:03d}", "case_id": current.case_id, "arm": arm,
                             "stratum": private["stratum"], "words": len(json.dumps(record).split())})
        rng.shuffle(records)
        positions = {x["case_id"]: i for i, x in enumerate(records)}
        for row in schedule[-len(records):]:
            row["order"] = positions[row["case_id"]]
        (expert_path / f"slot_{slot:03d}.jsonl").write_text("".join(json.dumps(x, allow_nan=False) + "\n" for x in records))
    write_json(private_path / "evaluation.json", [p for _, p in selected])
    write_json(private_path / "current_cases.json", [asdict(c) for c, _ in selected])
    table = pd.DataFrame(schedule)
    table.to_csv(private_path / "allocation.csv", index=False)
    table.groupby("arm").words.agg(["min", "mean", "max"]).to_csv(output / "presentation_lengths.csv")
    pd.DataFrame(columns=["slot", "case_id", "choice", "confidence_0_100", "elapsed_seconds", "request_review"]).to_csv(output / "ratings_EMPTY.csv", index=False)
    summary = {"status": "SOFTWARE_PREPARATION_ONLY", "source": "synthetic", "cases": len(selected),
               "pool_cases": len(pool), "allocation_slots": reviewer_slots, "human_participants": 0,
               "human_ratings": 0, "seed": seed, "primary_comparison": "arm4_vs_arm3",
               "strata": {s: sum(p["stratum"] == s for _, p in selected) for s in ("revised", "stable", "near_tie", "mc_false_negative")},
               "independent_outcome": "current-record entailment of A>=H for H in {0,1}; no model inputs to answer except A and observed H",
               "expert_validation": "NOT_RUN", "information_effort_matching": "NOT_VALIDATED_word_counts_saved"}
    write_json(output / "bundle_receipt.json", summary)
    return summary
