"""Independent checks and a single-slot preview for the synthetic F1 fixture."""
from itertools import permutations
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.decision_value import write_json
from financepaper.evaluation.decision_vignettes import (
    CurrentCase, observed_attributions, render_case, synthetic_pool,
)

REQUIRED = {"case_id", "task", "context", "available_record", "risk_probability",
            "explanation", "notice", "choices"}
OPTIONAL = {"completion_information", "completion_caveat"}
RECORD_FIELDS = {"resource_A", "context_B", "context_C", "requirement_H"}


def validate_public(row):
    """Reject unknown fields rather than forwarding private keys to a browser."""
    if not REQUIRED <= row.keys() or row.keys() - REQUIRED - OPTIONAL:
        raise ValueError("Unexpected public schema fields")
    if set(row["available_record"]) != RECORD_FIELDS:
        raise ValueError("Unexpected record fields")
    if bool(row.keys() & OPTIONAL) and not OPTIONAL <= row.keys():
        raise ValueError("Incomplete completion disclosure")
    if row["choices"] != ["supported", "review"]:
        raise ValueError("Unexpected task choices")
    for key in ("case_id", "task", "context", "notice", *[x for x in ("completion_caveat",) if x in row]):
        if not isinstance(row[key], str):
            raise ValueError("Public text must be a string")
    for key in ("explanation", *[x for x in ("completion_information",) if x in row]):
        if not isinstance(row[key], list) or not all(isinstance(x, str) for x in row[key]):
            raise ValueError("Public statements must be strings")
    for key, value in row["available_record"].items():
        if key == "requirement_H" and value is None:
            continue
        if type(value) not in (float, int) or not np.isfinite(value):
            raise ValueError("Nonfinite record")
    if row["available_record"]["requirement_H"] not in (None, 0., 1.):
        raise ValueError("Requirement outside declared domain")
    probability = row["risk_probability"]
    if type(probability) not in (float, int) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid public probability")
    return row


def read_public(path):
    rows = [validate_public(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Empty slot or repeated case")
    return rows


def build_preview(public_jsonl, template):
    """Only a single already-public slot is an input; no private files are opened."""
    rows = read_public(public_jsonl)
    payload = json.dumps(rows, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = Path(template).read_text()
    if document.count("__CASE_PAYLOAD__") != 1:
        raise ValueError("Preview template must have exactly one payload slot")
    return document.replace("__CASE_PAYLOAD__", payload)


def exact_shapley(a, b, h, c):
    """Independent 24-permutation, zero-reference evaluator; no closed-form reuse."""
    vector = np.array([a, b, h, c], float)
    phi = np.zeros(4)
    def f(x):
        return x[0] * (1 - x[2]) + x[1] * x[2] + x[3]
    for order in permutations(range(4)):
        x = np.zeros(4)
        for j in order:
            previous = f(x)
            x[j] = vector[j]
            phi[j] += (f(x) - previous) / 24
    return phi


def enumerated_answer(a, h):
    """Evaluate every feasible world from current facts, independently of the helper."""
    worlds = (0., 1.) if h is None else (h,)
    return "supported" if all(a >= value for value in worlds) else "review"


def visible_word_count(row):
    """Variable visible content only; omit shared chrome and JSON schema keys."""
    fields = [row["task"], row["context"], row["notice"], *row["explanation"],
              *row.get("completion_information", []), row.get("completion_caveat", "")]
    return len(" ".join(fields).split())


def audit_fixture(bundle, seed, output):
    pool = synthetic_pool(seed)
    checks, errors = 0, []
    for current, private in tqdm(pool, desc="Independent fixture audit", unit="case"):
        a, b, c = current.resource_a, current.context_b, current.context_c
        for h in (0. if current.requirement_h is None else current.requirement_h, private["full_record"]["H"]):
            phi = exact_shapley(a, b, h, c)
            errors.append(float(np.max(abs(phi[[0, 1, 3]] - observed_attributions(a, b, c, h)))))
            if not np.isclose(phi.sum(), a * (1 - h) + b * h + c, atol=1e-12, rtol=0):
                raise ValueError("Exact attribution efficiency failed")
        if enumerated_answer(a, current.requirement_h) != private["answer"]:
            raise ValueError("Rubric disagrees with feasible-world enumeration")
        checks += 1
    if max(errors) > 1e-12:
        raise ValueError("Attribution formulas disagree with exact enumeration")
    selected = json.loads((bundle / "private/current_cases.json").read_text())
    labels = {x["case_id"]: x for x in json.loads((bundle / "private/evaluation.json").read_text())}
    schedule = pd.read_csv(bundle / "private/allocation.csv")
    if schedule.duplicated(["slot", "case_id"]).any():
        raise ValueError("Reviewer would see a repeated case")
    if not set(schedule.groupby(["case_id", "arm"]).size()) == {4}:
        raise ValueError("Unbalanced per-case allocation")
    current_lookup = {x["case_id"]: CurrentCase(**x) for x in selected}
    if set(current_lookup) != set(labels) or set(schedule.case_id) != set(labels):
        raise ValueError("Private/current/schedule identities disagree")
    pool_lookup = {(c.resource_a, c.context_b, c.context_c): (c, p) for c, p in pool}
    for current in current_lookup.values():
        expected_current, expected_private = pool_lookup[(current.resource_a, current.context_b, current.context_c)]
        normalize = lambda c: json.dumps({k: v for k, v in asdict(c).items() if k != "case_id"}, sort_keys=True)
        if normalize(current) != normalize(expected_current):
            raise ValueError("Selected current evidence differs from the frozen pool")
        if any(labels[current.case_id][key] != expected_private[key] for key in expected_private):
            raise ValueError("Selected private labels differ from the independently checked pool")
    for slot, group in schedule.groupby("slot"):
        public = read_public(bundle / "expert" / f"{slot}.jsonl")
        for row in group.itertuples():
            expected = render_case(current_lookup[row.case_id], int(row.arm))
            if public[int(row.order)] != expected:
                raise ValueError("Public case differs from current-only renderer")
    cohorts = [("synthetic_pool", [(c, p) for c, p in pool]),
               ("selected_fixture", [(c, labels[c.case_id]) for c in current_lookup.values()])]
    baseline_rows, strata_rows, length_rows = [], [], []
    for cohort, cases in cohorts:
        for baseline in ("always_review", "always_supported", "current_record_rule"):
            correct = 0
            for current, private in cases:
                predicted = (enumerated_answer(current.resource_a, current.requirement_h) if baseline == "current_record_rule"
                             else baseline.removeprefix("always_"))
                correct += predicted == private["answer"]
            baseline_rows.append(dict(cohort=cohort, baseline=baseline, n_cases=len(cases), correct=correct,
                                      accuracy=correct / len(cases), evidence="deterministic_software_check_not_human_accuracy"))
        for stratum in ("revised", "stable", "near_tie", "mc_false_negative"):
            block = [p for _, p in cases if p["stratum"] == stratum]
            strata_rows.append(dict(cohort=cohort, stratum=stratum, n_cases=len(block),
                                    supported=sum(p["answer"] == "supported" for p in block),
                                    review=sum(p["answer"] == "review" for p in block)))
    for current in current_lookup.values():
        for arm in range(1, 5):
            length_rows.append(dict(arm=arm, visible_words=visible_word_count(render_case(current, arm))))
    tables = {"rule_baselines.csv": pd.DataFrame(baseline_rows), "stratum_counts.csv": pd.DataFrame(strata_rows),
              "visible_lengths.csv": pd.DataFrame(length_rows).groupby("arm").visible_words.agg(["min", "mean", "max"]).reset_index()}
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    summary = {"pool_cases_checked": checks, "selected_cases_checked": len(selected),
               "public_renderings_checked": len(schedule), "shapley_comparisons": len(errors),
               "max_abs_shapley_error": max(errors), "rule_solver_accuracy": 1.0,
               "answer_determined_by_arm1_current_facts": True,
               "new_information_required_for_correct_answer": False,
               "interpretation": "Any human benefit would be a presentation/cognitive effect; human performance unmeasured",
               "human_study": "NOT_RUN", "expert_review": "NOT_RUN", "power": "NOT_RUN",
               "readiness": "UI_FIXTURE_ONLY_EXPERT_REVIEW_PENDING"}
    write_json(output / "audit.json", summary)
    return [output / name for name in [*tables, "audit.json"]]
