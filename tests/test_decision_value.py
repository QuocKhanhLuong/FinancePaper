"""Scientific contracts for the new audit, not benchmark accuracy claims."""
import json
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financepaper.evaluation.decision_value import (
    break_even, check_hashes, cluster_weights, digest, pareto_flags, ratio_samples,
)
from financepaper.evaluation.decision_vignettes import (
    CurrentCase, export_synthetic_bundle, independent_answer, render_case, synthetic_pool,
)


def test_paired_cluster_resampling_keeps_all_masks_and_folds():
    meta = pd.DataFrame({"cluster": [7, 7, 9, 9, 11, 11], "fold": [0, 0, 0, 0, 1, 1]})
    w, inv = cluster_weights(meta, 100, 1)
    assert np.array_equal(inv, [0, 0, 1, 1, 2, 2])
    assert np.all(w[:, :2].sum(1) == 2)
    assert np.all(w[:, 2] == 1)
    assert np.array_equal(w[:, inv[0]], w[:, inv[1]])
    assert np.array_equal(w, cluster_weights(meta, 100, 1)[0])
    assert not np.array_equal(w, cluster_weights(meta, 100, 2)[0])


def test_utility_crossing_and_frontier_include_ties_and_strong_baseline():
    counts = pd.DataFrame([
        dict(dataset="d", condition="c", method="release_all", valid_reasons=100, failed_reasons=10),
        dict(dataset="d", condition="c", method="mask_warning", valid_reasons=100, failed_reasons=10),
        dict(dataset="d", condition="c", method="stable_both", valid_reasons=80, failed_reasons=5),
        dict(dataset="d", condition="c", method="bad", valid_reasons=70, failed_reasons=6),
    ])
    result = break_even(counts).set_index("method")
    assert result.loc["stable_both", "c_over_b"] == 4
    assert np.isnan(result.loc["mask_warning", "c_over_b"])
    assert result.loc["mask_warning", "interpretation"] == "identical gross utility at every ratio"
    assert pareto_flags(counts.valid_reasons, counts.failed_reasons).tolist() == [True, True, True, False]
    assert 100 - 2 * 10 > 80 - 2 * 5
    assert 100 - 10 * 10 < 80 - 10 * 5
    assert np.isnan(ratio_samples(np.array([0.]), np.array([0.]))[0])


def test_hash_drift_is_rejected(tmp_path):
    p = tmp_path / "a"; p.write_text("original")
    entries = {str(p): digest(p)}
    check_hashes(entries)
    p.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        check_hashes(entries)


def test_outcome_independent_of_model_and_hidden_future():
    assert independent_answer(.5, None) == "review"
    assert independent_answer(1., None) == "supported"
    assert independent_answer(.5, 0.) == "supported"
    assert independent_answer(.5, 1.) == "review"
    pool = synthetic_pool(20261007)
    for case, private in pool:
        assert private["answer"] == independent_answer(case.resource_a, case.requirement_h)
        changed = replace(case, risk_probability=.999, contributions=(-90., 0., 90.), completion_support_counts=(0, 8, 8))
        assert independent_answer(changed.resource_a, changed.requirement_h) == private["answer"]
        before = render_case(case, 4)
        private.update(answer="POISON", restored_probability="POISON", full_record="POISON")
        assert render_case(case, 4) == before
        assert "POISON" not in json.dumps(before)


def test_public_contract_rejects_private_fields_and_invalid_evidence():
    values = dict(case_id="x", resource_a=1., context_b=.5, context_c=.1, requirement_h=None,
                  risk_probability=.5, contributions=(.1, .2, .3), completion_support_counts=(1, 2, 3))
    with pytest.raises(TypeError):
        CurrentCase(**values, full_record={"H": 1})
    with pytest.raises(ValueError, match="numeric"):
        CurrentCase(**{**values, "risk_probability": float("nan")})
    with pytest.raises(ValueError, match="support count"):
        CurrentCase(**{**values, "completion_support_counts": (9, 0, 0)})


def test_bundle_balanced_blind_no_repeat_and_reproducible(tmp_path):
    a, b, c = (tmp_path / x for x in ("a", "b", "c"))
    receipt = export_synthetic_bundle(a)
    export_synthetic_bundle(b)
    export_synthetic_bundle(c, seed=20261008)
    assert receipt["human_ratings"] == receipt["human_participants"] == 0
    assert receipt["strata"] == dict(revised=4, stable=4, near_tie=4, mc_false_negative=4)
    schedule = pd.read_csv(a / "private/allocation.csv")
    assert not schedule.duplicated(["slot", "case_id"]).any()
    assert set(schedule.groupby(["case_id", "arm"]).size()) == {4}
    assert set(schedule.groupby(["slot", "arm", "stratum"]).size()) == {1}
    assert (a / "private/allocation.csv").read_bytes() == (b / "private/allocation.csv").read_bytes()
    assert (a / "private/allocation.csv").read_bytes() != (c / "private/allocation.csv").read_bytes()
    forbidden = {"answer", "stratum", "full_record", "arm", "computational_revision", "mc_false_negative",
                 "restored_probability", "restored_contributions", "model_identity", "policy_score"}
    currents = json.loads((a / "private/current_cases.json").read_text())
    assert len({(x["resource_a"], x["context_b"], x["context_c"]) for x in currents}) == 16
    for p in (a / "expert").glob("*.jsonl"):
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert len({x["case_id"] for x in rows}) == 16
        for row in rows:
            assert not forbidden.intersection(row)
            text = json.dumps(row).lower()
            assert not any(x in text for x in ("low income", "poor payment", "shap", "stable_core"))
    for case in currents:
        case = CurrentCase(**case)
        arms = [render_case(case, arm) for arm in range(1, 5)]
        assert all(x["risk_probability"] == arms[0]["risk_probability"] for x in arms)
        assert all(x["available_record"] == arms[0]["available_record"] for x in arms)
        assert not arms[0]["explanation"]
        assert "completion_information" not in arms[2]
        assert "completion_information" in arms[3]
    assert len(pd.read_csv(a / "ratings_EMPTY.csv")) == 0


def test_stage_resume_verifies_outputs_without_rerunning(tmp_path):
    spec = importlib.util.spec_from_file_location("pilot_runner", Path("scripts/run_decision_value_pilot.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    def work():
        calls.append(1)
        p = tmp_path / "result.csv"; p.write_text("metric,value\nx,1\n")
        return [p]
    module.stage(tmp_path, "example", work, resume=False)
    module.stage(tmp_path, "example", work, resume=True)
    assert len(calls) == 1
    (tmp_path / "result.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.stage(tmp_path, "example", work, resume=True)


def test_consolidated_provenance_paths_exist():
    from financepaper.experiments.robustness_followup import source_hashes
    hashes = source_hashes()
    assert "archive/20261007/docs/ROBUSTNESS_FOLLOWUP_PROTOCOL.md" in hashes
    assert "archive/20261007/docs/TEMPORAL_MODEL_SPEC.md" in hashes
