"""Known-answer checks for the finite audit; not tests of a proposed method."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pivot_audit", ROOT/"scripts/audit_method_pivot.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_conditional_operator_known_answers_and_hidden_query_invariance():
    states = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    weights = [.1, .2, .3, .4]
    values = [0., 1., 2., 3.]
    assert audit.conditional_mean(states, weights, values, {}) == pytest.approx(2.)
    assert audit.conditional_mean(states, weights, values, {0: -1}) == pytest.approx(2/3)
    # Both hidden query states have the identical serving observation dictionary.
    for hidden in (-1, 1):
        query = (-1, hidden)
        assert audit.conditional_mean(states, weights, values, {0: query[0]}) == pytest.approx(2/3)


@pytest.mark.parametrize("weights", [[.4, .4], [-.1, 1.1], [float('nan'), .5]])
def test_invalid_probability_laws_fail(weights):
    with pytest.raises(ValueError):
        audit.conditional_mean([(-1,), (1,)], weights, [0., 1.], {})


def test_unsupported_or_duplicate_states_fail():
    with pytest.raises(ValueError, match="outside"):
        audit.conditional_mean([(-1,), (1,)], [.5, .5], [0., 1.], {0: 2})
    with pytest.raises(ValueError, match="unique"):
        audit.conditional_mean([(-1,), (-1,)], [.5, .5], [0., 1.], {})
    with pytest.raises(ValueError, match="coordinate"):
        audit.conditional_mean([(-1,), (1,)], [.5, .5], [0., 1.], {1: 1})


def test_expected_bernoulli_losses_known_answer():
    losses = audit.expected_losses([1.], [.25], [.5])
    assert losses["brier"] == pytest.approx(.25)
    assert losses["log_loss"] == pytest.approx(math.log(2))


def test_refinement_identity_includes_nonzero_bias():
    result = audit.refinement_losses([(-1,), (1,)], [.5, .5], [.1, .9], [.6, .6], subset=())
    assert result["independent_pair"] == pytest.approx(.01)
    assert result["single_refinement"] == pytest.approx(.17)
    assert result["legitimate_update_variance"] == pytest.approx(.16)


def test_reduction_no_interaction_and_determinism():
    cfg = json.loads((ROOT/audit.CONFIG).read_text())
    cfg["interaction"] = 0.
    first = audit.run_finite_audit(cfg)
    assert first == audit.run_finite_audit(cfg)
    assert first["worlds"]["additive"] == first["worlds"]["interaction"]


def test_config_support_budget_is_enforced():
    cfg = json.loads((ROOT/audit.CONFIG).read_text())
    cfg["max_support_states"] = 4
    with pytest.raises(ValueError, match="budget"):
        audit.run_finite_audit(cfg)


def test_historical_missing_artifacts_are_not_fabricated(tmp_path):
    result = audit.historical_audit(tmp_path)
    assert len(result) == 5
    assert all(r["status"] == "UNAVAILABLE" and "checked_rows" not in r for r in result.values())


def test_corrupt_historical_region_is_rejected(tmp_path):
    path = tmp_path/audit.HISTORICAL["region_b"]
    path.parent.mkdir(parents=True)
    path.write_text("dataset,variant,condition,cutoff,probability_scale,stable_n,B,B_given_stable\n"
                    "taiwan,group2,mcar30,0.02,prob_shift,727,99,0.1\n"
                    "polish,group2,mcar30,0.02,prob_shift,552,85,0.1539855072\n")
    with pytest.raises(AssertionError, match="Region B"):
        audit.historical_audit(tmp_path)


def test_artifact_hashes_and_no_overwrite(tmp_path):
    output = tmp_path/"audit.json"
    audit.main(["--output", str(output)])
    result = json.loads(output.read_text())
    assert result["provenance"]["source_sha256"] == audit.digest(ROOT/"scripts/audit_method_pivot.py")
    assert result["provenance"]["protocol_sha256"] == audit.digest(ROOT/audit.PROTOCOL)
    assert result["provenance"]["trained_parameters"] == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        audit.main(["--output", str(output)])
    assert output.read_bytes() == before
