"""Numerical and persistence contracts, not evidence about human accuracy."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from financepaper.evaluation.decision_inference import (
    analyze_trial, balanced_assignment, bootstrap_weights, calibrate, marginal_mean,
    replicate, scenarios, summarize, validate_config, weighted_contrasts, wilson,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/decision_inference_sim.json").read_text())


@pytest.mark.parametrize("reviewers,cases", [(16, 16), (48, 32), (96, 64)])
def test_rotation_balances_each_stratum_and_case(reviewers, cases):
    arms, strata = balanced_assignment(reviewers, cases, np.random.default_rng(7))
    for arm in range(1, 5):
        assert np.all((arms == arm).sum(axis=0) == reviewers // 4)
        for s in range(4):
            assert np.all((arms[:, strata == s] == arm).sum(axis=1) == cases // 16)


def test_weighted_bootstrap_matches_literal_duplicate_rows_and_columns():
    rng = np.random.default_rng(123)
    arms, strata = balanced_assignment(8, 16, rng)
    y = rng.binomial(1, .6, arms.shape)
    observed = rng.random(arms.shape) > .15
    rows, cols = bootstrap_weights(8, strata, 40, rng)
    fast = weighted_contrasts(y, observed, arms, strata, rows, cols)
    slow = []
    for row_counts, col_counts in zip(rows, cols, strict=True):
        ri = np.repeat(np.arange(8), row_counts.astype(int))
        ci = np.repeat(np.arange(16), col_counts.astype(int))
        expanded_y = y[np.ix_(ri, ci)]
        expanded_obs = observed[np.ix_(ri, ci)]
        expanded_arms = arms[np.ix_(ri, ci)]
        diffs = []
        for s in range(4):
            rates = []
            for arm in (3, 4):
                sample = expanded_y[expanded_obs & (expanded_arms == arm) & (strata[ci][None, :] == s)]
                rates.append(sample.mean() if len(sample) else np.nan)
            diffs.append(rates[1]-rates[0])
        slow.append(np.mean(diffs))
    np.testing.assert_allclose(fast, slow, atol=1e-14, equal_nan=True)
    assert np.isfinite(fast).any()
    assert np.all(rows.sum(axis=1) == 8)
    for s in range(4):
        assert np.all(cols[:, strata == s].sum(axis=1) == 4)


def test_population_calibration_not_conditional_intercept(config):
    validate_config(config)
    for s in scenarios(config):
        fitted = calibrate(s, config["stratum_logits"])
        assert abs(fitted["arm3_128node_error"]) < 1e-9
        assert abs(fitted["arm4_128node_error"]) < 1e-9
        if s["profile"] == "slopes_missing" and s["effect"] == 0:
            assert fitted["arm4_intercept"] != fitted["arm3_intercept"]
    assert marginal_mean(0, 1.4, [-.3, -.1, .1, .3], 128) == pytest.approx(.5)


def test_bootstrap_invalid_on_missing_arm_and_naive_closed_form(config):
    arms, strata = balanced_assignment(16, 16, np.random.default_rng(1))
    y = np.zeros(arms.shape)
    # Exactly half correct in every observed arm/stratum.
    for s in range(4):
        for arm in (3, 4):
            ri, ci = np.where((arms == arm) & (strata[None, :] == s))
            y[ri[:len(ri)//2], ci[:len(ci)//2]] = 1
    observed = np.ones(arms.shape, dtype=bool)
    rows = analyze_trial(y, observed, arms, strata, config, np.random.default_rng(2))
    naive = rows[1]
    # Four strata, n=16 per arm/stratum: variance=4*(.25/16+.25/16)/16.
    assert naive["estimate"] == 0
    assert naive["upper"] == pytest.approx(1.959963984540054 * np.sqrt(1 / 128))
    observed[arms == 4] = False
    bad = analyze_trial(y, observed, arms, strata, config, np.random.default_rng(2))
    assert not any(row["valid"] for row in bad)
    assert bad[0]["finite_bootstrap_fraction"] == 0


def test_replication_is_index_addressed_and_rng_streams_separate(config):
    s = scenarios(config)[0]
    calibration = calibrate(s, config["stratum_logits"])
    a = pd.DataFrame(replicate(config, s, calibration, 3))
    replicate(config, s, calibration, 9)
    b = pd.DataFrame(replicate(config, s, calibration, 3))
    pd.testing.assert_frame_equal(a, b)
    fewer = dict(config, bootstrap_draws=31)
    c = pd.DataFrame(replicate(fewer, s, calibration, 3))
    pd.testing.assert_series_equal(a.estimate, c.estimate)
    pd.testing.assert_series_equal(a.primary_observed, c.primary_observed)


def test_invalid_replicates_remain_in_primary_denominator(config):
    cfg = dict(config, replicates=2, designs=config["designs"][:1], profiles=config["profiles"][:1], marginal_effects=[0.0])
    records = []
    for method in ("two_way_percentile", "naive_wald"):
        for index, valid in enumerate((True, False)):
            records.append(dict(scenario=0, replicate=index, method=method, valid=valid, reject=valid, cover=False,
                                width=.2 if valid else np.nan, estimate=.2, primary_observed=128,
                                finite_bootstrap_fraction=1.0 if valid else 0.0))
    summary, gates = summarize(pd.DataFrame(records), cfg)
    assert list(summary.rejection_rate) == [.5, .5]
    assert list(summary.rejection_given_valid) == [1, 1]
    assert list(summary.invalid_fraction) == [.5, .5]
    assert not gates.null_screen_pass.any()
    with pytest.raises(ValueError, match="replicate"):
        summarize(pd.DataFrame(records[:-1]), cfg)


def test_wilson_known_boundary():
    lo, hi = wilson(0, 1000)
    assert lo == 0
    assert hi == pytest.approx(0.0038267584855551234)
    lo, hi = wilson(1000, 1000)
    assert hi == 1
    for total in (50, 1000):
        for successes in (0, 1, total//2, total-1, total):
            lower, upper = wilson(successes, total)
            assert lower <= successes/total <= upper
    lo, hi = wilson(50, 1000)
    assert lo < .05 < hi


def test_figure_handles_zero_and_unit_rejection_rates(tmp_path, config, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    spec = importlib.util.spec_from_file_location("inference_figure_test", ROOT / "scripts/run_decision_inference_sim.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    rows = []
    for s in scenarios(config):
        rate = 0.0 if s["effect"] == 0 else 1.0
        lower, upper = wilson(int(rate * 1000), 1000)
        for method in ("two_way_percentile", "naive_wald"):
            rows.append(dict(profile=s["profile"], reviewers=s["reviewers"], cases=s["cases"], effect=s["effect"],
                             method=method, rejection_rate=rate, rejection_wilson_lower=lower, rejection_wilson_upper=upper))
    runner.draw_figure(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "rejection_rates.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "rejection_rates.pdf").read_bytes().startswith(b"%PDF")


def test_runner_partial_resume_drift_and_output_corruption(tmp_path, monkeypatch, config):
    spec = importlib.util.spec_from_file_location("inference_runner_test", ROOT / "scripts/run_decision_inference_sim.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Isolated fixture root: no real run files or repository state are mutated.
    for name in ("src/financepaper/evaluation/decision_inference.py", "src/financepaper/evaluation/decision_value.py",
                 "tests/test_decision_inference.py", "docs/DECISION_INFERENCE_SIM_PROTOCOL.md", "pyproject.toml", "uv.lock"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    monkeypatch.chdir(tmp_path)
    original_check_output = runner.subprocess.check_output
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda args, **kw: "FIXTURE_ONLY\n" if args[0] == "git" else original_check_output(args, **kw))
    cfg = dict(config, replicates=2, chunk_size=1, bootstrap_draws=20,
               designs=config["designs"][:1], profiles=config["profiles"][:1], marginal_effects=[0.0])
    cfg_path = tmp_path / "fixture.json"
    cfg_path.write_text(json.dumps(cfg))
    output = Path("runs/decision_value_pilot/fixture")
    runner.run(cfg_path, output, stop_after_chunks=1)
    first = output / "chunk_00_0000_0001.csv"
    original = first.read_bytes()
    runner.run(cfg_path, output, resume=True, stop_after_chunks=1)
    assert first.read_bytes() == original
    assert (output / "chunk_00_0001_0002.csv").exists()
    events = [json.loads(x) for x in (output / "progress.jsonl").read_text().splitlines()]
    assert any(e["event"] == "resume_verified" and e["stage"] == "chunk_00_0000_0001" for e in events)
    cfg_path.write_text(json.dumps(dict(cfg, seed=42)))
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.run(cfg_path, output, resume=True, stop_after_chunks=1)
    cfg_path.write_text(json.dumps(cfg))
    first.write_bytes(original + b"tamper\n")
    with pytest.raises(ValueError, match="(?i)hash|changed|mismatch"):
        runner.run(cfg_path, output, resume=True, stop_after_chunks=1)
