"""Run frozen, resumable F2 inference simulations, without participant data."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.decision_inference import calibrate, replicate, scenarios, summarize, validate_config
from financepaper.evaluation.decision_value import check_hashes, digest, write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(output, event, *, console=True, **values):
    row = dict(utc=utc(), event=event, **values)
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    if console:
        tqdm.write(json.dumps(row, allow_nan=False))


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def complete_stage(output, name, work, resume):
    receipt_path = output / f"{name}_complete.json"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"Stage exists: {name}")
        receipt = json.loads(receipt_path.read_text())
        check_hashes(receipt["outputs"])
        log(output, "resume_verified", console=False, stage=name)
        return receipt, False
    started = time.perf_counter()
    paths = work()
    receipt = dict(stage=name, elapsed_seconds=time.perf_counter()-started, completed_utc=utc(),
                   outputs={str(p): digest(p) for p in paths}, device="cpu")
    write_json(receipt_path, receipt)
    log(output, "stage_complete", stage=name, elapsed_seconds=receipt["elapsed_seconds"])
    return receipt, True


def table(frame):
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def draw_figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profiles = list(summary.profile.unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = {"two_way_percentile": "#0066A1", "naive_wald": "#C46600"}
    markers = {"two_way_percentile": "o", "naive_wald": "s"}
    for ax, profile in zip(axes.flat, profiles, strict=True):
        part = summary[summary.profile == profile]
        for method in colors:
            for j, ((r, c), group) in enumerate(part[part.method == method].groupby(["reviewers", "cases"])):
                group = group.sort_values("effect")
                y = group.rejection_rate.to_numpy()
                ax.errorbar(group.effect, y,
                            yerr=np.array([y-group.rejection_wilson_lower, group.rejection_wilson_upper-y]),
                            color=colors[method], marker=markers[method], linestyle=(":", "--", "-")[j],
                            capsize=2, label=f"{method}: R={r}, C={c}")
        ax.axhline(.05, color="gray", linewidth=.8, alpha=.7)
        ax.axhline(.8, color="gray", linewidth=.8, alpha=.4)
        ax.set_title(profile)
        ax.set_xticks([0, .05, .1], ["0 (null)", ".05", ".10"])
        ax.set_ylim(0, 1)
        ax.grid(alpha=.15)
    for ax in axes[1]:
        ax.set_xlabel("Assumed population accuracy gain")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rejection / all 1,000 replicates")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Assumption-only method feasibility — 95% Monte Carlo Wilson intervals\n"
                 "Invalid analyses count as non-rejections; no human power or sample-size recommendation", fontsize=12)
    fig.tight_layout(rect=(0, .1, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"rejection_rates.{suffix}", dpi=180)
    plt.close(fig)


def write_report(output, config, chunk_receipts):
    manifest = json.loads((output / "manifest.json").read_text())
    summary = pd.read_csv(output / "scenario_summary.csv")
    gates = pd.read_csv(output / "null_gates.csv")
    calibration = pd.read_csv(output / "calibration.csv")
    null_table = gates[["reviewers", "cases", "profile", "method", "rejection_rate", "rejection_wilson_upper",
                        "invalid_fraction", "null_screen_pass", "conservative_flag"]].copy()
    for col in ("rejection_rate", "rejection_wilson_upper", "invalid_fraction"):
        null_table[col] = null_table[col].map(lambda x: f"{x:.4f}")
    power_table = summary[summary.effect > 0][["reviewers", "cases", "profile", "effect", "method", "rejection_rate",
                                            "rejection_wilson_lower", "rejection_wilson_upper", "null_screen_pass"]].copy()
    for col in ("rejection_rate", "rejection_wilson_lower", "rejection_wilson_upper"):
        power_table[col] = power_table[col].map(lambda x: f"{x:.4f}")
    boot = gates[gates.method == "two_way_percentile"]
    naive = gates[gates.method == "naive_wald"]
    seconds = sum(r["elapsed_seconds"] for r in chunk_receipts)
    events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    resumed = sum(e["event"] == "resume_verified" and e["stage"].startswith("chunk_") for e in events)
    paused = any(e["event"] == "intentional_pause" for e in events)
    report = f"""# F2 inference feasibility — actual simulation, no human evidence

## Decision and scope

Executed all {len(scenarios(config))} prespecified hypothetical scenarios, each
with {config['replicates']} independent outer replicates and {config['bootstrap_draws']}
two-way bootstrap draws. The candidate bootstrap passes the prespecified
inflation/invalid screening gate in **{int(boot.null_screen_pass.sum())}/{len(boot)}**
null settings; **{int(boot.conservative_flag.sum())}/{len(boot)}** are flagged
conservative (null rejection below .025). The naive comparator passes
**{int(naive.null_screen_pass.sum())}/{len(naive)}** null settings.

These results diagnose this secondary candidate procedure under the frozen laws.
They do not validate the planned primary logistic mixed model, choose a method,
establish human benefit, or justify a final sample size. A gate pass is not proof
of exact .05 type-I error. No cutoff, method or scenario was tuned after results.

## VERIFIED — actually run in this stage

- CPU run `{output.name}`, started `{manifest['created_utc']}`;
  base `{manifest['base_git_head']}`, branch `{manifest['branch']}`.
- {len(scenarios(config))*config['replicates']:,} simulated trials, two analyses each;
  {len(scenarios(config))*config['replicates']*config['bootstrap_draws']:,} bootstrap draws attempted.
  Sum of measured simulation-chunk elapsed times: **{seconds:.2f} seconds**.
  This excludes tests, plotting, setup, pauses and work before the run.
- Python `{manifest['python']}`, platform `{manifest['platform']}`, actual device CPU.
  Dependency versions and all source/config/protocol hashes: `execution_receipt.json`.
- Marginal arm probabilities calibrated by 64-node Gaussian quadrature and checked
  with 128 nodes; largest target error
  **{calibration[['arm3_128node_error','arm4_128node_error']].abs().to_numpy().max():.3g}**.
  Random slopes at gain zero implement a weak marginal null, not a sharp null.
- Real intentional pause recorded: **{paused}**; verified reused chunk receipts:
  **{resumed}** at report creation. Index-addressed independent data/bootstrap RNG
  streams, 50-replicate checkpoints, SHA256 checks, JSONL progress and tqdm/ETA.
- No participant observations, predictor training, Taiwan/Polish model selection,
  new clinical/financial dataset or simulated individual-rating export.
  Tests and final preservation/push checks are recorded in `validation_receipt.json`.
  That receipt also lists development failures and repeated attempts; repeated
  same-seed executions are not independent confirmation.

## Null calibration — all prespecified settings

Two-sided alpha .05; Wilson upper <=.075 and invalid fraction <=.01 is the
frozen diagnostic gate. Invalid analyses count as non-rejections in all 1,000.
Conservative rejection can coexist with a gate pass and imply weak power.

{table(null_table)}

## Conditional power under assumed effects

Effects .05 and .10 are sensitivity values, **not an expert-approved SESOI**.
Rows with a failed matched null screen must not be used for sample-size claims.
Wilson intervals describe finite outer-simulation uncertainty; 499 inner draws
also add Monte Carlo noise. Estimates are about this bootstrap/Wald procedure,
not about the unrun primary mixed model. No design is selected here.

{table(power_table)}

![Rejection rates with Monte Carlo uncertainty](rejection_rates.png)

`scenario_summary.csv` additionally includes coverage, bias, interval width,
valid-only rejection, invalid fraction, smallest finite-bootstrap fraction and
mean observed primary ratings. Coverage conditional on validity is labeled;
`coverage_all` counts failures as noncoverage. Small complete or MCAR samples
and a conservative procedure require both calibration and power checks.

## REPORTED from prior reports — not rerun

F0 utility audit, the 245-case rule baseline, selected 16-case fixture audit,
arm text-length differences and incomplete browser QA are historical evidence
from commits `eb6689c` and `f5ae1de`. This stage did not repeat those experiments
or repair/claim browser testing. No archive item became a new execution queue.

## PROPOSED and NOT RUN

- **NOT RUN:** human study, expert approval of task/rubric/SESOI, consent or
  recruitment, actual human timings/accuracy, primary GLMM fit/convergence/power,
  MAR/MNAR or differential attrition, larger sensitivity grids and any pilot.
- **PROPOSED:** retain this as a falsification/feasibility receipt for the
  secondary analysis; freeze a separate primary-analysis specification only
  after the task and meaningful effect are agreed. Do not tune using these
  outcomes and relabel an exploratory repair as this frozen run.
- **One next action:** domain expert/supervisor reviews the existing fixture
  and rubric with `docs/DECISION_VALUE_REVIEW_GUIDE.md`, including the fact that
  all answers are already recoverable from the current record in arm 1.

## Reproduce / resume

From the repository root with the existing locked environment:

```sh
rtk proxy .venv/bin/python scripts/run_decision_inference_sim.py --output runs/decision_value_pilot/inference_NEW
rtk proxy .venv/bin/python scripts/run_decision_inference_sim.py --output runs/decision_value_pilot/inference_NEW --resume
```

`--stop-after-chunks 1` provides a real pause after one newly completed chunk.
Resume rejects source/config/dependency drift and altered completed outputs.
Only an explicit aggregate allowlist is published; local replicate statistics,
logs and run receipts stay in the ignored run directory. Historical results and
configs, lockfile and main remain unchanged by this stage.

## Method references and limitations

The product-weight row/column construction follows the crossed-data resampling
idea in Owen (2007), [The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111),
and Owen & Eckles (2012), [Bootstrapping data arrays of arbitrary order](https://arxiv.org/abs/1106.2125).
Their results are not a guarantee for this finite binary percentile interval.
Case resampling is stratified; reviewer/case balance can break within bootstrap
draws. MCAR, Gaussian effects, fixed four-stratum weights and Bernoulli outcomes
are assumptions, not measured properties of a future study.
"""
    (output / "RESULTS.md").write_text(report)


def publish_results(output, destination):
    root = Path("reports/decision_inference_sim").resolve()
    if not destination.resolve().is_relative_to(root):
        raise ValueError("Publish only inside reports/decision_inference_sim")
    names = ("RESULTS.md", "scenario_summary.csv", "null_gates.csv", "calibration.csv",
             "rejection_rates.png", "rejection_rates.pdf", "execution_receipt.json")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source, target = output / name, destination / name
        if target.exists() and digest(target) != digest(source):
            raise FileExistsError(f"Refusing to overwrite published result: {target}")
        shutil.copy2(source, target)
    write_json(destination / "publication_receipt.json", dict(
        source_run=str(output), copied_utc=utc(), scope="AGGREGATE_ASSUMPTION_ONLY_SIMULATION",
        outputs={str(destination / name): digest(destination / name) for name in names}))


def run(config_path, output, *, resume=False, stop_after_chunks=None, publish=None):
    started = time.perf_counter()
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Runs must stay in ignored runs/decision_value_pilot")
    if stop_after_chunks is not None and stop_after_chunks < 1:
        raise ValueError("Stop count must be positive")
    config = json.loads(config_path.read_text())
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
    sources = [config_path, Path(__file__), Path("src/financepaper/evaluation/decision_inference.py"),
               Path("src/financepaper/evaluation/decision_value.py"), Path("tests/test_decision_inference.py"),
               Path("docs/DECISION_INFERENCE_SIM_PROTOCOL.md"), Path("pyproject.toml"), Path("uv.lock")]
    frozen = dict(config=config, sources={str(p): digest(p) for p in sources},
                  dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "tqdm", "matplotlib")},
                  python=platform.python_version(), device="cpu", numeric_threads=1)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume or manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: require --resume and identical code/config/dependencies")
    else:
        if resume:
            raise ValueError("Cannot resume a nonexistent run")
        manifest = dict(created_utc=utc(), frozen=frozen, python=platform.python_version(), platform=platform.platform(),
                        base_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                        status="RUNNING", invocations=[], human_study="NOT_RUN", primary_glmm="NOT_RUN")
        write_json(manifest_path, manifest)
    log(output, "invocation_started", resume=resume, total_replicates=len(scenarios(config))*config["replicates"])
    calibrations = {s["scenario"]: calibrate(s, config["stratum_logits"]) for s in scenarios(config)}

    def calibration_work():
        atomic_csv(pd.DataFrame([dict(scenario=k, **v) for k, v in calibrations.items()]), output / "calibration.csv")
        return [output / "calibration.csv"]

    complete_stage(output, "calibration", calibration_work, resume)
    chunk_paths, chunk_receipts = [], []
    new_chunks = 0
    with tqdm(total=len(scenarios(config))*config["replicates"], desc="Frozen F2 simulation", unit="trial", mininterval=2) as bar:
        for scenario in scenarios(config):
            sid = scenario["scenario"]
            bar.set_postfix(profile=scenario["profile"], R=scenario["reviewers"], C=scenario["cases"], effect=scenario["effect"], refresh=False)
            for start in range(0, config["replicates"], config["chunk_size"]):
                end = min(start+config["chunk_size"], config["replicates"])
                name = f"chunk_{sid:02d}_{start:04d}_{end:04d}"
                path = output / f"{name}.csv"

                def work():
                    records = []
                    for index in range(start, end):
                        records.extend(replicate(config, scenario, calibrations[sid], index))
                        bar.update(1)
                    atomic_csv(pd.DataFrame(records), path)
                    return [path]

                receipt, new = complete_stage(output, name, work, resume)
                chunk_paths.append(path)
                chunk_receipts.append(receipt)
                if not new:
                    bar.update(end-start)
                else:
                    new_chunks += 1
                if stop_after_chunks is not None and new_chunks >= stop_after_chunks:
                    manifest["status"] = "PAUSED"
                    manifest["invocations"].append(dict(completed_utc=utc(), elapsed_seconds=time.perf_counter()-started, intentional_pause=True))
                    write_json(manifest_path, manifest)
                    log(output, "intentional_pause", new_chunks=new_chunks, verified_replicates=int(bar.n))
                    return

    def aggregate_work():
        records = pd.concat([pd.read_csv(p) for p in chunk_paths], ignore_index=True)
        summary, gates = summarize(records, config)
        atomic_csv(summary, output / "scenario_summary.csv")
        atomic_csv(gates, output / "null_gates.csv")
        draw_figure(summary, output)
        write_json(output / "execution_receipt.json", dict(
            run=str(output), created_utc=manifest["created_utc"], completed_utc=utc(),
            base_git_head=manifest["base_git_head"], branch=manifest["branch"], platform=manifest["platform"],
            frozen=frozen, chunks=len(chunk_paths), outer_replicates=len(records)//2,
            simulation_chunk_elapsed_seconds=sum(r["elapsed_seconds"] for r in chunk_receipts),
            chunk_receipt_hashes={str(output / f"{r['stage']}_complete.json"): digest(output / f"{r['stage']}_complete.json") for r in chunk_receipts},
            human_study="NOT_RUN", primary_glmm="NOT_RUN"))
        write_report(output, config, chunk_receipts)
        return [output / name for name in ("scenario_summary.csv", "null_gates.csv", "rejection_rates.png",
                                           "rejection_rates.pdf", "execution_receipt.json", "RESULTS.md")]

    complete_stage(output, "aggregate", aggregate_work, resume)
    check_hashes(frozen["sources"])
    manifest.update(status="COMPLETE", completed_utc=utc())
    manifest["invocations"].append(dict(completed_utc=utc(), elapsed_seconds=time.perf_counter()-started, intentional_pause=False))
    write_json(manifest_path, manifest)
    log(output, "run_complete", chunks=len(chunk_paths), invocation_elapsed_seconds=time.perf_counter()-started)
    if publish is not None:
        publish_results(output, publish)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/decision_inference_sim.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    run(args.config, args.output, resume=args.resume, stop_after_chunks=args.stop_after_chunks, publish=args.publish)
