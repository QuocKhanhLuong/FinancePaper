"""F0 artifact audit + F1 synthetic export. No fit, SHAP recomputation or people."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import pandas as pd
from tqdm.auto import tqdm

from financepaper.evaluation.decision_value import (
    audit_dataset, break_even, check_hashes, digest, input_inventory, plot_frontiers, write_json,
)
from financepaper.evaluation.decision_vignettes import export_synthetic_bundle


def utc():
    return datetime.now(timezone.utc).isoformat()


def stage(output, name, run, *, resume):
    receipt = output / f"{name}_complete.json"
    if receipt.exists():
        if not resume:
            raise FileExistsError(f"Stage {name} exists; pass --resume")
        saved = json.loads(receipt.read_text())
        check_hashes(saved["outputs"])
        log(output, name, "resume_verified", original_elapsed_seconds=saved["elapsed_seconds"])
        return
    started = time.perf_counter()
    log(output, name, "started")
    paths = run()
    saved = {"stage": name, "elapsed_seconds": time.perf_counter() - started,
             "completed_utc": utc(), "outputs": {str(p): digest(p) for p in paths}, "device": "cpu"}
    write_json(receipt, saved)
    log(output, name, "complete", elapsed_seconds=saved["elapsed_seconds"])


def log(output, stage_name, state, **values):
    event = {"utc": utc(), "stage": stage_name, "state": state, **values}
    with (output / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def run(config_path, output, resume=False, stop_after=None):
    started = time.perf_counter()
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("New runs must be inside ignored runs/decision_value_pilot")
    output.mkdir(parents=True, exist_ok=True)
    # Keep plotting caches local and writable; they are not study outputs.
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.resolve() / ".cache"))
    config = json.loads(config_path.read_text())
    sources = [config_path, Path(__file__), Path("uv.lock"), Path("pyproject.toml"),
               Path("src/financepaper/evaluation/decision_value.py"),
               Path("src/financepaper/evaluation/decision_vignettes.py"),
               Path("src/financepaper/reliability/reason_sets.py"), Path("docs/DECISION_VALUE_PROTOCOL.md")]
    log(output, "preflight", "hashing_inputs")
    frozen = {"config": config, "sources": {str(p): digest(p) for p in sources},
              "inputs": input_inventory(config)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError("Existing run requires --resume")
        if manifest["frozen"] != frozen:
            raise ValueError("Resume rejected: config, code or input hashes changed; use a new run")
    else:
        manifest = {"created_utc": utc(), "frozen": frozen,
                    "base_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                    "python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
                    "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "joblib", "tqdm", "matplotlib", "pytest", "torch")},
                    "status": "RUNNING", "human_study": "NOT_RUN", "predictor_training": "NOT_RUN",
                    "warnings": ["Exploratory reuse; inspected cohorts", "Utility assumptions are not welfare",
                                 "Compute/delay not monetized", "Polish exact-feature clusters are not verified company IDs"]}
        write_json(manifest_path, manifest)
    stages = [*config["datasets"], "frontier", "vignettes"]
    for name in tqdm(stages, desc="Decision-value stages", unit="stage"):
        if name in config["datasets"]:
            def work(dataset=name):
                audit_dataset(config, dataset, output)
                return sorted(output.glob(f"utility_{dataset}_*.csv"))
        elif name == "frontier":
            def work():
                counts = pd.concat([pd.read_csv(output / f"utility_{d}_counts.csv") for d in config["datasets"]])
                utility = pd.concat([pd.read_csv(output / f"utility_{d}_grid.csv") for d in config["datasets"]])
                counts.to_csv(output / "utility_counts.csv", index=False)
                utility.to_csv(output / "utility_grid.csv", index=False)
                break_even(counts).to_csv(output / "utility_break_even.csv", index=False)
                plot_frontiers(counts, utility, output)
                return [output / n for n in ("utility_counts.csv", "utility_grid.csv", "utility_break_even.csv", "utility_frontier.png", "utility_frontier.pdf")]
        else:
            def work():
                export_synthetic_bundle(output / "case_bundle", config["seed"])
                return sorted(p for p in (output / "case_bundle").rglob("*") if p.is_file())
        stage(output, name, work, resume=resume)
        if stop_after == name:
            log(output, "run", "intentional_pause_after_stage", completed_stage=name)
            return
    # Verify the inputs were not modified while auditing them.
    check_hashes(frozen["inputs"])
    manifest.update(status="COMPLETE", completed_utc=utc(), invocation_elapsed_seconds=time.perf_counter() - started,
                    stage_receipts={name: json.loads((output / f"{name}_complete.json").read_text()) for name in stages})
    write_json(manifest_path, manifest)
    log(output, "run", "complete", elapsed_seconds=manifest["invocation_elapsed_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/decision_value_pilot.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=["taiwan", "polish", "frontier", "vignettes"])
    args = parser.parse_args()
    run(args.config, args.output, args.resume, args.stop_after)
