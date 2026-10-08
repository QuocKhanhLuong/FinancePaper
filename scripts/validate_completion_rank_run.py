"""Run final software checks and exercise C5 resume guards on disposable copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from financepaper.evaluation.decision_value import check_hashes, digest, write_json


def run(run_dir, output):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Validation output must be in the ignored run root")
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((run_dir/"execution_receipt.json").read_text())
    check_hashes(receipt["frozen"]["sources"])
    check_hashes(receipt["outputs"])
    check_hashes(receipt["task_receipts"])
    records = []
    runner = [sys.executable, "scripts/run_completion_rank_audit.py"]

    def command(label, args, expected=0, contains=None):
        tick = time.perf_counter()
        process = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = output/f"{label}.log"
        path.write_text(process.stdout)
        assert process.returncode == expected or (expected == "nonzero" and process.returncode != 0), process.stdout[-2000:]
        assert contains is None or contains in process.stdout, process.stdout[-2000:]
        record = dict(check=label, returncode=process.returncode, elapsed_seconds=time.perf_counter()-tick,
                      log=str(path), sha256=digest(path), passed=True)
        records.append(record)
        print(json.dumps(record), flush=True)

    command("completed_resume", runner+["--output", str(run_dir), "--resume"], contains='"new": 0, "reused": 94')
    config = output/"scratch_config.json"
    original = Path("configs/completion_rank_audit.json").read_bytes()
    config.write_bytes(original)
    scratch = output/"scratch_run"
    args = runner+["--config", str(config), "--output", str(scratch)]
    command("intentional_pause", args+["--stop-after-tasks", "1"], contains='"intentional_pause"')
    config.write_bytes(original+b"\n")
    command("config_drift_rejected", args+["--resume"], expected="nonzero", contains="Resume rejected")
    config.write_bytes(original)
    task_receipt = next(scratch.glob("*_complete.json"))
    task = Path(next(iter(json.loads(task_receipt.read_text())["outputs"])))
    task.write_bytes(task.read_bytes()+b"\n")
    command("completed_output_corruption_rejected", args+["--resume"], expected="nonzero", contains="hash mismatch")
    write_json(output/"guard_checks.json", dict(checks=records, actual_research_outputs_modified=False))

    env = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[variable] = "1"
    tick = time.perf_counter()
    process = subprocess.run([sys.executable, "-m", "pytest", "-q"], env=env,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = output/"pytest_final.txt"
    path.write_text(process.stdout)
    write_json(output/"pytest_final.json", dict(returncode=process.returncode,
        wall_seconds=time.perf_counter()-tick, command=[sys.executable, "-m", "pytest", "-q"],
        environment={k: env[k] for k in env if k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), final_lines=process.stdout.splitlines()[-16:]))
    print(process.stdout, flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)
