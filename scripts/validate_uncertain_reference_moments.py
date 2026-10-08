"""Exercise uncertain-reference-moment audit resume/integrity guards and repository tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output, run):
    if not output.resolve().is_relative_to(Path("runs/decision_value_pilot").resolve()):
        raise ValueError("Use an ignored validation directory")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    commands = []

    def execute(label, args, expected_error=None):
        tick = time.perf_counter()
        print(f"Validation {len(commands)+1}/5: {label}", flush=True)
        result = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = output / f"{label}.log"
        path.write_text(result.stdout)
        success = result.returncode == 0 if expected_error is None else (
            result.returncode != 0 and expected_error in result.stdout)
        commands.append(dict(label=label, argv=args, returncode=result.returncode,
            expected_error=expected_error, accepted=success, seconds=time.perf_counter()-tick,
            log=str(path), log_sha256=sha(path)))
        if not success:
            raise RuntimeError(f"Unexpected outcome: {label}; see {path}")
        return result.stdout

    runner = [sys.executable, "scripts/run_uncertain_reference_moments_v2.py"]
    execute("complete_resume", runner+["--output", str(run), "--resume"])
    events = [json.loads(line) for line in (run/"progress.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "complete" and events[-1]["new"] == 0 and events[-1]["reused"] == 1716
    config = output/"scratch_config.json"
    original = Path("configs/uncertain_reference_moments.json").read_bytes()
    config.write_bytes(original)
    scratch = output/"scratch_run"
    args = runner+["--config", str(config), "--output", str(scratch)]
    execute("scratch_pause", args+["--stop-after-tasks", "1"])
    assert len(list(scratch.glob("*_complete.json"))) == 1
    config.write_bytes(original+b"\n")
    execute("config_drift", args+["--resume"], "Resume rejected: source/config/dependency drift")
    config.write_bytes(original)
    task_receipt = json.loads(next(scratch.glob("*_complete.json")).read_text())
    payload = Path(next(iter(task_receipt["outputs"])))
    payload.write_bytes(payload.read_bytes()+b"\n")
    execute("task_corruption", args+["--resume"], "artifact hash mismatch")
    test_output = execute("full_pytest", [sys.executable, "-m", "pytest", "-q"])
    summary = re.search(r"\d+ passed[^\n]*", test_output)
    assert summary is not None
    receipt = dict(completed_utc=datetime.now(timezone.utc).isoformat(), device="cpu",
        numeric_threads=1, validator_sha256=sha(__file__), commands=commands,
        complete_resume=dict(new=0,reused=1716), pytest_summary=summary.group(0),
        real_run=str(run), real_receipt_sha256=sha(run/"execution_receipt.json"),
        scratch_corruption_intentional=True)
    (output/"validation_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    main(args.output,args.run)
