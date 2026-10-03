"""Reproducible, separately gated validation stages (CPU, no new architecture)."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")
import argparse
from pathlib import Path
from financepaper.experiments.decisive_validation import freeze, run_taiwan, run_external_fit, run_external_evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["freeze", "taiwan", "external-fit", "external-evaluate"])
    p.add_argument("--config", default="configs/decisive_validation.yaml")
    p.add_argument("--output", default="outputs/decisive_validation")
    args = p.parse_args()
    if args.stage == "freeze":
        freeze(args.config)
    else:
        {"taiwan": run_taiwan, "external-fit": run_external_fit, "external-evaluate": run_external_evaluate}[args.stage](Path(args.output))
