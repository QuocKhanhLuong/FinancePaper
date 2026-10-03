#!/usr/bin/env python3
"""Run the frozen bounded follow-up; numerical thread limits precede imports."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
from pathlib import Path
from financepaper.experiments.robustness_followup import evaluate, fit_all, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robustness_followup.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--stage", choices=["fit", "evaluate", "all"], default="all")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    output = args.output or Path(cfg["output_dir"])
    if args.stage in ("fit", "all"):
        fit_all(args.config, output, args.device)
    if args.stage in ("evaluate", "all"):
        evaluate(output, args.device)
    print(output)


if __name__ == "__main__":
    main()
