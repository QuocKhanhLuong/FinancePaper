"""Run the bounded, frozen, evaluator-only acquisition headroom gate."""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
from pathlib import Path
from financepaper.experiments.verification_headroom import freeze, evaluate, report

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","evaluate","report"))
    parser.add_argument("--config",default="configs/verification_headroom.yaml")
    parser.add_argument("--output",default="outputs/verification_headroom")
    args=parser.parse_args()
    if args.stage=="freeze":freeze(args.config)
    elif args.stage=="evaluate":evaluate(Path(args.output))
    else:report(Path(args.output))
