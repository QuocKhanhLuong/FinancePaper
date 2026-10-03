"""Bounded audit, conditional fitting, and frozen outer assessment."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name,"1")
import argparse
from pathlib import Path
from financepaper.experiments.revision_study import run_audit
from financepaper.experiments.revision_selection import run_baselines
from financepaper.experiments.revision_assessment import run_freeze,run_assessment

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/revision_study.yaml")
    parser.add_argument("--output",default="outputs/revision_study")
    parser.add_argument("--device",default=None)
    parser.add_argument("--stage",choices=["audit","baselines","freeze","assess"],default="audit")
    args=parser.parse_args()
    {"audit":run_audit,"baselines":run_baselines,"freeze":run_freeze,"assess":run_assessment}[args.stage](Path(args.config),Path(args.output),args.device)
