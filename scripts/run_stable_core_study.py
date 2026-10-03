import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
from pathlib import Path
from financepaper.experiments.stable_core_study import freeze, calibration, evaluate

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["freeze","calibrate","evaluate"])
    p.add_argument("--config",default="configs/stable_core_study.yaml")
    p.add_argument("--output",default="outputs/stable_core_study")
    a=p.parse_args()
    if a.stage=="freeze":freeze(a.config)
    else:{"calibrate":calibration,"evaluate":evaluate}[a.stage](Path(a.output))
