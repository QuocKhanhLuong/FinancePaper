#!/usr/bin/env python3
"""Command-line entry point for the bounded temporal pilot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


# Keep the optional PyTorch/XGBoost/SHAP native runtimes from competing for
# Apple Silicon thread pools.  Users can override these explicitly in the
# shell; this is a CLI-local reproducibility default, not a package-wide change.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from financepaper.experiments.temporal_pilot import print_device_info, run_temporal_pilot  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible temporal credit-risk pilot")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "temporal_pilot.yaml")
    parser.add_argument("--output", type=Path, help="New output directory for predict/all; existing directory for explain")
    parser.add_argument("--data", type=Path, help="Override the local Taiwan .xls/.csv path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--stage", choices=("predict", "explain", "all"), default="all")
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="Print PyTorch/device/model-size diagnostics without loading the dataset",
    )
    args = parser.parse_args(argv)
    if args.device_info:
        print_device_info(args.device)
        return 0
    output = run_temporal_pilot(
        args.config,
        output_dir=args.output,
        data_path=args.data,
        device=args.device,
        stage=args.stage,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
