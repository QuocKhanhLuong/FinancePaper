"""Explicit download and experiment entry points; imports never fetch data."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="financepaper")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download and verify the official UCI workbook")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    pilot = sub.add_parser("pilot", help="Run the single-seed restoration pilot")
    pilot.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    pilot.add_argument("--data", type=Path, help="Override the local workbook/CSV path")
    pilot.add_argument("--output", type=Path, help="Use a new, empty output directory")
    args = parser.parse_args()
    if args.command == "download":
        from .data.taiwan import download_taiwan

        print(download_taiwan(args.raw_dir))
    else:
        from .experiments.pilot import run_pilot

        print(run_pilot(args.config, data_path=args.data, output_dir=args.output))
