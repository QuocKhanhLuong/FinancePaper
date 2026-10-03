"""Prepare user-acquired official files; never downloads or accepts provider terms."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from financepaper.data.freddie import (
    ROLE_BY_YEAR, FreddieDataError, prepare_vintage, sha256,
    validate_analysis_freeze, validate_partition_metadata, validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stage", choices=["pilot", "development", "confirmation"], required=True)
    parser.add_argument("--freeze-manifest", type=Path,
                        help="Required for confirmation; hashes all predictors/completions/policies")
    args = parser.parse_args()
    try:
        receipt = validate_receipt(args.receipt)  # Before touching raw files/output.
        development = ROOT / "data" / "processed" / "freddie_r47_development"
        freeze = None
        prior_targets = []
        if args.stage == "confirmation":
            if args.freeze_manifest is None:
                raise FreddieDataError("Fit and freeze on development before opening assessment archives")
            freeze = validate_analysis_freeze(args.freeze_manifest, ROOT, development / "intake_manifest.json")
            prior = json.loads((development / "intake_manifest.json").read_text())
            if prior.get("status") != "PREPARED_NOT_TRAINED" or prior.get("receipt_sha256") != sha256(args.receipt):
                raise FreddieDataError("Development intake or acquisition receipt mismatch")
            for year, role in ROLE_BY_YEAR.items():
                if role != "test":
                    prior_targets.append(pd.read_csv(development / f"targets_{year}.csv.gz", index_col="loan_id",
                                                     dtype={"landmark": str, "horizon_end": str, "label": "Int8"}))
        years = ([2000, 2001] if args.stage == "pilot" else
                 sorted(y for y, role in ROLE_BY_YEAR.items() if (role == "test") == (args.stage == "confirmation")))
        paths = [args.raw_dir / f"sample_{year}.zip" for year in years]
        if not all(p.is_file() for p in paths):
            raise FreddieDataError("Required official annual sample archives are absent")
        output = ROOT / "data" / "processed" / f"freddie_r47_{args.stage}"
        if output.exists():
            raise FreddieDataError("Output exists; preserve it and audit before a new intake")
        output.mkdir(parents=True, mode=0o700)
        manifest = {"status": "INCOMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "receipt": receipt, "receipt_sha256": sha256(args.receipt),
                    "stage": args.stage, "vintages": [], "features_only_not_an_experiment": True,
                    "assessment_freeze": freeze}
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        files = ["configs/freddie_validation.yaml", "docs/FREDDIE_TARGET_DEFINITION.md",
                 "docs/FREDDIE_DATA_PROTOCOL.md", "docs/FREDDIE_LEAKAGE_AUDIT.md",
                 "src/financepaper/data/freddie.py", "scripts/prepare_freddie.py"]
        manifest["implementation_sha256"] = {f: sha256(ROOT / f) for f in files}
        manifest_path = output / "intake_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        target_blocks = prior_targets
        for year, path in zip(years, paths):
            prepared = prepare_vintage(path, year)  # No source-count override in CLI.
            prepared.features.to_csv(output / f"features_{year}.csv.gz", compression={"method": "gzip", "mtime": 0})
            prepared.targets.to_csv(output / f"targets_{year}.csv.gz", compression={"method": "gzip", "mtime": 0})
            np.savez_compressed(output / f"natural_mask_{year}.npz",
                                mask=prepared.features.isna().to_numpy(),
                                loan_id=prepared.features.index.to_numpy(dtype=str),
                                columns=prepared.features.columns.to_numpy(dtype=str))
            manifest["vintages"].append(prepared.manifest)
            target_blocks.append(prepared.targets)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        targets = pd.concat(target_blocks)
        validate_partition_metadata(targets)
        labelled = int(targets.label.notna().sum())
        counts = targets.groupby("role").label.sum().fillna(0).astype(int).to_dict()
        manifest.update(status="PREPARED_NOT_TRAINED", labelled_loans=labelled,
                        observed_events_by_role=counts,
                        confirmation_scale_met=labelled >= 500000,
                        confirmation_event_support_met=counts.get("train", 0) >= 100 and counts.get("test", 0) >= 100)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Prepared local restricted artifacts: {output}. No models trained; inspect intake gates locally.")
    except (FreddieDataError, OSError, ValueError, KeyError) as exc:
        # Never print raw records, IDs, or parsing exceptions containing field contents.
        parser.exit(2, f"Intake stopped ({type(exc).__name__}); no experiment authorized. "
                      "Review the receipt, schema and local intake manifest.\n")


if __name__ == "__main__":
    main()
