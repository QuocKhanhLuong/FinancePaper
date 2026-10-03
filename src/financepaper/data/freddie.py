"""Offline Release-47 intake. No network, credentials, inference, or imputation.

Only synthetic fixtures have been exercised until official files are supplied.
See docs/FREDDIE_TARGET_DEFINITION.md for the fixed reporting-time estimand.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ORIGINATION_COLUMNS = tuple("""fico first_payment_date first_time_homebuyer maturity_date
msa mi_percentage units occupancy original_cltv original_dti original_upb original_ltv
original_rate channel prepayment_penalty amortization_type property_state property_type
postal_code loan_id loan_purpose original_term borrowers seller super_conforming
pre_harp_loan_id special_eligibility harp_indicator valuation_method interest_only
vantage_score4""".split())
PERFORMANCE_COLUMNS = tuple("""loan_id period current_upb delinquency_status loan_age
remaining_maturity defect_date modification_flag zero_balance_code zero_balance_date
current_rate deferred_upb ddlpi mi_recoveries net_sales_proceeds non_mi_recoveries
total_expenses legal_costs maintenance_costs taxes_insurance misc_expenses actual_loss
cumulative_modification_cost step_indicator payment_deferral estimated_ltv zero_balance_upb
delinquent_interest disaster_flag assistance_plan current_modification_cost
interest_bearing_upb mi_cancellation servicer bankruptcy_cramdown_cost""".split())
NUMERIC_FIELDS = ("fico", "original_dti", "original_ltv", "original_cltv", "original_upb",
                  "original_rate", "original_term", "mi_percentage", "units")
CATEGORY_LEVELS = {
    "first_time_homebuyer": ("N", "Y"), "occupancy": ("P", "I", "S"),
    "channel": ("R", "B", "C", "T"), "property_type": ("CP", "CO", "PU", "SF", "MH"),
    "loan_purpose": ("P", "C", "N", "R"),
}
CATEGORICAL_FIELDS = (*CATEGORY_LEVELS, "borrowers")
HISTORY_FIELDS = ("delinquency", "upb", "rate")
FEATURES = (*NUMERIC_FIELDS, *CATEGORICAL_FIELDS,
            *(f"{field}_month_{m}" for field in HISTORY_FIELDS for m in range(1, 7)))
GROUPS = {
    "credit_score": ("fico",), "debt_burden": ("original_dti",),
    "collateral_leverage": ("original_ltv", "original_cltv", "mi_percentage"),
    "contract": ("original_upb", "original_rate", "original_term"),
    "property": ("units", "occupancy", "property_type"),
    "origination_context": ("first_time_homebuyer", "channel", "loan_purpose", "borrowers"),
    **{f"{f}_history": tuple(f"{f}_month_{m}" for m in range(1, 7)) for f in HISTORY_FIELDS},
}
TERMINATIONS = {1, 2, 3, 9, 15, 16, 96}
ROLES = {"train": list(range(2000, 2009)), "development": [2011],
         "probability_calibration": [2014], "release_calibration": [2017],
         "test": [2020, 2021, 2022]}
ROLE_BY_YEAR = {y: role for role, years in ROLES.items() for y in years}
RELEASE = {"provider": "Freddie Mac", "release": 47, "release_date": "2026-07-29",
           "performance_cutoff": "2026-03-31", "schema": "july_2026",
           "terms_version": "2025-11-03",
           "source_url": "https://claritydownload.fmapps.freddiemac.com/",
           "purpose": "noncommercial academic mortgage credit-performance research"}


class FreddieDataError(ValueError):
    """Intake stops; error messages intentionally omit loan-level contents."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_receipt(path: Path) -> dict:
    """Attestation gate, not proof of origin or permission to bypass registration."""
    receipt = json.loads(path.read_text())
    if any(receipt.get(k) != v for k, v in RELEASE.items()):
        raise FreddieDataError("Receipt does not match the frozen release/source/terms/purpose")
    if (receipt.get("researcher_accepted_terms") is not True
            or receipt.get("applicable_clickthrough_reviewed") is not True
            or receipt.get("clickthrough_conflicts_with_audit") is not False):
        raise FreddieDataError("Official acquisition and applicable terms must be attested first")
    try:
        acquired = date.fromisoformat(receipt["download_date"])
    except (ValueError, TypeError, KeyError) as exc:
        raise FreddieDataError("Receipt requires an actual ISO download date") from exc
    if not date(2026, 7, 29) <= acquired <= date.today():
        raise FreddieDataError("Invalid release/download chronology")
    return receipt


def month_number(value: str) -> int:
    if not re.fullmatch(r"\d{6}", value) or not 1 <= int(value[4:]) <= 12:
        raise FreddieDataError("Invalid YYYYMM field")
    return int(value[:4]) * 12 + int(value[4:]) - 1


def month_text(number: int) -> str:
    year, month = divmod(number, 12)
    return f"{year:04d}{month + 1:02d}"


def _number(value: str, missing=()) -> float:
    if not value or value in missing:
        return np.nan
    try:
        result = float(value)
    except ValueError as exc:
        raise FreddieDataError("Invalid numeric field") from exc
    if not np.isfinite(result):
        raise FreddieDataError("Non-finite source numeric field")
    return result


def delinquency(value: str) -> float:
    if value in ("", "XX"):
        return np.nan
    if value == "RA":
        return 100.0  # Internal sentinel; never reaches model features.
    if not re.fullmatch(r"\d{1,2}", value):
        raise FreddieDataError("Unsupported delinquency status")
    return float(value)


def _termination(code: str, effective: str, period: int) -> int:
    if not code:
        if effective:
            raise FreddieDataError("Termination date without code")
        return 0
    if not code.isdigit() or int(code) not in TERMINATIONS:
        raise FreddieDataError("Unsupported termination code")
    # Refuse delayed/backdated termination rather than silently crossing an exit.
    if not effective or month_number(effective) != period:
        raise FreddieDataError("Termination period requires a documented alignment audit")
    return int(code)


def _origination_features(row: dict) -> list[float]:
    sentinels = {"fico": ("9999",), "units": ("99",),
                 **{k: ("999",) for k in ("original_dti", "original_ltv", "original_cltv", "mi_percentage")}}
    values = [_number(row[f], sentinels.get(f, ())) for f in NUMERIC_FIELDS]
    for field, levels in CATEGORY_LEVELS.items():
        value = row[field]
        if value in ("", "9", "99"):
            values.append(np.nan)
        elif value in levels:
            values.append(float(levels.index(value)))
        else:
            raise FreddieDataError("Unsupported origination category")
    borrowers = _number(row["borrowers"], ("99",))
    if np.isfinite(borrowers) and (borrowers < 1 or borrowers > 10 or int(borrowers) != borrowers):
        raise FreddieDataError("Unsupported borrower count")
    values.append(min(borrowers, 2) if np.isfinite(borrowers) else np.nan)
    return values


@dataclass(frozen=True)
class Outcome:
    label: int | None
    reason: str


def verification_target(status: np.ndarray, terminal: np.ndarray, present: np.ndarray) -> Outcome:
    """Label-only API for exactly t+1..t+12. Unknowns are never negative evidence."""
    if any(np.shape(x) != (12,) for x in (status, terminal, present)):
        raise FreddieDataError("Target requires exactly twelve future reporting months")
    complete = True
    for s, z, exists in zip(status, terminal, present):
        if not exists:
            complete = False
            continue
        if np.isfinite(s) and 3 <= s <= 99:
            return Outcome(1, "observed_90plus")
        if s == 100:
            return Outcome(None, "reo_without_numeric_event")
        complete &= bool(np.isfinite(s) and 0 <= s < 3)
        if z:
            if z == 1 and complete:
                return Outcome(0, "competing_voluntary_payoff")
            return Outcome(None, "terminal_incomplete_or_other")
    return Outcome(0, "complete_negative") if complete else Outcome(None, "incomplete_followup")


def _rows(archive: zipfile.ZipFile, member: str, columns: tuple, manifests: dict):
    """Stream and hash exact member bytes; never extract to a caller-chosen path."""
    digest, count = hashlib.sha256(), 0
    with archive.open(member) as stream:
        for raw in stream:
            digest.update(raw)
            try:
                row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")], delimiter="|"))
            except (UnicodeError, csv.Error) as exc:
                raise FreddieDataError("Invalid delimited input") from exc
            if len(row) != len(columns):
                raise FreddieDataError(f"Layout mismatch: expected {len(columns)} fields")
            count += 1
            yield dict(zip(columns, (v.strip() for v in row)))
    manifests[member] = {"sha256": digest.hexdigest(), "rows": count,
                         "bytes": archive.getinfo(member).file_size, "columns": len(columns)}


@dataclass
class PreparedVintage:
    features: pd.DataFrame
    targets: pd.DataFrame
    manifest: dict


def prepare_vintage(path: Path, year: int, *, expected_loans: int = 50000) -> PreparedVintage:
    """Library parser. CLI must validate receipt before invoking this local-only API.

    expected_loans is injectable for explicit synthetic tests; production CLI fixes 50k.
    Memory is O(source loans * 18), independent of all-history performance length.
    """
    if year not in ROLE_BY_YEAR or path.name != f"sample_{year}.zip":
        raise FreddieDataError("Unexpected vintage or archive filename")
    members, records, positions, exclusions = {}, [], {}, Counter()
    with zipfile.ZipFile(path) as archive:
        orig_name, perf_name = f"sample_orig_{year}.txt", f"sample_perf_{year}.txt"
        if sorted(archive.namelist()) != sorted([orig_name, perf_name]):
            raise FreddieDataError("Unexpected or duplicate archive members")
        for row in _rows(archive, orig_name, ORIGINATION_COLUMNS, members):
            identity = row["loan_id"]
            if not re.fullmatch(r"[FA]\d{2}Q[1-4]\d{7}", identity) or identity[1:3] != str(year)[2:]:
                raise FreddieDataError("Loan ID does not match the declared vintage/schema")
            if identity in positions:
                raise FreddieDataError("Duplicate origination entity")
            positions[identity] = len(records)
            first = month_number(row["first_payment_date"]) if row["first_payment_date"] else -1
            eligible = "eligible"
            if row["amortization_type"] != "FRM":
                eligible = "not_fixed_rate"
            elif row["harp_indicator"] == "Y" or row["pre_harp_loan_id"]:
                eligible = "known_refinance_link"
            elif not year * 12 <= first <= (year + 1) * 12 + 2:
                eligible = "first_payment_outside_window"
            records.append((identity, first, eligible, _origination_features(row)))
        if len(records) != expected_loans:
            raise FreddieDataError("Source cohort count differs from frozen annual sample")
        n = len(records)
        history = np.full((n, 6, 3), np.nan)
        status = np.full((n, 18), np.nan)
        terminal = np.zeros((n, 18), np.int16)
        present = np.zeros((n, 18), bool)
        cutoff = month_number("202603")
        for row in _rows(archive, perf_name, PERFORMANCE_COLUMNS, members):
            if row["loan_id"] not in positions:
                raise FreddieDataError("Performance entity missing from origination cohort")
            i = positions[row["loan_id"]]
            period = month_number(row["period"])
            if period > cutoff:
                raise FreddieDataError("Performance exceeds Release 47 cutoff")
            offset = period - records[i][1]
            if not 0 <= offset < 18:
                continue  # No future amount/loss values enter a feature reducer.
            if present[i, offset]:
                raise FreddieDataError("Duplicate loan/month within study window")
            present[i, offset] = True
            status[i, offset] = delinquency(row["delinquency_status"])
            terminal[i, offset] = _termination(row["zero_balance_code"], row["zero_balance_date"], period)
            if offset < 6:
                history[i, offset] = [status[i, offset], _number(row["current_upb"]),
                                     _number(row["current_rate"])]
    feature_rows, target_rows, kept_ids = [], [], []
    for i, (identity, first, eligibility, orig_features) in enumerate(records):
        if eligibility == "eligible":
            if not present[i, :6].all():
                eligibility = "incomplete_history"
            elif (status[i, :6] >= 3).any() or terminal[i, :6].any():
                eligibility = "prior_event_or_termination"
            elif not (np.isfinite(status[i, 5]) and status[i, 5] < 3 and history[i, 5, 1] > 0):
                eligibility = "unknown_or_inactive_landmark"
        if eligibility != "eligible":
            exclusions[eligibility] += 1
            continue
        outcome = verification_target(status[i, 6:], terminal[i, 6:], present[i, 6:])
        kept_ids.append(identity)
        feature_rows.append(orig_features + history[i].T.ravel().tolist())
        target_rows.append({"vintage": year, "role": ROLE_BY_YEAR[year], "landmark": month_text(first + 5),
                            "horizon_end": month_text(first + 17), "label": outcome.label,
                            "outcome_status": outcome.reason})
    index = pd.Index(kept_ids, name="loan_id")
    features = pd.DataFrame(feature_rows, columns=FEATURES, index=index, dtype=float)
    targets = pd.DataFrame(target_rows, columns=["vintage", "role", "landmark", "horizon_end",
                                                "label", "outcome_status"], index=index)
    targets["label"] = targets.label.astype("Int8")
    manifest = {"filename": path.name, "sha256": sha256(path), "release": RELEASE,
                "members": members, "source_loans": n, "eligible_loans": len(features),
                "exclusions": dict(exclusions), "outcomes": dict(Counter(targets.outcome_status)),
                "natural_missing_cells": int(features.isna().to_numpy().sum())}
    return PreparedVintage(features, targets, manifest)


def validate_partition_metadata(targets: pd.DataFrame) -> None:
    if targets.index.has_duplicates:
        raise FreddieDataError("Loan crosses vintages/partitions")
    previous_end = None
    for role in ROLES:
        block = targets[targets.role == role]
        if block.empty:
            continue
        start = min(map(month_number, block.landmark))
        end = max(map(month_number, block.horizon_end))
        if previous_end is not None and previous_end >= start:
            raise FreddieDataError("Earlier-role outcomes overlap later-role landmarks")
        previous_end = end


def validate_analysis_freeze(path: Path, root: Path, development_manifest: Path) -> dict:
    """Fail closed before opening assessment archives; hashes bind fitted artifacts.

    The producer of these artifacts is still a future experiment runner. This
    verifies an explicit freeze, not that its producer followed the full protocol.
    """
    freeze = json.loads(path.read_text())
    if freeze.get("release") != 47 or freeze.get("development_intake_sha256") != sha256(development_manifest):
        raise FreddieDataError("Assessment requires the completed development intake freeze")
    required = {"predictor_lr", "predictor_xgb", "preprocessor", "shap_background",
                "donor_completion", "conditional_completion", "release_policies", "protocol"}
    artifacts = freeze.get("artifacts", {})
    if set(artifacts) != required:
        raise FreddieDataError("Assessment freeze is missing required fitted artifacts")
    for item in artifacts.values():
        local = (root / item["path"]).resolve()
        if not local.is_relative_to(root.resolve()) or not local.is_file() or sha256(local) != item["sha256"]:
            raise FreddieDataError("Frozen artifact path or checksum mismatch")
    return freeze


def artificial_mask(values: np.ndarray, *, seed: int, rate: float) -> np.ndarray:
    if not 0 <= rate <= 1:
        raise ValueError("Missing rate must be between zero and one")
    return (np.random.default_rng(seed).random(values.shape) < rate) & np.isfinite(values)


def restore_artificial(partial: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if partial.shape != original.shape or mask.shape != original.shape or mask.dtype != bool:
        raise ValueError("Verification arrays/mask must align")
    if np.any(mask & ~np.isfinite(original)):
        raise ValueError("Natural missing cells have no verification truth")
    return np.where(mask, original, partial)
