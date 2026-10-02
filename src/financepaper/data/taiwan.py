"""Pinned official UCI workbook and explicit-schema local CSV loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
WORKBOOK_NAME = "default of credit card clients.xls"
ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
WORKBOOK_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


def _verify(content: bytes, expected: str) -> None:
    actual = sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"Dataset SHA256 mismatch: expected {expected}, got {actual}")


def download_taiwan(raw_dir: Path) -> Path:
    """Fetch explicitly; validate archive and workbook before an atomic write."""
    raw_dir = Path(raw_dir)
    destination = raw_dir / WORKBOOK_NAME
    if destination.exists():
        _verify(destination.read_bytes(), WORKBOOK_SHA256)
        return destination
    with urlopen(SOURCE_URL, timeout=180) as response:
        archive = response.read()
    _verify(archive, ARCHIVE_SHA256)
    with ZipFile(BytesIO(archive)) as zipped:
        content = zipped.read(WORKBOOK_NAME)  # Do not extract arbitrary archive paths.
    _verify(content, WORKBOOK_SHA256)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=raw_dir, prefix=".taiwan-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


@dataclass(frozen=True)
class TaiwanDataset:
    X: pd.DataFrame
    y: pd.Series
    sha256: str
    source_path: Path


def load_taiwan(path: Path) -> TaiwanDataset:
    """Read complete original records; ID is an audit index, never a predictor.

    XLS input must match the pinned UCI workbook. CSV is an explicit local
    alternative (also used for offline synthetic tests) and is fingerprinted,
    validated, and labeled as nonofficial in the experiment manifest.
    Unusual EDUCATION/MARRIAGE/repayment codes are retained without recoding.
    """
    path = Path(path)
    content = path.read_bytes()
    fingerprint = sha256(content).hexdigest()
    if path.suffix.lower() == ".xls":
        _verify(content, WORKBOOK_SHA256)
        frame = pd.read_excel(BytesIO(content), header=1, engine="xlrd")
    elif path.suffix.lower() == ".csv":
        header = next(csv.reader(content.decode("utf-8-sig").splitlines()), [])
        if len(set(header)) != len(header):
            raise ValueError("Dataset contains duplicate column names")
        frame = pd.read_csv(BytesIO(content))
    else:
        raise ValueError("Expected the official .xls workbook or an explicit-schema .csv")
    expected = {"ID", TARGET, *FEATURE_NAMES}
    if frame.columns.has_duplicates or set(frame.columns) != expected:
        raise ValueError(f"Dataset columns must be exactly ID, target and schema fields; got {list(frame.columns)}")
    if len(frame) == 0:
        raise ValueError("Dataset is empty")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Restoration benchmark source must be finite and complete")
    for name in ("ID", TARGET, *CATEGORICAL_FEATURES):
        if not np.equal(frame[name], np.floor(frame[name])).all():
            raise ValueError(f"{name} must contain integer codes")
    if frame["ID"].duplicated().any() or (frame["ID"] <= 0).any():
        raise ValueError("ID must be unique and positive")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both nondefault=0 and default=1")
    index = pd.Index(frame["ID"].astype(int), name="record_id")
    X = frame.loc[:, FEATURE_NAMES].astype(float)
    X.index = index
    y = frame[TARGET].astype(int)
    y.index = index
    return TaiwanDataset(X, y, fingerprint, path.resolve())
