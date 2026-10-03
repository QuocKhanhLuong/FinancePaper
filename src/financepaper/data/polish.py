"""Pinned officially licensed fifth-year cohort; no unknown values are truth."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedGroupKFold
from financepaper.reliability.validation import POLISH_NAMES

SOURCE = "https://archive.ics.uci.edu/static/public/365/polish%2Bcompanies%2Bbankruptcy%2Bdata.zip"
ARCHIVE_SHA = "17377929aa0b204bbf957e56462cf827c19fe4e2ce89f27dfbc77f9ea2bb16c9"
MEMBER_SHA = "cb3f6f250ac46bd8d18e9a222f489fe8ee3e396fcec18959f5a0ef8e8169b2fc"


def download_polish(path):
    path = Path(path)
    if not path.exists():
        with urlopen(SOURCE, timeout=180) as response:
            content = response.read()
        if sha256(content).hexdigest() != ARCHIVE_SHA:
            raise ValueError("official archive checksum mismatch")
        member = ZipFile(BytesIO(content)).read("5year.arff")
        if sha256(member).hexdigest() != MEMBER_SHA:
            raise ValueError("official member checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(member)
    if sha256(path.read_bytes()).hexdigest() != MEMBER_SHA:
        raise ValueError("local ARFF checksum mismatch")
    return path


def load_polish(path):
    path = Path(path)
    if sha256(path.read_bytes()).hexdigest() != MEMBER_SHA:
        raise ValueError("only the pinned official 5year.arff is accepted")
    data, _ = arff.loadarff(path)
    frame = pd.DataFrame(data)
    if tuple(frame.columns) != (*POLISH_NAMES, "class") or len(frame) != 5910:
        raise ValueError("unexpected fifth-year schema")
    X, y = frame.loc[:, POLISH_NAMES].astype(float), frame["class"].astype(int)
    X.index = pd.Index(np.arange(1, len(X)+1), name="record_id")
    y.index = X.index
    if np.isinf(X.to_numpy()).any() or y.sum() != 410:
        raise ValueError("invalid official values/class distribution")
    # Identical known values + missing pattern stay together, independent of y.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    return X, y, groups


def polish_partitions(X, y, groups, seed=20261004):
    assignment = np.full(len(X), -1)
    for fold, (_, idx) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=seed).split(X, y, groups)):
        assignment[idx] = fold
    ranges = {"train": range(10), "selector_train": range(10,12), "default_calibration": [12],
              "revision_calibration": [13], "release_calibration": [14,15], "outer": range(16,20)}
    parts = {name: np.flatnonzero(np.isin(assignment, list(folds))) for name, folds in ranges.items()}
    seen = set()
    for idx in parts.values():
        ids = set(groups[idx])
        if ids & seen:
            raise AssertionError("duplicate-customer group crossed partitions")
        seen |= ids
    return parts


def verification_masks(X, seed):
    natural = X.isna().to_numpy()
    u = np.random.default_rng(seed).random(X.shape)
    return {"complete": np.zeros(X.shape, bool), "mcar10": (u < .1) & ~natural,
            "mcar30": (u < .3) & ~natural}


def restore_artificial(partial, original, artificial):
    artificial = np.asarray(artificial)
    if artificial.dtype != bool or artificial.shape != partial.shape or not partial.index.equals(original.index):
        raise ValueError("misaligned artificial verification mask")
    if np.any(artificial & original.isna().to_numpy()):
        raise ValueError("natural missingness has no restoration truth")
    if not np.array_equal(partial.to_numpy()[~artificial], original.to_numpy()[~artificial], equal_nan=True):
        raise ValueError("observed/naturally missing state changed")
    result = partial.where(~artificial, original)
    if not result.isna().equals(original.isna()):
        raise AssertionError("natural unknowns must remain unknown")
    return result
