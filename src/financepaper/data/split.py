"""Split original records before any learned transformation or masking."""

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_FRACTIONS = {
    "train": 0.50, "development": 0.10, "probability_calibration": 0.10,
    "risk_calibration": 0.15, "test": 0.15,
}


def split_indices(y, seed: int = 42) -> dict[str, np.ndarray]:
    """Return disjoint positional indices; reserve both calibration splits unused.

    Fixed sequential stratification gives exact sizes for Taiwan's 30,000 rows.
    For smaller fixtures, integer floor sizes allocate rounding residue to test.
    """
    labels = np.asarray(y)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected a one-dimensional binary target containing both classes")
    remaining = np.arange(len(labels))
    result = {}
    for offset, (name, fraction) in enumerate(list(SPLIT_FRACTIONS.items())[:-1]):
        size = int(len(labels) * fraction)
        if size < 2:
            raise ValueError("Dataset is too small for five stratified splits")
        chosen, remaining = train_test_split(
            remaining, train_size=size, stratify=labels[remaining],
            random_state=seed + offset,
        )
        result[name] = np.sort(chosen)
    result["test"] = np.sort(remaining)
    if any(set(np.unique(labels[idx])) != {0, 1} for idx in result.values()):
        raise ValueError("Each split must contain both target classes")
    return result
