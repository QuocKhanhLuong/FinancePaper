import numpy as np
import pandas as pd
import pytest

from financepaper.data.schema import CATEGORICAL_FEATURES, FEATURE_NAMES


@pytest.fixture
def credit_frame():
    """Small, complete synthetic credit-shaped fixture; never research evidence."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(240, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    for name in CATEGORICAL_FEATURES:
        X[name] = rng.choice([-2, 0, 1, 2], len(X)).astype(float)
    X.index = pd.Index(np.arange(1, len(X) + 1), name="record_id")
    y = pd.Series((X.LIMIT_BAL * X.AGE + 0.2 * X.PAY_0 > 0.3).astype(int), index=X.index)
    return X, y
