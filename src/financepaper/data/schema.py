"""Canonical UCI field order; repayment codes are categorical, not distances."""

NUMERIC_FEATURES = (
    "LIMIT_BAL", "AGE",
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
)
CATEGORICAL_FEATURES = (
    "SEX", "EDUCATION", "MARRIAGE", "PAY_0",
    *(f"PAY_{i}" for i in range(2, 7)),
)
FEATURE_NAMES = (
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0",
    *(f"PAY_{i}" for i in range(2, 7)),
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
)
TARGET = "default payment next month"
