"""Leakage-safe temporal views of the Taiwan credit-card records.

The source workbook is a wide table.  This module is the single place where
the six monthly columns are put into chronological order and where training
statistics used by the temporal models are fitted.  A ``TemporalBatch`` keeps
the encoded values and the availability metadata together so that a model
cannot accidentally receive a hidden value while still being told that the
value is missing.

The public arrays are NumPy ``float32`` arrays.  PyTorch is intentionally an
optional runtime dependency for this module; it is imported only by
``TemporalBatch.to_torch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import FEATURE_NAMES


# The source fields are ordered from the oldest observed month to the newest.
# PAY_0 is September in the official UCI documentation; PAY_6 is April.
MONTH_NAMES = (
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
)

TEMPORAL_FIELDS = (
    ("PAY_6", "BILL_AMT6", "PAY_AMT6"),
    ("PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("PAY_0", "BILL_AMT1", "PAY_AMT1"),
)

STATIC_FIELDS = (
    "LIMIT_BAL",
    "AGE",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
)

_TEMPORAL_STATUS_FIELDS = tuple(fields[0] for fields in TEMPORAL_FIELDS)
_TEMPORAL_NUMERIC_FIELDS = tuple(fields[j] for fields in TEMPORAL_FIELDS for j in (1, 2))
_STATIC_NUMERIC_FIELDS = ("LIMIT_BAL", "AGE")
_STATIC_CATEGORICAL_FIELDS = ("SEX", "EDUCATION", "MARRIAGE")
_ALL_TEMPORAL_SOURCE_FIELDS = tuple(field for fields in TEMPORAL_FIELDS for field in fields)


def _validate_columns(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if list(X.columns) != list(FEATURE_NAMES):
        raise ValueError(
            "X columns must be in the canonical Taiwan schema order; "
            f"expected {list(FEATURE_NAMES)}, got {list(X.columns)}"
        )
    if X.index.has_duplicates:
        raise ValueError("X index must identify records uniquely")


def _as_finite_frame(X: pd.DataFrame, *, allow_nan: bool) -> np.ndarray:
    try:
        values = X.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric feature values") from exc
    if not allow_nan and np.isnan(values).any():
        raise ValueError("TemporalPreprocessor.fit requires complete training records")
    if np.isinf(values).any():
        raise ValueError("X cannot contain infinite values")
    return values


def _validate_hidden_mask(
    X: pd.DataFrame,
    hidden_mask: pd.DataFrame,
) -> np.ndarray:
    if not isinstance(hidden_mask, pd.DataFrame):
        raise TypeError("hidden_mask must be a pandas DataFrame")
    if not X.index.equals(hidden_mask.index) or not X.columns.equals(hidden_mask.columns):
        raise ValueError("hidden_mask axes must exactly match X")
    if hidden_mask.isna().any().any() or not all(dtype == bool for dtype in hidden_mask.dtypes):
        raise ValueError("hidden_mask must contain boolean, non-missing values")
    return hidden_mask.to_numpy(dtype=bool)


def elapsed_delta(observed: np.ndarray) -> np.ndarray:
    """Return elapsed monthly gaps from an observed mask.

    ``observed`` uses ``True`` for an available value.  The first month has an
    explicitly unknown pre-history and therefore receives zero.  For each
    later month the previous month's availability controls whether the gap is
    extended.  The function accepts either ``[B, T, F]`` or ``[T, F]`` input
    and preserves the corresponding rank in its output.
    """

    values = np.asarray(observed)
    if values.ndim not in (2, 3):
        raise ValueError("observed must have shape [T,F] or [B,T,F]")
    if values.shape[-1] <= 0 or values.shape[-2] <= 0:
        raise ValueError("observed must contain at least one timestep and feature")
    if not np.issubdtype(values.dtype, np.bool_):
        # Numeric 0/1 arrays are accepted only when they really are binary;
        # silently treating arbitrary floats as masks is a common data bug.
        if not np.isfinite(values.astype(float)).all() or not np.isin(values, (0, 1)).all():
            raise ValueError("observed must be boolean or contain only 0/1 values")
        values = values.astype(bool)
    else:
        values = values.astype(bool, copy=False)

    was_2d = values.ndim == 2
    if was_2d:
        values = values[None, ...]
    result = np.zeros(values.shape, dtype=np.float32)
    for timestep in range(1, values.shape[1]):
        result[:, timestep, :] = 1.0 + (
            (~values[:, timestep - 1, :]).astype(np.float32)
            * result[:, timestep - 1, :]
        )
    return result[0] if was_2d else result


@dataclass(frozen=True)
class TemporalBatch:
    """Aligned encoded values and availability metadata for a record batch.

    ``delta`` stores elapsed months in the natural scale.  Model inputs divide
    it by five, and ``flatten`` uses the same normalized representation so flat
    controls receive exactly the same metadata scale.
    """

    temporal: np.ndarray
    static: np.ndarray
    temporal_observed: np.ndarray
    static_observed: np.ndarray
    delta: np.ndarray
    record_ids: np.ndarray | None = None
    temporal_origins: tuple[str, ...] = ()
    static_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        arrays = {
            "temporal": self.temporal,
            "static": self.static,
            "temporal_observed": self.temporal_observed,
            "static_observed": self.static_observed,
            "delta": self.delta,
        }
        converted: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            converted[name] = array

        n_rows = converted["temporal"].shape[0] if converted["temporal"].ndim else -1
        if converted["temporal"].ndim != 3:
            raise ValueError("temporal must have shape [B,6,F]")
        if converted["temporal"].shape[1] != len(TEMPORAL_FIELDS):
            raise ValueError("temporal must contain exactly six chronological steps")
        if converted["static"].ndim != 2:
            raise ValueError("static must have shape [B,F]")
        if converted["temporal_observed"].shape != (n_rows, len(TEMPORAL_FIELDS), 3):
            raise ValueError("temporal_observed must have shape [B,6,3]")
        if converted["delta"].shape != converted["temporal_observed"].shape:
            raise ValueError("delta must have the same shape as temporal_observed")
        if converted["static_observed"].shape != (n_rows, len(STATIC_FIELDS)):
            raise ValueError("static_observed must have shape [B,5]")
        if converted["temporal_observed"].shape[0] != converted["static"].shape[0]:
            raise ValueError("all batch fields must have the same number of records")
        if not np.isin(converted["temporal_observed"], (0.0, 1.0)).all():
            raise ValueError("temporal_observed must contain exact 0/1 values")
        if not np.isin(converted["static_observed"], (0.0, 1.0)).all():
            raise ValueError("static_observed must contain exact 0/1 values")
        if np.any(converted["delta"] < 0):
            raise ValueError("delta cannot be negative")
        if len(self.temporal_origins) != converted["temporal"].shape[1] * converted["temporal"].shape[2]:
            raise ValueError("temporal_origins must describe every encoded temporal column")
        if len(self.static_origins) != converted["static"].shape[1]:
            raise ValueError("static_origins must describe every encoded static column")

        if self.record_ids is None:
            record_ids = None
        else:
            record_ids = np.asarray(self.record_ids)
            if record_ids.ndim != 1 or len(record_ids) != n_rows:
                raise ValueError("record_ids must be a one-dimensional aligned array")
            record_ids = record_ids.copy()

        for name, value in converted.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if record_ids is not None:
            record_ids.setflags(write=False)
        object.__setattr__(self, "record_ids", record_ids)
        object.__setattr__(self, "temporal_origins", tuple(self.temporal_origins))
        object.__setattr__(self, "static_origins", tuple(self.static_origins))

    @property
    def n_samples(self) -> int:
        return int(self.temporal.shape[0])

    def __len__(self) -> int:
        return self.n_samples

    @property
    def temporal_dim(self) -> int:
        return int(self.temporal.shape[-1])

    @property
    def static_dim(self) -> int:
        return int(self.static.shape[-1])

    @property
    def flat_origins(self) -> tuple[str, ...]:
        """Names aligned with ``flatten(include_metadata=True)`` columns."""

        temporal_mask_origins = tuple(
            f"__observed_{field}" for fields in TEMPORAL_FIELDS for field in fields
        )
        static_mask_origins = tuple(f"__observed_{field}" for field in STATIC_FIELDS)
        delta_origins = tuple(f"__delta_{field}" for fields in TEMPORAL_FIELDS for field in fields)
        return (
            tuple(self.temporal_origins)
            + tuple(self.static_origins)
            + temporal_mask_origins
            + static_mask_origins
            + delta_origins
        )

    def flatten(self, include_metadata: bool = False):
        """Flatten a batch for a matched tree/linear control.

        The first two blocks are encoded values.  When ``include_metadata`` is
        true, observed masks and elapsed deltas divided by five are appended in
        raw chronological order.  With metadata enabled the return value is a
        ``(values, origins)`` pair; otherwise it is only the value matrix.
        """

        values = [self.temporal.reshape(self.n_samples, -1), self.static]
        if include_metadata:
            values.extend(
                (
                    self.temporal_observed.reshape(self.n_samples, -1),
                    self.static_observed,
                    self.delta.reshape(self.n_samples, -1) / np.float32(5.0),
                )
            )
        flat = np.concatenate(values, axis=1).astype(np.float32, copy=False)
        if include_metadata:
            origins = self.flat_origins
            if flat.shape[1] != len(origins):  # defensive invariant for future fields
                raise RuntimeError("flattened origin metadata is misaligned")
            return flat, origins
        return flat

    def take(self, indices: Sequence[int] | np.ndarray) -> "TemporalBatch":
        """Return a positionally indexed batch while retaining all metadata."""

        index = np.asarray(indices)
        if index.ndim != 1:
            raise ValueError("indices must be a one-dimensional positional index")
        if np.issubdtype(index.dtype, np.bool_):
            if len(index) != self.n_samples:
                raise ValueError("boolean indices must match the batch length")
        elif not np.issubdtype(index.dtype, np.integer):
            raise TypeError("indices must be integer positions or a boolean mask")
        fields = {
            name: getattr(self, name)[index]
            for name in ("temporal", "static", "temporal_observed", "static_observed", "delta")
        }
        record_ids = None if self.record_ids is None else self.record_ids[index]
        return TemporalBatch(
            **fields,
            record_ids=record_ids,
            temporal_origins=self.temporal_origins,
            static_origins=self.static_origins,
        )

    def to_torch(self, device: Any = None) -> dict[str, Any]:
        """Convert aligned fields to float32 tensors on ``device``.

        The import is local so data loading and tests remain usable without the
        optional temporal PyTorch dependency installed.
        """

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "TemporalBatch.to_torch requires the optional PyTorch dependency"
            ) from exc
        return {
            # Arrays are intentionally read-only in TemporalBatch.  Copying
            # here avoids PyTorch's warning about non-writable NumPy buffers
            # and prevents tensor writes from mutating a supposedly immutable
            # batch through shared memory.
            "temporal": torch.tensor(self.temporal, dtype=torch.float32, device=device),
            "static": torch.tensor(self.static, dtype=torch.float32, device=device),
            "temporal_observed": torch.tensor(
                self.temporal_observed, dtype=torch.float32, device=device
            ),
            "static_observed": torch.tensor(
                self.static_observed, dtype=torch.float32, device=device
            ),
            "delta": torch.tensor(self.delta, dtype=torch.float32, device=device),
        }


class TemporalPreprocessor:
    """Fit train-only encodings and transform complete or masked records."""

    def __init__(self) -> None:
        self._fitted = False
        self.train_ids: tuple[Any, ...] = ()
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.categorical_modes: dict[str, float] = {}
        self.pay_vocabulary: tuple[float, ...] = ()
        self.static_vocabularies: dict[str, tuple[float, ...]] = {}
        self.temporal_origins: tuple[str, ...] = ()
        self.static_origins: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def temporal_dim(self) -> int:
        self._require_fitted()
        return len(self.pay_vocabulary) + 2

    @property
    def static_dim(self) -> int:
        self._require_fitted()
        return 2 + sum(len(self.static_vocabularies[field]) for field in _STATIC_CATEGORICAL_FIELDS)

    @property
    def output_dimensions(self) -> dict[str, int]:
        return {"temporal": self.temporal_dim, "static": self.static_dim}

    @staticmethod
    def _mode(values: pd.Series, field: str) -> float:
        counts = values.value_counts(dropna=True)
        if counts.empty:
            raise ValueError(f"training field {field} contains no observed values")
        max_count = counts.max()
        # Numeric sorting makes ties deterministic, including unusual PAY codes.
        winners = [float(value) for value, count in counts.items() if count == max_count]
        return min(winners)

    @staticmethod
    def _vocabulary(values: Iterable[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in sorted(set(float(v) for v in values)))

    def fit(self, X: pd.DataFrame, train_ids: Sequence[Any] | None = None) -> "TemporalPreprocessor":
        """Fit all imputers, scalers and vocabularies on complete train records."""

        _validate_columns(X)
        if len(X) == 0:
            raise ValueError("TemporalPreprocessor.fit requires at least one training record")
        values = _as_finite_frame(X, allow_nan=False)
        if train_ids is not None and len(train_ids) != len(X):
            raise ValueError("train_ids must align with X")
        self.train_ids = tuple(X.index.tolist() if train_ids is None else list(train_ids))

        numeric_fields = (*_STATIC_NUMERIC_FIELDS, *_TEMPORAL_NUMERIC_FIELDS)
        for field in numeric_fields:
            column = X[field].to_numpy(dtype=float)
            median = float(np.median(column))
            mean = float(np.mean(column))
            std = float(np.std(column, ddof=0))
            self.numeric_medians[field] = median
            self.numeric_means[field] = mean
            self.numeric_scales[field] = std if std > 0 else 1.0

        categorical_fields = (*_TEMPORAL_STATUS_FIELDS, *_STATIC_CATEGORICAL_FIELDS)
        for field in categorical_fields:
            self.categorical_modes[field] = self._mode(X[field], field)

        self.pay_vocabulary = self._vocabulary(
            value for field in _TEMPORAL_STATUS_FIELDS for value in X[field].to_numpy(dtype=float)
        )
        if not self.pay_vocabulary:
            raise ValueError("training records contain no repayment-status categories")
        self.static_vocabularies = {
            field: self._vocabulary(X[field].to_numpy(dtype=float))
            for field in _STATIC_CATEGORICAL_FIELDS
        }

        self.temporal_origins = tuple(
            origin
            for status, bill, payment in TEMPORAL_FIELDS
            for origin in ((status,) * len(self.pay_vocabulary) + (bill, payment))
        )
        self.static_origins = (
            "LIMIT_BAL",
            "AGE",
            *(field for field in _STATIC_CATEGORICAL_FIELDS for _ in self.static_vocabularies[field]),
        )
        self._fitted = True
        # ``values`` is deliberately consumed only to validate finiteness; all
        # statistics above are calculated from X itself, never from target data.
        del values
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TemporalPreprocessor must be fitted before use")

    def _prepare_input(
        self,
        X: pd.DataFrame,
        hidden_mask: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _validate_columns(X)
        raw = _as_finite_frame(X, allow_nan=True)
        nan_mask = np.isnan(raw)
        if hidden_mask is None:
            hidden = nan_mask
        else:
            hidden = _validate_hidden_mask(X, hidden_mask)
            # A supplied mask can hide a complete value; it cannot excuse a NaN
            # in a cell declared observed.
            if np.any(nan_mask & ~hidden):
                raise ValueError("X contains NaNs outside hidden_mask")
        observed = ~hidden
        if not np.isfinite(raw[observed]).all():
            raise ValueError("observed X values must be finite")
        prepared = raw.copy()
        prepared[hidden] = np.nan
        return prepared, observed

    def _scaled_numeric(self, values: np.ndarray, field: str) -> np.ndarray:
        filled = np.where(np.isnan(values), self.numeric_medians[field], values)
        return (filled - self.numeric_means[field]) / self.numeric_scales[field]

    @staticmethod
    def _one_hot(values: np.ndarray, vocabulary: tuple[float, ...]) -> np.ndarray:
        result = np.zeros((len(values), len(vocabulary)), dtype=np.float32)
        if not vocabulary:
            return result
        lookup = {value: position for position, value in enumerate(vocabulary)}
        for row, value in enumerate(values):
            if not np.isfinite(value):
                continue
            position = lookup.get(float(value))
            if position is not None:
                result[row, position] = 1.0
        return result

    def transform(
        self,
        X: pd.DataFrame,
        hidden_mask: pd.DataFrame | None = None,
    ) -> TemporalBatch:
        """Encode records, replacing hidden values with train-only statistics.

        If a complete ``X`` is supplied with ``hidden_mask``, the hidden source
        cells are discarded before imputation.  This explicit step prevents a
        caller from accidentally leaking restoration truth into ordinary
        prediction merely by passing both arguments.
        """

        self._require_fitted()
        prepared, observed = self._prepare_input(X, hidden_mask)

        n_rows = len(X)
        temporal = np.zeros((n_rows, len(TEMPORAL_FIELDS), self.temporal_dim), dtype=np.float32)
        temporal_observed = np.zeros((n_rows, len(TEMPORAL_FIELDS), 3), dtype=np.float32)
        for timestep, (status, bill, payment) in enumerate(TEMPORAL_FIELDS):
            status_values = prepared[:, FEATURE_NAMES.index(status)]
            # Missing categorical values are imputed by the training mode; an
            # observed category outside the train vocabulary intentionally maps
            # to an all-zero vector.
            status_values = np.where(
                np.isnan(status_values), self.categorical_modes[status], status_values
            )
            temporal[:, timestep, : len(self.pay_vocabulary)] = self._one_hot(
                status_values, self.pay_vocabulary
            )
            temporal[:, timestep, len(self.pay_vocabulary)] = self._scaled_numeric(
                prepared[:, FEATURE_NAMES.index(bill)], bill
            ).astype(np.float32)
            temporal[:, timestep, len(self.pay_vocabulary) + 1] = self._scaled_numeric(
                prepared[:, FEATURE_NAMES.index(payment)], payment
            ).astype(np.float32)
            temporal_observed[:, timestep, :] = np.column_stack(
                [
                    observed[:, FEATURE_NAMES.index(status)],
                    observed[:, FEATURE_NAMES.index(bill)],
                    observed[:, FEATURE_NAMES.index(payment)],
                ]
            ).astype(np.float32)

        static = np.zeros((n_rows, self.static_dim), dtype=np.float32)
        static_observed = np.zeros((n_rows, len(STATIC_FIELDS)), dtype=np.float32)
        static[:, 0] = self._scaled_numeric(
            prepared[:, FEATURE_NAMES.index("LIMIT_BAL")], "LIMIT_BAL"
        ).astype(np.float32)
        static[:, 1] = self._scaled_numeric(
            prepared[:, FEATURE_NAMES.index("AGE")], "AGE"
        ).astype(np.float32)
        static_offset = 2
        for position, field in enumerate(_STATIC_CATEGORICAL_FIELDS):
            column = prepared[:, FEATURE_NAMES.index(field)]
            column = np.where(np.isnan(column), self.categorical_modes[field], column)
            encoded = self._one_hot(column, self.static_vocabularies[field])
            static[:, static_offset : static_offset + encoded.shape[1]] = encoded
            static_offset += encoded.shape[1]
        for position, field in enumerate(STATIC_FIELDS):
            static_observed[:, position] = observed[:, FEATURE_NAMES.index(field)].astype(np.float32)

        delta = elapsed_delta(temporal_observed > 0.5).astype(np.float32)
        return TemporalBatch(
            temporal=temporal,
            static=static,
            temporal_observed=temporal_observed,
            static_observed=static_observed,
            delta=delta,
            record_ids=np.asarray(X.index),
            temporal_origins=self.temporal_origins,
            static_origins=self.static_origins,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fitted metadata for an experiment manifest."""

        self._require_fitted()

        def json_number(value: float) -> int | float:
            return int(value) if float(value).is_integer() else float(value)

        return {
            "train_ids": [
                value.item() if isinstance(value, np.generic) else value for value in self.train_ids
            ],
            "temporal_fields": [list(fields) for fields in TEMPORAL_FIELDS],
            "month_names": list(MONTH_NAMES),
            "static_fields": list(STATIC_FIELDS),
            "pay_vocabulary": [json_number(value) for value in self.pay_vocabulary],
            "static_vocabularies": {
                field: [json_number(value) for value in values]
                for field, values in self.static_vocabularies.items()
            },
            "categorical_modes": {
                field: json_number(value) for field, value in self.categorical_modes.items()
            },
            "numeric_medians": {
                field: float(value) for field, value in self.numeric_medians.items()
            },
            "numeric_means": {
                field: float(value) for field, value in self.numeric_means.items()
            },
            "numeric_scales": {
                field: float(value) for field, value in self.numeric_scales.items()
            },
            "temporal_origins": list(self.temporal_origins),
            "static_origins": list(self.static_origins),
            "output_dimensions": dict(self.output_dimensions),
        }


__all__ = [
    "MONTH_NAMES",
    "TEMPORAL_FIELDS",
    "STATIC_FIELDS",
    "TemporalBatch",
    "TemporalPreprocessor",
    "elapsed_delta",
]
