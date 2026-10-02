"""Stable binary objectives used by the optional temporal branch."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def _target_tensor(target: Any, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(target, Tensor):
        y = target.to(device=device, dtype=dtype)
    else:
        y = torch.as_tensor(target, device=device, dtype=dtype)
    y = y.reshape(-1)
    if y.numel() == 0:
        raise ValueError("binary targets cannot be empty")
    if not torch.isfinite(y).all():
        raise ValueError("binary targets must be finite")
    if torch.any((y < 0) | (y > 1)):
        raise ValueError("binary targets must lie in [0, 1]")
    return y


def positive_class_weight(target: Any) -> tuple[float, int, int]:
    """Return ``n_negative / n_positive`` and the train counts.

    The counts are deliberately computed by the caller from training labels;
    no validation/development/test labels enter this function.
    """

    if isinstance(target, Tensor):
        values = target.detach().cpu().reshape(-1).numpy()
    else:
        values = np.asarray(target).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("binary targets must be non-empty and finite")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("positive-class counts require exact binary targets")
    positive = int(np.count_nonzero(values == 1))
    negative = int(values.size - positive)
    if positive == 0 or negative == 0:
        raise ValueError("both classes are required to compute pos_weight")
    return float(negative / positive), positive, negative


# A short alias is convenient for experiment code and keeps the count-bearing
# function discoverable under the name used in the protocol.
compute_pos_weight = positive_class_weight


def binary_objective(
    logits: Tensor,
    target: Any,
    *,
    kind: str = "bce",
    pos_weight: float | Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> Tensor:
    """Compute a finite BCE, weighted BCE, or focal objective.

    ``alpha`` is the focal positive-class weight; the negative class receives
    ``1-alpha``.  Weighted BCE and focal objectives are intentionally separate
    configurations so their calibration effects can be measured independently.
    """

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    logits = logits.reshape(-1)
    if logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("logits must be non-empty and finite")
    y = _target_tensor(target, device=logits.device, dtype=logits.dtype)
    if y.shape != logits.shape:
        raise ValueError("logits and targets must have the same number of elements")

    objective = str(kind).lower()
    # The softplus form is the same stable BCE-with-logits equation, but keeps
    # the optional branch independent of the backend-specific BCE kernel.  This
    # matters when the repository's XGBoost/SHAP tests have already initialized
    # their native thread pools in the same Python process.
    bce_terms = (1.0 - y) * F.softplus(logits) + y * F.softplus(-logits)
    if objective in {"bce", "binary_cross_entropy", "unweighted_bce"}:
        value = bce_terms.mean()
    elif objective in {"weighted_bce", "bce_weighted"}:
        if pos_weight is None:
            raise ValueError("weighted_bce requires a training-only pos_weight")
        pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pw.numel() != 1 or not torch.isfinite(pw).all() or pw.item() <= 0:
            raise ValueError("pos_weight must be a positive finite scalar")
        weighted_terms = (1.0 - y) * F.softplus(logits) + y * pw * F.softplus(-logits)
        value = weighted_terms.mean()
    elif objective == "focal":
        if pos_weight is not None:
            raise ValueError("focal loss cannot be combined with pos_weight")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("focal gamma must be finite and non-negative")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("focal alpha must lie strictly between zero and one")
        # BCE-with-logits is stable for extreme logits.  p_t is formed only to
        # apply the focal modulation and is clamped away from exact endpoints.
        bce = bce_terms
        probability = torch.sigmoid(logits)
        p_t = (probability * y + (1.0 - probability) * (1.0 - y)).clamp(1e-8, 1.0)
        alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
        value = (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
    else:
        raise ValueError(f"unsupported binary objective: {kind!r}")

    if not torch.isfinite(value):
        raise FloatingPointError(f"non-finite {objective} objective")
    return value


# Explicitly named convenience wrapper for callers that prefer a loss-style
# API.  It has no mutable state and keeps the experiment code readable.
def binary_loss(
    logits: Tensor,
    target: Any,
    *,
    kind: str = "bce",
    pos_weight: float | Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> Tensor:
    return binary_objective(
        logits,
        target,
        kind=kind,
        pos_weight=pos_weight,
        gamma=gamma,
        alpha=alpha,
    )


__all__ = [
    "binary_loss",
    "binary_objective",
    "compute_pos_weight",
    "positive_class_weight",
]
