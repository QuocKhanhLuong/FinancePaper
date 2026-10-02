"""Integrated Gradients for the temporal model.

The public function in this module deliberately returns encoded attributions.
The temporal preprocessor owns the encoded-to-original-field mapping, which is
then supplied to :func:`aggregate_original`.  Keeping that mapping outside the
explainer avoids silently attributing a category or a missingness diagnostic
to a financial field with a different encoding.

Only value tensors are integrated.  The observed masks and elapsed-time
tensors in ``batch`` are copied unchanged for every quadrature point.  Thus an
attribution is conditional on the endpoint's information state.  A caller
that compares a masked endpoint with a restored endpoint must run this
function twice and report the mask/delta change separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # Torch is an optional dependency for the temporal branch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment]


_STANDARD_STEPS = (32, 64, 128)
_REQUIRED_VALUE_KEYS = ("temporal", "static")


def _require_torch() -> Any:
    if torch is None:
        raise ImportError(
            "Integrated Gradients requires PyTorch; install the temporal extra"
        )
    return torch


def _model_device(model: Any, fallback: Any) -> Any:
    """Return the model parameter device, or ``fallback`` for parameterless models."""

    for parameter in model.parameters():
        return parameter.device
    return fallback


def _model_dtype(model: Any, fallback: Any) -> Any:
    """Return the model floating dtype, or ``fallback`` for parameterless models."""

    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return fallback


def _extract_logit(output: Any, batch_size: int) -> Any:
    """Extract a flat raw-risk-logit tensor from the model's structured output."""

    if isinstance(output, Mapping):
        try:
            output = output["risk"]["logit"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Model output must contain risk.logit") from exc
    if torch is None or not torch.is_tensor(output):  # pragma: no cover - torch guard
        raise TypeError("Model risk.logit must be a torch tensor")
    if output.ndim == 0 and batch_size == 1:
        output = output.reshape(1)
    elif output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    elif output.ndim != 1:
        raise ValueError("Model risk.logit must have shape [batch] or [batch, 1]")
    if output.shape[0] != batch_size:
        raise ValueError("Model risk.logit batch dimension does not match inputs")
    return output


def _move_batch(batch: Mapping[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    """Copy tensor values to the model device without mutating the caller's mapping."""

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch is not None and torch.is_tensor(value):
            moved = value.to(device=device)
            # The temporal model consumes one float dtype for values, masks,
            # and deltas.  Do not coerce integer metadata or IDs that a future
            # caller may carry in batch.
            if moved.is_floating_point():
                moved = moved.to(dtype=dtype)
            result[key] = moved
        else:
            result[key] = value
    return result


def _slice_batch(batch: Mapping[str, Any], start: int, stop: int, batch_size: int) -> dict[str, Any]:
    """Slice per-record tensors while retaining scalar/shared metadata."""

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch is not None and torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[start:stop]
        else:
            result[key] = value
    return result


def _as_value_baseline(value: Any, name: str, expected_shape: tuple[int, ...], batch_size: int,
                       device: Any, dtype: Any) -> Any:
    """Validate and broadcast a value baseline to the batch."""

    if torch is None:  # pragma: no cover - guarded by public entry point
        raise ImportError("PyTorch is required")
    baseline = torch.as_tensor(value, device=device, dtype=dtype)
    if not torch.is_floating_point(baseline):
        baseline = baseline.to(dtype=dtype)
    if baseline.ndim == len(expected_shape):
        if tuple(baseline.shape) != expected_shape:
            raise ValueError(f"{name} baseline must have shape {expected_shape}")
        return baseline.unsqueeze(0).expand((batch_size,) + expected_shape)
    if baseline.ndim == len(expected_shape) + 1:
        if tuple(baseline.shape) != (batch_size,) + expected_shape:
            raise ValueError(
                f"{name} batch baseline must have shape {(batch_size,) + expected_shape}"
            )
        return baseline
    raise ValueError(f"{name} baseline has incompatible rank")


def _validate_steps(steps: int, max_steps: int) -> list[int]:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < steps:
        raise ValueError("max_steps must be an integer greater than or equal to steps")
    if max_steps > 128:
        raise ValueError("max_steps cannot exceed the supported 128-point quadrature")
    candidates = [int(steps)]
    # The pilot uses 32/64/128 Gauss-Legendre nodes.  Supporting a smaller
    # first value is useful for a quick unit test while retaining the same
    # retry semantics.
    for candidate in _STANDARD_STEPS:
        if steps < candidate <= max_steps:
            candidates.append(candidate)
    if max_steps not in candidates:
        candidates.append(int(max_steps))
    return sorted(set(candidates))


def _quadrature(n: int, device: Any, dtype: Any) -> tuple[Any, Any]:
    """Return Gauss-Legendre nodes and weights mapped from [-1,1] to [0,1]."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    alpha = torch.as_tensor((nodes + 1.0) / 2.0, device=device, dtype=dtype)
    weight = torch.as_tensor(weights / 2.0, device=device, dtype=dtype)
    return alpha, weight


def _run_logit(model: Any, batch: Mapping[str, Any], batch_size: int, *, grad: bool) -> Any:
    if grad:
        return _extract_logit(model(batch), batch_size)
    with torch.no_grad():
        return _extract_logit(model(batch), batch_size)


def _integrate_one_steps(model: Any, batch: Mapping[str, Any], baseline_t: Any,
                         baseline_s: Any, input_t: Any, input_s: Any,
                         node_count: int, batch_size: int) -> tuple[Any, Any]:
    """Compute encoded IG for one node count."""

    B = input_t.shape[0]
    phi_t = torch.zeros_like(input_t)
    phi_s = torch.zeros_like(input_s)
    alpha, weights = _quadrature(node_count, input_t.device, input_t.dtype)

    for node, weight in zip(alpha, weights, strict=True):
        for start in range(0, B, batch_size):
            stop = min(start + batch_size, B)
            # The endpoint masks/deltas remain exactly those supplied by the
            # caller.  Only value tensors follow the baseline-to-input path.
            t = (baseline_t[start:stop] + node * (input_t[start:stop] - baseline_t[start:stop])).detach()
            s = (baseline_s[start:stop] + node * (input_s[start:stop] - baseline_s[start:stop])).detach()
            t.requires_grad_(True)
            s.requires_grad_(True)
            local = _slice_batch(batch, start, stop, B)
            local["temporal"] = t
            local["static"] = s
            logits = _run_logit(model, local, stop - start, grad=True)
            gradients = torch.autograd.grad(
                logits.sum(), (t, s), allow_unused=True, retain_graph=False
            )
            grad_t, grad_s = gradients
            if grad_t is None:
                grad_t = torch.zeros_like(t)
            if grad_s is None:
                grad_s = torch.zeros_like(s)
            if not torch.isfinite(grad_t).all() or not torch.isfinite(grad_s).all():
                raise FloatingPointError("Integrated Gradients encountered a non-finite gradient")
            phi_t[start:stop] += weight * grad_t
            phi_s[start:stop] += weight * grad_s

    phi_t = phi_t * (input_t - baseline_t)
    phi_s = phi_s * (input_s - baseline_s)
    return phi_t, phi_s


def integrated_gradients(
    model: Any,
    batchdict: Mapping[str, Any],
    baseline_temporal: Any,
    baseline_static: Any,
    steps: int = 32,
    max_steps: int = 128,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Compute raw-logit Integrated Gradients for a temporal model.

    Parameters
    ----------
    model:
        A PyTorch module returning ``{"risk": {"logit": tensor[B]}}``.
        A ``[B, 1]`` logit tensor is also accepted for small test models.
    batchdict:
        Tensor dictionary containing at least ``temporal`` and ``static``.
        Any mask, observed-state, delta, or other model inputs are held fixed
        at their endpoint values throughout every path.
    baseline_temporal, baseline_static:
        Encoded value baselines shaped ``[6, Tdim]`` and ``[Sdim]``. A batch
        baseline with the corresponding leading ``B`` dimension is accepted.
    steps, max_steps:
        Initial and maximum Gauss-Legendre node counts. The pilot retries at
        64 and 128 nodes when the completeness tolerance fails.
    batch_size:
        Maximum number of records in one gradient call.

    Returns
    -------
    dict
        ``temporal`` and ``static`` contain encoded signed attributions;
        ``baseline_logit``, ``input_logit``, and ``residual`` are per-record
        arrays; ``valid`` explicitly marks the completeness check; and
        ``steps_used`` records the final node count. Aliases ending in
        ``_phi`` are included for callers that use attribution terminology.
    """

    _require_torch()
    if not isinstance(batchdict, Mapping):
        raise TypeError("batchdict must be a mapping of model input tensors")
    if any(key not in batchdict for key in _REQUIRED_VALUE_KEYS):
        raise ValueError("batchdict must contain temporal and static tensors")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    candidates = _validate_steps(steps, max_steps)

    temporal = batchdict["temporal"]
    static = batchdict["static"]
    if not torch.is_tensor(temporal) or not torch.is_tensor(static):
        raise TypeError("temporal and static batch values must be torch tensors")
    if temporal.ndim != 3 or static.ndim != 2:
        raise ValueError("Expected temporal [B, T, D] and static [B, S] tensors")
    B, n_months, temporal_dim = temporal.shape
    if static.shape[0] != B:
        raise ValueError("Temporal and static batch dimensions must match")
    if B == 0:
        raise ValueError("Cannot explain an empty batch")
    if not torch.is_floating_point(temporal) or not torch.is_floating_point(static):
        raise TypeError("Value inputs must be floating-point tensors")
    if not torch.isfinite(temporal).all() or not torch.isfinite(static).all():
        raise ValueError("Value inputs must be finite")

    fallback_device = temporal.device
    device = _model_device(model, fallback_device)
    dtype = _model_dtype(model, temporal.dtype)
    batch = _move_batch(batchdict, device, dtype)
    input_t = batch["temporal"].to(dtype=dtype)
    input_s = batch["static"].to(dtype=dtype)
    if input_t.device != device or input_s.device != device:
        input_t, input_s = input_t.to(device), input_s.to(device)
    baseline_t = _as_value_baseline(
        baseline_temporal, "temporal", (n_months, temporal_dim), B, device, dtype
    )
    baseline_s = _as_value_baseline(
        baseline_static, "static", (static.shape[1],), B, device, dtype
    )
    batch["temporal"] = input_t
    batch["static"] = input_s
    if not torch.isfinite(baseline_t).all() or not torch.isfinite(baseline_s).all():
        raise ValueError("Attribution baselines must be finite")

    was_training = bool(model.training)
    model.eval()
    try:
        input_logit_t = _run_logit(model, batch, B, grad=False)
        baseline_batch = dict(batch)
        baseline_batch["temporal"] = baseline_t
        baseline_batch["static"] = baseline_s
        baseline_logit_t = _run_logit(model, baseline_batch, B, grad=False)
        input_logit = input_logit_t.detach()
        baseline_logit = baseline_logit_t.detach()
        if not torch.isfinite(input_logit).all() or not torch.isfinite(baseline_logit).all():
            raise FloatingPointError("Model logits are non-finite")

        tolerance = 0.002 + 0.001 * torch.abs(input_logit - baseline_logit)
        final_phi_t = None
        final_phi_s = None
        final_residual = None
        final_valid = None
        final_steps = candidates[-1]
        for node_count in candidates:
            phi_t, phi_s = _integrate_one_steps(
                model, batch, baseline_t, baseline_s, input_t, input_s,
                node_count, int(batch_size)
            )
            residual = (phi_t.flatten(1).sum(dim=1) + phi_s.sum(dim=1)
                        - (input_logit - baseline_logit))
            valid = torch.isfinite(residual) & (torch.abs(residual) <= tolerance)
            final_phi_t, final_phi_s = phi_t, phi_s
            final_residual, final_valid, final_steps = residual, valid, node_count
            if bool(valid.all()):
                break
    finally:
        model.train(was_training)

    assert final_phi_t is not None and final_phi_s is not None
    assert final_residual is not None and final_valid is not None
    phi_t_np = final_phi_t.detach().cpu().numpy()
    phi_s_np = final_phi_s.detach().cpu().numpy()
    baseline_np = baseline_logit.detach().cpu().numpy()
    input_np = input_logit.detach().cpu().numpy()
    residual_np = final_residual.detach().cpu().numpy()
    valid_np = final_valid.detach().cpu().numpy().astype(bool)
    steps_np = np.full(B, int(final_steps), dtype=np.int64)
    return {
        "temporal": phi_t_np,
        "static": phi_s_np,
        "temporal_phi": phi_t_np,
        "static_phi": phi_s_np,
        "baseline_logit": baseline_np,
        "input_logit": input_np,
        "residual": residual_np,
        "valid": valid_np,
        "steps_used": steps_np,
        "tolerance": (0.002 + 0.001 * np.abs(input_np - baseline_np)),
        "failed_indices": np.flatnonzero(~valid_np),
    }


def _normalise_temporal_origins(origins: Sequence[Any], n_months: int, n_dim: int) -> np.ndarray:
    arr = np.asarray(origins, dtype=object)
    if arr.ndim == 1 and arr.shape[0] == n_months * n_dim:
        # TemporalBatch stores the canonical mapping as one flat tuple in
        # chronological (month, encoded-dimension) order.
        arr = arr.reshape(n_months, n_dim)
        return arr
    if arr.ndim == 1 and arr.shape[0] == n_dim:
        # A short mapping is intentionally supported for callers whose same
        # encoded field schema is reused at every month.
        arr = np.tile(arr.reshape(1, -1), (n_months, 1))
    if arr.ndim != 2 or arr.shape != (n_months, n_dim):
        raise ValueError("temporal_origins must have shape [months, encoded_temporal_dim]")
    return arr


def aggregate_original(
    temporal_phi: Any,
    static_phi: Any,
    temporal_origins: Sequence[Any],
    static_origins: Sequence[Any],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Sum encoded temporal/static attributions into original fields.

    ``temporal_origins`` may be the canonical flat tuple in
    ``TemporalBatch.temporal_origins`` order, a ``[months,
    encoded_temporal_dim]`` list/array, or one short encoded mapping reused at
    every month. Origins may repeat because one original field can have several
    one-hot dimensions. The returned matrix has shape ``[N,
    len(feature_names)]`` and preserves signed additivity.
    """

    t = np.asarray(temporal_phi, dtype=float)
    s = np.asarray(static_phi, dtype=float)
    names = tuple(feature_names)
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("feature_names must be nonempty and unique")
    if t.ndim != 3 or s.ndim != 2 or t.shape[0] != s.shape[0]:
        raise ValueError("Expected temporal [N, months, dim] and static [N, dim] attributions")
    if not np.isfinite(t).all() or not np.isfinite(s).all():
        raise ValueError("Attributions must be finite")
    temporal_names = _normalise_temporal_origins(temporal_origins, t.shape[1], t.shape[2])
    static_names = np.asarray(static_origins, dtype=object)
    if static_names.ndim != 1 or static_names.shape[0] != s.shape[1]:
        raise ValueError("static_origins must align with the static encoded dimension")
    allowed = set(names)
    if any(origin not in allowed for origin in temporal_names.flat) or any(origin not in allowed for origin in static_names):
        raise ValueError("Attribution origins must be present in feature_names")

    indices = {name: i for i, name in enumerate(names)}
    output = np.zeros((t.shape[0], len(names)), dtype=float)
    for month in range(t.shape[1]):
        for dim, origin in enumerate(temporal_names[month]):
            output[:, indices[origin]] += t[:, month, dim]
    for dim, origin in enumerate(static_names):
        output[:, indices[origin]] += s[:, dim]
    return output


__all__ = ["aggregate_original", "integrated_gradients"]
