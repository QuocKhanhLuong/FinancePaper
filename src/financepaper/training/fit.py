"""Small deterministic PyTorch fitting loop for the temporal pilot."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss
from torch import Tensor, nn

from .device import parameter_count, select_device
from .losses import binary_objective, positive_class_weight


DEFAULT_CONFIG: dict[str, Any] = {
    "epochs": 25,
    "patience": 5,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss": "bce",
    "gamma": 2.0,
    "alpha": 0.75,
    "gradient_clip": 5.0,
}

SELECTION_CONDITIONS = ("complete", "mcar10", "mcar30")
_BATCH_FIELDS = ("temporal", "static", "temporal_observed", "static_observed", "delta")


def _is_batch_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(name in value for name in _BATCH_FIELDS)
    return any(hasattr(value, name) for name in _BATCH_FIELDS)


def _field(batch: Any, name: str, default: Any = None) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _batch_length(batch: Any) -> int:
    if isinstance(batch, Mapping):
        for name in _BATCH_FIELDS:
            value = batch.get(name)
            if value is not None:
                return int(len(value))
    elif _is_batch_like(batch):
        for name in _BATCH_FIELDS:
            value = getattr(batch, name, None)
            if value is not None:
                return int(len(value))
        try:
            return int(len(batch))
        except TypeError:
            pass
    else:
        try:
            return int(len(batch))
        except TypeError:
            pass
    raise TypeError("batch must expose a batch dimension")


def _slice_value(value: Any, indices: np.ndarray) -> Any:
    if isinstance(value, Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, np.ndarray):
        return value[indices]
    if isinstance(value, (list, tuple)):
        selected = [value[int(index)] for index in indices]
        return type(value)(selected)
    # Metadata and scalars are carried through unchanged.  The model ignores
    # those fields; TemporalBatch.take handles aligned metadata itself.
    return value


def _slice_batch(batch: Any, indices: np.ndarray) -> Any:
    if hasattr(batch, "take") and callable(batch.take):
        try:
            return batch.take(indices)
        except TypeError:
            return batch.take(indices.tolist())
    if isinstance(batch, Mapping):
        return {key: _slice_value(value, indices) for key, value in batch.items()}
    if _is_batch_like(batch):
        # A fallback for lightweight test fixtures that expose attributes but
        # no ``take`` method.  A real TemporalBatch should use its own method.
        values = {
            name: _slice_value(getattr(batch, name), indices)
            for name in _BATCH_FIELDS
            if hasattr(batch, name)
        }
        return values
    raise TypeError("batch must be a TemporalBatch-like object or tensor mapping")


def _to_torch_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to_torch") and callable(batch.to_torch):
        try:
            return batch.to_torch(device)
        except TypeError:
            return batch.to_torch(device=device)
    if isinstance(batch, Mapping):
        converted: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                converted[key] = value.to(device=device)
            elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                converted[key] = torch.tensor(value, device=device)
            else:
                converted[key] = value
        return converted
    return batch


def _as_labels(values: Any) -> np.ndarray:
    if isinstance(values, Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("labels must be non-empty and finite")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("labels must lie in [0, 1]")
    return array


def _record_ids(batch: Any) -> np.ndarray | None:
    """Return aligned original-record IDs when a TemporalBatch exposes them."""

    value = _field(batch, "record_ids")
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        risk = output.get("risk", output)
        if isinstance(risk, Mapping):
            logits = risk.get("logit")
        else:
            logits = risk
    else:
        logits = getattr(output, "logit", None)
    if not isinstance(logits, Tensor):
        raise TypeError("model output must contain risk.logit")
    logits = logits.reshape(-1)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("model produced non-finite risk logits")
    return logits


def _iter_batches(batches: Any) -> list[Any]:
    if _is_batch_like(batches):
        return [batches]
    if isinstance(batches, Mapping):
        # A mapping without temporal fields is an explicit iterable of batches.
        return list(batches.values())
    if isinstance(batches, Iterable) and not isinstance(batches, (str, bytes)):
        return list(batches)
    return [batches]


def predict_logits(
    model: nn.Module,
    batches: Any,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 256,
) -> Tensor:
    """Return CPU logits for one TemporalBatch or an iterable of batches.

    Input batches are split into ``batch_size`` chunks before inference, so a
    full training partition can be supplied without exceeding unified memory.
    The model is evaluated with dropout disabled and its previous training mode
    is restored on return.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = select_device(device)
    was_training = model.training
    model = model.to(selected)
    model.eval()
    logits: list[Tensor] = []
    try:
        with torch.no_grad():
            for batch in _iter_batches(batches):
                n = _batch_length(batch)
                for start in range(0, n, batch_size):
                    indices = np.arange(start, min(start + batch_size, n), dtype=np.int64)
                    mini = _to_torch_batch(_slice_batch(batch, indices), selected)
                    logits.append(_extract_logits(model(mini, debug=False)).detach().cpu())
    finally:
        model.train(was_training)
    if not logits:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(logits, dim=0)


def _metric_pair(labels: np.ndarray, logits: Tensor) -> tuple[float, float]:
    values = np.asarray(logits.detach().cpu(), dtype=np.float64).reshape(-1)
    if len(labels) != len(values):
        raise ValueError("prediction and label lengths differ")
    probability = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    try:
        ap = float(average_precision_score(labels, probability))
    except ValueError:
        ap = float("nan")
    try:
        loss = float(log_loss(labels, probability, labels=[0, 1]))
    except ValueError:
        loss = float("inf")
    return ap, loss


def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(DEFAULT_CONFIG)
    if config:
        # Accept either the flat function contract or a ``training`` section
        # from the YAML runner without changing the frozen numerical defaults.
        source = config.get("training", config)
        if isinstance(source, Mapping):
            values.update(source)
    for name in ("epochs", "patience", "batch_size"):
        values[name] = int(values[name])
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("lr", "weight_decay", "gamma", "alpha", "gradient_clip"):
        values[name] = float(values[name])
        if not np.isfinite(values[name]) or values[name] < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if values["lr"] == 0:
        raise ValueError("lr must be positive")
    if values["gradient_clip"] == 0:
        raise ValueError("gradient_clip must be positive")
    values["loss"] = str(values["loss"]).lower()
    if values["loss"] not in {"bce", "weighted_bce", "focal"}:
        raise ValueError("loss must be bce, weighted_bce, or focal")
    return values


def _condition_labels(y_dev: Any, condition: str, fallback: np.ndarray | None = None) -> np.ndarray:
    if isinstance(y_dev, Mapping):
        if condition not in y_dev:
            raise KeyError(f"missing labels for development condition {condition!r}")
        return _as_labels(y_dev[condition])
    if fallback is None:
        return _as_labels(y_dev)
    return fallback


def fit_model(
    model: nn.Module,
    train_batch: Any,
    y_train: Any,
    dev_batches: Mapping[str, Any],
    y_dev: Any,
    augmentation_callback: Any,
    config: Mapping[str, Any] | None,
    seed: int,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Fit a temporal model and restore the checkpoint selected on development.

    ``augmentation_callback(epoch)`` must return a fresh TemporalBatch with the
    same original-record order and row count as ``train_batch``.  When
    ``train_batch`` exposes ``record_ids``, the callback must return the same
    IDs in the same order and this function checks that identity directly;
    without IDs the caller owns the partition/row-identity guarantee. Labels
    always remain the original ``y_train``.
    """

    if not isinstance(dev_batches, Mapping):
        raise TypeError("dev_batches must map condition names to batches")
    missing_conditions = [name for name in SELECTION_CONDITIONS if name not in dev_batches]
    if missing_conditions:
        raise KeyError(f"dev_batches missing selection conditions: {missing_conditions}")

    options = _resolve_config(config)
    labels = _as_labels(y_train)
    n_train = _batch_length(train_batch)
    if n_train != len(labels):
        raise ValueError("train_batch and y_train lengths differ")
    train_ids = _record_ids(train_batch)
    if train_ids is not None and len(train_ids) != n_train:
        raise ValueError("train_batch record_ids length differs from its rows")
    selected = select_device(device)
    model = model.to(selected)

    # This seed controls optimizer/dropout-side PyTorch randomness.  The
    # per-epoch shuffle stream is independent and the callback owns its mask
    # stream, as required by the experiment protocol.
    torch.manual_seed(int(seed))
    # GRUCell, Linear, AdamW and the scalar losses used here are deterministic
    # on the CPU for a fixed seed.  Avoid changing PyTorch's process-global
    # deterministic-algorithm switch: callers may run a later MPS evaluation in
    # the same process and the training helper should not leak global state.
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(seed) + 1_000_003)

    pos_weight, positive_count, negative_count = positive_class_weight(labels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"]
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch: int | None = None
    best_ap = float("-inf")
    best_logloss = float("inf")
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(options["epochs"]):
        model.train()
        if augmentation_callback is None:
            epoch_batch = train_batch
        else:
            epoch_batch = augmentation_callback(epoch)
            if epoch_batch is None:
                raise ValueError("augmentation_callback must return a TemporalBatch")
            if _batch_length(epoch_batch) != n_train:
                raise ValueError("augmented train batch must preserve the training row count")
            augmented_ids = _record_ids(epoch_batch)
            if train_ids is not None:
                if augmented_ids is None:
                    raise ValueError(
                        "augmented train batch must retain record_ids when training has them"
                    )
                if not np.array_equal(augmented_ids, train_ids):
                    raise ValueError("augmented train batch record_ids must match training order")

        permutation = torch.randperm(n_train, generator=shuffle_generator).numpy()
        total_loss = 0.0
        total_examples = 0
        for start in range(0, n_train, options["batch_size"]):
            indices = permutation[start : start + options["batch_size"]]
            mini = _to_torch_batch(_slice_batch(epoch_batch, indices), selected)
            target = torch.as_tensor(labels[indices], device=selected, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(mini, debug=False)
            logits = _extract_logits(output)
            loss = binary_objective(
                logits,
                target,
                kind=options["loss"],
                pos_weight=pos_weight if options["loss"] == "weighted_bce" else None,
                gamma=options["gamma"],
                alpha=options["alpha"],
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), options["gradient_clip"]
            )
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_examples += count

        model.eval()
        dev_metrics: dict[str, dict[str, float]] = {}
        ap_values: list[float] = []
        logloss_values: list[float] = []
        fallback_labels = _as_labels(y_dev) if not isinstance(y_dev, Mapping) else None
        for condition, batch in dev_batches.items():
            condition_labels = _condition_labels(y_dev, condition, fallback_labels)
            logits = predict_logits(
                model,
                batch,
                device=selected,
                batch_size=options["batch_size"],
            )
            ap, dev_logloss = _metric_pair(condition_labels, logits)
            dev_metrics[condition] = {"average_precision": ap, "log_loss": dev_logloss}
            if condition in SELECTION_CONDITIONS:
                ap_values.append(ap)
                logloss_values.append(dev_logloss)

        mean_ap = float(np.nanmean(ap_values)) if ap_values else float("nan")
        mean_logloss = (
            float(np.nanmean(logloss_values)) if logloss_values else float("inf")
        )
        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_examples, 1),
            "mean_average_precision": mean_ap,
            "mean_log_loss": mean_logloss,
            "dev": dev_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        for condition, metrics in dev_metrics.items():
            epoch_record[f"{condition}_average_precision"] = metrics["average_precision"]
            epoch_record[f"{condition}_log_loss"] = metrics["log_loss"]
        history.append(epoch_record)

        better = False
        if np.isfinite(mean_ap):
            if mean_ap > best_ap + 1e-12:
                better = True
            elif abs(mean_ap - best_ap) <= 1e-12 and mean_logloss < best_logloss - 1e-12:
                better = True
        if better or best_state is None:
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            best_ap = mean_ap
            best_logloss = mean_logloss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= options["patience"]:
                break

    if best_state is None or best_epoch is None:
        raise RuntimeError("training completed without a finite development checkpoint")
    model.load_state_dict(best_state)
    model.to(selected)
    model.eval()
    elapsed = time.perf_counter() - started
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_mean_average_precision": best_ap,
        "best_mean_log_loss": best_logloss,
        "best_ap": best_ap,
        "best_log_loss": best_logloss,
        "elapsed_seconds": elapsed,
        "positive_weight": pos_weight,
        "pos_weight": pos_weight,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "n_positive": positive_count,
        "n_negative": negative_count,
        "checkpoint": best_state,
        "checkpoint_state_dict": best_state,
        "device": str(selected),
        "parameter_count": parameter_count(model),
    }


__all__ = ["DEFAULT_CONFIG", "fit_model", "predict_logits"]
