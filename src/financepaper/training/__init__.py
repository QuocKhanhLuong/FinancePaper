"""Optional PyTorch training helpers for the temporal pilot.

Imports are lazy so importing :mod:`financepaper` or the static baseline
modules does not require the optional PyTorch extra.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "fit_model": ("financepaper.training.fit", "fit_model"),
    "predict_logits": ("financepaper.training.fit", "predict_logits"),
    "binary_loss": ("financepaper.training.losses", "binary_loss"),
    "binary_objective": ("financepaper.training.losses", "binary_objective"),
    "compute_pos_weight": ("financepaper.training.losses", "compute_pos_weight"),
    "device_info": ("financepaper.training.device", "device_info"),
    "select_device": ("financepaper.training.device", "select_device"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
