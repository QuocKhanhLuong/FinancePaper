"""Device selection and reproducibility diagnostics for the temporal branch."""

from __future__ import annotations

from typing import Any

import torch


def mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def select_device(requested: str | torch.device | None = None) -> torch.device:
    """Choose MPS, then CUDA, then CPU, unless an explicit device is given."""

    if isinstance(requested, torch.device):
        name = requested.type
    elif requested is None:
        name = "auto"
    else:
        name = str(requested).strip().lower()
        if name.startswith("torch.device("):
            name = name.split("'", 2)[1]

    if name in {"", "auto", "default"}:
        if mps_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps":
        if not mps_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable")
        return torch.device("mps")
    if name in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device override: {requested!r}")


def parameter_count(model: Any) -> int:
    """Count trainable and frozen parameters in a PyTorch module."""

    if model is None:
        return 0
    return int(sum(parameter.numel() for parameter in model.parameters()))


def device_info(
    model: Any = None,
    device: str | torch.device | None = None,
    *,
    print_info: bool = True,
) -> dict[str, Any]:
    """Return and optionally print the required local hardware diagnostics."""

    selected = select_device(device)
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(selected),
        "mps_available": mps_available(),
        "parameter_count": parameter_count(model),
    }
    if print_info:
        print(
            "PyTorch version: {torch_version}\n"
            "device: {device}\n"
            "MPS availability: {mps_available}\n"
            "model parameter count: {parameter_count}".format(**info)
        )
    return info


__all__ = ["device_info", "mps_available", "parameter_count", "select_device"]
