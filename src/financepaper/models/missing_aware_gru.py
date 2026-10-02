"""Mask-and-delta recurrent model for the temporal pilot."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .temporal_gru import DELTA_SCALE_MONTHS, TemporalGRU


class MissingAwareGRU(TemporalGRU):
    """TemporalGRU with explicit observed-mask and elapsed-delta channels.

    This is the frozen primary architecture (often called a GRU-Simple
    mask/delta input).  It deliberately has no learned GRU-D decay and no
    imputation or reconstruction head.  ``use_mask`` and ``use_delta`` are
    structural ablations; both default to enabled.
    """

    def __init__(
        self,
        temporal_dim: int,
        static_dim: int,
        hidden_size: int = 64,
        static_hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.1,
        *,
        use_static: bool = True,
        use_temporal: bool = True,
        use_mask: bool = True,
        use_delta: bool = True,
    ) -> None:
        self.use_mask = bool(use_mask)
        self.use_delta = bool(use_delta)
        # Mask and delta retain the three original monthly fields.  The parent
        # constructs its recurrent cell using ``temporal_dim``; replace it
        # below with the augmented input width after the common branch setup.
        super().__init__(
            temporal_dim=temporal_dim,
            static_dim=static_dim,
            hidden_size=hidden_size,
            static_hidden=static_hidden,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
            use_static=use_static,
            use_temporal=use_temporal,
            use_static_mask=self.use_mask,
        )
        if self.use_temporal:
            input_width = temporal_dim
            if self.use_mask:
                input_width += 3
            if self.use_delta:
                input_width += 3
            # Replace the vanilla cell while retaining the parent branch
            # dimensions and initialization semantics.
            self.temporal_cell = torch.nn.GRUCell(input_width, hidden_size)

    def _temporal_input(
        self,
        temporal: Tensor,
        temporal_observed: Tensor,
        delta: Tensor,
    ) -> Tensor:
        parts = [temporal]
        if self.use_mask:
            parts.append(temporal_observed)
        if self.use_delta:
            parts.append(delta / DELTA_SCALE_MONTHS)
        return torch.cat(parts, dim=-1)


__all__ = ["MissingAwareGRU"]
