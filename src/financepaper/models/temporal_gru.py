"""Compact temporal GRU models for the incomplete-credit pilot.

The module intentionally keeps PyTorch imports local to this optional model
branch.  The static LR/XGBoost package can therefore still be imported in an
environment without PyTorch installed.

``TemporalGRU`` is the vanilla control.  It consumes encoded temporal values
and encoded static values; availability masks and elapsed deltas are retained
in the diagnostic output but do not affect its prediction.  The
``MissingAwareGRU`` subclass adds those fields to the recurrent input.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn


# The Taiwan tensor has three original fields per month and five static
# original fields.  Encoded values may be wider because PAY fields are one-hot
# encoded, while the mask/delta remain at the original-field granularity.
TEMPORAL_MASK_DIM = 3
STATIC_MASK_DIM = 5
DELTA_SCALE_MONTHS = 5.0


def _field(batch: Any, name: str, default: Any = None) -> Any:
    """Read a field from a tensor dictionary or a TemporalBatch-like object."""

    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    # ``TemporalBatch`` may expose read-only NumPy views after a dataframe
    # transformation.  Copying here avoids PyTorch's undefined write warning
    # while keeping the model's optional direct-dictionary API convenient.
    return torch.tensor(value, device=device, dtype=dtype)


def _check_batch_size(reference: Tensor, value: Tensor | None, name: str) -> None:
    if value is not None and value.ndim > 0 and value.shape[0] != reference.shape[0]:
        raise ValueError(f"{name} batch dimension does not match temporal values")


def _validate_binary_mask(mask: Tensor, name: str) -> None:
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must contain finite binary availability values")


class TemporalGRU(nn.Module):
    """A small static-plus-temporal GRU risk model.

    Parameters match the frozen temporal specification.  ``use_temporal`` and
    ``use_static`` are structural ablations; at least one branch must remain
    enabled.  ``forward`` accepts either a mapping or an object exposing the
    ``TemporalBatch`` fields ``temporal``, ``static``,
    ``temporal_observed``, ``static_observed`` and ``delta``.
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
        use_static_mask: bool = False,
    ) -> None:
        super().__init__()
        if temporal_dim < 0 or static_dim < 0:
            raise ValueError("temporal_dim and static_dim must be non-negative")
        if hidden_size <= 0 or static_hidden <= 0 or fusion_hidden <= 0:
            raise ValueError("hidden sizes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not use_static and not use_temporal:
            raise ValueError("at least one of use_static/use_temporal must be enabled")
        if use_temporal and temporal_dim <= 0:
            raise ValueError("temporal_dim must be positive when use_temporal=True")
        if use_static and static_dim <= 0:
            raise ValueError("static_dim must be positive when use_static=True")

        self.temporal_dim = int(temporal_dim)
        self.static_dim = int(static_dim)
        self.hidden_size = int(hidden_size)
        self.static_hidden = int(static_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.dropout_rate = float(dropout)
        self.use_static = bool(use_static)
        self.use_temporal = bool(use_temporal)
        self.use_static_mask = bool(use_static_mask)

        self.temporal_cell = (
            nn.GRUCell(self.temporal_dim, self.hidden_size) if self.use_temporal else None
        )
        # Static masks are at original-field granularity (LIMIT_BAL, AGE,
        # SEX, EDUCATION, MARRIAGE), even when categorical values are encoded.
        self.static_encoder = (
            nn.Sequential(
                nn.Linear(
                    self.static_dim + (STATIC_MASK_DIM if self.use_static_mask else 0),
                    self.static_hidden,
                ),
                nn.ReLU(),
            )
            if self.use_static
            else None
        )

        fusion_input = (self.hidden_size if self.use_temporal else 0) + (
            self.static_hidden if self.use_static else 0
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.risk_head = nn.Linear(self.fusion_hidden, 1)

    def _temporal_input(
        self,
        temporal: Tensor,
        temporal_observed: Tensor,
        delta: Tensor,
    ) -> Tensor:
        """Build the vanilla recurrent input.

        Subclasses override this to append mask/delta channels.  Keeping the
        hook separate makes the vanilla control's invariance to simulated
        missingness explicit and testable.
        """

        del temporal_observed, delta
        return temporal

    def _prepare_inputs(self, batch: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        required = ("temporal", "static", "temporal_observed", "static_observed", "delta")
        missing = [name for name in required if _field(batch, name) is None]
        if missing:
            raise KeyError(f"batch is missing required fields: {', '.join(missing)}")

        temporal = _as_tensor(_field(batch, "temporal"), device=self.risk_head.weight.device)
        static = _as_tensor(_field(batch, "static"), device=temporal.device)
        temporal_observed = _as_tensor(
            _field(batch, "temporal_observed"), device=temporal.device
        )
        static_observed = _as_tensor(
            _field(batch, "static_observed"), device=temporal.device
        )
        delta = _as_tensor(_field(batch, "delta"), device=temporal.device)

        if temporal.ndim != 3 or temporal.shape[1] != 6:
            raise ValueError("temporal values must have canonical shape [B, 6, D]")
        if temporal.shape[-1] != self.temporal_dim:
            raise ValueError(
                f"temporal feature width {temporal.shape[-1]} != configured {self.temporal_dim}"
            )
        if static.ndim != 2 or static.shape[-1] != self.static_dim:
            raise ValueError(
                f"static values must have shape [B, {self.static_dim}]"
            )
        if temporal.shape[0] == 0:
            raise ValueError("batch must contain at least one record")
        _check_batch_size(temporal, static, "static")

        if temporal_observed.shape != (temporal.shape[0], 6, TEMPORAL_MASK_DIM):
            raise ValueError("temporal_observed must have canonical shape [B, 6, 3]")
        if static_observed.shape != (temporal.shape[0], STATIC_MASK_DIM):
            raise ValueError("static_observed must have canonical shape [B, 5]")
        if delta.shape != temporal_observed.shape:
            raise ValueError("delta must have canonical shape [B, 6, 3]")
        _validate_binary_mask(temporal_observed, "temporal_observed")
        _validate_binary_mask(static_observed, "static_observed")
        if not torch.isfinite(delta).all() or torch.any(delta < 0):
            raise ValueError("delta must contain finite nonnegative elapsed months")
        if not torch.isfinite(temporal).all() or not torch.isfinite(static).all():
            raise ValueError("encoded values must be finite")

        return temporal, static, temporal_observed, static_observed, delta

    @staticmethod
    def _missingness(
        temporal_observed: Tensor,
        static_observed: Tensor,
    ) -> dict[str, Tensor]:
        temporal_fraction = 1.0 - temporal_observed.mean(dim=(1, 2))
        static_fraction = 1.0 - static_observed.mean(dim=1)
        total_observed = temporal_observed.sum(dim=(1, 2)) + static_observed.sum(dim=1)
        total_fields = temporal_observed[0].numel() + static_observed[0].numel()
        fraction = 1.0 - total_observed / float(total_fields)
        per_month = 1.0 - temporal_observed.mean(dim=2)
        return {
            "fraction": fraction,
            "temporal_fraction": temporal_fraction,
            "static_fraction": static_fraction,
            "per_month": per_month,
        }

    def forward(self, batch: Any, debug: bool = False) -> dict[str, Any]:
        temporal, static, temporal_observed, static_observed, delta = self._prepare_inputs(batch)
        device = self.risk_head.weight.device
        batch_size = temporal.shape[0]

        if self.use_temporal:
            assert self.temporal_cell is not None
            if temporal.shape[1] <= 0:
                raise ValueError("temporal sequence must contain at least one step")
            hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=temporal.dtype)
            hidden_steps: list[Tensor] = []
            recurrent_input = self._temporal_input(temporal, temporal_observed, delta)
            if recurrent_input.shape[-1] != self.temporal_cell.input_size:
                raise ValueError(
                    "recurrent input width does not match model; check mask/delta dimensions"
                )
            for step in range(recurrent_input.shape[1]):
                hidden = self.temporal_cell(recurrent_input[:, step, :], hidden)
                if debug:
                    hidden_steps.append(hidden)
            temporal_embedding = hidden
            hidden_states = torch.stack(hidden_steps, dim=1) if debug else None
        else:
            temporal_embedding = torch.zeros(
                batch_size,
                0,
                device=device,
                dtype=temporal.dtype,
            )
            hidden_states = None

        if self.use_static:
            assert self.static_encoder is not None
            static_input = (
                torch.cat((static, static_observed), dim=-1)
                if self.use_static_mask
                else static
            )
            static_embedding = self.static_encoder(static_input)
        else:
            static_embedding = torch.zeros(
                batch_size,
                0,
                device=device,
                dtype=temporal.dtype,
            )

        fused = torch.cat((temporal_embedding, static_embedding), dim=-1)
        fused_embedding = self.fusion(fused)
        logit = self.risk_head(fused_embedding).squeeze(-1)
        probability = torch.sigmoid(logit)

        output: dict[str, Any] = {
            "risk": {
                "logit": logit,
                "prob_raw": probability,
                "prob_calibrated": None,
            },
            "missingness": self._missingness(temporal_observed, static_observed),
            "temporal": None,
            "embeddings": None,
            "explanation": None,
            "uncertainty": None,
            "reliability": {"revision_risk": None, "release": None},
        }
        if debug:
            output["temporal"] = {
                "hidden_states": hidden_states,
                "summary_embedding": temporal_embedding,
            }
            output["embeddings"] = {
                "static": static_embedding,
                "fused": fused_embedding,
            }
        return output


__all__ = ["TemporalGRU", "TEMPORAL_MASK_DIM", "STATIC_MASK_DIM", "DELTA_SCALE_MONTHS"]
