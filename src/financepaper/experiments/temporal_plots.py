"""Small, reproducible diagnostic figures; no model or threshold selection."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "xgb_aug": "XGBoost + masking",
    "vanilla_aug_bce": "Vanilla GRU + masking",
    "mask_delta_aug_bce": "Mask/delta GRU + BCE",
    "mask_delta_aug_weighted_bce": "Mask/delta GRU + weighted BCE",
    "mask_delta_aug_focal": "Mask/delta GRU + focal",
}


def plot_temporal_results(output: Path) -> None:
    """Render calibration and descriptive revision-risk curves from saved rows."""
    output = Path(output)
    calibration = pd.read_csv(output / "calibration_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, condition in zip(axes, ("mcar30", "mar30"), strict=True):
        selected = calibration[(calibration.condition == condition)
                               & (calibration.probability == "calibrated")]
        ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        for model, label in LABELS.items():
            rows = selected[(selected.model == model) & (selected.n > 0)]
            if len(rows):
                ax.plot(rows.mean_probability, rows.default_fraction, ".-", label=label)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean calibrated probability",
               ylabel="Observed default fraction", title=condition.upper())
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Held-out probability calibration • full test partition")
    fig.savefig(output / "temporal_calibration.png", dpi=160)
    plt.close(fig)

    curve_path = output / "risk_coverage.csv"
    if not curve_path.exists():
        return
    curves = pd.read_csv(curve_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, condition in zip(axes, ("mcar30", "mar30"), strict=True):
        selected = curves[curves.condition == condition]
        for model, label in LABELS.items():
            rows = selected[selected.model == model].sort_values("coverage")
            if len(rows):
                ax.step(rows.coverage, rows.revision_rate, where="post", label=label)
        ax.axhline(.10, linestyle="--", color="0.6", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Fraction of attribution subset",
               ylabel="Observed reason-revision rate", title=condition.upper())
        ax.grid(alpha=.15)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Missing-fraction ranking • descriptive subset curves, no release guarantee")
    fig.savefig(output / "temporal_risk_coverage.png", dpi=160)
    plt.close(fig)
