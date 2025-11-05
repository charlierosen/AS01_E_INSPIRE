from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config.models import ModelParameters
from .kfold import KFoldResults


def _set_matplotlib_defaults() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "Computer Modern",
            "figure.dpi": 300,
            "font.size": 15,
        }
    )


def plot_combined_regression_and_residuals(
    results: KFoldResults,
    config: ModelParameters,
    *,
    output_path: Optional[Path] = None,
):
    _set_matplotlib_defaults()
    fold_colors = plt.cm.tab10(np.linspace(0, 1, len(results.folds)))

    fig, (ax_reg, ax_res) = plt.subplots(
        2, 1, figsize=(10, 16), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1}
    )

    for idx, fold in enumerate(results.folds):
        mask = results.fold_indices == (idx + 1)
        ax_reg.scatter(
            results.all_pred[mask],
            results.all_true[mask],
            alpha=0.7,
            s=35,
            color=fold_colors[idx],
            edgecolors="k",
            linewidths=0.5,
            label=f"Fold {idx + 1} (R²: {fold.r2:.4f})",
        )

    limits = results.plot_limits
    ax_reg.plot(
        [limits["min"], limits["max"]],
        [limits["min"], limits["max"]],
        "r--",
        label="Perfect prediction",
        linewidth=1.5,
    )

    for boundary in (config.small_boundary, config.large_boundary):
        ax_reg.axvline(
            x=boundary,
            color="gray",
            alpha=0.2,
            linestyle="-.",
            linewidth=2.0,
        )
        ax_reg.text(
            boundary + 0.01,
            limits["max"],
            f"DoR = {boundary}",
            style="italic",
            color="gray",
            fontsize=10,
            va="top",
            rotation=90,
        )

    ax_reg.set_xlim(limits["min"], limits["max"])
    ax_reg.set_ylim(limits["min"], limits["max"])
    ax_reg.set_ylabel(f"True {config.target}", fontsize=25)
    ax_reg.legend(loc="lower right", fontsize=10)
    props = dict(boxstyle="square", facecolor="white", alpha=0.8, edgecolor="lightgray")
    ax_reg.text(
        0.05,
        0.95,
        r"$\mathrm{Mean}\ R^2 = %.4f\ (\pm%.4f)$"
        % (results.mean_r2(), results.std_r2()),
        transform=ax_reg.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    ax_reg.grid(True, alpha=0.3)
    ax_reg.tick_params(axis="both", which="both", direction="in", labelsize=14)
    ax_reg.minorticks_on()
    for spine in ax_reg.spines.values():
        spine.set_color("lightgray")
    plt.setp(ax_reg.get_xticklabels(), visible=False)

    for idx, fold in enumerate(results.folds):
        mask = results.fold_indices == (idx + 1)
        ax_res.scatter(
            results.all_pred[mask],
            results.all_residuals[mask],
            alpha=0.7,
            s=35,
            color=fold_colors[idx],
            edgecolors="k",
            linewidths=0.5,
            label=f"Fold {idx + 1}",
        )

    ax_res.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="Perfect prediction")
    for boundary in (config.small_boundary, config.large_boundary):
        ax_res.axvline(
            x=boundary,
            color="gray",
            alpha=0.2,
            linestyle="-.",
            linewidth=2.0,
        )

    ymin, ymax = ax_res.get_ylim()
    for boundary in (config.small_boundary, config.large_boundary):
        ax_res.text(
            boundary + 0.01,
            ymin * 0.9,
            f"DoR = {boundary}",
            style="italic",
            color="gray",
            fontsize=10,
            va="bottom",
            rotation=90,
        )

    rmse = results.overall_rmse()
    mae = results.overall_mae()

    props = dict(boxstyle="square", facecolor="white", alpha=0.8, edgecolor="lightgray")
    ax_res.text(
        0.05,
        0.95,
        "\n".join(
            [
                r"$\mathrm{Overall\ RMSE} = %.4f$" % rmse,
                r"$\mathrm{Overall\ MAE} = %.4f$" % mae,
            ]
        ),
        transform=ax_res.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax_res.set_xlabel(f"Predicted {config.target}", fontsize=16)
    ax_res.set_ylabel("Residuals (True - Predicted)", fontsize=16)
    ax_res.set_xlim(limits["min"], limits["max"])
    ax_res.grid(True, alpha=0.3)
    ax_res.tick_params(axis="both", which="both", direction="in", labelsize=14)
    ax_res.minorticks_on()
    ax_res.legend(loc="upper right", fontsize=10)
    for spine in ax_res.spines.values():
        spine.set_color("lightgray")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_residuals(
    results: KFoldResults,
    config: ModelParameters,
    *,
    output_path: Optional[Path] = None,
):
    _set_matplotlib_defaults()
    fig, ax = plt.subplots(figsize=(10, 8))
    fold_colors = plt.cm.tab10(np.linspace(0, 1, len(results.folds)))

    for idx, fold in enumerate(results.folds):
        mask = results.fold_indices == (idx + 1)
        ax.scatter(
            results.all_pred[mask],
            results.all_residuals[mask],
            alpha=0.7,
            s=35,
            color=fold_colors[idx],
            edgecolors="k",
            linewidths=0.5,
            label=f"Fold {idx + 1}",
        )

    ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="Perfect prediction")

    limits = results.plot_limits
    ax.set_xlim(limits["min"], limits["max"])
    for boundary in (config.small_boundary, config.large_boundary):
        ax.axvline(
            x=boundary,
            color="gray",
            alpha=0.2,
            linestyle="-.",
            linewidth=2.0,
        )

    small_mask = results.all_pred <= config.small_boundary
    mid_mask = (results.all_pred > config.small_boundary) & (
        results.all_pred < config.large_boundary
    )
    large_mask = results.all_pred >= config.large_boundary

    rmse = lambda mask: float(
        np.sqrt(np.mean((results.all_true[mask] - results.all_pred[mask]) ** 2))
    ) if np.any(mask) else float("nan")

    props = dict(boxstyle="square", facecolor="white", alpha=0.8, edgecolor="lightgray")
    ax.text(
        0.05,
        0.95,
        "\n".join(
            [
                r"$\mathrm{Overall\ RMSE} = %.4f$" % results.overall_rmse(),
                r"$\mathrm{RMSE\ (DoR \leq %.2f)} = %.4f$"
                % (config.small_boundary, rmse(small_mask)),
                r"$\mathrm{RMSE\ (%.2f < DoR < %.2f)} = %.4f$"
                % (config.small_boundary, config.large_boundary, rmse(mid_mask)),
                r"$\mathrm{RMSE\ (DoR \geq %.2f)} = %.4f$"
                % (config.large_boundary, rmse(large_mask)),
            ]
        ),
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel(f"Predicted {config.target}", fontsize=16)
    ax.set_ylabel("Residuals (True - Predicted)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=14)
    ax.minorticks_on()
    ax.legend(loc="upper right", fontsize=10)

    for spine in ax.spines.values():
        spine.set_color("lightgray")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_feature_importance(
    results: KFoldResults,
    config: ModelParameters,
    *,
    output_path: Optional[Path] = None,
):
    _set_matplotlib_defaults()
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_items = sorted(
        results.feature_importances.items(), key=lambda item: item[1], reverse=True
    )
    feature_names = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    display_names = [
        config.feature_display_names.get(name, name) for name in feature_names
    ]

    bars = ax.barh(range(len(feature_names)), importances, height=0.6, align="center")
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(feature_names)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Feature Importance (MDI)", fontsize=16)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=14)

    for spine in ax.spines.values():
        spine.set_color("lightgray")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
