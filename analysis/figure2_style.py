"""Build SCI Figure 2: quantitative benchmark of CSMap and three baselines.

The figure uses the saved Chapter 5 metric tables. Summary metrics are
single-run point estimates; SMLE panels show per-spot distributions and must
not be interpreted as replicate-level uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ALGORITHMS = ["CSMap", "CytoSPACE", "Tangram", "CellTrek"]
METRICS = ["CMR", "WCMR", "WRR", "CU", "OP_Score"]
METRIC_LABELS = {
    "CMR": "CMR",
    "WCMR": "WCMR",
    "WRR": "WRR",
    "CU": "CU",
    "OP_Score": "OP-Score",
}

# Okabe-Ito-inspired, color-vision-deficiency-friendly palette.
COLORS = {
    "CSMap": "#0072B2",
    "CytoSPACE": "#E69F00",
    "Tangram": "#009E73",
    "CellTrek": "#CC79A7",
}
MARKERS = {"CSMap": "o", "CytoSPACE": "s", "Tangram": "^", "CellTrek": "D"}
OFFSETS = {"CSMap": -0.24, "CytoSPACE": -0.08, "Tangram": 0.08, "CellTrek": 0.24}

DATASETS = [
    {
        "name": "MOB",
        "subtitle": "2,561 cells · 260 spots · 6 cell types",
    },
    {
        "name": "Cerebellum (n = 5)",
        "subtitle": "23,372 cells · 4,761 spots · 11 cell types",
    },
    {
        "name": "Cerebellum (n = 15)",
        "subtitle": "23,372 cells · 1,544 spots · 11 cell types",
    },
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    sns.set_theme(style="ticks", context="paper", rc=mpl.rcParams)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.19,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_summary(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    spec: dict,
    panel: str,
    show_metric_labels: bool,
) -> None:
    y_base = np.arange(len(METRICS))[::-1]
    for idx, y in enumerate(y_base):
        if idx % 2 == 0:
            ax.axhspan(y - 0.46, y + 0.46, color="#F3F6F8", zorder=0)

    for algorithm in ALGORITHMS:
        values = metrics.loc[algorithm, METRICS].astype(float).to_numpy()
        y = y_base + OFFSETS[algorithm]
        ax.scatter(
            values,
            y,
            s=34 if algorithm == "CSMap" else 27,
            marker=MARKERS[algorithm],
            color=COLORS[algorithm],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
            label=algorithm,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Score  (higher is better)")
    ax.set_yticks(y_base)
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS] if show_metric_labels else [])
    ax.grid(axis="x", color="#DCE3E8", linewidth=0.65, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_title(spec["name"], fontweight="bold", pad=17)
    ax.text(
        0.5,
        1.035,
        spec["subtitle"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#596B78",
    )
    add_panel_label(ax, panel)


def plot_smle(ax: plt.Axes, smle: pd.DataFrame, panel: str) -> None:
    sns.violinplot(
        data=smle,
        x="algorithm",
        y="log1p_SMLE",
        hue="algorithm",
        order=ALGORITHMS,
        hue_order=ALGORITHMS,
        palette=[COLORS[a] for a in ALGORITHMS],
        legend=False,
        inner=None,
        cut=0,
        linewidth=0.65,
        saturation=0.85,
        width=0.82,
        ax=ax,
    )
    sns.boxplot(
        data=smle,
        x="algorithm",
        y="log1p_SMLE",
        order=ALGORITHMS,
        width=0.22,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#263845", "linewidth": 0.65},
        whiskerprops={"color": "#263845", "linewidth": 0.65},
        capprops={"color": "#263845", "linewidth": 0.65},
        medianprops={"color": "#263845", "linewidth": 1.1},
        ax=ax,
    )

    raw_ticks = np.array([0, 1, 2, 5, 10, 20, 50, 100], dtype=float)
    max_raw = float(smle["SMLE"].max())
    use_ticks = raw_ticks[raw_ticks <= max(10, max_raw * 1.08)]
    ax.set_yticks(np.log10(1.0 + use_ticks))
    ax.set_yticklabels([f"{int(v)}" for v in use_ticks])
    ax.set_ylim(-0.04, np.log10(1 + max_raw) * 1.06)
    ax.set_xlabel("")
    ax.set_ylabel("SMLE  (log scale; lower is better)")
    ax.set_xticks(range(len(ALGORITHMS)), ["CSMap", "Cyto-\nSPACE", "Tangram", "Cell-\nTrek"])
    ax.grid(axis="y", color="#E1E7EB", linewidth=0.65)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3, labelsize=6.3)

    medians = smle.groupby("algorithm", observed=False)["SMLE"].median().reindex(ALGORITHMS)
    for x, algorithm in enumerate(ALGORITHMS):
        transformed = np.log10(1 + medians.loc[algorithm])
        ax.text(
            x,
            min(ax.get_ylim()[1] - 0.03, transformed + 0.085),
            f"{medians.loc[algorithm]:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#263845",
            fontweight="bold" if algorithm == "CSMap" else "normal",
        )
    add_panel_label(ax, panel)


