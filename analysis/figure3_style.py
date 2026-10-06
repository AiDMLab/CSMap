"""Build SCI Figure 3: spatial reconstruction across three benchmark settings."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


METHODS = ["Ground Truth", "CSMap", "CytoSPACE", "Tangram", "CellTrek"]

MOB_COLORS = {
    "Astrocytes": "#4E79A7",
    "Ependymal": "#F28E2B",
    "Immune": "#E15759",
    "Neurons": "#76B7B2",
    "Oligos": "#59A14F",
    "Vascular": "#B07AA1",
}

CERE_COLORS = {
    "Astrocyte": "#4E79A7",
    "Bergmann": "#F28E2B",
    "Choroid": "#E15759",
    "Endothelial": "#76B7B2",
    "Fibroblast": "#59A14F",
    "Granule": "#EDC948",
    "Microglia": "#B07AA1",
    "Nnat": "#FF9DA7",
    "Oligodendrocyte": "#9C755F",
    "Purkinje": "#BAB0AC",
    "Pvalb": "#2F4B7C",
}

JITTER_SEEDS = {
    "Ground Truth": 7,
    "CSMap": 11,
    "CytoSPACE": 17,
    "Tangram": 23,
    "CellTrek": 29,
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def median_spacing(coords: pd.DataFrame) -> float:
    pts = coords[["x", "y"]].drop_duplicates().astype(float).to_numpy()
    if len(pts) < 2:
        return 1.0
    distances = cdist(pts, pts)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def jitter_cells(df: pd.DataFrame, spacing: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half_width = spacing * 0.43
    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()
    return (
        x + rng.uniform(-half_width, half_width, len(df)),
        y + rng.uniform(-half_width, half_width, len(df)),
    )


def panel_limits(dataset: dict) -> tuple[tuple[float, float], tuple[float, float], float]:
    coords = dataset["spot_coords"]
    spacing = median_spacing(coords)
    pad = spacing * 1.0
    return (
        (float(coords["x"].min() - pad), float(coords["x"].max() + pad)),
        (float(coords["y"].min() - pad), float(coords["y"].max() + pad)),
        spacing,
    )


