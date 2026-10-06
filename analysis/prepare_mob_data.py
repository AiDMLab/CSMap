"""Prepare MOB inputs required by CSMap (Seurat_cellfracs + ground truth)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

from config import CELL_TYPES, DATA_DIR, MOB_FILES, MOB_RDATA, OUTPUT_DIR, ROOT


def load_ground_truth() -> pd.DataFrame:
    parsed = rdata.read_rda(str(MOB_RDATA))
    meta = parsed["sc_meta"].copy()
    meta.index = meta["cell_id_new"].astype(str)
    meta["spot"] = meta["spot"].astype(str)
    meta["cellType"] = meta["cellType"].astype(str)
    meta = meta[meta["cellType"].isin(CELL_TYPES)]
    return meta


def build_spot_fractions(meta: pd.DataFrame) -> pd.DataFrame:
    counts = (
        meta.groupby(["spot", "cellType"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CELL_TYPES, fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    fractions.index.name = "SpotID"
    return fractions


def build_global_fractions(meta: pd.DataFrame) -> pd.DataFrame:
    counts = meta["cellType"].value_counts().reindex(CELL_TYPES, fill_value=0)
    fracs = counts / counts.sum()
    return pd.DataFrame([fracs.values], columns=CELL_TYPES)


def write_seurat_cellfracs(fractions: pd.DataFrame, out_path: Path) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fractions.to_csv(out_path, sep="\t")


def write_ground_truth_files(meta: pd.DataFrame, coords: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    gt = meta[["spot", "cellType"]].copy()
    gt.columns = ["true_spot", "cellType"]
    gt.index.name = "cell_id"
    gt.to_csv(DATA_DIR / "mob_ground_truth.csv")

    spot_coords = coords.copy()
    spot_coords.index = spot_coords.index.astype(str)
    spot_coords.to_csv(DATA_DIR / "mob_spot_coords.csv")

    spot_counts = meta.groupby("spot").size().rename("n_cells")
    spot_counts.to_csv(DATA_DIR / "mob_spot_cell_counts.csv", header=True)


def prepare_all() -> None:
    parsed = rdata.read_rda(str(MOB_RDATA))
    meta = load_ground_truth()
    fractions = build_spot_fractions(meta)

    # CSMap reads testresult/Seurat_cellfracs.txt after csmap_runner copies it.
    testresult_frac = ROOT / "testresult" / "Seurat_cellfracs.txt"
    write_seurat_cellfracs(fractions, testresult_frac)

    data_frac = DATA_DIR / "Seurat_cellfracs.txt"
    write_seurat_cellfracs(fractions, data_frac)

    global_frac = build_global_fractions(meta)
    global_path = DATA_DIR / "Seurat_weights.txt"
    out = global_frac.copy()
    out.index = ["Fraction"]
    out.to_csv(global_path, sep="\t")
    (ROOT / "testresult").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "testresult" / "Seurat_weights.txt", sep="\t")

    coords = parsed["spatial_location"][["x", "y"]].copy()
    coords.index = coords.index.astype(str)
    write_ground_truth_files(meta, coords)

    print(f"Ground truth cells: {len(meta)}")
    print(f"Spot fractions written to: {testresult_frac.resolve()}")
    print(f"Also saved to: {data_frac}")


if __name__ == "__main__":
    prepare_all()
