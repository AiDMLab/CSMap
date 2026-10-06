"""Evaluation metrics from thesis section 5.2.2.

Definitions match the thesis text and author's cerebellum summaries:
  CMR  = correct mapped instances / total mapped instances
         (each assignment row after duplication counts)
  CU   = unique OriginalCIDs used / total input cells
         (duplicated copies of the same cell count once)
  WCMR = Σ_s Precision_s * True_s / Σ_s True_s
  WRR  = Σ_s Recall_s * True_s / Σ_s True_s
  OP   = CMR * CU

Per spot:
  Precision_s = CorrectMapped_s / TotalMapped_s
  Recall_s    = CorrectMapped_s / TrueCellCount_s
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    ground_truth: pd.DataFrame,
    assigned: pd.DataFrame,
    spot_coords: pd.DataFrame | None = None,
    total_input_cells: int | None = None,
) -> dict:
    """
    ground_truth: index=cell_id, columns=[true_spot, cellType]
    assigned: index=cell_id (OriginalCID), columns=[pred_spot, x, y, cellType]
              May contain duplicate index values after CSMap/CytoSPACE upsampling.
    spot_coords: index=spot_id, columns=[x, y]
    """
    gt = ground_truth.copy()
    pred = assigned.copy()
    gt["true_spot"] = gt["true_spot"].astype(str)
    pred["pred_spot"] = pred["pred_spot"].astype(str)

    if total_input_cells is None:
        total_input_cells = len(gt)

    # Keep every assignment row for CMR; unique OriginalCID for CU.
    pred_work = pred.copy().reset_index(drop=True)
    pred_work["cell_id"] = pred.index.astype(str)
    pred_work["pred_spot"] = pred_work["pred_spot"].astype(str)

    gt_work = gt.copy().reset_index(drop=True)
    gt_work["cell_id"] = gt.index.astype(str)
    gt_work["true_spot"] = gt_work["true_spot"].astype(str)

    n_mapped_instances = len(pred_work)
    n_unique_original = int(pred_work["cell_id"].nunique())
    cu = n_unique_original / total_input_cells if total_input_cells else 0.0

    merged = pred_work.merge(gt_work[["cell_id", "true_spot"]], on="cell_id", how="inner")
    if len(merged) == 0:
        return {
            "CMR": 0.0,
            "CU": cu,
            "WCMR": 0.0,
            "WRR": 0.0,
            "OP_Score": 0.0,
            "SMLE_per_spot": [],
            "n_mapped": n_mapped_instances,
            "n_unique_original": n_unique_original,
            "n_correct": 0,
        }

    is_correct = merged["true_spot"].values == merged["pred_spot"].values
    n_correct = int(is_correct.sum())
    # CMR: thesis uses 总有效映射细胞数 = assignment instances after sampling.
    cmr = n_correct / n_mapped_instances if n_mapped_instances else 0.0

    true_counts = gt.groupby("true_spot").size().astype(float)
    all_spots = true_counts.index.astype(str)

    merged = merged.copy()
    merged["correct"] = is_correct
    mapped_counts = merged.groupby("pred_spot").size().astype(float)
    correct_by_pred = merged.loc[merged["correct"]].groupby("pred_spot").size().astype(float)

    precision = {}
    recall = {}
    for spot in all_spots:
        true_n = float(true_counts.get(spot, 0.0))
        mapped_n = float(mapped_counts.get(spot, 0.0))
        correct_n = float(correct_by_pred.get(spot, 0.0))
        precision[spot] = (correct_n / mapped_n) if mapped_n > 0 else 0.0
        recall[spot] = (correct_n / true_n) if true_n > 0 else 0.0

    weights = np.array([true_counts[s] for s in all_spots], dtype=float)
    wcmr = float(np.average([precision[s] for s in all_spots], weights=weights))
    wrr = float(np.average([recall[s] for s in all_spots], weights=weights))
    op_score = cmr * cu

    smle_values = []
    if spot_coords is not None and {"x", "y"}.issubset(merged.columns):
        for spot, grp in merged.groupby("true_spot"):
            if spot not in spot_coords.index:
                continue
            true_xy = spot_coords.loc[spot, ["x", "y"]].astype(float).values
            dists = np.sqrt(((grp[["x", "y"]].astype(float).values - true_xy) ** 2).sum(axis=1))
            smle_values.append(float(dists.mean()))

    return {
        "CMR": cmr,
        "CU": cu,
        "WCMR": wcmr,
        "WRR": wrr,
        "OP_Score": op_score,
        "SMLE_per_spot": smle_values,
        "n_mapped": n_mapped_instances,
        "n_unique_original": n_unique_original,
        "n_correct": n_correct,
    }


def load_assigned_csv(path: str, spot_coords: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Soft/unbalanced CSMap writes every reference cell for auditability.  Only
    # rows that received a tissue location are mapping instances for CMR/CU.
    if "AssignmentStatus" in df.columns:
        df = df.loc[df["AssignmentStatus"].astype(str) == "mapped"].copy()
    df = df.set_index("OriginalCID")
    df.index = df.index.astype(str)
    df["pred_spot"] = df["SpotID"].astype(str)
    if "CellType" in df.columns:
        df["cellType"] = df["CellType"]
    # Spot ids and mob_spot_coords use the order ``row x col``.  Keep that
    # same orientation here; swapping the two axes makes even an exactly
    # matched cell appear far from its true spot in SMLE.
    if {"rol", "col"}.issubset(df.columns):
        df["x"] = pd.to_numeric(df["rol"], errors="coerce")
        df["y"] = pd.to_numeric(df["col"], errors="coerce")
    elif {"row", "col"}.issubset(df.columns):
        df["x"] = pd.to_numeric(df["row"], errors="coerce")
        df["y"] = pd.to_numeric(df["col"], errors="coerce")
    if spot_coords is not None and "x" not in df.columns:
        df["x"] = spot_coords.reindex(df["pred_spot"])["x"].astype(float).values
        df["y"] = spot_coords.reindex(df["pred_spot"])["y"].astype(float).values
    return df
