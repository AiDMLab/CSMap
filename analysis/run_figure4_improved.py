#!/usr/bin/env python3
"""Run the expanded CSMap constraint ablation used for revised SCI Figure 4.

The experiment keeps the published/raw CSMap cost scale and evaluates a broad
lambda grid.  It also creates reproducible corrupted composition priors for the
MOB robustness controls.  Cerebellum priors are derived from the paired
simulation ground truth; they are used only for this benchmark ablation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from csmap_runner import run_csmap


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "analysis"
RUN_ROOT = WORK / "output" / "figure4_improved"

DATASETS = {
    "mob": {
        "sc": ROOT / "inputs" / "mob" / "MOBsc.csv",
        "labels": ROOT / "inputs" / "mob" / "MOBcelltype.csv",
        "st": ROOT / "inputs" / "mob" / "MOBst.csv",
        "coords": ROOT / "inputs" / "mob" / "MOBstxy.csv",
        "gt": WORK / "data" / "mob_ground_truth.csv",
        "mean_cells": 10,
    },
    "cere_n5": {
        "sc": ROOT / "inputs" / "cerebellum" / "test1" / "scRNA_Noise05.csv",
        "labels": ROOT / "inputs" / "cerebellum" / "test1" / "cell_labels.csv",
        "st": ROOT / "inputs" / "cerebellum" / "test1" / "ST_expression.csv",
        "coords": ROOT / "inputs" / "cerebellum" / "test1" / "Coordinates_non_empty.csv",
        "gt": ROOT / "data" / "figure3" / "cere_n5_ground_truth.csv",
        "mean_cells": 5,
    },
    "cere_n15": {
        "sc": ROOT / "inputs" / "cerebellum" / "test2" / "scRNA_Noise05.csv",
        "labels": ROOT / "inputs" / "cerebellum" / "test2" / "cell_labels.csv",
        "st": ROOT / "inputs" / "cerebellum" / "test2" / "ST_expression.csv",
        "coords": ROOT / "inputs" / "cerebellum" / "test2" / "Coordinates.csv",
        "gt": ROOT / "data" / "figure3" / "cere_n15_ground_truth.csv",
        "mean_cells": 15,
    },
}

LAMBDA_GRID = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 3.0]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def _ground_truth_table(dataset: str) -> pd.DataFrame:
    path = Path(DATASETS[dataset]["gt"])
    gt = pd.read_csv(path)
    if dataset == "mob":
        gt = gt.rename(columns={gt.columns[0]: "cell_id", "true_spot": "spot", "cellType": "cell_type"})
    else:
        gt = gt.rename(columns={"cell_id_new": "cell_id", "CellType": "cell_type"})
    return gt[["cell_id", "spot", "cell_type"]].astype(str)


def prepare_priors(dataset: str) -> tuple[Path, Path]:
    out = RUN_ROOT / "priors" / dataset
    out.mkdir(parents=True, exist_ok=True)
    frac_path = out / "Seurat_cellfracs.txt"
    weight_path = out / "Seurat_weights.txt"
    if frac_path.exists() and weight_path.exists():
        return frac_path, weight_path

    gt = _ground_truth_table(dataset)
    labels = pd.read_csv(DATASETS[dataset]["labels"])
    label_types = labels.iloc[:, 1].astype(str).drop_duplicates().tolist()
    counts = gt.groupby(["spot", "cell_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=label_types, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    fractions.index.name = "SpotID"
    fractions.to_csv(frac_path, sep="\t")

    global_counts = labels.iloc[:, 1].astype(str).value_counts().reindex(label_types, fill_value=0)
    weights = (global_counts / global_counts.sum()).to_frame().T
    weights.index = ["Fraction"]
    weights.to_csv(weight_path, sep="\t")
    return frac_path, weight_path


def prepare_mob_noisy_prior(level: float, seed: int = 2026) -> tuple[Path, Path]:
    base_frac, base_weights = prepare_priors("mob")
    out = RUN_ROOT / "priors" / "mob_noise" / f"noise_{level:g}"
    out.mkdir(parents=True, exist_ok=True)
    frac_path = out / "Seurat_cellfracs.txt"
    weight_path = out / "Seurat_weights.txt"
    if frac_path.exists() and weight_path.exists():
        return frac_path, weight_path

    frac = pd.read_csv(base_frac, sep="\t", index_col=0)
    rng = np.random.default_rng(seed)
    permuted = frac.iloc[rng.permutation(len(frac))].copy()
    permuted.index = frac.index
    corrupted = (1.0 - level) * frac + level * permuted
    corrupted = corrupted.div(corrupted.sum(axis=1), axis=0).fillna(0.0)
    corrupted.index.name = "SpotID"
    corrupted.to_csv(frac_path, sep="\t")
    shutil.copy2(base_weights, weight_path)
    return frac_path, weight_path


def _lambda_tag(value: float) -> str:
    return "0" if value == 0 else f"{value:.0e}".replace("+", "")


def run_one(dataset: str, variant: str, penalty: float, frac: Path, weights: Path) -> Path:
    cfg = DATASETS[dataset]
    out = RUN_ROOT / "runs" / dataset / variant
    target = out / f"{dataset}_assigned_locations.csv"
    if target.exists():
        print(f"Using existing {target}", flush=True)
        return target
    print(f"Running {dataset}/{variant}: lambda={penalty:g}", flush=True)
    result = run_csmap(
        output_folder=out,
        output_prefix=f"{dataset}_",
        lambda_1=penalty,
        mean_cell_numbers=int(cfg["mean_cells"]),
        seed=1,
        plot_off=True,
        scRNA_path=cfg["sc"],
        cell_type_path=cfg["labels"],
        st_path=cfg["st"],
        coordinates_path=cfg["coords"],
        cellfracs_path=frac,
        weights_path=weights,
    )
    return result


def run_grid(datasets: list[str], lambdas: list[float]) -> None:
    manifest: list[dict] = []
    for dataset in datasets:
        frac, weights = prepare_priors(dataset)
        for penalty in lambdas:
            variant = f"lambda_{_lambda_tag(penalty)}"
            result = run_one(dataset, variant, penalty, frac, weights)
            manifest.append({"dataset": dataset, "variant": variant, "lambda": penalty, "assigned": str(result)})
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_noise(penalty: float) -> None:
    for level in NOISE_LEVELS:
        frac, weights = prepare_mob_noisy_prior(level)
        run_one("mob", f"noise_{level:g}_lambda_{_lambda_tag(penalty)}", penalty, frac, weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDA_GRID)
    parser.add_argument("--noise-lambda", type=float)
    args = parser.parse_args()
    run_grid(args.datasets, args.lambdas)
    if args.noise_lambda is not None:
        run_noise(args.noise_lambda)


if __name__ == "__main__":
    main()
