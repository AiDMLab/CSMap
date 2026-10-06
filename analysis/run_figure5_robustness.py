"""Run MOB robustness and within-method efficiency experiments for Figure 5."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from csmap_runner import run_csmap
from metrics import compute_metrics, load_assigned_csv


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "analysis"
MOB = ROOT / "inputs" / "mob"
EXP = WORK / "output" / "figure5_robustness"
GENE_INPUTS = EXP / "gene_inputs"
RUNS = EXP / "runs"
TIMING_CSV = EXP / "timing.csv"

CAPACITIES = [5, 8, 10, 12, 15]
SEEDS = [1, 2, 3, 4, 5]
GENE_COUNTS = [500, 1000, 2000, 5000]


def prepare_gene_inputs() -> None:
    expected = [GENE_INPUTS / f"g{n}" / "MOBsc.csv" for n in GENE_COUNTS]
    if all(path.exists() for path in expected):
        print("Gene-subset inputs already exist", flush=True)
        return

    print("Reading full MOB matrices to select high-variance shared genes", flush=True)
    sc = pd.read_csv(MOB / "MOBsc.csv", index_col=0).astype(np.float32)
    st = pd.read_csv(MOB / "MOBst.csv", index_col=0).astype(np.float32)
    shared = sc.index.intersection(st.index)
    sc = sc.loc[shared]
    st = st.loc[shared]

    # Variance after log1p stabilizes extreme count differences. Ties retain
    # input order, making gene selection deterministic.
    log_sc = np.log1p(sc)
    variance = log_sc.var(axis=1).sort_values(ascending=False, kind="mergesort")
    del log_sc
    for count in GENE_COUNTS:
        out = GENE_INPUTS / f"g{count}"
        out.mkdir(parents=True, exist_ok=True)
        genes = variance.index[:count]
        sc.loc[genes].to_csv(out / "MOBsc.csv")
        st.loc[genes].to_csv(out / "MOBst.csv")
        print(f"Wrote {count}-gene input matrices", flush=True)


def run_single(family: str, value: str) -> None:
    if family == "capacity":
        mean_cells = int(value)
        seed = 1
        sc_path = MOB / "MOBsc.csv"
        st_path = MOB / "MOBst.csv"
        name = f"capacity_{mean_cells}"
    elif family == "seed":
        mean_cells = 10
        seed = int(value)
        sc_path = MOB / "MOBsc.csv"
        st_path = MOB / "MOBst.csv"
        name = f"seed_{seed}"
    elif family == "genes":
        mean_cells = 10
        seed = 1
        count = int(value)
        sc_path = GENE_INPUTS / f"g{count}" / "MOBsc.csv"
        st_path = GENE_INPUTS / f"g{count}" / "MOBst.csv"
        name = f"genes_{count}"
    else:
        raise ValueError(f"Unknown family: {family}")

    out = RUNS / name
    target = out / "mob_assigned_locations.csv"
    if target.exists():
        print(f"Using existing {target}")
        return
    run_csmap(
        output_folder=out,
        output_prefix="mob_",
        lambda_1=1e-6,
        mean_cell_numbers=mean_cells,
        seed=seed,
        plot_off=True,
        scRNA_path=sc_path,
        cell_type_path=MOB / "MOBcelltype.csv",
        st_path=st_path,
        coordinates_path=MOB / "MOBstxy.csv",
        cellfracs_path=WORK / "data" / "Seurat_cellfracs.txt",
        weights_path=WORK / "data" / "Seurat_weights.txt",
    )


def parse_time_output(text: str) -> tuple[float, float]:
    real = re.search(r"([0-9.]+)\s+real", text)
    rss = re.search(r"([0-9]+)\s+maximum resident set size", text)
    if not real or not rss:
        raise RuntimeError("Could not parse /usr/bin/time output")
    return float(real.group(1)), int(rss.group(1)) / (1024**2)


def load_timing() -> pd.DataFrame:
    if TIMING_CSV.exists():
        return pd.read_csv(TIMING_CSV)
    return pd.DataFrame(columns=["family", "value", "runtime_s", "peak_rss_mb"])


def run_condition(family: str, value: int) -> None:
    timing = load_timing()
    existing = (timing["family"].astype(str) == family) & (timing["value"].astype(str) == str(value))
    target = RUNS / f"{family}_{value}" / "mob_assigned_locations.csv"
    if existing.any() and target.exists():
        print(f"Completed {family}={value} (cached)", flush=True)
        return

    print(f"Running {family}={value}", flush=True)
    cmd = [
        "/usr/bin/time",
        "-l",
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        family,
        str(value),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    log_dir = EXP / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{family}_{value}.stdout.log").write_text(result.stdout)
    (log_dir / f"{family}_{value}.stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Run failed for {family}={value}; see {log_dir}")
    runtime_s, peak_rss_mb = parse_time_output(result.stderr)
    timing = timing.loc[~existing].copy()
    timing = pd.concat(
        [
            timing,
            pd.DataFrame(
                [{"family": family, "value": value, "runtime_s": runtime_s, "peak_rss_mb": peak_rss_mb}]
            ),
        ],
        ignore_index=True,
    )
    EXP.mkdir(parents=True, exist_ok=True)
    timing.to_csv(TIMING_CSV, index=False)
    print(f"  runtime={runtime_s:.2f}s peak_RSS={peak_rss_mb:.1f}MB", flush=True)


def collect_metrics() -> None:
    gt = pd.read_csv(WORK / "data" / "mob_ground_truth.csv", index_col=0)
    gt.index = gt.index.astype(str)
    gt["true_spot"] = gt["true_spot"].astype(str)
    coords = pd.read_csv(WORK / "data" / "mob_spot_coords.csv", index_col=0)
    coords.index = coords.index.astype(str)
    timing = load_timing()

    rows = []
    conditions = [("capacity", v) for v in CAPACITIES] + [("seed", v) for v in SEEDS[1:]] + [
        ("genes", v) for v in GENE_COUNTS
    ]
    for family, value in conditions:
        path = RUNS / f"{family}_{value}" / "mob_assigned_locations.csv"
        assigned = load_assigned_csv(str(path), coords)
        metrics = compute_metrics(gt, assigned, coords, total_input_cells=len(gt))
        time_row = timing[(timing["family"] == family) & (timing["value"].astype(int) == value)].iloc[-1]
        rows.append(
            {
                "family": family,
                "value": value,
                **{k: metrics[k] for k in ["CMR", "CU", "WCMR", "WRR", "OP_Score"]},
                "n_assignments": len(assigned),
                "n_unique_cells": assigned.index.nunique(),
                "runtime_s": float(time_row["runtime_s"]),
                "peak_rss_mb": float(time_row["peak_rss_mb"]),
            }
        )

    # The capacity=10, seed=1, full-gene run is the common reference for the
    # seed and gene-series panels.
    baseline = next(row for row in rows if row["family"] == "capacity" and row["value"] == 10)
    for family, value in [("seed", 1), ("genes", 18263)]:
        copy = dict(baseline)
        copy["family"] = family
        copy["value"] = value
        rows.append(copy)

    pd.DataFrame(rows).sort_values(["family", "value"]).to_csv(EXP / "robustness_metrics.csv", index=False)
    print(f"Wrote {EXP / 'robustness_metrics.csv'}", flush=True)


def orchestrate() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    prepare_gene_inputs()
    for value in CAPACITIES:
        run_condition("capacity", value)
    for value in SEEDS[1:]:
        run_condition("seed", value)
    for value in GENE_COUNTS:
        run_condition("genes", value)
    collect_metrics()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=2, metavar=("FAMILY", "VALUE"))
    args = parser.parse_args()
    if args.single:
        run_single(args.single[0], args.single[1])
    else:
        orchestrate()


if __name__ == "__main__":
    main()
