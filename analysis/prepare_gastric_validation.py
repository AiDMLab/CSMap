#!/usr/bin/env python3
"""Prepare cross-donor gastric-cancer inputs for fixed-parameter CSMap."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, sparse
from scipy.io import mmread


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gastric_cancer"
RAW_SC = DATA / "raw_sc"
RAW_ST = DATA / "raw_st"
PREP = DATA / "prepared"

SC_SAMPLES = ("D03T", "D04T", "D05T")
ST_SAMPLES = {
    "D02T1": {
        "prefix": "GSM8349863_D02T1",
        "title": "Donor 02 · Stage III · Indeterminate · Core",
    },
    "D18T2": {
        "prefix": "GSM8349879_D18T2",
        "title": "Donor 18 · Stage III · Diffuse · In-between",
    },
    "D24T2": {
        "prefix": "GSM8349886_D24T2",
        "title": "Donor 24 · Stage II · Intestinal · In-between",
    },
}

PRIOR_MARKERS = {
    "Tumor epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "TFF3"],
    "T/NK": ["CD3D", "CD3E", "TRBC1", "NKG7", "GNLY"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "CD74", "CD37"],
    "Plasma cells": ["MZB1", "JCHAIN", "SDC1", "DERL3", "IGKC"],
    "Myeloid": ["LST1", "TYROBP", "FCER1G", "CTSS", "C1QA"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "COL6A1"],
    "Endothelial": ["VWF", "PECAM1", "EMCN", "KDR", "ENG"],
    "Mast cells": ["TPSAB1", "TPSB2", "KIT", "CPA3", "MS4A2"],
    "Smooth muscle": ["ACTA2", "TAGLN", "RGS5", "MCAM", "CSPG4"],
}

HELDOUT_MARKERS = {
    "Tumor epithelial": ["KRT7", "KRT17", "KRT23", "CLDN4", "CEACAM6"],
    "T/NK": ["IL7R", "CD247", "LCK", "CCL5", "TRAC"],
    "B cells": ["CD19", "CD22", "BANK1", "HLA-DQA1", "CD83"],
    "Plasma cells": ["XBP1", "SSR4", "FKBP11", "SEC11C", "TNFRSF17"],
    "Myeloid": ["AIF1", "CSF1R", "FCGR3A", "CSTA", "LILRB1"],
    "Fibroblasts": ["COL5A1", "COL5A2", "PDGFRA", "LUM", "C7"],
    "Endothelial": ["RAMP2", "PLVAP", "EPAS1", "CA4", "KLF2"],
    "Mast cells": ["HDC", "GATA2", "HPGDS", "SLC18A2", "VWA5A"],
    "Smooth muscle": ["MYL9", "CNN1", "MYH11", "CALD1", "TPM2"],
}


def first_file(directory: Path, suffix: str) -> Path:
    matches = list(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one *{suffix} in {directory}, found {matches}")
    return matches[0]


def load_sc_sample(sample: str) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    directory = RAW_SC / sample
    matrix = mmread(first_file(directory, "_matrix.mtx.gz")).tocsr().astype(np.float32)
    features = pd.read_csv(
        first_file(directory, "_features.tsv.gz"), sep="\t", header=None
    )
    genes = features.iloc[:, 1 if features.shape[1] > 1 else 0].astype(str).tolist()
    barcodes = pd.read_csv(
        first_file(directory, "_barcodes.tsv.gz"), sep="\t", header=None
    ).iloc[:, 0].astype(str)
    cells = [f"{sample}:{barcode}" for barcode in barcodes]
    return matrix, genes, cells


def collapse_duplicate_genes(
    matrix: sparse.csr_matrix, genes: list[str]
) -> tuple[sparse.csr_matrix, list[str]]:
    codes, unique = pd.factorize(pd.Index(genes), sort=False)
    if len(unique) == len(genes):
        return matrix, unique.astype(str).tolist()
    aggregator = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (codes, np.arange(len(codes)))),
        shape=(len(unique), len(codes)),
    )
    return (aggregator @ matrix).tocsr(), unique.astype(str).tolist()


def load_sc_reference() -> tuple[sparse.csr_matrix, list[str], list[str], list[str]]:
    matrices, all_cells, donors = [], [], []
    genes_ref = None
    for sample in SC_SAMPLES:
        matrix, genes, cells = load_sc_sample(sample)
        matrix, genes = collapse_duplicate_genes(matrix, genes)
        if genes_ref is None:
            genes_ref = genes
        elif genes != genes_ref:
            lookup = {gene: i for i, gene in enumerate(genes)}
            take = np.array([lookup.get(gene, -1) for gene in genes_ref])
            if (take < 0).any():
                raise ValueError(f"Gene rows differ for {sample}")
            matrix = matrix[take]
        matrices.append(matrix)
        all_cells.extend(cells)
        donors.extend([sample] * len(cells))
    combined = sparse.hstack(matrices, format="csr")
    return combined, genes_ref or [], all_cells, donors


def normalize_log(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    scaled = matrix @ sparse.diags(1e4 / np.maximum(totals, 1.0))
    scaled.data = np.log1p(scaled.data)
    return scaled.tocsr()


def module_scores(
    log_matrix: sparse.csr_matrix, genes: list[str], modules: dict[str, list[str]]
) -> pd.DataFrame:
    lookup = {gene: i for i, gene in enumerate(genes)}
    values = {}
    for label, markers in modules.items():
        rows = [lookup[gene] for gene in markers if gene in lookup]
        if len(rows) < 2:
            raise ValueError(f"Too few markers found for {label}: {rows}")
        values[label] = np.asarray(log_matrix[rows].mean(axis=0)).ravel()
    return pd.DataFrame(values)


def annotate_reference(
    counts: sparse.csr_matrix, genes: list[str], cells: list[str], donors: list[str]
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    totals = np.asarray(counts.sum(axis=0)).ravel()
    mito_rows = [i for i, gene in enumerate(genes) if gene.upper().startswith("MT-")]
    mito = (
        np.asarray(counts[mito_rows].sum(axis=0)).ravel() / np.maximum(totals, 1.0)
        if mito_rows
        else np.zeros(counts.shape[1])
    )
    keep = (detected >= 200) & (detected <= 7500) & (mito <= 0.25)
    counts = counts[:, keep]
    cells = np.asarray(cells)[keep].tolist()
    donors = np.asarray(donors)[keep].tolist()
    log = normalize_log(counts)

    scores = module_scores(log, genes, PRIOR_MARKERS)
    zscores = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
    ordered = np.sort(zscores.to_numpy(), axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    top_score = zscores.max(axis=1).to_numpy()
    labels = zscores.idxmax(axis=1).to_numpy()
    labels[(margin < 0.10) | (top_score < 0.20)] = "Unknown"

    meta = pd.DataFrame(
        {
            "cell_id": cells,
            "donor": donors,
            "cell_type": labels,
            "marker_score_margin": margin,
            "marker_score_z": top_score,
            "n_genes": detected[keep],
            "total_counts": totals[keep],
            "mitochondrial_fraction": mito[keep],
        }
    ).set_index("cell_id")
    return counts, meta


def balanced_reference(
    counts: sparse.csr_matrix, meta: pd.DataFrame, max_per_type: int = 900
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    selected = []
    for cell_type in PRIOR_MARKERS:
        idx = np.flatnonzero(meta["cell_type"].to_numpy() == cell_type)
        if not len(idx):
            continue
        selected.extend(rng.choice(idx, min(len(idx), max_per_type), replace=False))
    selected = np.array(sorted(selected), dtype=int)
    if len(selected) < 4000:
        raise RuntimeError(f"Only {len(selected)} confidently annotated cells retained")
    return counts[:, selected].tocsr(), meta.iloc[selected].copy()


def read_10x_h5(path: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        ).tocsr().astype(np.float32)
        features = group["features"]
        key = "name" if "name" in features else "id"
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in features[key][:]]
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in group["barcodes"][:]]
    return collapse_duplicate_genes(matrix, genes)[0], list(pd.Index(genes).drop_duplicates()), barcodes


def otsu_threshold(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    hist, edges = np.histogram(finite, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1)
    )[::-1]
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[np.argmax(variance)])


def histology_capacities(
    image: np.ndarray,
    coords: pd.DataFrame,
    spot_diameter: float,
    n_cells: int,
) -> tuple[pd.Series, pd.DataFrame]:
    rgb = image[..., :3].astype(np.float32)
    od = -np.log((rgb + 1.0) / 256.0)
    # A compact hematoxylin-like optical-density score.
    hematoxylin = 0.65 * od[..., 0] + 0.70 * od[..., 1] + 0.29 * od[..., 2]
    smoothed = ndimage.gaussian_filter(hematoxylin, 0.8)
    threshold = max(otsu_threshold(smoothed), float(np.quantile(smoothed, 0.70)))
    maxima = smoothed == ndimage.maximum_filter(smoothed, size=5)
    peaks = np.argwhere(maxima & (smoothed >= threshold))
    peak_y, peak_x = peaks[:, 0], peaks[:, 1]

    radius = max(3.0, 0.47 * spot_diameter)
    raw = []
    for x, y in coords[["image_x", "image_y"]].to_numpy():
        raw.append(np.sum((peak_x - x) ** 2 + (peak_y - y) ** 2 <= radius**2))
    raw = np.asarray(raw, dtype=float)
    weights = np.maximum(raw, 1.0)
    scaled = weights / weights.sum() * (n_cells - len(weights))
    capacities = np.floor(scaled).astype(int) + 1
    remainder = n_cells - int(capacities.sum())
    residual = scaled - np.floor(scaled)
    if remainder > 0:
        capacities[np.argsort(-residual)[:remainder]] += 1
    elif remainder < 0:
        eligible = np.flatnonzero(capacities > 1)
        capacities[eligible[np.argsort(residual[eligible])[: -remainder]]] -= 1
    result = pd.Series(capacities, index=coords.index, name="n_cells")
    peaks_frame = pd.DataFrame({"image_x": peak_x, "image_y": peak_y})
    if result.sum() != n_cells or (result < 1).any():
        raise RuntimeError("Histology capacity reconciliation failed")
    return result, peaks_frame


def prior_from_markers(
    st_counts: sparse.csr_matrix, genes: list[str], spots: list[str]
) -> pd.DataFrame:
    log = normalize_log(st_counts)
    scores = module_scores(log, genes, PRIOR_MARKERS)
    z = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
    values = np.exp(np.clip(z.to_numpy() / 1.5, -10, 10))
    values /= values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, index=spots, columns=scores.columns)


def save_sparse_gene_cell(
    matrix: sparse.csr_matrix, genes: list[str], cells: list[str], directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "matrix_gene_by_cell.npz", matrix)
    pd.Series(genes).to_csv(directory / "genes.tsv", sep="\t", index=False, header=False)
    pd.Series(cells).to_csv(directory / "barcodes.tsv", sep="\t", index=False, header=False)


def main() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    counts, genes, cells, donors = load_sc_reference()
    counts, meta = annotate_reference(counts, genes, cells, donors)
    counts, meta = balanced_reference(counts, meta)
    meta.to_csv(PREP / "reference_cell_metadata.csv")
    pd.DataFrame(
        meta.groupby(["donor", "cell_type"]).size().unstack(fill_value=0)
    ).to_csv(PREP / "reference_cell_counts.csv")

    heldout = {gene for markers in HELDOUT_MARKERS.values() for gene in markers}
    rng = np.random.default_rng(2026)
    for sample, info in ST_SAMPLES.items():
        directory = RAW_ST / sample
        st_counts, st_genes, spots = read_10x_h5(
            directory / f"{info['prefix']}_filtered_feature_bc_matrix.h5"
        )
        positions = pd.read_csv(
            directory / f"{info['prefix']}_tissue_positions_list.csv.gz",
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_fullres",
                "pxl_col_fullres",
            ],
        ).set_index("barcode")
        keep_spots = [
            i for i, spot in enumerate(spots)
            if spot in positions.index and int(positions.loc[spot, "in_tissue"]) == 1
        ]
        st_counts = st_counts[:, keep_spots]
        spots = [spots[i] for i in keep_spots]
        positions = positions.loc[spots]

        with gzip.open(directory / f"{info['prefix']}_scalefactors_json.json.gz", "rt") as h:
            scale = json.load(h)
        with gzip.open(directory / f"{info['prefix']}_tissue_hires_image.png.gz", "rb") as h:
            image = np.asarray(Image.open(h).convert("RGB"))
        coords = pd.DataFrame(
            {
                "array_row": positions["array_row"],
                "array_col": positions["array_col"],
                "image_x": positions["pxl_col_fullres"] * scale["tissue_hires_scalef"],
                "image_y": positions["pxl_row_fullres"] * scale["tissue_hires_scalef"],
            },
            index=spots,
        )

        sc_lookup = {gene: i for i, gene in enumerate(genes)}
        st_lookup = {gene: i for i, gene in enumerate(st_genes)}
        common = [gene for gene in genes if gene in st_lookup and gene not in heldout]
        # Select high-variance genes using the cross-donor reference only.
        sc_rows = np.array([sc_lookup[gene] for gene in common])
        sc_common = counts[sc_rows]
        mean = np.asarray(sc_common.mean(axis=1)).ravel()
        var = np.asarray(sc_common.power(2).mean(axis=1)).ravel() - mean**2
        n_genes = min(3000, len(common))
        chosen = np.argpartition(var, -n_genes)[-n_genes:]
        chosen = chosen[np.argsort(var[chosen])[::-1]]
        map_genes = [common[i] for i in chosen]
        sc_map = counts[[sc_lookup[g] for g in map_genes]]
        st_map = st_counts[[st_lookup[g] for g in map_genes]]

        # Use the same cross-donor balanced reference in each section, with a
        # deterministic order permutation to prevent ordering artefacts.
        order = rng.permutation(sc_map.shape[1])
        sc_map = sc_map[:, order]
        section_meta = meta.iloc[order].copy()

        sample_out = PREP / sample
        save_sparse_gene_cell(sc_map, map_genes, section_meta.index.tolist(), sample_out / "sc")
        save_sparse_gene_cell(st_map, map_genes, spots, sample_out / "st")
        section_meta[["cell_type", "donor"]].to_csv(sample_out / "cell_metadata.csv")
        coords.to_csv(sample_out / "coordinates.csv")
        prior = prior_from_markers(st_counts, st_genes, spots)
        prior.to_csv(sample_out / "composition_prior.csv")
        capacities, peaks = histology_capacities(
            image,
            coords,
            scale["spot_diameter_fullres"] * scale["tissue_hires_scalef"],
            sc_map.shape[1],
        )
        capacities.to_csv(sample_out / "histology_capacities.csv")
        peaks.to_csv(sample_out / "histology_nuclear_peaks.csv", index=False)
        Image.fromarray(image).save(sample_out / "histology_hires.png")
        with (sample_out / "metadata.json").open("w") as handle:
            json.dump(
                {
                    **info,
                    "n_spots": len(spots),
                    "n_reference_cells": int(sc_map.shape[1]),
                    "n_mapping_genes": len(map_genes),
                    "capacity_source": "relative H&E nuclear-peak density, reconciled to reference size",
                    "heldout_markers": HELDOUT_MARKERS,
                    "random_seed": 2026,
                },
                handle,
                indent=2,
            )
        print(
            f"{sample}: {len(spots):,} spots, {sc_map.shape[1]:,} cells, "
            f"{len(map_genes):,} mapping genes"
        )


if __name__ == "__main__":
    main()
