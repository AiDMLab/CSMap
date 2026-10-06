#!/usr/bin/env python3
"""Prepare a story-focused gastric immune-niche validation for CSMap.

LAR/TLS genes and all evaluation markers are excluded from both mapping and
prior construction.  The single-cell reference is refined to distinguish
activated CD8 T cells and LAMP3 dendritic cells before spatial mapping.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

import prepare_gastric_validation as base


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gastric_cancer"
PREP = DATA / "prepared_biology"

FINE_PRIOR_MARKERS = {
    "Tumor epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "TFF3"],
    "Activated CD8 T": ["CD8A", "CD8B", "CD27", "PDCD1", "GZMK"],
    "Other T/NK": ["CD3D", "CD3E", "IL7R", "NKG7", "GNLY"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "CD74", "CD37"],
    "Plasma cells": ["MZB1", "JCHAIN", "SDC1", "DERL3", "IGKC"],
    "LAMP3 DC": ["LAMP3", "CD70", "CCR7", "FSCN1", "CD83"],
    "Other myeloid": ["LST1", "TYROBP", "FCER1G", "C1QA", "FCN1"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "COL6A1"],
    "Endothelial": ["VWF", "PECAM1", "EMCN", "KDR", "ENG"],
    "Mast cells": ["TPSAB1", "TPSB2", "KIT", "CPA3", "MS4A2"],
    "Smooth muscle": ["ACTA2", "TAGLN", "RGS5", "MCAM", "CSPG4"],
}

FINE_HELDOUT_MARKERS = {
    "Tumor epithelial": ["KRT7", "KRT17", "KRT23", "CLDN4", "CEACAM6"],
    "Activated CD8 T": ["CCL5", "LAG3", "TIGIT", "TOX", "PRF1"],
    "Other T/NK": ["TRAC", "LCK", "LTB", "CST7", "CD247"],
    "B cells": ["CD19", "CD22", "BANK1", "HLA-DQA1", "CD40"],
    "Plasma cells": ["XBP1", "SSR4", "FKBP11", "SEC11C", "TNFRSF17"],
    "LAMP3 DC": ["CCL17", "CCL22", "IDO1", "CD274", "MARCKSL1"],
    "Other myeloid": ["AIF1", "CSF1R", "FCGR3A", "CSTA", "LILRB1"],
    "Fibroblasts": ["COL5A1", "COL5A2", "PDGFRA", "LUM", "C7"],
    "Endothelial": ["RAMP2", "PLVAP", "EPAS1", "CA4", "KLF2"],
    "Mast cells": ["HDC", "GATA2", "HPGDS", "SLC18A2", "VWA5A"],
    "Smooth muscle": ["MYL9", "CNN1", "MYH11", "CALD1", "TPM2"],
}

LAR_MARKERS = ["CCL19", "FDCSP", "CXCL13", "CCL21"]
FINE_TYPES = list(FINE_PRIOR_MARKERS)


def qc_reference():
    counts, genes, cells, donors = base.load_sc_reference()
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    totals = np.asarray(counts.sum(axis=0)).ravel()
    mito_rows = [i for i, gene in enumerate(genes) if gene.upper().startswith("MT-")]
    mito = np.asarray(counts[mito_rows].sum(axis=0)).ravel() / np.maximum(totals, 1)
    keep = (detected >= 200) & (detected <= 7500) & (mito <= 0.25)
    return (
        counts[:, keep].tocsr(),
        genes,
        np.asarray(cells)[keep],
        np.asarray(donors)[keep],
        detected[keep],
        totals[keep],
        mito[keep],
    )


def initial_broad_labels(log: sparse.csr_matrix, genes: list[str]) -> np.ndarray:
    scores = base.module_scores(log, genes, base.PRIOR_MARKERS)
    z = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
    ordered = np.sort(z.to_numpy(), axis=1)
    labels = z.idxmax(axis=1).to_numpy()
    ambiguous = ((ordered[:, -1] - ordered[:, -2]) < 0.10) | (
        z.max(axis=1).to_numpy() < 0.20
    )
    labels[ambiguous] = "Unknown"
    return labels


def refine_pair(
    counts: sparse.csr_matrix,
    log: sparse.csr_matrix,
    genes: list[str],
    broad_labels: np.ndarray,
    broad_type: str,
    target: str,
    other: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(broad_labels == broad_type)
    modules = {target: FINE_PRIOR_MARKERS[target], other: FINE_PRIOR_MARKERS[other]}
    scores = base.module_scores(log[:, idx], genes, modules)
    z = (scores - scores.mean()) / scores.std(ddof=0).replace(0, 1)
    predicted = z.idxmax(axis=1).to_numpy()
    margin = np.abs(z.iloc[:, 0].to_numpy() - z.iloc[:, 1].to_numpy())
    lookup = {gene: i for i, gene in enumerate(genes)}
    marker_rows = [lookup[gene] for gene in modules[target] if gene in lookup]
    detected = np.asarray((counts[marker_rows][:, idx] > 0).sum(axis=0)).ravel()
    strict_target = (predicted == target) & (margin >= 0.20) & (detected >= 2)
    refined = np.full(len(idx), other, dtype=object)
    refined[strict_target] = target
    return idx, refined, margin


def annotate_reference():
    counts, genes, cells, donors, detected, totals, mito = qc_reference()
    log = base.normalize_log(counts)
    labels = initial_broad_labels(log, genes)
    margins = np.full(len(labels), np.nan)
    for broad_type, target, other in [
        ("T/NK", "Activated CD8 T", "Other T/NK"),
        ("Myeloid", "LAMP3 DC", "Other myeloid"),
    ]:
        idx, refined, margin = refine_pair(
            counts, log, genes, labels, broad_type, target, other
        )
        labels[idx] = refined
        margins[idx] = margin
    meta = pd.DataFrame(
        {
            "cell_id": cells,
            "donor": donors,
            "cell_type": labels,
            "fine_marker_margin": margins,
            "n_genes": detected,
            "total_counts": totals,
            "mitochondrial_fraction": mito,
        }
    ).set_index("cell_id")
    return counts, genes, meta


def balanced_reference(
    counts: sparse.csr_matrix, meta: pd.DataFrame, maximum: int = 800
):
    rng = np.random.default_rng(2027)
    selected = []
    for cell_type in FINE_TYPES:
        idx = np.flatnonzero(meta["cell_type"].to_numpy() == cell_type)
        if not len(idx):
            raise RuntimeError(f"No reference cells for {cell_type}")
        selected.extend(rng.choice(idx, min(len(idx), maximum), replace=False))
    selected = np.asarray(sorted(selected), dtype=int)
    return counts[:, selected].tocsr(), meta.iloc[selected].copy()


def main() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    counts, genes, meta = annotate_reference()
    counts, meta = balanced_reference(counts, meta)
    meta.to_csv(PREP / "reference_cell_metadata.csv")
    (
        meta.groupby(["donor", "cell_type"]).size().unstack(fill_value=0)
        .reindex(columns=FINE_TYPES, fill_value=0)
        .to_csv(PREP / "reference_cell_counts.csv")
    )

    excluded = set(LAR_MARKERS)
    excluded.update(
        gene for markers in FINE_HELDOUT_MARKERS.values() for gene in markers
    )
    overlap = excluded.intersection(
        gene for markers in FINE_PRIOR_MARKERS.values() for gene in markers
    )
    if overlap:
        raise RuntimeError(f"Prior/evaluation marker leakage: {sorted(overlap)}")

    rng = np.random.default_rng(2027)
    for sample, info in base.ST_SAMPLES.items():
        raw = base.RAW_ST / sample
        st_counts, st_genes, spots = base.read_10x_h5(
            raw / f"{info['prefix']}_filtered_feature_bc_matrix.h5"
        )
        positions = pd.read_csv(
            raw / f"{info['prefix']}_tissue_positions_list.csv.gz",
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
        keep = [
            i
            for i, spot in enumerate(spots)
            if spot in positions.index and int(positions.loc[spot, "in_tissue"]) == 1
        ]
        st_counts = st_counts[:, keep]
        spots = [spots[i] for i in keep]
        positions = positions.loc[spots]
        with gzip.open(raw / f"{info['prefix']}_scalefactors_json.json.gz", "rt") as h:
            scale = json.load(h)
        with gzip.open(raw / f"{info['prefix']}_tissue_hires_image.png.gz", "rb") as h:
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
        common = [
            gene for gene in genes if gene in st_lookup and gene not in excluded
        ]
        sc_common = counts[[sc_lookup[gene] for gene in common]]
        mean = np.asarray(sc_common.mean(axis=1)).ravel()
        variance = np.asarray(sc_common.power(2).mean(axis=1)).ravel() - mean**2
        n_genes = min(3000, len(common))
        chosen = np.argpartition(variance, -n_genes)[-n_genes:]
        chosen = chosen[np.argsort(variance[chosen])[::-1]]
        map_genes = [common[i] for i in chosen]
        sc_map = counts[[sc_lookup[gene] for gene in map_genes]]
        st_map = st_counts[[st_lookup[gene] for gene in map_genes]]
        order = rng.permutation(sc_map.shape[1])
        sc_map = sc_map[:, order]
        section_meta = meta.iloc[order].copy()

        out = PREP / sample
        base.save_sparse_gene_cell(
            sc_map, map_genes, section_meta.index.tolist(), out / "sc"
        )
        base.save_sparse_gene_cell(st_map, map_genes, spots, out / "st")
        section_meta[["cell_type", "donor"]].to_csv(out / "cell_metadata.csv")
        coords.to_csv(out / "coordinates.csv")
        log_st = base.normalize_log(st_counts)
        prior_scores = base.module_scores(log_st, st_genes, FINE_PRIOR_MARKERS)
        z = (prior_scores - prior_scores.mean()) / prior_scores.std(ddof=0).replace(0, 1)
        prior_values = np.exp(np.clip(z.to_numpy() / 1.5, -10, 10))
        prior_values /= prior_values.sum(axis=1, keepdims=True)
        pd.DataFrame(
            prior_values, index=spots, columns=prior_scores.columns
        ).to_csv(out / "composition_prior.csv")
        capacities, peaks = base.histology_capacities(
            image,
            coords,
            scale["spot_diameter_fullres"] * scale["tissue_hires_scalef"],
            sc_map.shape[1],
        )
        capacities.to_csv(out / "histology_capacities.csv")
        peaks.to_csv(out / "histology_nuclear_peaks.csv", index=False)
        Image.fromarray(image).save(out / "histology_hires.png")
        with (out / "metadata.json").open("w") as handle:
            json.dump(
                {
                    **info,
                    "n_spots": len(spots),
                    "n_reference_cells": int(sc_map.shape[1]),
                    "n_mapping_genes": len(map_genes),
                    "fine_types": FINE_TYPES,
                    "lar_markers_held_out": LAR_MARKERS,
                    "evaluation_markers_held_out": FINE_HELDOUT_MARKERS,
                    "random_seed": 2027,
                },
                handle,
                indent=2,
            )
        print(
            f"{sample}: {len(spots):,} spots, {sc_map.shape[1]:,} cells, "
            f"{len(map_genes):,} genes"
        )


if __name__ == "__main__":
    main()
