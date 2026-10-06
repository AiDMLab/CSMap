"""Held-out expression and spatial summaries used in the manuscript."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import seaborn as sns
import prepare_gastric_validation as broad
import prepare_gastric_biology as bio
ROOT = Path(__file__).resolve().parent
PREP = ROOT / 'data' / 'gastric_cancer' / 'prepared_biology'
RESULTS = ROOT / 'output' / 'gastric_cancer' / 'evaluation_inputs'
OUT = ROOT.parent / 'results' / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = ('D02T1', 'D18T2', 'D24T2')
EXEMPLAR = 'D18T2'
COLORS = {'Tumor epithelial': '#D55E00', 'Activated CD8 T': '#0072B2', 'Other T/NK': '#56B4E9', 'B cells': '#009E73', 'Plasma cells': '#CC79A7', 'LAMP3 DC': '#E69F00', 'Other myeloid': '#B07AA1', 'Fibroblasts': '#6B8E23', 'Endothelial': '#5E3C99', 'Mast cells': '#A6761D', 'Smooth muscle': '#666666'}
LAR_DISPLAY_TYPES = ['B cells', 'Activated CD8 T', 'Other T/NK', 'LAMP3 DC', 'Fibroblasts', 'Tumor epithelial', 'Plasma cells']

def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result

def load_original_st(sample: str):
    info = broad.ST_SAMPLES[sample]
    raw = broad.RAW_ST / sample
    matrix, genes, spots = broad.read_10x_h5(raw / f"{info['prefix']}_filtered_feature_bc_matrix.h5")
    positions = pd.read_csv(raw / f"{info['prefix']}_tissue_positions_list.csv.gz", header=None, names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_fullres', 'pxl_col_fullres']).set_index('barcode')
    keep = [i for i, spot in enumerate(spots) if spot in positions.index and int(positions.loc[spot, 'in_tissue']) == 1]
    return (matrix[:, keep], genes, [spots[i] for i in keep])

def spatial_graph(coords: pd.DataFrame, k: int=6):
    points = coords[['image_x', 'image_y']].to_numpy(float)
    _, neighbors = cKDTree(points).query(points, k=min(k + 1, len(points)))
    rows = np.repeat(np.arange(len(points)), neighbors.shape[1] - 1)
    cols = neighbors[:, 1:].ravel()
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(points)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    with_self = graph.copy()
    with_self.setdiag(1)
    row_sum = np.asarray(with_self.sum(axis=1)).ravel()
    smoother = sparse.diags(1.0 / np.maximum(row_sum, 1)) @ with_self
    return (graph, smoother)

def sample_analysis(sample: str) -> dict:
    coords = pd.read_csv(PREP / sample / 'coordinates.csv', index_col=0)
    coords.index = coords.index.astype(str)
    assigned = pd.read_csv(RESULTS / sample / f'{sample.lower()}_biology_assigned_locations.csv')
    counts = assigned.groupby(['SpotID', 'CellType']).size().unstack(fill_value=0).reindex(index=coords.index, columns=bio.FINE_TYPES, fill_value=0).astype(float)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    matrix, genes, spots = load_original_st(sample)
    if spots != coords.index.tolist():
        raise ValueError(f'Spot order differs for {sample}')
    log_st = broad.normalize_log(matrix)
    lar_score = broad.module_scores(log_st, genes, {'LAR score': bio.LAR_MARKERS})['LAR score'].to_numpy(float)
    heldout_scores = broad.module_scores(log_st, genes, bio.FINE_HELDOUT_MARKERS)
    heldout_scores.index = spots
    graph, smoother = spatial_graph(coords)
    smooth_lar = np.asarray(smoother @ lar_score).ravel()
    smooth_fraction = pd.DataFrame(np.asarray(smoother @ fractions.to_numpy(dtype=np.float64)), index=spots, columns=fractions.columns)
    threshold = float(np.quantile(smooth_lar, 0.9))
    hotspot = smooth_lar >= threshold
    lymphoid_types = ['Activated CD8 T', 'Other T/NK', 'B cells', 'Plasma cells']
    lymphoid_niche = smooth_fraction[lymphoid_types].sum(axis=1).to_numpy()
    rng = np.random.default_rng(2029 + SAMPLES.index(sample))
    points = coords[['image_x', 'image_y']].to_numpy(float)
    tree = cKDTree(points)
    centers = rng.integers(0, len(points), size=1999)
    window_indices = [np.atleast_1d(tree.query(points[center], k=int(hotspot.sum()))[1]) for center in centers]
    enrichment_rows = []
    for cell_type in bio.FINE_TYPES:
        values = smooth_fraction[cell_type].to_numpy()
        observed = float(values[hotspot].mean() - values[~hotspot].mean())
        log2_fold = float(np.log2((values[hotspot].mean() + 0.0001) / (values[~hotspot].mean() + 0.0001)))
        null = np.empty(len(window_indices))
        for iteration, indices in enumerate(window_indices):
            window = np.zeros(len(values), dtype=bool)
            window[indices] = True
            null[iteration] = values[window].mean() - values[~window].mean()
        p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
        enrichment_rows.append({'sample': sample, 'cell_type': cell_type, 'lar_hotspot_mean': float(values[hotspot].mean()), 'non_lar_mean': float(values[~hotspot].mean()), 'difference': observed, 'log2_enrichment': log2_fold, 'permutation_p': p_value, 'null_model': 'random contiguous spatial window with matched spot count'})
    correlation_rows = []
    for cell_type in bio.FINE_TYPES:
        rho, p_value = spearmanr(fractions[cell_type], heldout_scores[cell_type])
        correlation_rows.append({'sample': sample, 'cell_type': cell_type, 'heldout_marker_rho': rho, 'p_value': p_value, 'markers': ';'.join(bio.FINE_HELDOUT_MARKERS[cell_type])})
    deciles = pd.qcut(smooth_lar, 10, labels=False, duplicates='drop')
    decile_frame = pd.DataFrame({'sample': sample, 'LAR_decile': deciles + 1, 'B cells': smooth_fraction['B cells'].to_numpy(), 'Activated CD8 T': smooth_fraction['Activated CD8 T'].to_numpy(), 'Tumor epithelial': smooth_fraction['Tumor epithelial'].to_numpy(), 'Lymphoid niche': lymphoid_niche}).groupby(['sample', 'LAR_decile'], as_index=False).mean()
    spot_frame = pd.DataFrame({'sample': sample, 'spot_id': spots, 'LAR_score_raw': lar_score, 'LAR_score_smoothed': smooth_lar, 'LAR_hotspot': hotspot, 'lymphoid_niche_fraction': lymphoid_niche})
    rho, p_value = spearmanr(smooth_lar, lymphoid_niche)
    niche_summary = {'sample': sample, 'LAR_lymphoid_rho': float(rho), 'LAR_lymphoid_p': float(p_value), 'n_spots': len(spots), 'n_hotspots': int(hotspot.sum())}
    return {'coords': coords, 'assigned': assigned, 'counts': counts, 'fractions': fractions, 'smooth_fraction': smooth_fraction, 'lar_score': lar_score, 'smooth_lar': smooth_lar, 'hotspot': hotspot, 'lymphoid_niche': lymphoid_niche, 'enrichment': pd.DataFrame(enrichment_rows), 'correlation': pd.DataFrame(correlation_rows), 'deciles': decile_frame, 'spots': spot_frame, 'summary': niche_summary}
