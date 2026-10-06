"""Build revised SCI Figure 4 from the expanded MOB constraint ablation."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metrics import compute_metrics, load_assigned_csv
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'analysis'
RUNS = WORK / 'output' / 'figure4_improved' / 'runs' / 'mob'
PRIORS = WORK / 'output' / 'figure4_improved' / 'priors'
OUT = ROOT / 'results' / 'evaluation'
OUT.mkdir(parents=True, exist_ok=True)
LAMBDA_VALUES = [0.0, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 3.0]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
CELL_TYPES = ['Astrocytes', 'Ependymal', 'Immune', 'Neurons', 'Oligos', 'Vascular']
COLORS = {'Expression only': '#667784', 'Weak soft': '#0072B2', 'Hard-limit': '#D55E00'}

def lambda_tag(value: float) -> str:
    return '0' if value == 0 else f'{value:.0e}'.replace('+', '')

def load_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(WORK / 'data' / 'mob_ground_truth.csv', index_col=0)
    gt.index = gt.index.astype(str)
    gt['true_spot'] = gt['true_spot'].astype(str)
    gt['cellType'] = gt['cellType'].astype(str)
    coords = pd.read_csv(WORK / 'data' / 'mob_spot_coords.csv', index_col=0)
    coords.index = coords.index.astype(str)
    return (gt, coords)

def load_run(folder: str, gt: pd.DataFrame, coords: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    path = RUNS / folder / 'mob_assigned_locations.csv'
    assigned = load_assigned_csv(path, coords)
    metrics = compute_metrics(gt, assigned, coords, total_input_cells=len(gt))
    pred = assigned.reset_index(drop=False).rename(columns={assigned.index.name or 'index': 'cell_id'})
    pred['cell_id'] = pred['cell_id'].astype(str)
    truth = gt.reset_index(drop=False).rename(columns={gt.index.name or 'index': 'cell_id'})
    truth['cell_id'] = truth['cell_id'].astype(str)
    merged = pred.merge(truth[['cell_id', 'true_spot', 'cellType']], on='cell_id', suffixes=('_pred', '_true'))
    merged['correct'] = merged['pred_spot'].astype(str) == merged['true_spot'].astype(str)
    return (assigned, metrics, merged)

def bootstrap_cmr(merged: pd.DataFrame, seed: int=42, n_boot: int=2000) -> tuple[float, float]:
    grouped = merged.groupby('true_spot')['correct'].agg(['sum', 'count'])
    values = grouped[['sum', 'count']].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sampled = values[rng.integers(0, len(values), len(values))]
        estimates[i] = sampled[:, 0].sum() / sampled[:, 1].sum()
    return tuple(np.quantile(estimates, [0.025, 0.975]))

def composition_tv(assigned: pd.DataFrame, reference: pd.DataFrame) -> float:
    pred = assigned.reset_index(drop=False)
    type_col = 'CellType' if 'CellType' in pred.columns else 'cellType'
    counts = pred.groupby(['pred_spot', type_col]).size().unstack(fill_value=0)
    counts.index = counts.index.astype(str)
    pred_frac = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    cols = sorted(set(reference.columns.astype(str)) | set(pred_frac.columns.astype(str)))
    ref = reference.reindex(columns=cols, fill_value=0.0).astype(float)
    pred_frac = pred_frac.reindex(index=ref.index.astype(str), columns=cols, fill_value=0.0)
    return float((0.5 * (pred_frac - ref).abs().sum(axis=1)).mean())

def collect() -> dict[str, pd.DataFrame]:
    gt, coords = load_truth()
    truth_frac = pd.read_csv(PRIORS / 'mob' / 'Seurat_cellfracs.txt', sep='\t', index_col=0)
    truth_frac.index = truth_frac.index.astype(str)
    lambda_rows = []
    run_cache = {}
    for penalty in LAMBDA_VALUES:
        folder = f'lambda_{lambda_tag(penalty)}'
        assigned, metrics, merged = load_run(folder, gt, coords)
        run_cache[folder] = (assigned, metrics, merged)
        lambda_rows.append({'lambda': penalty, **{k: metrics[k] for k in ['CMR', 'WCMR', 'CU', 'OP_Score']}, 'true_composition_TV': composition_tv(assigned, truth_frac)})
    lambda_df = pd.DataFrame(lambda_rows)
    component_specs = [('Expression only', 'lambda_0'), ('Weak soft', 'lambda_1e-06'), ('Hard-limit', 'lambda_3e00'), ('Shuffled hard', 'noise_1_lambda_3e00')]
    component_rows = []
    for label, folder in component_specs:
        assigned, metrics, merged = run_cache.get(folder, (None, None, None))
        if assigned is None:
            assigned, metrics, merged = load_run(folder, gt, coords)
        low, high = bootstrap_cmr(merged)
        component_rows.append({'variant': label, 'CMR': metrics['CMR'], 'ci_low': low, 'ci_high': high})
    component_df = pd.DataFrame(component_rows)
    per_type_rows = []
    for label, folder in [('Weak soft', 'lambda_1e-06'), ('Hard-limit', 'lambda_3e00')]:
        merged = run_cache[folder][2]
        rates = merged.groupby('cellType_true')['correct'].mean()
        base = run_cache['lambda_0'][2].groupby('cellType_true')['correct'].mean()
        for ct in CELL_TYPES:
            per_type_rows.append({'variant': label, 'cell_type': ct, 'delta_CMR': rates.get(ct, np.nan) - base.get(ct, np.nan)})
    per_type_df = pd.DataFrame(per_type_rows)
    spot_counts = gt.groupby(['true_spot', 'cellType']).size().unstack(fill_value=0)
    spot_frac = spot_counts.div(spot_counts.sum(axis=1), axis=0)
    entropy = -(spot_frac.where(spot_frac > 0) * np.log(spot_frac.where(spot_frac > 0))).sum(axis=1)
    entropy = entropy / np.log(spot_frac.shape[1])
    quartile = pd.qcut(entropy.rank(method='first'), 4, labels=['Q1 low', 'Q2', 'Q3', 'Q4 high'])
    complexity_rows = []
    for label, folder in [('Expression only', 'lambda_0'), ('Weak soft', 'lambda_1e-06'), ('Hard-limit', 'lambda_3e00')]:
        merged = run_cache[folder][2].copy()
        merged['complexity'] = merged['true_spot'].map(quartile)
        for q, group in merged.groupby('complexity', observed=False):
            complexity_rows.append({'variant': label, 'complexity': str(q), 'CMR': group['correct'].mean(), 'n': len(group)})
    complexity_df = pd.DataFrame(complexity_rows)
    noise_rows = []
    for penalty, label in [(1e-06, 'Weak soft'), (3.0, 'Hard-limit')]:
        for level in NOISE_LEVELS:
            folder = f'noise_{level:g}_lambda_{lambda_tag(penalty)}'
            assigned, metrics, merged = load_run(folder, gt, coords)
            noisy_prior = pd.read_csv(PRIORS / 'mob_noise' / f'noise_{level:g}' / 'Seurat_cellfracs.txt', sep='\t', index_col=0)
            noisy_prior.index = noisy_prior.index.astype(str)
            noise_rows.append({'variant': label, 'noise': level, 'CMR': metrics['CMR'], 'WCMR': metrics['WCMR'], 'OP_Score': metrics['OP_Score'], 'true_composition_TV': composition_tv(assigned, truth_frac), 'prior_adherence_TV': composition_tv(assigned, noisy_prior)})
    noise_df = pd.DataFrame(noise_rows)
    return {'lambda': lambda_df, 'components': component_df, 'per_type': per_type_df, 'complexity': complexity_df, 'noise': noise_df}
