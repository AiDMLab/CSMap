"""Replot the final Figure 2 from frozen manuscript source tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure2_style as q
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
def draw(loaded):
    with plt.rc_context():
        q.configure_style()
        fig = plt.figure(figsize=(7.6, 5.5), facecolor='white')
        grid = fig.add_gridspec(2, 3, left=.095, right=.985, bottom=.10,
                               top=.825, hspace=.46, wspace=.34)
        for i, (spec, metrics, smle) in enumerate(loaded):
            q.plot_summary(fig.add_subplot(grid[0, i]), metrics, spec, chr(65+i), i == 0)
            ax = fig.add_subplot(grid[1, i])
            q.plot_smle(ax, smle, chr(68+i))
            ax.set_ylabel('SMLE (log scale; lower is better)', fontsize=7)
        handles = [plt.Line2D([], [], marker=q.MARKERS[m], color='none',
                   markerfacecolor=q.COLORS[m], markeredgecolor='white',
                   markeredgewidth=.6, markersize=6, label=m) for m in q.ALGORITHMS]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .955),
                   ncol=4, frameon=False, columnspacing=1.7, handletextpad=.45)
        save(fig, 'Figure2_Matched_source_benchmarks')



if __name__=='__main__':
    loaded=[]
    for n,spec in zip([0,5,15],q.DATASETS):
        m=pd.read_csv(ROOT/f'data/figure2/benchmark_n{n}_metrics.csv',index_col=0)
        e=pd.read_csv(ROOT/f'data/figure2/historical_n{n}_smle.csv')
        e['log1p_SMLE']=np.log10(1+e.SMLE)
        loaded.append((spec,m,e))
    draw(loaded)
    print(OUT)
