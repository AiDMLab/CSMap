"""Generate synthetic inputs and exercise the original solver end to end."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from csmap_runner import run_csmap
from original_quota_fix import install

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/demo'));a=p.parse_args()
    out=a.out.resolve();inp=out/'inputs';inp.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(21)
    genes=[f'g{i}' for i in range(120)];cells=[f'c{i}' for i in range(24)];spots=[f's{i}' for i in range(8)]
    counts=rng.poisson(5,size=(120,24));counts[:60,:12]+=rng.poisson(14,size=(60,12));counts[60:,12:]+=rng.poisson(14,size=(60,12))
    sc=pd.DataFrame(counts,index=genes,columns=cells)
    st=pd.DataFrame(np.column_stack([counts[:,i*3:(i+1)*3].sum(axis=1) for i in range(8)]),index=genes,columns=spots)
    sc.to_csv(inp/'sc.csv');st.to_csv(inp/'st.csv')
    pd.DataFrame({'cell_type':['A']*12+['B']*12},index=cells).to_csv(inp/'types.csv')
    pd.DataFrame({'x':np.arange(8),'y':np.zeros(8)},index=spots).to_csv(inp/'coordinates.csv')
    pd.DataFrame({'A':[1]*4+[0]*4,'B':[0]*4+[1]*4},index=spots).to_csv(inp/'fractions.tsv',sep='\t')
    pd.DataFrame({'A':[.5],'B':[.5]},index=['Fraction']).to_csv(inp/'weights.tsv',sep='\t')
    pd.DataFrame({'n_cells':[3]*8},index=spots).to_csv(inp/'capacities.tsv',sep='\t')
    install()
    result=run_csmap(out/'mapping',output_prefix='demo_',lambda_1=1e-6,mean_cell_numbers=3,seed=1,plot_off=True,scRNA_path=inp/'sc.csv',st_path=inp/'st.csv',cell_type_path=inp/'types.csv',coordinates_path=inp/'coordinates.csv',cellfracs_path=inp/'fractions.tsv',weights_path=inp/'weights.tsv',n_cells_per_spot_path=inp/'capacities.tsv')
    df=pd.read_csv(result)
    assert len(df)==24 and df.UniqueCID.is_unique
    assert df.SpotID.value_counts().eq(3).all()
    assert df.OriginalCID.nunique()==24
    print('PASS: 24 assignments; 8 spots with exactly 3 cells; unique reference cells retained.')
if __name__=='__main__':main()
