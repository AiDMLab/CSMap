"""Command-line entry point for the manuscript's original dense CSMap."""
from pathlib import Path
import argparse
from csmap_runner import run_csmap
from original_quota_fix import install

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sc','st','cell-types','coordinates','fractions','weights','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--capacities',type=Path)
    p.add_argument('--penalty',type=float,default=None,help='Omit for the original adaptive rule; 0 disables the mismatch penalty.')
    p.add_argument('--mean-cells',type=int,default=10)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--prefix',default='csmap_')
    p.add_argument('--geometry',choices=['square','honeycomb'],default='square')
    p.add_argument('--quota-guard',action='store_true',help='Enable the negative-quota guard used for gastric data.')
    a=p.parse_args()
    if a.quota_guard:install()
    print(run_csmap(a.out.resolve(),output_prefix=a.prefix,lambda_1=a.penalty,mean_cell_numbers=a.mean_cells,seed=a.seed,geometry=a.geometry,plot_off=True,scRNA_path=a.sc,st_path=a.st,cell_type_path=a.cell_types,coordinates_path=a.coordinates,cellfracs_path=a.fractions,weights_path=a.weights,n_cells_per_spot_path=a.capacities))
if __name__=='__main__':main()
