"""Run the original dense model and paired lambda-zero gastric controls."""
from pathlib import Path
import argparse,json,contextlib,shutil
import numpy as np
import pandas as pd
from scipy import sparse,io
from csmap_runner import run_csmap
from original_quota_fix import install,REPAIRS
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/original_cancer_quota_fixed_20260916'
def prepare(sample):
    target = OUT / sample / 'inputs'
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/gastric_cancer/prepared_biology'/sample
    for mode in ['sc','st']:
        dest = target/mode
        dest.mkdir(exist_ok=True)
        io.mmwrite(str(dest/'matrix.mtx'), sparse.load_npz(source/mode/'matrix_gene_by_cell.npz'))
        for file in ['genes.tsv','barcodes.tsv']:
            shutil.copy2(source/mode/file,dest/file)
    sc,st = target/'sc/matrix.mtx',target/'st/matrix.mtx'
    types,coords = target/'cell_types.txt',target/'coordinates.txt'
    pd.read_csv(source/'cell_metadata.csv',index_col=0)[['cell_type']].to_csv(types,sep='\t')
    pd.read_csv(source/'coordinates.csv',index_col=0)[['image_x','image_y']].rename(columns={'image_x':'rol','image_y':'col'}).to_csv(coords,sep='\t')
    prior = pd.read_csv(source/'composition_prior.csv',index_col=0)
    caps = pd.read_csv(source/'histology_capacities.csv',index_col=0).iloc[:,0]
    # Original API expects one global fraction row and a separate spot fraction file.
    # This is an input conversion, not a patch to the original integer rounding.
    caps = caps.reindex(prior.index)
    assert not caps.isna().any()
    prior.to_csv(target/'Seurat_cellfracs.txt',sep='\t')
    glob = prior.mul(caps,axis=0).sum(axis=0)
    (glob/glob.sum()).to_frame().T.to_csv(target/'Seurat_weights.txt',sep='\t')
    caps.to_frame('n_cells').to_csv(target/'n_cells_per_spot.txt',sep='\t')
    return dict(scRNA_path=sc,st_path=st,cell_type_path=types,coordinates_path=coords,
                cellfracs_path=target/'Seurat_cellfracs.txt',weights_path=target/'Seurat_weights.txt',
                n_cells_per_spot_path=target/'n_cells_per_spot.txt'), int(caps.sum())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--samples',nargs='+',choices=['D02T1','D18T2','D24T2'],default=['D02T1','D18T2','D24T2'])
    args=p.parse_args();install()
    for sample in args.samples:
        kwargs,n=prepare(sample)
        for penalty,dest in [(None,OUT/sample),(0.,ROOT/'output/value_ablation_20260920/runs'/sample)]:
            dest.mkdir(parents=True,exist_ok=True);REPAIRS.clear()
            with (dest/'run.log').open('w') as log,contextlib.redirect_stdout(log):
                result=run_csmap(dest/'mapping',output_prefix=sample.lower()+'_',lambda_1=penalty,mean_cell_numbers=2,seed=2026,plot_off=True,geometry='honeycomb',**kwargs)
            status={'sample':sample,'status':'success','lambda':penalty,'lambda_1':penalty,'seed':2026,'result':str(result.resolve()),'input_directory':str((OUT/sample/'inputs').resolve()),'quota_repairs':REPAIRS.copy()}
            (dest/'status.json').write_text(json.dumps(status,indent=2))
            print(sample,penalty,result)
