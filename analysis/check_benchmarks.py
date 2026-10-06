from pathlib import Path
import json
import numpy as np
import pandas as pd
from metrics import compute_metrics
ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'data/figure3/mapping_audit.json').read_text())
for n in [0,5,15]:
    if n==0:
        gt=pd.read_csv(ROOT/'data/figure3/mob_ground_truth.csv',index_col=0)
        coords=pd.read_csv(ROOT/'data/figure3/mob_spot_coords.csv',index_col=0)
    else:
        gt=pd.read_csv(ROOT/f'data/figure3/cere_n{n}_ground_truth.csv').rename(columns={'cell_id_new':'cell_id','CellType':'cellType','spot':'true_spot'}).set_index('cell_id')
        coords=gt.groupby('true_spot')[['x','y']].mean()
    gt.index=gt.index.astype(str);gt.true_spot=gt.true_spot.astype(str);coords.index=coords.index.astype(str)
    saved=pd.read_csv(ROOT/f'data/figure2/benchmark_n{n}_metrics.csv',index_col=0)
    label='MOB' if n==0 else f'Cerebellum (n = {n})'
    for row in [r for r in audit if r['dataset']==label]:
        raw=pd.read_csv(ROOT/row['source']);ids=raw.OriginalCID.astype(str);spots=raw.SpotID.astype(str)
        missing=raw.SpotID.isna()
        if missing.any():
            lookup={tuple(v):k for k,v in zip(coords.index,coords[['x','y']].to_numpy())}
            spots.loc[missing]=[lookup[tuple(v)] for v in raw.loc[missing,['x','y']].to_numpy()]
        xy=coords.loc[spots,['x','y']].to_numpy()
        pred=pd.DataFrame({'pred_spot':spots.to_numpy(),'x':xy[:,0],'y':xy[:,1],'cellType':gt.loc[ids,'cellType'].to_numpy()},index=ids)
        m=compute_metrics(gt,pred,coords,len(gt))
        for key in ['CMR','WCMR','WRR','CU','OP_Score']:
            np.testing.assert_allclose(m[key],saved.loc[row['method'],key],rtol=0,atol=1e-12)
        print(label,row['method'],'PASS')
print('All 60 benchmark scores match. Historical SMLE is preserved separately, not recomputed here.')
