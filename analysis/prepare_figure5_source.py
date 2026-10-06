"""Three-section Figure 5: audited coordinates, explicitly labelled display smoothing."""
from pathlib import Path
import json, gzip, hashlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import fitz

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'results/figure5_source'
OUT.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected={}
sources={}
def tracked(p):
    p=Path(p); sources[str(p)]=sha(p); return p
def read(p,**kw):return pd.read_csv(tracked(p),**kw)
samples=['D02T1','D18T2','D24T2']
types=['Tumor epithelial','Activated CD8 T','Other T/NK','B cells','Plasma cells',
       'LAMP3 DC','Other myeloid','Fibroblasts','Endothelial','Mast cells','Smooth muscle']
lymphoid=['Activated CD8 T','Other T/NK','B cells','Plasma cells']
sections={};audits=[]
for sample in samples:
    prep=ROOT/'data/gastric_cancer/prepared_biology'/sample
    coords=read(prep/'coordinates.csv',index_col=0)
    run=ROOT/'output/original_cancer_quota_fixed_20260916'/sample
    status=json.loads(tracked(run/'status.json').read_text());assert status['status']=='success'
    mapping=read(status['result'])
    mapping.SpotID=mapping.SpotID.str.replace(r'^SPOT_','',regex=True)
    mapping.CellType=mapping.CellType.str.replace(r'^TYPE_','',regex=True)
    assert mapping.CellType.isin(types).all() and set(mapping.SpotID)<=set(coords.index)
    cap=read(run/'inputs/n_cells_per_spot.txt',sep='\t',index_col=0).iloc[:,0]
    assert set(cap.index)<=set(coords.index)
    assert mapping.SpotID.value_counts().reindex(cap.index,fill_value=0).eq(cap).all()
    raw=ROOT/'data/gastric_cancer/raw_st'/sample
    pp=next(raw.glob('*_tissue_positions_list.csv.gz'))
    positions=read(pp,header=None,names=['barcode','in_tissue','row','col','y','x'],index_col=0)
    with gzip.open(tracked(next(raw.glob('*_scalefactors_json.json.gz'))),'rt') as f:scale=json.load(f)
    img=np.asarray(Image.open(tracked(prep/'histology_hires.png')).convert('RGB'))
    with gzip.open(tracked(next(raw.glob('*_tissue_hires_image.png.gz'))),'rb') as f:
        source_img=np.asarray(Image.open(f).convert('RGB'))
    assert np.array_equal(img,source_img)
    expected=positions.loc[coords.index,['x','y']].to_numpy()*scale['tissue_hires_scalef']
    xy=coords[['image_x','image_y']].to_numpy()
    error=float(np.abs(xy-expected).max());assert error<1e-8
    assert positions.loc[coords.index,'in_tissue'].eq(1).all()
    radius=scale['spot_diameter_fullres']*scale['tissue_hires_scalef']/2
    H,W=img.shape[:2]
    assert ((xy[:,0]>=radius)&(xy[:,0]<W-radius)&(xy[:,1]>=radius)&(xy[:,1]<H-radius)).all()
    counts=pd.crosstab(mapping.SpotID,mapping.CellType).reindex(index=coords.index,columns=types,fill_value=0)
    valid=counts.sum(axis=1)>0
    fractions=counts.loc[valid].div(counts.loc[valid].sum(axis=1),axis=0)
    xy=xy[valid]
    # One display-only pass. Average spot fractions, not pooled counts.
    dist,idx=cKDTree(xy).query(xy,k=min(7,len(xy)))
    cutoff=1.25*float(np.median(dist[:,1]))
    weights=(dist<=cutoff).astype(float);weights/=weights.sum(axis=1,keepdims=True)
    smooth=np.einsum('ij,ijk->ik',weights,fractions.to_numpy()[idx])
    smooth=pd.DataFrame(smooth,index=fractions.index,columns=types)
    assert np.allclose(smooth.sum(axis=1),1) and (smooth.to_numpy()>=0).all()
    audits.append(dict(sample=sample,effective_spots=int(valid.sum()),zero_assignment_spots=int((~valid).sum()),
        assigned_instances=len(mapping),coordinate_max_error_pixels=error,image_identical=True,
        full_image_width=W,full_image_height=H,spot_diameter_pixels=2*radius,
        smoothing_distance_limit_pixels=cutoff,neighbours_excluding_self_min=int((weights>0).sum(axis=1).min()-1),
        neighbours_excluding_self_max=int((weights>0).sum(axis=1).max()-1)))
    sections[sample]=dict(img=img,xy=xy,radius=radius,raw=fractions,smooth=smooth)


# Reconstruct the selected spatial panels from mappings and held-out expression.
import prepare_gastric_validation as broad
import prepare_gastric_biology as bio
from gastric_evaluation import spatial_graph
for sample,d in sections.items():
    raw=ROOT/'data/gastric_cancer/raw_st'/sample
    matrix,genes,barcodes=broad.read_10x_h5(next(raw.glob('*_filtered_feature_bc_matrix.h5')))
    log=broad.normalize_log(matrix)
    ix=pd.Index(barcodes).get_indexer(d['raw'].index)
    assert (ix>=0).all()
    lar=broad.module_scores(log[:,ix],genes,{'LAR score':bio.LAR_MARKERS})['LAR score'].to_numpy(float)
    coords=pd.DataFrame(d['xy'],index=d['raw'].index,columns=['image_x','image_y'])
    graph,smoother=spatial_graph(coords)
    smooth_lar=np.asarray(smoother@lar).ravel()
    smooth_fraction=np.asarray(smoother@d['raw'].to_numpy())
    lymph=smooth_fraction[:,[types.index(t) for t in lymphoid]].sum(axis=1)
    pitch=float(np.median(cKDTree(d['xy']).query(d['xy'],k=2)[0][:,1]))
    d.update(lar=smooth_lar,larhot=smooth_lar>=np.quantile(smooth_lar,.9),lymph=lymph,pitch=pitch)
    if sample=='D02T1':
        expr=log[[genes.index(g) for g in ['COL5A1','COL5A2','LUM']]][:,ix].toarray()
        score=expr.mean(axis=0)
        dist,nn=cKDTree(d['xy']).query(d['xy'],k=7)
        w=(dist<=1.25*pitch).astype(float);w/=w.sum(axis=1,keepdims=True)
        d['ecm']=(score[nn]*w).sum(axis=1)
        d['fibhot']=d['smooth'].Fibroblasts.ge(d['smooth'].Fibroblasts.quantile(.9)).to_numpy()
        np.save(OUT/'D02T1_ecm_expression.npy',expr)
for sample in ['D02T1','D18T2']:
    d=sections[sample]
    arrays={k:(v.to_numpy() if isinstance(v,pd.DataFrame) else v) for k,v in d.items() if k!='img'}
    arrays['spot_ids']=d['raw'].index.to_numpy(dtype=str)
    np.savez_compressed(OUT/(sample+'.npz'),**arrays)
    Image.fromarray(d['img']).save(OUT/(sample+'_histology.png'))
(OUT/'coordinate_audit.json').write_text(json.dumps(audits,indent=2))
print(OUT)
