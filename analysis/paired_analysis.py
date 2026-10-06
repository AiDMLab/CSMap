"""Paired audits and transparent marginal-value analysis; no model tuning."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import fitz
import mob_evaluation as mob
import gastric_evaluation as gastric

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/value_ablation_20260920'
FIG=ROOT.parent/'results/paired_analysis'
OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)

def paired_frames(a,b):
    assert a.UniqueCID.is_unique and b.UniqueCID.is_unique
    a=a.set_index('UniqueCID').sort_index();b=b.set_index('UniqueCID').sort_index()
    assert a.index.equals(b.index)
    for col in ['OriginalCID','CellType']:
        assert a[col].equals(b[col]),f'Non-paired sampling: {col}'
    return a,b

def paired_ci(delta,cluster,seed=20260920):
    frame=pd.DataFrame({'delta':np.asarray(delta),'cluster':np.asarray(cluster)})
    group=frame.groupby('cluster').delta.agg(['sum','count']).to_numpy()
    rng=np.random.default_rng(seed); picks=rng.integers(len(group),size=(2000,len(group)))
    boot=group[picks,0].sum(axis=1)/group[picks,1].sum(axis=1)
    return float(frame.delta.mean()),*np.quantile(boot,[.025,.975]).tolist()

def load_mob():
    gt,coords=mob.load_truth()
    paths={n:mob.RUNS/folder/'mob_assigned_locations.csv' for n,folder in
           [('Expression only','lambda_0'),('Weak soft','lambda_1e-06'),('Hard-limit','lambda_3e00')]}
    raw={n:pd.read_csv(p) for n,p in paths.items()}
    base=raw['Expression only']; totals=[];effects=[];pertype=[];transitions=[]
    counts=gt.groupby(['true_spot','cellType']).size().unstack(fill_value=0)
    truefrac=counts.div(counts.sum(axis=1),axis=0)
    def details(frame):
        t=gt.reindex(frame.OriginalCID)
        assert not t.isna().any().any()
        correct=frame.SpotID.to_numpy()==t.true_spot.to_numpy()
        xy=coords.loc[frame.SpotID,['x','y']].to_numpy();truthxy=coords.loc[t.true_spot,['x','y']].to_numpy()
        dist=np.linalg.norm(xy-truthxy,axis=1)
        pred=frame.groupby(['SpotID','CellType']).size().unstack(fill_value=0).reindex(index=truefrac.index,columns=truefrac.columns,fill_value=0)
        pred=pred.div(pred.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
        tv=.5*(pred-truefrac).abs().sum(axis=1)
        return correct,dist,t,tv
    for name,frame in raw.items():
        a,b=paired_frames(base,frame); ca,da,truth,ta=details(a);cb,db,_,tb=details(b)
        totals.append({'variant':name,'CMR':float(cb.mean()),'mean_spot_distance':float(db.mean()),'true_composition_TV':float(tb.mean()),'n_instances':len(b),'n_unique':b.OriginalCID.nunique(),'reference_coverage':b.OriginalCID.nunique()/len(gt)})
        transitions.append({'variant':name,'corrected':int((~ca&cb).sum()),'new_errors':int((ca&~cb).sum()),'relocated':int((a.SpotID!=b.SpotID).sum())})
        for metric,delta,cluster in [('CMR',cb.astype(float)-ca,truth.true_spot),('mean_spot_distance',db-da,truth.true_spot),('true_composition_TV',tb.to_numpy()-ta.to_numpy(),ta.index)]:
            mean,lo,hi=paired_ci(delta,cluster);effects.append({'variant':name,'metric':metric,'delta':mean,'low':lo,'high':hi})
        for ct in sorted(truth.cellType.unique()):
            sel=truth.cellType.to_numpy()==ct
            mean,lo,hi=paired_ci(cb[sel].astype(float)-ca[sel],truth.true_spot.to_numpy()[sel])
            pertype.append({'variant':name,'cell_type':ct,'delta_CMR':mean,'low':lo,'high':hi,'n_instances':int(sel.sum())})
    for name,data in [('mob_summary',totals),('mob_paired_effects',effects),('mob_transitions',transitions),('mob_all_celltypes',pertype)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    # All available prespecified lambda/noise conditions retained in tables.
    old=mob.collect()
    for name in ['lambda','noise']:old[name].to_csv(OUT/('mob_'+name+'_all.csv'),index=False)
    (OUT/'mob_source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},indent=2))
    return pd.DataFrame(totals),pd.DataFrame(effects),pd.DataFrame(transitions),old

def gastric_eval():
    fullroot=ROOT/'output/original_cancer_quota_fixed_20260916'
    alias=OUT/'gastric_evaluation_inputs';allrows=[];summaries=[];audit=[]
    for sample in gastric.SAMPLES:
        status=json.loads((OUT/'runs'/sample/'status.json').read_text())
        assert status['status']=='success',status
        fullstatus=json.loads((fullroot/sample/'status.json').read_text())
        source=Path(status['input_directory'])
        comparison=Path(fullstatus.get('reused_from',str(fullroot/sample)))/'inputs'
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            other=comparison/path.relative_to(source)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==hashlib.sha256(other.read_bytes()).hexdigest(),f'Input changed: {path}'
            audit.append({'sample':sample,'input':str(path.relative_to(source)),'sha256':digest})
        excluded={g for genes in gastric.bio.FINE_HELDOUT_MARKERS.values() for g in genes}
        for mode in ['sc','st']:
            used=set((source/mode/'genes.tsv').read_text().splitlines())
            assert not used.intersection(excluded),'Evaluation markers present in mapping inputs'
        full=pd.read_csv(fullroot/sample/'mapping'/(sample.lower()+'_assigned_locations.csv'))
        zero=pd.read_csv(status['result']); a,b=paired_frames(zero,full)
        cap=pd.read_csv(fullroot/sample/'inputs/n_cells_per_spot.txt',sep='\t',index_col=0).iloc[:,0]
        for variant,frame in [('Expression only',zero),('Weak soft',full)]:
            frame=frame.copy();frame.SpotID=frame.SpotID.str.replace(r'^SPOT_','',regex=True)
            frame.CellType=frame.CellType.str.replace(r'^TYPE_','',regex=True)
            actual=frame.SpotID.value_counts().reindex(cap.index,fill_value=0)
            assert np.array_equal(actual.to_numpy(),cap.to_numpy())
            dest=alias/variant/sample;dest.mkdir(parents=True,exist_ok=True)
            frame.to_csv(dest/(sample.lower()+'_biology_assigned_locations.csv'),index=False)
            gastric.RESULTS=alias/variant
            result=gastric.sample_analysis(sample)
            table=result['correlation'].copy();table['variant']=variant;allrows.append(table)
            summaries.append({'sample':sample,'variant':variant,'mean_marker_rho':float(table.heldout_marker_rho.mean()),'n_defined_markers':int(table.heldout_marker_rho.notna().sum()),'n_instances':len(frame),'capacity_violations':0,'relocated_instances':int((a.SpotID!=b.SpotID).sum())})
    data=pd.concat(allrows,ignore_index=True)
    pairs=data.pivot(index=['sample','cell_type'],columns='variant',values='heldout_marker_rho').reset_index()
    pairs['delta_rho']=pairs['Weak soft']-pairs['Expression only']
    data.to_csv(OUT/'gastric_all_heldout_correlations.csv',index=False)
    pairs.to_csv(OUT/'gastric_paired_heldout.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'gastric_summary.csv',index=False)
    (OUT/'gastric_input_hash_audit.json').write_text(json.dumps(audit,indent=2))
    return pairs,pd.DataFrame(summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group()
    g.add_argument('--mob-only',action='store_true')
    g.add_argument('--gastric-only',action='store_true')
    args=p.parse_args()
    if not args.gastric_only:load_mob()
    if not args.mob_only:gastric_eval()
