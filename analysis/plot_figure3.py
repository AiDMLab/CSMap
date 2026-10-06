"""Plot-only Figure 3 refresh from the audited historical mappings."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
import figure3_style as s

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'results/figures'
OUT.mkdir(parents=True,exist_ok=True)
STEM = 'Figure3_Matched_source_spatial_maps'
MOB = dict(zip(s.MOB_COLORS, ['#2864DC','#FF8A00','#E63956','#00B8AA','#66B82B','#AB45CE']))
CERE = dict(zip(s.CERE_COLORS, ['#0057B8','#FF8A00','#DD365D','#00B8AA','#87B82F',
                              '#279FE8','#963FC4','#EF74B5','#A45A2A','#FFD02F','#183660']))


def main(preview=False, final_compact=False):
    preview = final_compact = True
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {}
    audit_path = ROOT.parent/'data/figure3/mapping_audit.json'
    audit = json.loads(audit_path.read_text())
    provenance = {'algorithm_rerun':False, 'sources':{str(audit_path):digest(audit_path)},
                  'palette_MOB':MOB, 'palette_cerebellum':CERE,
                  'display_jitter':'Unchanged seeds and spacing * 0.43; display only', 'panels':[]}
    def read(p, **kw):
        provenance['sources'][str(p)] = digest(p)
        return pd.read_csv(p, **kw)
    s.configure_style()
    fig = plt.figure(figsize=(10.8,8.5) if preview else (9,9.7), facecolor='white', dpi=180 if preview else 600)
    left, gap, width = .095,.025,.275
    height = .157
    if preview:
        left,gap,width=.035,.015,.178
        for col,m in enumerate(s.METHODS):
            fig.text(left+col*(width+gap)+width/2,.98,m,ha='center',va='top',fontsize=10.5,fontweight='bold',color='#253646')
        for title,ty in [('MOB',.937),('Cerebellum (n = 5)',.630),('Cerebellum (n = 15)',.335)]:
            fig.text(left,ty,title,ha='left',va='center',fontsize=10,fontweight='bold',color='#253646')
            fig.add_artist(plt.Line2D([.26,.985],[ty,ty],transform=fig.transFigure,color='#DCE3E8',lw=.7))
    else:
        for col,title in enumerate(['MOB','Cerebellum (n = 5)','Cerebellum (n = 15)']):
            fig.text(left+col*(width+gap)+width/2,.987,title,ha='center',va='top',fontsize=11,fontweight='bold',color='#253646')
        for row,m in enumerate(s.METHODS):
            fig.text(.031,.795-row*.17+height/2,m,rotation=90,ha='center',va='center',fontsize=10.5,fontweight='bold',color='#253646')
    for r,n in enumerate((0,5,15)):
        name = 'MOB' if n == 0 else f'Cerebellum (n = {n})'
        if n == 0:
            gt = read(ROOT.parent/'data/figure3/mob_ground_truth.csv').rename(columns={'true_spot':'spot'})
            coords = read(ROOT.parent/'data/figure3/mob_spot_coords.csv',index_col=0)
            coords.index = coords.index.astype(str)
            gt.spot = gt.spot.astype(str)
            gt[['x','y']] = coords.loc[gt.spot,['x','y']].to_numpy()
        else:
            gt = read(ROOT.parent/f'data/figure3/cere_n{n}_ground_truth.csv').rename(columns={'cell_id_new':'cell_id','CellType':'cellType'})
            gt.spot = gt.spot.astype(str)
            coords = gt.groupby('spot')[['x','y']].mean()
        gt.cell_id = gt.cell_id.astype(str)
        truth = gt.set_index('cell_id')
        colors = MOB if n == 0 else CERE
        dataset = {'spot_coords':coords}
        xlim,ylim,spacing = s.panel_limits(dataset)
        for c,m in enumerate(s.METHODS):
            y = .795-c*.17
            if preview:
                y,height=[(.708,.20),(.372,.24),(.077,.24)][r]
            if m == 'Ground Truth':
                points = gt[['cell_id','x','y','cellType']].copy()
            else:
                entry = next(a for a in audit if a['dataset']==name and a['method']==m)
                raw = read(ROOT.parent/entry['source'])
                ids = raw.OriginalCID.astype(str)
                spots = raw.SpotID.astype(str)
                missing = raw.SpotID.isna()
                if missing.any():
                    assert not coords.duplicated(['x','y']).any()
                    lookup = {tuple(v):k for k,v in zip(coords.index,coords[['x','y']].to_numpy())}
                    spots.loc[missing] = [lookup[tuple(v)] for v in raw.loc[missing,['x','y']].to_numpy()]
                assert len(raw)==entry['instances'] and ids.nunique()==entry['unique_cells']
                points = pd.DataFrame({'cell_id':ids.to_numpy(),'x':coords.loc[spots,'x'].to_numpy(),
                    'y':coords.loc[spots,'y'].to_numpy(),'cellType':truth.loc[ids,'cellType'].to_numpy()})
            assert points.cellType.isin(colors).all()
            panel_x=left+(c if preview else r)*(width+gap)
            letter=chr(65+r*5+c) if preview else chr(65+c*3+r)
            ax = fig.add_axes([panel_x,y,width,height])
            ax.scatter(coords.x,coords.y,s=.28,c='#E6EBF0',linewidths=0,rasterized=True,zorder=0)
            jx,jy = s.jitter_cells(points,spacing,s.JITTER_SEEDS[m])
            order = np.random.default_rng(s.JITTER_SEEDS[m]+101).permutation(len(points))
            ax.scatter(jx[order],jy[order],s=3.0 if n==0 else .8,
                       c=points.cellType.map(colors).to_numpy()[order],alpha=1,
                       linewidths=0,rasterized=True,zorder=2)
            ax.set(xlim=xlim,ylim=ylim,aspect='equal')
            ax.axis('off')
            fig.text(panel_x-.008,y+height-.008,letter,fontsize=10,fontweight='bold',color='#253646')
            provenance['panels'].append({'panel':letter,'dataset':name,'method':m,
                'instances':len(points),'point_size':3.0 if n==0 else .8})
    legends = [(MOB,.51,None,6,.69),(CERE,.51,None,6,.065)] if preview else [
        (MOB,left+width/2,'MOB cell types',2,.098),
        (CERE,left+width+gap+(2*width+gap)/2,'Cerebellum cell types',4,.098)]
    for colors,x,title,ncol,legend_y in legends:
        handles=[plt.Line2D([],[],marker='o',linestyle='',color=color,markersize=5,label=label) for label,color in colors.items()]
        fig.legend(handles=handles,title=title,loc='upper center',bbox_to_anchor=(x,legend_y),ncol=ncol,
                   frameon=False,fontsize=8.5 if final_compact else 7.8,title_fontsize=8.5,handletextpad=.35,columnspacing=1.05 if final_compact else .85,
                   labelspacing=.5,borderaxespad=0)
    if final_compact:
        dest=OUT
        dest.mkdir(exist_ok=True)
        stem='Figure3_Spatial_reconstruction_compact_refined'
        for ext in ('.pdf','.svg','.png'):
            fig.savefig(dest/(stem+ext),dpi=600 if ext!='.png' else 400,facecolor='white')
        plt.close(fig)
        with fitz.open(dest/(stem+'.pdf')) as pdf:
            assert len(pdf)==1
            pdf[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(dest/'Figure3_PDF_review.png')
        assert all(digest(p)==h for p,h in before.items())
        provenance.update({'layout':'3 rows x 5 columns', 'size_inches':[10.8,8.5],
                           'existing_figures_unchanged':True,'protected_sha256':{str(p):h for p,h in before.items()}})
        (dest/'Figure3_provenance.json').write_text(json.dumps(provenance,indent=2))
        print(dest/(stem+'.pdf'))
        return

if __name__=='__main__':
    main(final_compact=True)
