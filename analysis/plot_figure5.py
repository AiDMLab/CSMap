"""Selected final Figure 5 (A-E, H-I), expanded from the approved v20 builder."""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
import seaborn as sns
import fitz
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/figure5'
OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
samples=['D02T1','D18T2','D24T2']
types=['Tumor epithelial','Activated CD8 T','Other T/NK','B cells','Plasma cells','LAMP3 DC','Other myeloid','Fibroblasts','Endothelial','Mast cells','Smooth muscle']
lymphoid=['Activated CD8 T','Other T/NK','B cells','Plasma cells']
paired=pd.read_csv(DATA/'marker_correlations_all33.csv')
sections={}
for sample in ['D02T1','D18T2']:
    a=np.load(DATA/(sample+'.npz'),allow_pickle=False)
    d={key:a[key] for key in a.files}
    for key in ['raw','smooth']:
        d[key]=pd.DataFrame(d[key],index=d['spot_ids'],columns=types)
    d['img']=np.asarray(Image.open(DATA/(sample+'_histology.png')).convert('RGB'))
    sections[sample]=d
expr=np.load(DATA/'D02T1_ecm_expression.npy',allow_pickle=False)
plt.rcParams.update({'font.family':'Arial','font.size':16,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(16,17.2))
grid=fig.add_gridspec(4,6,left=.07,right=.98,bottom=.045,top=.96,wspace=.9,hspace=.40,height_ratios=[1.16,1.30,1.16,1.12])
gs={(0,0):grid[0,:4],(0,2):grid[0,4:],(1,0):grid[2,:4],(1,2):grid[2,4:],
    (0,1):grid[1,:],(2,1):grid[3,:3],(2,2):grid[3,3:]}

def heading(spec,letter,title):
 p=spec.get_position(fig);fig.text(p.x0-.022,p.y1+.024,letter,fontsize=22,fontweight='bold');fig.text(p.x0+.003,p.y1+.024,title,fontsize=16.5,fontweight='bold')
def spatial(ax,d,values=None,cmap='viridis',vmax=1,mask=None):
 if values is None:ax.imshow(d['img'])
 xy=d['xy'] if mask is None else d['xy'][mask]
 pc=PatchCollection([RegularPolygon(tuple(p),6,radius=d['pitch']/np.sqrt(3)) for p in xy],edgecolor='none',antialiased=False,facecolor='#00B99D' if values is None else None,cmap=cmap,rasterized=True)
 if values is not None:pc.set_array(np.asarray(values));pc.set_clim(0,vmax)
 ax.add_collection(pc);H,W=d['img'].shape[:2];ax.set_xlim(-.5,W-.5);ax.set_ylim(H-.5,-.5);ax.set_aspect('equal');ax.axis('off')
 if values is not None:
  # Bounds contain every mapped hexagon, independent of type or signal.
  lo=d['xy'].min(axis=0)-2*d['pitch'];hi=d['xy'].max(axis=0)+2*d['pitch']
  ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
 return pc
def pack_maps(spec,axes,gap=.14):
 # Work in physical inches. Equal height, native aspect, fixed narrow gap.
 box=spec.get_position(fig);fw,fh=fig.get_size_inches()
 ratios=[abs((a.get_xlim()[1]-a.get_xlim()[0])/(a.get_ylim()[1]-a.get_ylim()[0])) for a in axes]
 available_w=box.width*fw;available_h=box.height*fh
 height=min(available_h,(available_w-gap*(len(axes)-1))/sum(ratios))
 widths=[height*r for r in ratios];total=sum(widths)+gap*(len(axes)-1)
 left=box.x0+(available_w-total)/(2*fw);bottom=box.y0+(available_h-height)/(2*fh)
 for a,width in zip(axes,widths):
  a.set_position([left,bottom,width/fw,height/fh]);left+=(width+gap)/fw

def trio(spec,sample,region):
 d=sections[sample];sg=spec.subgridspec(1,3,wspace=.04)
 vals=[None,d['lar'] if region=='LAR' else d['ecm'],d['lymph'] if region=='LAR' else d['smooth'].Fibroblasts.to_numpy()]
 map_axes=[]
 titles=['H&E + region','Held-out '+region+' score','Mapped '+('lymphoid' if region=='LAR' else 'fibroblast')]
 for j in range(3):
  ax=fig.add_subplot(sg[j]);map_axes.append(ax);p=spatial(ax,d,vals[j],cmap='magma' if j==1 else 'viridis',vmax=float(vals[j].max()) if j==1 else 1,mask=(d['larhot'] if region=='LAR' else d['fibhot']) if j==0 else None);ax.set_title(titles[j],fontsize=16.5,pad=7)
  if j:
   ca=ax.inset_axes([.10,-.10,.8,.025]);fig.colorbar(p,cax=ca,orientation='horizontal',format='%.1f' if j==2 else '%.2f',ticks=[0,.5,1] if j==2 else [0,float(vals[j].max())/2,float(vals[j].max())]).ax.tick_params(labelsize=15,length=2)
 pack_maps(spec,map_axes,gap=.12)
heading(gs[0,0],'A','D18T2 | Lymphoid region');trio(gs[0,0],'D18T2','LAR')
def facets(spec,sample,labels):
 map_axes=[]
 sg=spec.subgridspec(1,4,wspace=.10)
 for j,t in enumerate(labels):
  ax=fig.add_subplot(sg[j]);map_axes.append(ax);p=spatial(ax,sections[sample],sections[sample]['raw'][t].to_numpy())
  ax.set_title(t,fontsize=16.5,pad=3)
 pack_maps(spec,map_axes,gap=.12)
 pos=spec.get_position(fig);ca=fig.add_axes([pos.x0+pos.width*.34,pos.y0-.006,pos.width*.32,.004])
 cb=fig.colorbar(p,cax=ca,orientation='horizontal',ticks=[0,.5,1]);cb.ax.tick_params(labelsize=15,length=2)
 cb.set_label('',fontsize=15,labelpad=1)
heading(gs[0,1],'C','D18T2 | Cell-type fractions')
facets(gs[0,1],'D18T2',['B cells','Activated CD8 T','LAMP3 DC','Tumor epithelial'])
heading(gs[0,2],'B','LAR-associated composition')
ax=fig.add_subplot(gs[0,2]);d=sections['D18T2'];q=pd.qcut(d['lar'],10,labels=False,duplicates='drop')
for typ,col in [('B cells','#009E73'),('Activated CD8 T','#0072B2'),('Tumor epithelial','#D55E00')]:
 v=d['raw'][typ].groupby(q).mean();ax.plot(v.index+1,v.values,'o-',ms=3,label=typ,color=col)
ax.set(xlabel='LAR-score quantile group',ylabel='Mean raw fraction');ax.legend(fontsize=15,frameon=False)
heading(gs[1,0],'D','D02T1 | Stromal region');trio(gs[1,0],'D02T1','ECM')
heading(gs[1,2],'E','ECM expression trend')
ax=fig.add_subplot(gs[1,2]);d=sections['D02T1'];q=pd.qcut(d['smooth'].Fibroblasts,10,labels=False,duplicates='drop')
for g,y,col in zip(['COL5A1','COL5A2','LUM'],expr,['#0072B2','#E69F00','#009E73']):
 zz=(y-y.mean())/y.std();v=pd.Series(zz).groupby(np.asarray(q)).mean();ax.plot(v.index+1,v,'o-',ms=3,label=g,color=col)
ax.set(xlabel='Fibroblast quantile group',ylabel='Mean raw gene score (SD)');ax.legend(fontsize=15,frameon=False)
heading(gs[2,1],'H','Held-out marker changes')
ax=fig.add_subplot(gs[2,1]);pos=ax.get_position();ax.set_position([pos.x0+.12,pos.y0,pos.width-.095,pos.height]);t=paired.pivot(index='cell_type',columns='sample',values='delta_rho').reindex(index=types,columns=['D02T1','D18T2']);lim=np.abs(t.to_numpy()).max()
sns.heatmap(t,ax=ax,yticklabels=True,cmap='vlag',vmin=-lim,vmax=lim,annot=True,fmt='+.3f',annot_kws={'fontsize':15},cbar=False,linewidths=.4);ax.set(xlabel='',ylabel='');ax.tick_params(labelsize=15,rotation=0)
pos=gs[2,2].get_position(fig)
fig.text(.56,pos.y1+.024,'I',fontsize=22,fontweight='bold')
fig.text(.585,pos.y1+.024,'Mean consistency',fontsize=16.5,fontweight='bold')
ax=fig.add_axes([.655,pos.y0,.325,pos.height])
for row,sample in enumerate(['D02T1','D18T2']):
 vals=paired.loc[paired['sample']==sample,['Expression only','Weak soft']].mean().to_numpy()
 y=1-row
 ax.plot(vals,[y,y],color='#9DA9B2',lw=2.8,zorder=2)
 ax.scatter(vals[0],y,s=78,color='#8C949C',edgecolors='white',linewidths=.8,zorder=3,label='No penalty' if row==0 else None)
 ax.scatter(vals[1],y,s=78,color='#0072B2',edgecolors='white',linewidths=.8,zorder=4,label='Original' if row==0 else None)
 ax.text(.175,y,f'{vals[1]-vals[0]:+.3f}',ha='right',va='center',fontsize=16,color='#0072B2')
ax.set_xlim(0,.18);ax.set_ylim(-.65,1.8)
ax.set_yticks([1,0],['D02T1','D18T2']);ax.tick_params(axis='y',length=0,pad=8,labelsize=15)
ax.set_xticks([0,.05,.10,.15],['0.00','0.05','0.10','0.15']);ax.tick_params(axis='x',labelsize=15)
ax.set_xlabel('Mean marker Spearman rho',fontsize=15,labelpad=8)
ax.spines[['left','top','right']].set_visible(False)
ax.legend(loc='upper left',bbox_to_anchor=(-.04,1.04),ncol=2,frameon=False,fontsize=14,handletextpad=.3,columnspacing=.8)
ax.text(.175,1.43,'Change',ha='right',fontsize=14,color='#526977')
for ax in fig.axes:
 if ax.axison:ax.spines[['top','right']].set_visible(False)
for ax in fig.axes:
 if ax.axison:
  if 'quantile' in ax.get_xlabel():ax.set_xticks([2,4,6,8,10] if ax.get_xlim()[1]>9.5 else [2,4,6,8])
  if 'Correlation change' in ax.get_xlabel():ax.set_xticks([-.02,0,.04,.08])
fig.text(.07,.009,'Selected sections: D02T1 and D18T2; full three-section results retained separately.',fontsize=15,color='#526977')
fig.canvas.draw()
stem=OUT/'Figure5_Gastric_dumbbell'
master=OUT/'_master.pdf'
fig.savefig(master,dpi=450,facecolor='white')
fig.savefig(stem.with_suffix('.svg'),dpi=450,facecolor='white')
plt.close(fig)

# Preserve vector elements while setting the actual paper dimensions.
from PIL import Image
width_mm=180.;height_mm=193.5
with fitz.open(master) as src:
 dst=fitz.open();page=dst.new_page(width=width_mm/25.4*72,height=height_mm/25.4*72)
 page.show_pdf_page(page.rect,src,0)
 dst.save(stem.with_suffix('.pdf'),garbage=4,deflate=True);dst.close()
master.unlink()
svg=stem.with_suffix('.svg');s=svg.read_text();s=re.sub(r'(<svg\b[^>]*\bwidth=")[^"]+',r'\g<1>180mm',s,count=1);s=re.sub(r'(<svg\b[^>]*\bheight=")[^"]+',r'\g<1>193.5mm',s,count=1);svg.write_text(s)
with fitz.open(stem.with_suffix('.pdf')) as doc:
 page=doc[0];pix=page.get_pixmap(dpi=600,alpha=False)
 im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
 im.save(stem.with_suffix('.png'),dpi=(600,600))
 im.save(stem.with_suffix('.tif'),dpi=(600,600),compression='tiff_lzw')
 page.get_pixmap(dpi=180).save(OUT/'Figure5_review.png')
 spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
 sizes=[s['size'] for s in spans]
 font_audit={'min_font_pt':min(sizes),'max_font_pt':max(sizes),'fonts':page.get_fonts()}



print(OUT)
