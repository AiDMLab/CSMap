"""Twelve-panel evidence overview, using saved results without rerunning models."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

BASE=Path(__file__).resolve().parents[1]/'data/figure4'
OUT=BASE.parents[1]/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
selected=refined=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected={}
sources={}
def read(name):
    p=BASE/name;sources[str(p)]=sha(p);return pd.read_csv(p)
grid=read('mob_lambda_all.csv').sort_values('lambda')
eff=read('mob_paired_effects.csv');noise=read('mob_noise_all.csv')
summary=read('mob_summary.csv');types=read('mob_all_celltypes.csv')
trans=read('mob_transitions.csv');rob=read('robustness.csv')
BLUE='#087FC1';ORANGE='#EE8737';INK='#263B4A'
variants=[('Weak soft','Weak',BLUE),('Hard-limit','Large',ORANGE)]
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(3,4,figsize=(14,10.8))
fig.subplots_adjust(left=.065,right=.95,bottom=.085,top=.865,wspace=.66,hspace=.78)
titles=['Penalty-strength response','Composition-localization trade-off','Paired localization effect','Paired composition effect',
        'Prior-corruption limits','Cell-type heterogeneity','Corrected and introduced errors','Reference-sampling stability',
        'Spot-capacity sensitivity','Gene-pool sensitivity','Runtime vs assigned instances','Gene-pool resource demand']
for i,(ax,t) in enumerate(zip(axs.flat,titles)):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E5EBF0',lw=.6)
    ax.set_title(t,loc='left',pad=14,fontweight='bold')
    ax.text(-.22,1.08,chr(65+i),transform=ax.transAxes,fontsize=12,fontweight='bold',color=INK)
ax=axs[0,0]
ax.plot(grid['lambda'],grid.CMR,'o-',color=BLUE,ms=4)
ax.set_xscale('symlog',linthresh=1e-8)
ax.set_xticks([0,1e-6,1e-3,3],['0',r'$10^{-6}$',r'$10^{-3}$','3'])
ax.set(xlabel=r'Penalty strength $\lambda$ (symlog)',ylabel='CMR (higher is better)')
ax=axs[0,1];ax.plot(grid.true_composition_TV,grid.CMR,'o-',color='#95A3AD',ms=3)
for v,label,c,marker in [(0,'No penalty','#536574','s'),(1e-6,'Weak',BLUE,'o'),(3,'Large',ORANGE,'D')]:
    row=grid[np.isclose(grid['lambda'],v,rtol=1e-8,atol=0)].iloc[0]
    axs[0,0].scatter(v,row.CMR,c=c,marker=marker,s=32,zorder=3)
    ax.scatter(row.true_composition_TV,row.CMR,c=c,marker=marker,s=32,label=label,zorder=3)
ax.legend(frameon=False,fontsize=7,loc='lower right')
ax.set(xlabel='Composition TV (lower is better)',ylabel='CMR (higher is better)')
for ax,metric,scale,xlabel in [(axs[0,2],'CMR',100,'Change in CMR (pp)'),(axs[0,3],'true_composition_TV',1,'Change in composition TV')]:
    for y,(key,label,c) in enumerate(variants):
        row=eff.query('variant==@key and metric==@metric').iloc[0]
        ax.errorbar(row.delta*scale,y,xerr=[[scale*(row.delta-row.low)],[scale*(row.high-row.delta)]],fmt='o',capsize=3,color=c)
    ax.set_yticks([0,1],['Weak','Large']);ax.set_ylim(1.65,-.65);ax.axvline(0,color='gray',ls='--',lw=.8)
    ax.set_xlabel(xlabel);ax.grid(False)
    ax.text(.02,.02,'95% paired bootstrap intervals',transform=ax.transAxes,fontsize=6.5,color='#536574')
ax=axs[1,0]
for key,label,c in variants:
    sub=noise.query('variant==@key').sort_values('noise')
    ax.plot(sub.noise*100,sub.true_composition_TV,'o-',color=c,label=label,ms=4)
ax.axhline(summary.query('variant=="Expression only"').iloc[0].true_composition_TV,color='gray',ls='--',label='No penalty')
ax.set(xlabel='Prior corruption (%)',ylabel='Composition TV');ax.legend(frameon=False,fontsize=7)
ax=axs[1,1];sub=types.query('variant=="Weak soft"')
ax.errorbar(sub.delta_CMR*100,range(len(sub)),xerr=[100*(sub.delta_CMR-sub.low),100*(sub.high-sub.delta_CMR)],fmt='o',capsize=3,color=BLUE)
ax.set_yticks(range(len(sub)),sub.cell_type,fontsize=7);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Change in CMR (pp)')
ax=axs[1,2];sub=trans.set_index('variant').loc[['Weak soft','Hard-limit']];x=np.arange(2)
for shift,col,label,c in [(-.18,'corrected','Corrected','#00A58A'),(.18,'new_errors','Introduced','#DD6266')]:
    bars=ax.bar(x+shift,sub[col],.36,label=label,color=c)
    ax.bar_label(bars,fontsize=7,padding=3)
ax.set_xticks(x,['Weak','Large']);ax.set(ylabel='Mapped cell instances',ylim=(0,155));ax.legend(frameon=False,fontsize=7)
metrics=[('CMR','CMR',BLUE,'o'),('WCMR','WCMR','#00A58A','s'),('CU','CU','#E59D00','D'),('OP_Score','OP-Score','#9467BD','^')]
for ax,family in [(axs[1,3],'seed'),(axs[2,0],'capacity'),(axs[2,1],'genes')]:
    sub=rob.query('family==@family').sort_values('value');x=np.arange(len(sub))
    for col,label,c,m in metrics:ax.plot(x,sub[col],color=c,marker=m,label=label,ms=3.5,lw=1.1)
    labels=['500','1k','2k','5k','18,263'] if family=='genes' else [str(int(v)) for v in sub.value]
    ax.set_xticks(x,labels);ax.set_ylim(0,1.05);ax.set_ylabel('Score')
    ax.set_xlabel({'seed':'Reference-sampling seed','capacity':'Assumed mean cells per spot','genes':'Input shared-gene pool'}[family])
    ax.legend(frameon=False,ncol=2,fontsize=6.5,loc='lower right')
    if family=='seed':ax.text(.03,.45,f'CMR: {sub.CMR.min():.4f}-{sub.CMR.max():.4f}\nFive seeds; one dataset',transform=ax.transAxes,fontsize=7,color='#536574')
ax=axs[2,2];sub=rob.query('family=="capacity"').sort_values('n_assignments')
ax.plot(sub.n_assignments,sub.runtime_s,'o-',color=BLUE,ms=4)
ax.set(xlabel='Assigned cell instances',ylabel='Wall-clock time (s)')
ax=axs[2,3];sub=rob.query('family=="genes"').sort_values('value');x=np.arange(len(sub))
ax.bar(x,sub.peak_rss_mb/1024,color='#9CB6C9',width=.6)
ax.set_xticks(x,['500','1k','2k','5k','18,263']);ax.set(xlabel='Input shared-gene pool',ylabel='Peak RSS (GiB)')
twin=ax.twinx();twin.plot(x,sub.runtime_s,'o-',color=ORANGE,ms=4)
twin.set_ylabel('Time (s)',color=ORANGE);twin.tick_params(axis='y',labelcolor=ORANGE);twin.spines['top'].set_visible(False)
fig.text(.065,.97,'CSMap: mechanism, operating limits and computational requirements',fontsize=16,fontweight='bold',color=INK)
fig.text(.065,.935,'Full evidence layout draft | Original-model MOB experiments | Gastric analysis presented separately',fontsize=9,color='#536574')
for y,label in [(.90,'1  PARAMETER EFFECTS AND PAIRED BENEFITS'),(.602,'2  LIMITS, ERROR TRANSITIONS AND STABILITY'),(.305,'3  INPUT SENSITIVITY AND RESOURCE REQUIREMENTS')]:
    fig.text(.065,y,label,fontsize=9,fontweight='bold',color='#37667E')
fig.text(.065,.025,'Ground-truth-derived MOB priors; weak penalty has no net CMR gain in the paired comparison.\n'
         'All conditions retained within each displayed analysis. Resource timings are single measurements, not a cross-method speed comparison.',fontsize=8,color='#536574',linespacing=1.5)
stem='Figure4_Full_evidence_draft'
if selected:
    chosen=[0,1,2,3,4,7,8,9]
    keep=[list(axs.flat)[i] for i in chosen]
    for ax in list(fig.axes):
        if ax not in keep:fig.delaxes(ax)
    for txt in list(fig.texts):txt.remove()
    fig.set_size_inches(13.5,7.4)
    layout=fig.add_gridspec(2,4,left=.065,right=.98,bottom=.12,top=.85,wspace=.66,hspace=.69)
    for i,ax in enumerate(keep):
        ax.set_subplotspec(layout[i//4,i%4])
        old=chr(65+chosen[i])
        for txt in ax.texts:
            if txt.get_text()==old:
                txt.set_text(chr(65+i));break
    fig.text(.065,.97,'CSMap: penalty effects, operating limits and input sensitivity',fontsize=15,fontweight='bold',color=INK)
    fig.text(.065,.925,'1  PARAMETER EFFECTS AND PAIRED COMPARISONS',fontsize=9,fontweight='bold',color='#37667E')
    fig.text(.065,.49,'2  PRIOR QUALITY, SAMPLING AND INPUT CONDITIONS',fontsize=9,fontweight='bold',color='#37667E')
    fig.text(.065,.027,'Original-model MOB experiments; ground-truth-derived composition priors. Weak penalty has no net CMR gain.\n'
             'Paired intervals and five sampling seeds quantify within-dataset variability, not independent-sample generalization.',
             fontsize=8,color='#536574',linespacing=1.5)
    stem='Figure4_Selected8_penalty_sensitivity_draft'
if refined:
    for txt in list(fig.texts):txt.remove()
    fig.set_size_inches(10.4,6.3)
    layout=fig.add_gridspec(2,4,left=.075,right=.982,bottom=.145,top=.895,wspace=.62,hspace=.64)
    short_titles=['Penalty response','Composition trade-off','Localization effect','Composition effect',
                  'Prior-corruption limits','Sampling stability','Spot-capacity sensitivity','Gene-pool sensitivity']
    for i,ax in enumerate(keep):
        ax.set_subplotspec(layout[i//4,i%4])
        ax.set_title(short_titles[i],loc='left',fontsize=9,fontweight='bold',pad=16,color=INK)
        for txt in list(ax.texts):
            if txt.get_text()==chr(65+i):
                txt.set_position((-.22,1.12));txt.set_fontsize(12)
            elif 'bootstrap' in txt.get_text():txt.remove()
        ax.tick_params(labelsize=7.5,length=3,width=.7)
        ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
        ax.spines['left'].set_linewidth(.7);ax.spines['bottom'].set_linewidth(.7)
        for line in ax.lines:
            if line.get_linestyle()=='-':line.set_linewidth(1.25)
    keep[0].set_xlabel(r'Penalty $\lambda$ (symlog)')
    keep[1].set_xlabel('Composition error (TV)')
    keep[0].set_ylabel('CMR');keep[1].set_ylabel('CMR')
    keep[2].set_xlabel('Change in CMR (pp)')
    keep[3].set_xlabel('Change in composition TV')
    keep[4].set_ylabel('Composition error (TV)')
    keep[5].set_xlabel('Sampling seed')
    keep[6].set_xlabel('Assumed cells per spot')
    keep[7].set_xlabel('Input shared-gene pool')
    for ax in keep[5:]:ax.get_legend().remove()
    handles,labels=keep[5].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.64,.022),ncol=4,
               frameon=False,fontsize=8,columnspacing=1.6,handlelength=1.5,handletextpad=.5)
    stem='Figure4_CSMap_penalty_and_sensitivity_refined'
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{stem}.{ext}',dpi=600 if refined else 250,facecolor='white')
plt.close(fig)
print(OUT)
