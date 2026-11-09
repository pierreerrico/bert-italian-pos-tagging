from pathlib import Path
import csv,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'report/figures'
RUNS=ROOT/'outputs/bert_upos_then_feats_v2'
COLORS={'frozen':'#81919d','linear':'#246b8e','crf':'#be763a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#b9c2c8','text.color':'#24343d','axes.labelcolor':'#24343d','xtick.color':'#53626a','ytick.color':'#24343d','pdf.fonttype':42,'savefig.facecolor':'white'})

def read(path):
 with path.open(newline='',encoding='utf8') as stream:return list(csv.DictReader(stream))
def stats(rows,system,metric):
 values=[100*float(row[metric]) for row in rows if row['system']==system]
 assert len(values)==3
 return statistics.mean(values),statistics.stdev(values)
def axis(ax,title,label):
 ax.set_title(title,loc='left',fontweight='bold',pad=12)
 ax.set_xlabel(label)
 ax.set_axisbelow(True);ax.grid(axis='x',color='#e5eaed',linewidth=.8)
 ax.tick_params(axis='y',length=0,pad=8)
 ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')))
 ax.xaxis.set_major_locator(MaxNLocator(4))
def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
 plt.close(fig)

rows=read(RUNS/'results.csv')
reference=next(row for row in read(ROOT/'outputs/references/udpipe_2_15/morphology_comparison.csv') if row['architecture']=='udpipe')
# Bar charts with explicitly truncated score axes to show small differences.
def bars(ax,labels,means,stds,colors,title,delta=False):
 ax.set_title(title,loc='left',fontweight='bold',pad=12)
 positions=list(range(len(labels)))
 ax.bar(positions,means,yerr=stds,color=colors,width=.58,capsize=4,error_kw={'ecolor':'#34434b','elinewidth':1.1})
 ax.set_xticks(positions,labels,fontsize=8)
 ax.set_axisbelow(True);ax.grid(axis='y',color='#e5eaed',linewidth=.8)
 ax.set_ylabel('Punti percentuali' if delta else 'Punteggio (%)')
 ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')))
 if delta:
  low=min(0,min(m-v for m,v in zip(means,stds)));high=max(m+v for m,v in zip(means,stds))
  ax.set_ylim(low-.08*(high-low),high+.23*(high-low));ax.axhline(0,color='#81919d',linewidth=.8)
 else:
  low=min(m-v for m,v in zip(means,stds));high=max(m+v for m,v in zip(means,stds))
  span=max(high-low,.04)
  ax.set_ylim(low-.20*span,high+.35*span)
  ax.yaxis.set_major_locator(MaxNLocator(5))
  # A visible break and explicit label distinguish these from zero-baseline bars.
  ax.spines['left'].set_visible(True)
  for shift in [-.012,.012]:
   ax.plot([-.012,.012],[shift-.008,shift+.008],transform=ax.transAxes,color='#53626a',clip_on=False,linewidth=1.2)
  ax.set_xlabel('Asse verticale troncato',fontsize=8,color='#53626a',labelpad=8)
 for x,mean,std in zip(positions,means,stds):
  text=(f'{mean:+.2f}' if delta else f'{mean:.2f}').replace('.',',')
  offset=(ax.get_ylim()[1]-ax.get_ylim()[0])*.035
  ax.text(x,mean+std+offset,text,ha='center',va='bottom',fontsize=9,fontweight='bold')

fig,axes=plt.subplots(1,2,figsize=(8.8,3.0),layout='constrained')
for ax,metric,title,ref in zip(axes,['upos_accuracy','upos_macro_f1_present'],['Accuracy UPOS','Macro-F1 UPOS - classi presenti'],['upos_accuracy_mean_pct','upos_macro_f1_present_mean_pct']):
 pairs=[stats(rows,system,metric) for system in ['linear_upos','linear_upos_feats']]
 bars(ax,['UDPipe 2','BERT\nsolo UPOS','BERT\nUPOS + FEATS'],[float(reference[ref])]+[p[0] for p in pairs],[0]+[p[1] for p in pairs],['#81919d','#7a9caf',COLORS['linear']],title)
save(fig,'upos_reference_comparison')

changes=read(RUNS/'upos_stage_changes.csv')
fig,axes=plt.subplots(1,2,figsize=(8.8,3.0),layout='constrained')
for ax,metric,title in zip(axes,['upos_accuracy','upos_macro_f1_present'],['Variazione di accuracy UPOS','Variazione di macro-F1 UPOS']):
 values=[[float(row[metric]) for row in changes if row['architecture']==arch] for arch in ['frozen','linear','crf']]
 assert all(len(v)==3 for v in values)
 bars(ax,['BERT\ncongelato','BERT\nfine-tuned','BERT + CRF'],[statistics.mean(v) for v in values],[statistics.stdev(v) for v in values],list(COLORS.values()),title,delta=True)
save(fig,'upos_stage_changes')

fig,axes=plt.subplots(1,2,figsize=(8.8,3.0),layout='constrained')
for ax,metric,title,ref in zip(axes,['feats_exact_accuracy','feats_macro_f1_present'],['Accuracy esatta FEATS','Macro-F1 FEATS - coppie presenti'],['feats_exact_accuracy_mean_pct','feats_macro_f1_present_mean_pct']):
 mean,std=stats(rows,'linear_upos_feats',metric)
 bars(ax,['UDPipe 2','BERT fine-tuned'],[float(reference[ref]),mean],[0,std],['#81919d',COLORS['linear']],title)
save(fig,'feats_reference_comparison')

# Learning curves: saved training logs, all three seeds of the principal model.
import numpy as np
fig,axes=plt.subplots(2,2,figsize=(8.8,5.6),layout='constrained')
for col,(directory,system,title) in enumerate([
 ('bert_upos_then_feats_v1','linear_upos','Fase 1 - solo UPOS'),
 ('bert_upos_then_feats_v2','linear_upos_feats','Fase 2 - UPOS + FEATS')]):
 histories=[read(ROOT/'outputs'/directory/system/f'seed_{seed}'/'training_history.csv') for seed in [42,123,456]]
 def curve(ax,key,label,color,linestyle='-',percent=False):
  series=[[(float(r['epoch']),float(r[key])) for r in history if r.get(key)] for history in histories]
  assert all(len(series[0])==len(x) for x in series)
  x=np.array([v[0] for v in series[0]])
  assert all(np.allclose(x,[v[0] for v in seq]) for seq in series)
  y=np.array([[v[1] for v in seq] for seq in series])*(100 if percent else 1)
  mean=y.mean(axis=0);std=y.std(axis=0,ddof=1)
  ax.plot(x,mean,label=label,color=color,linestyle=linestyle,linewidth=1.7,marker='o' if key.startswith('eval_') else None,markersize=3)
  ax.fill_between(x,np.maximum(mean-std,1e-5) if not percent else mean-std,mean+std,color=color,alpha=.12,linewidth=0)
 ax=axes[0,col];axis(ax,title,'Epoca nella fase');ax.set_ylabel('Loss (scala logaritmica)');ax.grid(axis='y',color='#e5eaed',linewidth=.7)
 curve(ax,'loss','Training','#246b8e');curve(ax,'eval_loss','Development','#be763a')
 ax.set_yscale('log');ax.set_xticks([1,2,3,4,5]);ax.legend(frameon=False,fontsize=8)
 ax=axes[1,col];axis(ax,'Metriche sul development','Epoca nella fase');ax.set_ylabel('Macro-F1 (%)');ax.grid(axis='y',color='#e5eaed',linewidth=.7)
 curve(ax,'eval_upos_macro_f1_all','UPOS · totale','#246b8e',percent=True)
 if col==1:
  curve(ax,'eval_feats_macro_f1_present','FEATS · coppie presenti','#be763a',percent=True)
  curve(ax,'eval_joint_macro_f1','Punteggio di selezione','#74816f',linestyle='--',percent=True)
 ax.set_xticks([1,2,3,4,5]);ax.legend(frameon=False,fontsize=8,loc='lower right')
 ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'.replace('.',',')))
save(fig,'training_dynamics')
