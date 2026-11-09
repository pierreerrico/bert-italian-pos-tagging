from pathlib import Path
import csv,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
out=ROOT/'report/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#24343d','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(13,6.3));ax.set_xlim(0,13);ax.set_ylim(0,6.4);ax.axis('off')
blue='#246b8e';orange='#be763a';grey='#81919d'
def box(x,y,w,h,text,color=blue,fill='#edf4f7',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.08',linewidth=1.3,edgecolor=color,facecolor=fill))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,linewidth=1.15,color='#60717b'))
box(.35,5.75,12.3,.55,'Parole e frasi gold  >  WordPiece  >  token e maschere',grey,'#f3f5f6')
for i,(title,frozen,crf) in enumerate([('BERT congelato',True,False),('BERT fine-tuned',False,False),('BERT + CRF',False,True)]):
 x=.35+i*4.3;c=x+1.85
 arrow(c,5.65,c,4.75)
 box(x,3.95,3.7,.75,title+'\nEncoder italiano: '+('pesi fissi' if frozen else 'pesi aggiornati'),grey if frozen else blue,'#f0f2f3' if frozen else '#e8f2f7')
 arrow(c,3.85,c,3.48)
 box(x,2.93,3.7,.48,'Prima subword di ogni parola',grey,'#f5f6f7',10)
 arrow(c,2.83,x+.78,2.43);arrow(c,2.83,x+2.85,2.43)
 box(x,1.80,1.56,.55,'Testa UPOS',blue,size=10)
 box(x+2.0,1.80,1.7,.55,'14 teste FEATS',orange,'#fff5eb',10)
 if crf:
  arrow(x+.78,1.70,x+.78,1.36);box(x,.83,1.56,.45,'CRF',blue,size=10);arrow(x+.78,.73,x+.78,.38)
 else:arrow(x+.78,1.70,x+.78,.38)
 arrow(x+2.85,1.70,x+2.85,.38)
 ax.text(x+.78,.2,'Tag UPOS',ha='center',fontsize=10)
 ax.text(x+2.85,.2,'Attributi FEATS',ha='center',fontsize=10)
fig.savefig(out/'model_architectures.pdf',bbox_inches='tight');fig.savefig(out/'model_architectures.png',dpi=170,bbox_inches='tight');plt.close(fig)
counts=collections.Counter()
root=ROOT/'outputs/bert_upos_then_feats_v2/linear_upos_feats'
for seed in [42,123,456]:
 with (root/f'seed_{seed}'/'predictions.csv').open(encoding='utf8',newline='') as f:
  for r in csv.DictReader(f):
   if r['true_label']!=r['predicted_label']:counts[(r['true_label'],r['predicted_label'])]+=1
importance=collections.Counter()
for (g,p),n in counts.items():importance[g]+=n;importance[p]+=n
tags=[tag for tag,n in importance.most_common(6)]
m=np.array([[counts[g,p] if g!=p else 0 for p in tags] for g in tags])
fig,ax=plt.subplots(figsize=(8.3,6.8));im=ax.imshow(np.ma.masked_where(np.eye(len(tags)),m),cmap='Blues',vmin=0,vmax=max(m.max(),1))
ax.set_xticks(range(6),tags,fontfamily='monospace');ax.set_yticks(range(6),tags,fontfamily='monospace');ax.set_xlabel('Tag predetto',labelpad=10);ax.set_ylabel('Tag gold',labelpad=10)
for i in range(6):
 for j in range(6):ax.text(j,i,'-' if i==j else str(m[i,j]),ha='center',va='center',fontsize=13,color='white' if m[i,j]>m.max()*.55 else '#24343d')
for spine in ax.spines.values():spine.set_visible(False)
ax.tick_params(length=0);fig.colorbar(im,ax=ax,shrink=.7,label='Errori aggregati sui tre seed')
fig.savefig(out/'upos_error_confusions.pdf',bbox_inches='tight');fig.savefig(out/'upos_error_confusions.png',dpi=170,bbox_inches='tight');plt.close(fig)
print('Anteprime create; categorie della matrice:',', '.join(tags))
