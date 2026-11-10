from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
out=Path(__file__).resolve().parent/'figures';out.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#24343d'})
fig,ax=plt.subplots(figsize=(12,6.3));ax.set_xlim(0,12);ax.set_ylim(-1.0,6.95);ax.axis('off')
blue='#246b8e';grey='#81919d';orange='#be763a'
def box(x,y,w,h,text,color=blue,fill='#edf4f7',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.06',facecolor=fill,edgecolor=color,linewidth=1.3))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(x1,y1,x2,y2,**kwargs):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color='#60717b',linewidth=1.2,**kwargs))
# Overall encoder, read top to bottom.
box(.35,5.60,4.6,.54,'Token WordPiece e token speciali',grey,'#f3f5f6')
arrow(2.65,5.52,2.65,5.16)
box(.35,4.13,4.6,.95,'Embedding dei token\n+ embedding di posizione\n+ embedding di segmento',size=11)
arrow(2.65,4.05,2.65,3.65)
box(.35,2.96,4.6,.61,'Blocco Encoder 1')
arrow(2.65,2.88,2.65,2.60)
box(.35,1.95,4.6,.57,'Blocco Encoder 2')
arrow(2.65,1.87,2.65,1.65)
ax.text(2.65,1.40,'...',ha='center',va='center',fontsize=22)
arrow(2.65,1.15,2.65,.97)
box(.35,.28,4.6,.61,'Blocco Encoder 12')
arrow(2.65,.20,2.65,-.08)
box(.35,-.88,4.6,.72,'Un vettore contestuale per token\n768 componenti',fill='#e2eff5')
# Expanded encoder block.
ax.add_patch(FancyBboxPatch((6.15,1.12),5.35,5.28,boxstyle='round,pad=.08',facecolor='#fafbfc',edgecolor='#b9c2c8',linewidth=1.2))
box(7.05,1.48,3.40,.65,'Multi-Head Self-Attention\nbidirezionale (12 teste)',size=10)
arrow(8.75,2.21,8.75,2.54)
box(7.05,2.64,3.40,.45,'Somma residua + LayerNorm',grey,'#f1f4f6',10)
arrow(8.75,3.17,8.75,3.53)
box(7.05,3.63,3.40,.74,'Rete feed-forward + GELU\n768 > 3072 > 768',orange,'#fff4e9',10)
arrow(8.75,4.45,8.75,4.78)
box(7.05,4.88,3.40,.45,'Somma residua + LayerNorm',grey,'#f1f4f6',10)
# Route the expanded block output back to Encoder 2 in the stack.
ax.plot([8.75,8.75,11.80,11.80,5.60,5.60],[5.41,6.70,6.70,-.15,-.15,2.235],color='#60717b',linewidth=1.2)
arrow(5.60,2.235,5.04,2.235)
ax.text(8.75,.30,'Struttura di ciascun blocco Encoder',ha='center',fontsize=10,color=grey)
arrow(8.75,1.17,8.75,1.40)
# Residual branches, one around each sublayer.
ax.plot([8.75,6.55,6.55],[1.25,1.25,2.86],color=grey,linewidth=1.1);arrow(6.55,2.86,6.97,2.86)
ax.plot([8.75,10.98,10.98],[3.35,3.35,5.10],color=grey,linewidth=1.1);arrow(10.98,5.10,10.53,5.10)
arrow(5.08,3.265,6.06,3.265,linestyle='--')
fig.savefig(out/'bert_encoder_structure.pdf',bbox_inches='tight',facecolor='white')
fig.savefig(out/'bert_encoder_structure.png',dpi=180,bbox_inches='tight',facecolor='white')
