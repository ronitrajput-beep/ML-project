import os
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(9,3.6));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor=color,edgecolor='#34495e',linewidth=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color='#34495e'))
box(.15,2.85,1.65,.8,'Training rows\n(labels available)','#e7eff8')
box(2.2,2.85,2.1,.8,'Rank channel groups\non training rows only','#e7eff8')
box(4.7,2.85,2.2,.8,'Clean + three\ngroup-masked copies','#e7eff8')
box(7.3,2.85,2.25,.8,'One weighted HGB\n160 boosting rounds','#e7eff8')
arrow((1.8,3.25),(2.2,3.25));arrow((4.3,3.25),(4.7,3.25));arrow((6.9,3.25),(7.3,3.25))
box(.15,1.5,2.1,.8,'Validation rows\n(clean + same outages)','#fef3d5')
box(3.0,1.5,2.4,.8,'Predict probabilities\nwith frozen fitted HGB','#fef3d5')
box(6.15,1.5,3.4,.8,'Select one threshold\nby equal-environment mean F1','#fef3d5')
arrow((2.25,1.9),(3,1.9));arrow((5.4,1.9),(6.15,1.9));arrow((8.4,2.85),(8.4,2.3))
box(.15,.15,2.1,.8,'Test / deployment rows\nlabels unseen by selector','#e7f3e8')
box(3.0,.15,2.4,.8,'Same fitted HGB\n(no adaptation)','#e7f3e8')
box(6.15,.15,3.4,.8,'Apply fixed threshold\nNormal / fault prediction','#e7f3e8')
arrow((2.25,.55),(3,.55));arrow((5.4,.55),(6.15,.55));arrow((7.85,1.5),(7.85,.95))
fig.tight_layout(pad=.3);out=ROOT/'paper/figures';out.mkdir(exist_ok=True);fig.savefig(out/'architecture.pdf');fig.savefig(out/'architecture.png',dpi=180)
