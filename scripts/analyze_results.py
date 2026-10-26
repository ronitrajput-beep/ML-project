#!/usr/bin/env python3
import os
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.study import metrics
MAIN=[('logistic','clean','Logistic'),('hgb','clean','HGB'),('group_hgb','environment','Group-HGB')]
ENV=['clean','group_1','group_2','group_3','group_4','group_5','mcar_10','mcar_30']

def bootstrap(dataset,comparison,replicates=1000):
 # APS uses same rows across seeds; resample one row index vector jointly.
 seeds=[11,29,47] if dataset=='aps' else [11]
 ys=[];preds=[]
 for seed in seeds:
  run=ROOT/'results'/f'{dataset}_{seed}';d=np.load(run/'predictions.npz');ys.append(d['y_test'])
  methods=[]
  for model,policy in comparison:
   import joblib
   artifact=joblib.load(ROOT/'models'/f'{dataset}_{seed}_{model}.joblib');t=artifact['thresholds'][policy]
   methods.append(np.stack([d[f'{model}__{e}']>=t for e in ENV[:4]]))
  preds.append(np.stack(methods))
 y=ys[0];assert all(np.array_equal(y,b) for b in ys)
 pred=np.stack(preds);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);rng=np.random.default_rng(1729)
 def difference(idx):
  labels=y[idx];z=pred[:,:,:,idx];tp=(z & (labels==1)).sum(-1);fp=(z & (labels==0)).sum(-1);fn=((~z)&(labels==1)).sum(-1)
  f=2*tp/np.maximum(2*tp+fp+fn,1);return float(f[:,0,:].mean()-f[:,1,:].mean())
 point=difference(np.arange(len(y)));samples=np.array([difference(np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]) for _ in range(replicates)])
 return dict(dataset=dataset,comparison=str(comparison),estimate=point,lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),bootstrap_replicates=replicates,scope='conditional row sampling; APS seed-averaged; AI4I seed 11 only')

def main():
 df=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'results').glob('*_*/metrics.csv'))],ignore_index=True)
 df.to_csv(ROOT/'results/all_metrics.csv',index=False)
 timings=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'results').glob('*_*/timings.csv'))],ignore_index=True);timings.to_csv(ROOT/'results/all_timings.csv',index=False)
 metrics_cols=['accuracy','precision','recall','f1','macro_f1','weighted_f1','auprc','auroc','aps_cost']
 grouped=df.groupby(['dataset','model','policy','environment'])[metrics_cols].agg(['mean','std']);grouped.columns=['_'.join(c) for c in grouped.columns];grouped.reset_index().to_csv(ROOT/'results/summary.csv',index=False)
 primary=df[df.environment.isin(ENV[:4])].groupby(['dataset','seed','model','policy'])[metrics_cols].mean().reset_index();primary.to_csv(ROOT/'results/primary_by_seed.csv',index=False)
 psummary=primary.groupby(['dataset','model','policy'])[metrics_cols].agg(['mean','std']);psummary.columns=['_'.join(c) for c in psummary.columns];psummary.reset_index().to_csv(ROOT/'results/primary_summary.csv',index=False)
 contrast=[]
 for dataset in ['aps','ai4i']:
  for comp in [[('group_hgb','environment'),('hgb','clean')],[('group_hgb','environment'),('cell_hgb','environment')],[('group_hgb','environment'),('group_hgb','clean')]]:contrast.append(bootstrap(dataset,comp))
 (ROOT/'results/bootstrap.json').write_text(json.dumps(contrast,indent=2))
 figs=ROOT/'paper/figures';figs.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(10,3.1),sharey=True)
 for ax,dataset in zip(axes,['aps','ai4i']):
  for model,policy,label in MAIN+[('cell_hgb','environment','Cell-HGB')]:
   sub=df[(df.dataset==dataset)&(df.model==model)&(df.policy==policy)];s=sub.groupby('environment').f1.agg(['mean','std']).reindex(ENV)
   ax.errorbar(range(len(ENV)),s['mean'],yerr=s['std'],marker='o',capsize=2,label=label,lw=1.3,ms=3)
  ax.set_title('APS (real)' if dataset=='aps' else 'AI4I (synthetic)');ax.set_xticks(range(len(ENV)),['Clean','G1','G2','G3','G4*','G5*','R10','R30']);ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xlabel('Outage environment (* = untrained full-group outage)')
 axes[0].set_ylabel('Positive-class F1');axes[1].legend(fontsize=7,loc='lower left');fig.tight_layout();fig.savefig(figs/'outage_f1.pdf');fig.savefig(figs/'outage_f1.png',dpi=180);plt.close(fig)
 # Confusion matrices for first predesignated APS split, clean and most-ranked outage.
 fig,axes=plt.subplots(2,3,figsize=(8,4.3))
 for j,(model,policy,label) in enumerate(MAIN):
  for i,env in enumerate(['clean','group_1']):
   r=df[(df.dataset=='aps')&(df.seed==11)&(df.model==model)&(df.policy==policy)&(df.environment==env)].iloc[0];m=np.array([[r.tn,r.fp],[r.fn,r.tp]],int);ax=axes[i,j];ax.imshow(np.log1p(m),cmap='Blues')
   for rr in range(2):
    for cc in range(2):ax.text(cc,rr,str(m[rr,cc]),ha='center',va='center',color='black' if m[rr,cc]<1000 else 'white',fontsize=9)
   ax.set_xticks([0,1],['Pred 0','Pred 1']);ax.set_yticks([0,1],['True 0','True 1']);ax.set_title(label+' / '+env,fontsize=9)
 fig.tight_layout();fig.savefig(figs/'confusion.pdf');fig.savefig(figs/'confusion.png',dpi=180);plt.close(fig)
 print('PRIMARY:');print(psummary[['f1_mean','f1_std','recall_mean','precision_mean']].round(4).to_string());print('BOOTSTRAP',json.dumps(contrast,indent=2))
if __name__=='__main__':main()
