#!/usr/bin/env python3
"""Train all fixed configurations; save every prediction and model."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='4'
os.environ.setdefault('MPLCONFIGDIR','/tmp/outage-mpl-cache')
import sys,json,time,hashlib,platform,subprocess,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,sklearn,scipy,joblib,matplotlib
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from src.study import make_logistic,make_hgb,rank_groups,environments,augment,choose_threshold,metrics
SEEDS=[11,29,47]

def load(dataset):
 if dataset=='aps':
  a=pd.read_csv(ROOT/'data/aps_train.csv',na_values='na');b=pd.read_csv(ROOT/'data/aps_test.csv',na_values='na')
  cols=list(a.columns[1:]);x=a[cols].to_numpy(np.float32);y=(a['class']=='pos').to_numpy(np.int8)
  xt=b[cols].to_numpy(np.float32);yt=(b['class']=='pos').to_numpy(np.int8)
  groups={g:[i for i,c in enumerate(cols) if c.split('_')[0]==g] for g in sorted(set(c.split('_')[0] for c in cols))}
  return x,y,xt,yt,cols,groups
 d=pd.read_csv(ROOT/'data/ai4i.csv');cols=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
 x=np.column_stack([d[cols].to_numpy(np.float32)]+[(d['Type']==k).to_numpy(np.float32) for k in ['L','M','H']]);y=d['Machine failure'].to_numpy(np.int8)
 groups={c:[i] for i,c in enumerate(cols)}
 return x,y,None,None,cols+['Type_L','Type_M','Type_H'],groups

def run(dataset,seed):
 start=time.time();x,y,xt,yt,cols,groups=load(dataset);ids=np.arange(len(y))
 if dataset=='ai4i':
  trainval,test_ids=train_test_split(ids,test_size=.2,random_state=seed,stratify=y);xt=x[test_ids];yt=y[test_ids]
  train_ids,val_ids=train_test_split(trainval,test_size=.25,random_state=seed,stratify=y[trainval])
 else:
  train_ids,val_ids=train_test_split(ids,test_size=.2,random_state=seed,stratify=y);test_ids=np.arange(len(yt))
 xtr,ytr=x[train_ids],y[train_ids];xv,yv=x[val_ids],y[val_ids]
 ranking=rank_groups(xtr,ytr,groups);valid=environments(xv,groups,ranking,seed+700,include_transfer=False)
 test=environments(xt,groups,ranking,seed+800)
 splitdir=ROOT/'results'/f'{dataset}_{seed}';splitdir.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(splitdir/'split_indices.npz',train=train_ids,validation=val_ids,test=test_ids)
 metadata=dict(dataset=dataset,seed=seed,train_rows=len(ytr),validation_rows=len(yv),test_rows=len(yt),train_positive=int(ytr.sum()),validation_positive=int(yv.sum()),test_positive=int(yt.sum()),features=cols,groups=groups,ranking=ranking,train_missing_fraction=float(np.isnan(xtr).mean()),test_missing_fraction=float(np.isnan(xt).mean()),protocol_sha256=hashlib.sha256((ROOT/'docs/PROTOCOL.md').read_bytes()).hexdigest())
 (splitdir/'manifest.json').write_text(json.dumps(metadata,indent=2))
 allmetrics=[];predictions={'y_test':yt,'test_ids':test_ids};timings=[]
 for modelname in ['logistic','hgb','group_hgb','cell_hgb']:
  print(f'START {dataset} seed={seed} model={modelname}',flush=True);t0=time.perf_counter()
  model=make_logistic(seed) if modelname=='logistic' else make_hgb(seed)
  if modelname in ['group_hgb','cell_hgb']:
   xa,ya,w=augment(xtr,ytr,groups,ranking,'group' if modelname=='group_hgb' else 'cell',seed+900)
   model.fit(xa,ya,sample_weight=w);del xa,ya,w
  else:model.fit(xtr,ytr)
  fit_seconds=time.perf_counter()-t0
  vp={e:model.predict_proba(z)[:,1] for e,z in valid.items()}
  clean_t,clean_score=choose_threshold(yv,vp,False);env_t,env_score=choose_threshold(yv,vp,True)
  thresholds={'clean':clean_t,'environment':env_t,'default':.5}
  selected={'clean':clean_score,'environment':env_score}
  artifact=dict(model=model,thresholds=thresholds,metadata=metadata,model_name=modelname,validation_objectives=selected)
  artifact_path=ROOT/'models'/f'{dataset}_{seed}_{modelname}.joblib';joblib.dump(artifact,artifact_path,compress=3)
  timing=dict(dataset=dataset,seed=seed,model=modelname,fit_seconds=fit_seconds,artifact_bytes=artifact_path.stat().st_size)
  for e,z in test.items():
   t0=time.perf_counter();p=model.predict_proba(z)[:,1];elapsed=time.perf_counter()-t0
   predictions[f'{modelname}__{e}']=p
   for policy,t in thresholds.items():
    if modelname=='logistic' and policy=='environment':continue
    row=dict(dataset=dataset,seed=seed,model=modelname,policy=policy,environment=e,threshold=t,predict_seconds=elapsed,**metrics(yt,p,t));allmetrics.append(row)
  timings.append(timing);print('DONE',json.dumps(timing),'thresholds',thresholds,flush=True)
 for e in test:
  allmetrics.append(dict(dataset=dataset,seed=seed,model='majority',policy='default',environment=e,threshold=.5,predict_seconds=0.,**metrics(yt,np.zeros(len(yt)),.5)))
 pd.DataFrame(allmetrics).to_csv(splitdir/'metrics.csv',index=False);pd.DataFrame(timings).to_csv(splitdir/'timings.csv',index=False)
 np.savez_compressed(splitdir/'predictions.npz',**predictions)
 print('SPLIT_FINISHED',dataset,seed,'seconds',time.time()-start,flush=True)

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['aps','ai4i']);ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);a=ap.parse_args()
 versions={m:__import__(m).__version__ for m in ['numpy','pandas','sklearn','scipy','joblib','matplotlib']}
 (ROOT/'results/environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),versions=versions,seeds=a.seeds,datasets=a.datasets,threads=4),indent=2))
 with threadpool_limits(limits=4):
  for dataset in a.datasets:
   for seed in a.seeds:run(dataset,seed)
if __name__=='__main__':main()
