"""Post-hoc representation-count control; never replaces original results."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from scripts.run_experiments import load
from src.study import make_hgb,environments,choose_threshold,metrics
rows=[];timings=[]
with threadpool_limits(limits=4):
 for ds in ['aps','ai4i']:
  x,y,xt,yt,cols,groups=load(ds)
  for seed in [11,29,47]:
   split=np.load(ROOT/f'results/{ds}_{seed}/split_indices.npz');m=json.loads((ROOT/f'results/{ds}_{seed}/manifest.json').read_text());ids=split['train'];vi=split['validation']
   testx,testy=(xt,yt) if ds=='aps' else (x[split['test']],y[split['test']])
   model=make_hgb(seed);t0=time.perf_counter();model.fit(np.tile(x[ids],(4,1)),np.tile(y[ids],4),sample_weight=np.full(len(ids)*4,.25));elapsed=time.perf_counter()-t0
   val=environments(x[vi],groups,m['ranking'],seed+700,False);vp={e:model.predict_proba(z)[:,1] for e,z in val.items()}
   thresholds={policy:choose_threshold(y[vi],vp,policy=='environment')[0] for policy in ['clean','environment']};thresholds['default']=.5
   dest=ROOT/'models/exploratory';dest.mkdir(exist_ok=True)
   joblib.dump(dict(model=model,thresholds=thresholds,metadata=m,model_name='clean_repeat_hgb',status='post-hoc control'),dest/f'{ds}_{seed}_clean_repeat_hgb.joblib',compress=3)
   pred={'y_test':testy}
   for e,z in environments(testx,groups,m['ranking'],seed+800).items():
    p=model.predict_proba(z)[:,1];pred[e]=p
    for policy,t in thresholds.items():rows.append(dict(dataset=ds,seed=seed,environment=e,model='clean_repeat_hgb',policy=policy,threshold=t,**metrics(testy,p,t)))
   np.savez_compressed(ROOT/f'results/{ds}_{seed}/exploratory_repeat_predictions.npz',**pred)
   timings.append(dict(dataset=ds,seed=seed,fit_seconds=elapsed));print('DONE',ds,seed,elapsed,flush=True)
pd.DataFrame(rows).to_csv(ROOT/'results/exploratory_clean_repeat.csv',index=False)
pd.DataFrame(timings).to_csv(ROOT/'results/exploratory_clean_repeat_timings.csv',index=False)
d=pd.DataFrame(rows);print(d[d.environment.isin(['clean','group_1','group_2','group_3'])].groupby(['dataset','policy']).f1.agg(['mean','std']).to_string())
