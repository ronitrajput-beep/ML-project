import os
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from scripts.run_experiments import load
from src.study import outage,metrics
from threadpoolctl import threadpool_limits
rows=[]
x,y,xt,yt,cols,groups=load('aps')
with threadpool_limits(limits=4):
 for seed in [11,29,47]:
  run=ROOT/f'results/aps_{seed}';manifest=json.loads((run/'manifest.json').read_text());ids=np.load(run/'split_indices.npz')['train'];ranking=manifest['ranking'][:5]
  selected=[groups[g][0] for g in ranking];corr=pd.DataFrame(x[ids][:,selected]).corr(min_periods=1000).abs().to_numpy()
  components=[];visited=set()
  for i in range(len(selected)):
   if i in visited:continue
   pending=[i];component=set()
   while pending:
    j=pending.pop()
    if j in component:continue
    component.add(j);pending += [k for k in range(len(selected)) if k!=j and corr[j,k]>.9999 and k not in component]
   visited |= component
   if len(component)>1:components.append(sorted(component))
  if not components:continue
  indices=[selected[i] for i in components[0]];z=outage(xt,indices)
  for modelname,policy in [('logistic','clean'),('hgb','clean'),('group_hgb','environment'),('cell_hgb','environment')]:
   a=joblib.load(ROOT/f'models/aps_{seed}_{modelname}.joblib');p=a['model'].predict_proba(z)[:,1]
   rows.append(dict(dataset='aps',seed=seed,model=modelname,policy=policy,masked_columns=';'.join(cols[j] for j in indices),**metrics(yt,p,a['thresholds'][policy])))
pd.DataFrame(rows).to_csv(ROOT/'results/exploratory_joint_outage.csv',index=False)
print(pd.DataFrame(rows)[['seed','model','masked_columns','accuracy','recall','f1']].to_string(index=False))
