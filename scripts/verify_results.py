"""Recompute metrics from predictions and audit split/protocol/artifact consistency."""
import os
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from src.study import metrics
from scripts.run_experiments import load
from threadpoolctl import threadpool_limits

def main():
 protocol=hashlib.sha256((ROOT/'docs/PROTOCOL.md').read_bytes()).hexdigest();checked=0;artifacts=0
 table=pd.read_csv(ROOT/'results/all_metrics.csv')
 assert len(table)==576,f'Expected 576 rows, found {len(table)}'
 with threadpool_limits(limits=4):
  for dataset in ['aps','ai4i']:
   x,y,xt,yt,cols,groups=load(dataset)
   for seed in [11,29,47]:
    run=ROOT/f'results/{dataset}_{seed}';meta=json.loads((run/'manifest.json').read_text());split=np.load(run/'split_indices.npz');d=np.load(run/'predictions.npz')
    assert meta['protocol_sha256']==protocol
    assert not set(split['train']) & set(split['validation'])
    if dataset=='ai4i':
     assert not (set(split['test']) & (set(split['train'])|set(split['validation'])))
     testx=x[split['test']];testy=y[split['test']]
     assert not any(c in cols for c in ['UDI','Product ID','TWF','HDF','PWF','OSF','RNF'])
    else:testx=xt;testy=yt
    np.testing.assert_array_equal(d['y_test'],testy)
    for model in ['logistic','hgb','group_hgb','cell_hgb']:
     a=joblib.load(ROOT/f'models/{dataset}_{seed}_{model}.joblib');np.testing.assert_allclose(a['model'].predict_proba(testx)[:,1],d[f'{model}__clean'],rtol=0,atol=1e-12);artifacts+=1
    sub=table[(table.dataset==dataset)&(table.seed==seed)]
    for _,row in sub.iterrows():
     prob=np.zeros(len(testy)) if row.model=='majority' else d[f'{row.model}__{row.environment}']
     result=metrics(testy,prob,row.threshold)
     for k,v in result.items():assert np.isclose(v,row[k],atol=1e-12,rtol=1e-10),(dataset,seed,k,v,row[k])
     checked+=1
 extra_checked=0;extra_artifacts=0
 extra=pd.read_csv(ROOT/'results/exploratory_clean_repeat.csv')
 with threadpool_limits(limits=4):
  for dataset in ['aps','ai4i']:
   x,y,xt,yt,cols,groups=load(dataset)
   for seed in [11,29,47]:
    split=np.load(ROOT/f'results/{dataset}_{seed}/split_indices.npz');d=np.load(ROOT/f'results/{dataset}_{seed}/exploratory_repeat_predictions.npz');a=joblib.load(ROOT/f'models/exploratory/{dataset}_{seed}_clean_repeat_hgb.joblib')
    testx=xt if dataset=='aps' else x[split['test']]
    np.testing.assert_allclose(a['model'].predict_proba(testx)[:,1],d['clean'],rtol=0,atol=1e-12);extra_artifacts+=1
    for _,row in extra[(extra.dataset==dataset)&(extra.seed==seed)].iterrows():
     for k,v in metrics(d['y_test'],d[row.environment],row.threshold).items():assert np.isclose(v,row[k],atol=1e-12,rtol=1e-10)
     extra_checked+=1
 assert extra_checked==144 and extra_artifacts==6
 report={'metric_rows_recomputed':checked,'exploratory_repeat_metric_rows_recomputed':extra_checked,'exploratory_repeat_models_prediction_checked':extra_artifacts,'saved_models_reloaded_and_prediction_checked':artifacts,'split_and_protocol_checks':'passed','protocol_sha256':protocol}
 (ROOT/'results/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
