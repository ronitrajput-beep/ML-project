"""Descriptive post-fit audit; no audit result changes model selection."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scripts.run_experiments import load

def main():
 result={}
 for dataset in ['aps','ai4i']:
  x,y,xt,yt,cols,groups=load(dataset)
  report={'source_rows':len(y),'source_positive':int(y.sum()),'source_positive_fraction':float(y.mean()),'features':len(cols),'groups':len(groups),'source_missing_fraction':float(np.isnan(x).mean())}
  if xt is not None:
   h1=pd.util.hash_pandas_object(pd.DataFrame(x),index=False);h2=pd.util.hash_pandas_object(pd.DataFrame(xt),index=False)
   report.update(test_rows=len(yt),test_positive=int(yt.sum()),test_positive_fraction=float(yt.mean()),test_feature_rows_also_present_in_source=int(h2.isin(set(h1)).sum()),source_duplicate_feature_rows=int(h1.duplicated().sum()),test_duplicate_feature_rows=int(h2.duplicated().sum()))
  else:report['source_duplicate_feature_rows']=int(pd.DataFrame(x).duplicated().sum())
  # Top-ranked feature groups and their pairwise observed-value correlations, seed 11 training only.
  m=json.loads((ROOT/f'results/{dataset}_11/manifest.json').read_text());ids=np.load(ROOT/f'results/{dataset}_11/split_indices.npz')['train']
  selected=[j for g in m['ranking'][:5] for j in groups[g]]
  corr=pd.DataFrame(x[ids][:,selected],columns=[cols[j] for j in selected]).corr()
  report['top_group_feature_correlations']=corr.round(6).to_dict()
  result[dataset]=report
 (ROOT/'results/data_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
