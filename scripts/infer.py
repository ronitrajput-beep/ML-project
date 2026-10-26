#!/usr/bin/env python3
"""Apply a saved model and its fixed validation-selected threshold to CSV rows."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',required=True);ap.add_argument('--csv',required=True);ap.add_argument('--output',required=True);ap.add_argument('--policy',choices=['clean','environment','default'],default='clean');a=ap.parse_args()
 artifact=joblib.load(a.artifact);frame=pd.read_csv(a.csv,na_values='na');metadata=artifact['metadata']
 if metadata['dataset']=='ai4i' and 'Type' in frame:
  for k in ['L','M','H']:frame['Type_'+k]=(frame['Type']==k).astype(float)
 # Absent numeric columns explicitly represent unavailable channels, not inferred values.
 for c in metadata['features']:
  if c not in frame:frame[c]=np.nan
 x=frame[metadata['features']].apply(pd.to_numeric,errors='raise').to_numpy(np.float32)
 with threadpool_limits(limits=4):p=artifact['model'].predict_proba(x)[:,1]
 t=artifact['thresholds'][a.policy]
 pd.DataFrame({'row':np.arange(len(p)),'fault_score':p,'threshold':t,'predicted_fault':(p>=t).astype(int)}).to_csv(a.output,index=False)
 print(f'Wrote {len(p)} predictions at threshold {t:.3f}. Scores are not guaranteed calibrated probabilities.')
if __name__=='__main__':main()
