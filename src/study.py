"""Fixed-protocol models, corruption operators and threshold selection.
No function that selects a model or threshold accepts test labels.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, roc_auc_score

GRID=np.arange(.005,1,.005)
def signed_log(x):return np.sign(x)*np.log1p(np.abs(x))
def make_logistic(seed):
 return Pipeline([
  ('log',FunctionTransformer(signed_log)),
  ('features',FeatureUnion([
   ('values',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])),
   ('missing',MissingIndicator(features='all',error_on_new=False))])),
  ('model',LogisticRegression(C=1,class_weight='balanced',max_iter=1500,random_state=seed))])
def make_hgb(seed):
 return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,class_weight='balanced',random_state=seed)

def rank_groups(x,y,groups):
 # Class-mean separation: training-only, no validation/test statistics.
 with np.errstate(invalid='ignore',divide='ignore'):
  a=np.nanmean(x[y==1],axis=0);b=np.nanmean(x[y==0],axis=0);s=np.nanstd(x,axis=0)
  scores=np.nan_to_num(np.abs(a-b)/np.maximum(s,1e-8),nan=0.,posinf=0.)
 return sorted(groups,key=lambda g:(-float(np.max(scores[groups[g]])),g))
def outage(x,indices):
 z=x.copy();z[:,indices]=np.nan;return z
def environments(x,groups,ranking,seed,include_transfer=True):
 env={'clean':x}
 for i,g in enumerate(ranking[:5] if include_transfer else ranking[:3]):env[f'group_{i+1}']=outage(x,groups[g])
 if include_transfer:
  for rate in [.1,.3]:
   rng=np.random.default_rng(seed+int(rate*10000));z=x.copy();cols=sorted(set(j for ids in groups.values() for j in ids))
   mask=rng.random((len(z),len(cols)))<rate;block=z[:,cols];block[mask]=np.nan;z[:,cols]=block;env[f'mcar_{int(rate*100)}']=z
 return env
def augment(x,y,groups,ranking,mode,seed):
 copies=[x];rng=np.random.default_rng(seed)
 numeric=sorted(set(j for ids in groups.values() for j in ids))
 for g in ranking[:3]:
  if mode=='group':z=outage(x,groups[g])
  elif mode=='cell':
   z=x.copy();block=z[:,numeric].copy();n=min(block.size,len(x)*len(groups[g]))
   # Exactly match number of selected cells, without replacement.
   ids=rng.choice(block.size,n,replace=False);block.flat[ids]=np.nan;z[:,numeric]=block
  else:raise ValueError(mode)
  copies.append(z)
 return np.concatenate(copies),np.tile(y,len(copies)),np.full(len(x)*len(copies),1/len(copies))
def grid_f1(y,p,thresholds=GRID):
 order=np.argsort(p);sortedp=p[order];sorted_y=y[order];prefix=np.r_[0,np.cumsum(sorted_y)]
 start=np.searchsorted(sortedp,thresholds,side='left');tp=prefix[-1]-prefix[start];fp=len(y)-start-tp;fn=prefix[-1]-tp
 return np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(thresholds),float),where=(2*tp+fp+fn)>0)
def choose_threshold(y,validation_probabilities,environment_balanced=False):
 ps=list(validation_probabilities.values()) if environment_balanced else [validation_probabilities['clean']]
 objective=np.mean([grid_f1(y,p) for p in ps],axis=0);best=np.flatnonzero(np.isclose(objective,objective.max(),atol=1e-12,rtol=0))[-1]
 return float(GRID[best]),float(objective[best])
def metrics(y,p,t):
 pred=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),macro_f1=f1_score(y,pred,average='macro',zero_division=0),weighted_f1=f1_score(y,pred,average='weighted',zero_division=0),macro_precision=precision_score(y,pred,average='macro',zero_division=0),weighted_precision=precision_score(y,pred,average='weighted',zero_division=0),macro_recall=recall_score(y,pred,average='macro',zero_division=0),weighted_recall=recall_score(y,pred,average='weighted',zero_division=0),auprc=average_precision_score(y,p),auroc=roc_auc_score(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),false_alarms_per_1000=1000*fp/(fp+tn),aps_cost=int(10*fp+500*fn))
