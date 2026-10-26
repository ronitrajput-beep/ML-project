from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'paper/tables';out.mkdir(exist_ok=True)
s=pd.read_csv(ROOT/'results/summary.csv');p=pd.read_csv(ROOT/'results/primary_summary.csv');t=pd.read_csv(ROOT/'results/all_timings.csv')
MAIN=[('logistic','clean','Logistic'),('hgb','clean','HGB'),('group_hgb','environment','Group-HGB')]
head=r'''\begin{table*}[t]
\centering\caption{Clean-test classification results, mean over three fixed split seeds. Metrics are percentages except AP, which is on $[0,1]$. These are our runs, not cross-paper comparisons.}
\label{tab:clean}\small
\begin{tabular}{llrrrrrrr}\hline
Dataset & Model & Accuracy & Precision & Recall & $F_1$ & Macro $F_1$ & Weighted $F_1$ & AP\\\hline
'''
rows=[]
for ds in ['aps','ai4i']:
 for model,policy,label in MAIN:
  r=s[(s.dataset==ds)&(s.model==model)&(s.policy==policy)&(s.environment=='clean')].iloc[0]
  vals=[f'{100*r[m+"_mean"]:.2f}' for m in ['accuracy','precision','recall','f1','macro_f1','weighted_f1']]+[f'{r.auprc_mean:.3f}']
  rows.append(('APS' if ds=='aps' else 'AI4I')+' & '+label+' & '+' & '.join(vals)+r'\\')
(out/'clean.tex').write_text(head+'\n'.join(rows)+'\n'+r'\hline\end{tabular}\end{table*}'+'\n')
head=r'''\begin{table}[t]\centering\caption{Positive-class $F_1$ over clean plus three development-family outage environments. Mean $\pm$ sample SD over three split seeds. ``Env'' selects a threshold using all four validation environments.}\label{tab:ablation}\small
\begin{tabular}{llrr}\hline
Training & Threshold & APS & AI4I\\\hline
'''
rows=[]
for model,policy,label,plabel in [('logistic','clean','Logistic','Clean'),('hgb','default','HGB','0.5'),('hgb','clean','HGB','Clean'),('hgb','environment','HGB','Env'),('group_hgb','default','Group-HGB','0.5'),('group_hgb','clean','Group-HGB','Clean'),('group_hgb','environment','Group-HGB','Env'),('cell_hgb','environment','Cell-HGB','Env')]:
 vals=[]
 for ds in ['aps','ai4i']:
  r=p[(p.dataset==ds)&(p.model==model)&(p.policy==policy)].iloc[0];vals.append(f'${r.f1_mean:.3f}\\pm{r.f1_std:.3f}$')
 rows.append(label+' & '+plabel+' & '+' & '.join(vals)+r'\\')
(out/'ablation.tex').write_text(head+'\n'.join(rows)+'\n'+r'\hline\end{tabular}\end{table}'+'\n')
head=r'''\begin{table}[t]\centering\caption{Transfer to untrained full-group outages (G4/G5), averaged over both groups and all three split seeds. Random cell masking is a control, not a newly proposed algorithm.}\label{tab:transfer}\small
\begin{tabular}{lrr}\hline
Model & APS $F_1$ & AI4I $F_1$\\\hline
'''
rows=[]
for model,policy,label in MAIN+[('cell_hgb','environment','Cell-HGB')]:
 vals=[]
 for ds in ['aps','ai4i']:
  r=s[(s.dataset==ds)&(s.model==model)&(s.policy==policy)&(s.environment.isin(['group_4','group_5']))];vals.append(f'{r.f1_mean.mean():.3f}')
 rows.append(label+' & '+' & '.join(vals)+r'\\')
(out/'transfer.tex').write_text(head+'\n'.join(rows)+'\n'+r'\hline\end{tabular}\end{table}'+'\n')
# Publisher metadata -> complete bibliography; no invented identifiers or authors.
corpus=json.loads((ROOT/'literature/corpus.json').read_text());bib=[]
for r in corpus:
 meta=json.loads((ROOT/'literature/metadata'/f'{r["key"]}.json').read_text())
 authors=meta.get('bibtex_author') or ' and '.join(a.get('given','')+' '+a.get('family','') for a in meta.get('author',[]))
 overrides={
 'lobo25a':'Reyero Lobo, Angel D. and Ayme, Alexis and Boyer, Claire and Scornet, Erwan',
 'cheng25e':'Cheng, Zi-Jian and Jia, Zi-Yi and Zhou, Zhi and Li, Yu-Feng and Guo, Lan-Zhe',
 'zhou23b':'Zhou, Helen and Balakrishnan, Sivaraman and Lipton, Zachary C.',
 'cai25j':'Cai, Hao-Run and Ye, Han-Jia',
 'stempfle24a':'Stempfle, Lena and Johansson, Fredrik D.'}
 authors=overrides.get(r['key'],authors)
 title=meta['title'].replace('&',r'\&').replace('—','--')
 volume=str(json.loads((ROOT/'literature/sources.json').read_text())[r['key']]['volume'])
 venue=json.loads((ROOT/'literature/volume_metadata.json').read_text())[volume]['booktitle']
 fields={'title':'{'+title+'}','author':authors,'booktitle':venue,'year':r['year'],'volume':volume,'pages':meta.get('page',''),'url':r['pdf_url']}
 bib.append('@inproceedings{'+r['key']+',\n'+',\n'.join(' '+k+' = {'+str(v)+'}' for k,v in fields.items() if v)+'\n}')
bib += [r'''@misc{apsdata,
 author = {{Scania CV AB}},
 title = {{APS Failure at Scania Trucks}},
 year = {2016},
 note = {Dataset description; public mirror pinned to commit 4cd68dd},
 url = {https://raw.githubusercontent.com/kev-wes/pdm_scania/4cd68dd63886416e8e60f1330d79ac7f5094ce9a/aps_failure_description.txt}
}''',r'''@misc{ai4idata,
 author = {{UCI Machine Learning Repository}},
 title = {{AI4I 2020 Predictive Maintenance Dataset}},
 year = {2020},
 note = {Dataset; acquired from public mirror pinned to commit 789f44a},
 url = {https://raw.githubusercontent.com/rockyfjord/predictive-maintenance/789f44a7336f173cea4d4879fb61993db0d42fdf/maintenance.csv}
}''']
(ROOT/'paper/references.bib').write_text('\n\n'.join(bib)+'\n')
print('Generated tables and bibliography from saved results and publisher metadata.')
