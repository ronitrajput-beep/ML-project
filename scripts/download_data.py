"""Pinned public mirrors; original datasets are credited in data/README.md."""
from pathlib import Path
import urllib.request, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
'ai4i.csv':'https://raw.githubusercontent.com/rockyfjord/predictive-maintenance/789f44a7336f173cea4d4879fb61993db0d42fdf/maintenance.csv',
'aps_train.csv':'https://raw.githubusercontent.com/kev-wes/pdm_scania/4cd68dd63886416e8e60f1330d79ac7f5094ce9a/aps_failure_training_set.csv',
'aps_test.csv':'https://raw.githubusercontent.com/kev-wes/pdm_scania/4cd68dd63886416e8e60f1330d79ac7f5094ce9a/aps_failure_test_set.csv'}
def main():
 (ROOT/'data').mkdir(exist_ok=True);manifest=[]
 previous=json.loads((ROOT/'data/manifest.json').read_text()) if (ROOT/'data/manifest.json').exists() else []
 expected={x['file']:x['sha256'] for x in previous}
 for name,url in SOURCES.items():
  p=ROOT/'data'/name
  if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=90).read())
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if name in expected and digest!=expected[name]:raise ValueError('Dataset checksum mismatch: '+name)
  manifest.append(dict(file=name,url=url,sha256=digest,bytes=p.stat().st_size))
 (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
