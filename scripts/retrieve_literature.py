"""Retrieve publisher-maintained open-access PDFs via the PMLR GitHub mirror."""
from pathlib import Path
import urllib.request,json,hashlib,re,concurrent.futures
import fitz,yaml
ROOT=Path(__file__).resolve().parents[1]
PAPERS=[(119,'muzellec20a'),(162,'jarrett22a'),(206,'zhou23b'),(206,'telyatnikov23a'),(238,'stempfle24a'),(235,'ayme24a'),(258,'lobo25a'),(267,'cheng25e'),(267,'stempfle25a'),(267,'cai25j')]

def retrieve(pair):
 volume,key=pair;directory=ROOT/'literature';directory.mkdir(exist_ok=True)
 for folder in ['pdfs','texts','metadata']:(directory/folder).mkdir(exist_ok=True)
 source=json.loads((ROOT/'literature/sources.json').read_text())[key]
 metadata_source=source['metadata']
 raw=urllib.request.urlopen(metadata_source,timeout=30).read().decode()
 meta=yaml.safe_load(raw.split('---',2)[1]);(directory/'metadata'/f'{key}.json').write_text(json.dumps(meta,indent=2,default=str))
 url=source['pdf'];p=directory/'pdfs'/f'{key}.pdf'
 if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=90).read())
 previous_path=ROOT/'literature/corpus.json'
 previous=json.loads(previous_path.read_text()) if previous_path.exists() else []
 old=next((r for r in previous if r['key']==key),None)
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 if old and digest!=old['sha256']:raise ValueError('PDF checksum mismatch: '+key)
 with fitz.open(p) as doc:pages=[page.get_text() for page in doc]
 (directory/'texts'/f'{key}.json').write_text(json.dumps(pages,indent=2))
 (directory/'texts'/f'{key}.txt').write_text('\n'.join(f'\n[PDF PAGE {i+1}]\n{page}' for i,page in enumerate(pages)))
 return dict(key=key,title=meta['title'],year=source['year'],authors=meta.get('author'),venue=meta.get('booktitle',f'PMLR {volume}'),pages=len(pages),url=meta.get('url'),pdf_url=url,metadata_url=metadata_source,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),read_status=old['read_status'] if old else 'full text retrieved; main-text review pending')
if __name__=='__main__':
 results=list(concurrent.futures.ThreadPoolExecutor(max_workers=3).map(retrieve,PAPERS));(ROOT/'literature/corpus.json').write_text(json.dumps(results,indent=2,default=str));print(json.dumps(results,indent=2,default=str))
