import urllib.request,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'outputs/references/isdt_commit_comparison'
out.mkdir(parents=True,exist_ok=True)
def sentences(data):
 result=[]
 for block in data.decode('utf8').strip().split('\n\n'):
  rows=[line.split('\t') for line in block.splitlines() if line and not line.startswith('#')]
  rows=[row for row in rows if row[0].isdigit()]
  if rows: result.append(rows)
 return result
report={}
for split in ['train','dev','test']:
 name=f'it_isdt-ud-{split}.conllu'
 url=f'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/f20fa2b/{name}'
 cached=out/name
 if not cached.exists():cached.write_bytes(urllib.request.urlopen(url,timeout=40).read())
 old=cached.read_bytes()
 current=(root/'data/UD_Italian-ISDT-r2.15'/name).read_bytes()
 a,b=sentences(old),sentences(current)
 ar=[row for sent in a for row in sent];br=[row for sent in b for row in sent]
 aligned=len(a)==len(b) and all([r[1] for r in x]==[r[1] for r in y] for x,y in zip(a,b))
 info={'source_url':url,'historical_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(current).hexdigest(),'byte_identical':old==current,'historical_sentences':len(a),'current_sentences':len(b),'historical_words':len(ar),'current_words':len(br),'forms_and_sentence_boundaries_identical':aligned}
 if aligned:
  for column,index in [('upos',3),('feats',5)]:info[column+'_differences']=sum(x[index]!=y[index] for x,y in zip(ar,br))
 report[split]=info
 print(split,json.dumps(info))
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
