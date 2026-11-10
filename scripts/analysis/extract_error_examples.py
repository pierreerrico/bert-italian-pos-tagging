"""Extract four illustrative cases, retaining predictions for every saved seed."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
TABLE_SEED=123
CASES=[('adjectival_participle',195,2),('institutional_title',319,3),
       ('shared_participle_error',30,18),('rare_degree',109,0)]

def read_csv(path):
 with path.open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))

def parse_conllu(path):
 sentences=[];tokens=[];metadata={}
 for line in path.read_text(encoding='utf8').splitlines()+['']:
  if not line.strip():
   if tokens:sentences.append((metadata,tokens));tokens=[];metadata={}
  elif line.startswith('# '):
   if ' = ' in line: k,v=line[2:].split(' = ',1);metadata[k]=v
  elif not line.startswith('#'):
   row=line.split('\t')
   if row[0].isdigit():tokens.append(row)
 return sentences

def main():
 sources=[ROOT/'data/UD_Italian-ISDT-r2.15/it_isdt-ud-test.conllu',
          ROOT/'outputs/references/udpipe_2_15/predictions.csv']
 gold=parse_conllu(sources[0]);reference=read_csv(sources[1])
 u={(int(row['sentence_index']),int(row['word_index'])):row for row in reference}
 responses=[]
 for path in sorted((ROOT/'outputs/references/udpipe_2_15').glob('response_*.conllu')):
  responses.extend(parse_conllu(path));sources.append(path)
 assert len(gold)==len(responses)==482
 bert={}
 for seed in [42,123,456]:
  path=ROOT/f'outputs/bert_upos_then_feats_v2/linear_upos_feats/seed_{seed}/predictions.csv'
  bert[seed]={(int(r['sentence_index']),int(r['word_index'])):r for r in read_csv(path)}
  sources.append(path)
 cases=[]
 for kind,sid,wid in CASES:
  metadata,words=gold[sid];g=words[wid];r=responses[sid][1][wid]
  assert [word[1] for word in words]==[word[1] for word in responses[sid][1]]
  assert g[1]==u[sid,wid]['word'] and g[3]==u[sid,wid]['true_label']
  assert r[3]==u[sid,wid]['predicted_label']
  case=dict(kind=kind,sentence_index=sid,word_index=wid,sent_id=metadata['sent_id'],
            sentence=metadata.get('text',' '.join(w[1] for w in words)),word=g[1],
            gold_upos=g[3],gold_feats=g[5],gold_relation=g[7],udpipe_upos=r[3],udpipe_feats=r[5],bert={})
  for seed in [42,123,456]:
   b=bert[seed][sid,wid]
   assert b['word']==g[1] and b['true_label']==g[3] and b['true_feats']==g[5]
   case['bert'][str(seed)]=dict(upos=b['predicted_label'],feats=b['predicted_feats'])
  case['bert_agrees_across_seeds']=len({json.dumps(v,sort_keys=True) for v in case['bert'].values()})==1
  cases.append(case)
 output=ROOT/'outputs/references/qualitative_examples.json'
 data=dict(selection='Four illustrative cases, not a representative sample. The report table shows seed 123; predictions for all seeds are retained and need not agree.',
           table_seed=TABLE_SEED,
           sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],cases=cases)
 output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 for case in cases:print(case['word'],case['gold_upos'],case['udpipe_upos'],case['bert'][str(TABLE_SEED)],case['sentence'])
 print('Saved',output)

if __name__=='__main__':main()
