import gzip
import hashlib
import json
import lzma
from pathlib import Path

DATA_ROOT=Path(__file__).resolve().parent


def token_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def load_manifest(root=DATA_ROOT):
    root=Path(root)
    value=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if value['format'] not in {'ep-opd-jsonl-xz-v1','ep-opd-jsonl-gzip-v1'}:
        raise ValueError('unsupported data manifest')
    return value


def read_rows(group,root=DATA_ROOT,verify=True):
    root=Path(root).resolve();manifest=load_manifest(root)
    prefix='data/'+group.strip('/')+'/'
    files=[row for row in manifest['files'] if row['path'].startswith(prefix)]
    if not files:
        raise ValueError('dataset group not found: '+group)
    for row in files:
        path=(root.parent/row['path']).resolve()
        if root not in path.parents:
            raise ValueError('dataset shard is outside data/')
        if verify and (path.stat().st_size!=row['bytes'] or
                       hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']):
            raise ValueError('dataset shard checksum mismatch: '+row['path'])
        opener=lzma.open if path.suffix=='.xz' else gzip.open
        count=0
        with opener(path,'rt',encoding='utf8') as stream:
            for line in stream:
                if line.strip():
                    count+=1;yield json.loads(line)
        if count!=row['records']:
            raise ValueError('dataset shard row count differs: '+row['path'])


def verify_splits(splits,counts):
    papers={};ids=set()
    for split,rows in splits.items():
        if len(rows)!=counts[split]:raise ValueError('paired split count differs: '+split)
        papers[split]={row['source']['paper_id'] for row in rows}
        for row in rows:
            if row['pair_id'] in ids:raise ValueError('duplicate pair ID across splits')
            ids.add(row['pair_id'])
            if set(row['documents'])!={'Dorig','Dcf','Dpara','Dirr','Ddrop'}:
                raise ValueError('paired record needs all five variants')
    names=list(papers)
    if any(papers[left]&papers[right] for i,left in enumerate(names) for right in names[i+1:]):
        raise ValueError('source papers overlap across paired splits')
