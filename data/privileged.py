import json
from pathlib import Path


def load_privileged(path,field,expected_ids,maximum,vocabulary):
    """Read externally supplied teacher token IDs; no privileged-data construction."""
    if field not in {'teacher_pi_ids','teacher_pi_span_ids'}:
        raise ValueError('unknown privileged input field')
    result={}
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row,dict) or 'id' not in row:
                raise ValueError('privileged input records require id')
            key=str(row['id']);tokens=row.get(field)
            if key in result:raise ValueError('duplicate privileged input ID: '+key)
            if not isinstance(tokens,list) or not tokens:
                raise ValueError('external privileged inputs require '+field+': '+key)
            if len(tokens)>maximum:raise ValueError('privileged prompt exceeds token budget: '+key)
            if any(type(token) is not int or token not in vocabulary for token in tokens):
                raise ValueError('invalid privileged token IDs: '+key)
            result[key]=tokens
    if set(result)!=set(expected_ids):
        raise ValueError('privileged inputs must match every baseline question ID exactly')
    return result
