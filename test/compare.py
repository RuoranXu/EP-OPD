"""Paired source-paper bootstrap for matched predictions after test.evaluate.

python -m test.compare --baseline outputs/sft_test.jsonl --candidate outputs/ep_test.jsonl
Optional CSP labels must completely cover both methods; no missing labels become zeros.
"""
import argparse
import json
from pathlib import Path

from test.bootstrap import ratios
from test.claims import read_rows, response_hash


def samples(path, include_csp):
    correctness=json.loads(path.with_suffix('.correctness.json').read_text(encoding='utf8'))
    predictions=read_rows(path)
    outputs={(row['pair_id'],row['variant']):row for row in predictions}
    if len(outputs)!=len(predictions):
        raise ValueError('duplicate predictions')
    variants=['Dorig','Dcf','Dpara','Dirr','Ddrop']
    expected={(row['pair_id'],variant) for row in correctness for variant in variants}
    if set(outputs)!=expected:
        raise ValueError('prediction/correctness coverage differs')
    labels={}
    if include_csp:
        raw=read_rows(path.with_suffix('.complete_claim_labels.jsonl'))
        labels={row['response_hash']:row for row in raw}
        if len(labels)!=len(raw) or set(labels)!={response_hash(row) for row in predictions}:
            raise ValueError('CSP labels do not cover every prediction exactly once')
    result=[]
    for row in correctness:
        values={variant+'_f1':(row[variant+'_f1'],1.) for variant in ['Dorig','Dcf','Dpara','Dirr']}
        values.update({variant+'_binary_correct':(float(row[variant]),1.) for variant in variants})
        values['bypass_rate']=(float(row['Dorig'] and row['persistence']),float(row['Dorig']))
        generated=[outputs[row['pair_id'],variant] for variant in variants]
        values['cap_rate']=(sum(bool(item.get('hit_cap')) for item in generated)/len(variants),1.)
        if include_csp:
            relevant=[claim for item in generated for claim in labels[response_hash(item)]['claims']
                      if claim['answer_relevant'] and claim['label']!='irrelevant']
            values['claim_support_precision']=((sum(c['label']=='supported' for c in relevant)/len(relevant),1.)
                                               if relevant else (0.,0.))
        result.append((row['pair_id'],row['paper_id'],values))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--replicates',type=int,default=10000)
    parser.add_argument('--output',default='outputs/comparison.json')
    args=parser.parse_args()
    if args.replicates<1:
        parser.error('replicates must be positive')
    paths=[Path(args.baseline),Path(args.candidate)]
    summaries=[json.loads(path.with_suffix('.summary.json').read_text(encoding='utf8')) for path in paths]
    for field in ['cache_sha256','scorer_version','original_references_sha256']:
        if summaries[0].get(field)!=summaries[1].get(field):
            raise ValueError('unmatched evaluation field: '+field)
    identities=[summary.get('identity') for summary in summaries]
    if any(identity is None for identity in identities):
        raise ValueError('both summaries need generation identities from test.generate')
    for field in ['input_sha256','split','stop_protocol','max_new_tokens','context','decoding',
                  'seed','dtype','batch_size','backend','backend_version','termination_assisted','budget_policy']:
        if identities[0].get(field)!=identities[1].get(field):
            raise ValueError('unmatched generation field: '+field)
    csp=[path.with_suffix('.complete_claim_labels.jsonl').exists() for path in paths]
    if csp[0]!=csp[1]:
        raise ValueError('CSP is present for only one method')
    result=ratios(dict(zip(['base','ep_opd'],[samples(path,all(csp)) for path in paths])),args.replicates)
    result['csp_status']='complete_labels_checked' if all(csp) else 'not_judged'
    result['method_paths']={'base':str(paths[0]),'ep_opd':str(paths[1])}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
