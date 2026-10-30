import hashlib
import json
from pathlib import Path
import random

from evidence_opd.paired_metrics import final_answer
from evidence_opd.qa_metrics import annotation_f1,token_f1


def prepare_probe(payload,config,cache_sha256,root):
    count=int(config['pairs']);cap=int(config['response_tokens'])
    if not 0<count<=len(payload['development']) or not 0<cap<=4096:
        raise ValueError('invalid fixed development probe size')
    pairs=random.Random(config['seed']).sample(sorted(payload['development'],key=lambda p:p['pair_id']),count)
    references={};reference_hash=None
    if config.get('original_references'):
        path=Path(root)/config['original_references']
        reference_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        references={row['id']:row['gold_answers'] for row in
                    (json.loads(line) for line in path.open() if line.strip())}
    jobs=[]
    for pair in pairs:
        for variant in ['Dorig','Dcf']:
            answer=pair['answers'][variant]
            jobs.append({'pair_id':pair['pair_id'],'variant':variant,
                'prompt_ids':pair['student_prompts'][variant],
                'references':references.get(pair['source'].get('question_id')) if variant=='Dorig' else None,
                'canonical':[answer['canonical'],*answer.get('aliases',[])]})
    identity={'version':'development-guard-v1','cache_sha256':cache_sha256,'config':config,
              'original_references_sha256':reference_hash,
              'jobs_sha256':hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest(),
              'prompt':'unchanged training answer-only prompt','decoding':'greedy',
              'purpose':'early behavior warning; not the complete development or test score'}
    identity['signature']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return jobs,identity


def repetition_fraction(tokens):
    tokens=tokens[-768:]
    if len(tokens)<128:return 0.
    grams=[tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    return 1-len(set(grams))/len(grams)


def summarize_probe(jobs,tokens,tokenizer,eos,requested_cap):
    if len(jobs)!=len(tokens):raise ValueError('development probe coverage differs')
    rows=[]
    for job,values in zip(jobs,tokens):
        actual_cap=min(requested_cap,32768-len(job['prompt_ids']))
        if not values or len(values)>actual_cap:raise ValueError('invalid development response length')
        text=tokenizer.decode(values,skip_special_tokens=True)
        answer=final_answer(text)
        f1=annotation_f1(answer,job['references']) if job['references'] else max(token_f1(answer,a) for a in job['canonical'])
        ended=values[-1] in eos
        rows.append({'pair_id':job['pair_id'],'variant':job['variant'],'text':text,'answer_f1':f1,
                     'generated_tokens':len(values),'ended_with_eos':ended,
                     'hit_cap':len(values)==actual_cap and not ended,
                     'repeated_4gram_fraction':repetition_fraction(values)})
    def avg(key,selected=rows):return sum(row[key] for row in selected)/len(selected)
    summary={'original_answer_f1':avg('answer_f1',[r for r in rows if r['variant']=='Dorig']),
             'counterfactual_answer_f1':avg('answer_f1',[r for r in rows if r['variant']=='Dcf']),
             'eos_rate':avg('ended_with_eos'),'cap_rate':avg('hit_cap'),
             'mean_generated_tokens':avg('generated_tokens'),
             'repetitive_response_fraction':sum(r['repeated_4gram_fraction']>.6 for r in rows)/len(rows)}
    return summary,rows


def assess(history,baseline,config):
    """Require sustained deterioration; a noisy single small probe only warns."""
    patience=int(config['patience'])
    if patience<1:raise ValueError('invalid development guard patience')
    def reasons(score):
        bad=[]
        for key in ['original_answer_f1','counterfactual_answer_f1']:
            if score[key]<baseline[key]-float(config['maximum_f1_drop']):bad.append(key)
        if score['eos_rate']<baseline['eos_rate']-float(config['maximum_eos_drop']):bad.append('eos_rate')
        if score['repetitive_response_fraction']>float(config['maximum_repetitive_fraction']):bad.append('repetition')
        return bad
    warnings=reasons(history[-1]['summary'])
    stop=len(history)>=patience and all(reasons(h['summary']) for h in history[-patience:])
    return {'warning':warnings,'stop':stop,'patience':patience,
            'checks':[h['completed_updates'] for h in history[-patience:]]}


def improves(score,best,baseline):
    if score['original_answer_f1']<baseline['original_answer_f1'] or score['counterfactual_answer_f1']<baseline['counterfactual_answer_f1']:
        return False
    if score['eos_rate']<baseline['eos_rate']-.05 or score['repetitive_response_fraction']>.1:return False
    prior=baseline if best is None else best['summary']
    return (score['original_answer_f1']+score['counterfactual_answer_f1']>
            prior['original_answer_f1']+prior['counterfactual_answer_f1'])


def generate_probe(client,jobs,config,eos,version=None,initial_base=False):
    prompts=[j['prompt_ids'] for j in jobs]
    cap=int(config['response_tokens']);caps=[min(cap,32768-len(p)) for p in prompts]
    seeds=[int(config['seed'])+i for i in range(len(prompts))]
    if initial_base:
        result=client.generate_base(prompts,seeds,cap,eos,caps)
    else:
        result=client.generate(prompts,version,seeds,cap,eos,temperature=0.,top_p=1.,caps=caps)
    return result['tokens']


def exchange_probe_tokens(rank,tokens,group):
    import torch.distributed as dist
    if rank==3:
        dist.send_object_list([tokens],dst=0,group=group)
    elif rank==0:
        holder=[None];dist.recv_object_list(holder,src=3,group=group)
        return holder[0]
    return None
