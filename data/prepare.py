from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from .dataset import DATA_ROOT,load_manifest,read_rows,token_digest,verify_splits

CACHE=DATA_ROOT/'cache'


@contextmanager
def preparation_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix(path.suffix+'.lock')
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise RuntimeError('another process is preparing '+str(path)) from None
    try:
        os.write(fd,str(os.getpid()).encode());yield
    finally:
        os.close(fd);lock.unlink()


def manifest_digest():
    return hashlib.sha256((DATA_ROOT/'manifest.json').read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)


def atomic_torch(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def user_prompt(row):
    blocks=[]
    for section in row['sections']:
        blocks.append(f"[{section['section_id']}] {section['section_name']}")
        blocks.extend(f"[{paragraph['paragraph_id']}] {paragraph['text']}" for paragraph in section['paragraphs'])
    if row.get('figures_and_tables'):
        blocks.append('Figures and tables')
        blocks.extend(f"[{item['float_id']}] {item['caption']}" for item in row['figures_and_tables'])
    paper='\n\n'.join(blocks)
    return (f"Title: {row['title']}\n\nAbstract: {row['abstract']}\n\n"
            f"Question: {row['question']}\n\nPaper (paragraph IDs are stable metadata):\n"
            f"{paper}\n\nQuestion: {row['question']}\nAnswer:")


def build_pair(raw,student,teacher):
    from evidence_opd.train.stop_contract import content,encode_prefix,END_MARKER,STUDENT_EOS,inspect_completion
    row={key:copy.deepcopy(value) for key,value in raw.items() if key!='token_sha256'}
    row.update(student_prompts={},teacher_prompts={},evaluation_prompts={},clean_targets={})
    for variant,document in row['documents'].items():
        text=content(user_prompt(document))
        ids=encode_prefix(student,text)
        row['evaluation_prompts'][variant]=ids
        evidence='NONE' if variant=='Ddrop' else ', '.join(row['evidence_closure_ids'])
        target=' '+row['answers'][variant]['canonical'].strip()+'\nEVIDENCE: '+evidence+'\n'+END_MARKER
        if not inspect_completion(target,row['retained_block_ids']).valid:
            raise ValueError('invalid canonical completion: '+row['pair_id'])
        row['clean_targets'][variant]=student.encode(target,add_special_tokens=False)+[STUDENT_EOS]
        if variant in {'Dorig','Dcf'}:
            row['student_prompts'][variant]=ids
            row['teacher_prompts'][variant]=encode_prefix(teacher,text,teacher=True)
        if max(len(ids),len(row['teacher_prompts'].get(variant,[])))+len(row['clean_targets'][variant])>32768:
            raise ValueError('unchanged packed example exceeds context')
    row['answer_ids']=[student.encode(' '+row['answers'][variant]['canonical'],add_special_tokens=False)
                       for variant in ['Dorig','Dcf']]
    for field,expected in raw['token_sha256'].items():
        observed=({variant:token_digest(ids) for variant,ids in row[field].items()}
                  if isinstance(row[field],dict) else token_digest(row[field]))
        if observed!=expected:
            raise ValueError('prompt/target token IDs changed: '+row['pair_id']+'/'+field)
    return row


def references(split):
    path=CACHE/f'qasper_{split}_references.jsonl'
    if path.exists():return path
    CACHE.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf8') as stream:
        for row in read_rows('references/'+split):stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    temporary.replace(path);return path


def paired_cache(student_path,teacher_path,output=None):
    from transformers import AutoTokenizer
    path=Path(output) if output else CACHE/'paired_cache.pt'
    if path.exists():return path
    with preparation_lock(path):
        student=AutoTokenizer.from_pretrained(student_path,local_files_only=True)
        teacher=AutoTokenizer.from_pretrained(teacher_path,local_files_only=True)
        if student.get_vocab()!=teacher.get_vocab():raise ValueError('teacher/student vocabularies differ')
        manifest=load_manifest()
        vocab=hashlib.sha256(json.dumps(student.get_vocab(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if vocab!=manifest['paired_metadata']['vocab_sha256']:raise ValueError('tokenizer differs from frozen data')
        raw={split:list(read_rows('paired/'+split)) for split in ['train','development','test']}
        verify_splits(raw,manifest['paired_counts'])
        result={'metadata':copy.deepcopy(manifest['paired_metadata'])}
        result['metadata'].update(portable_dataset_manifest_sha256=manifest_digest(),
            source_cache_sha256=manifest['paired_source_cache_sha256'],all_prompt_target_hashes_verified=True)
        for split,rows in raw.items():
            result[split]=[build_pair(row,student,teacher) for row in rows]
        atomic_torch(path,result)
        atomic_json(path.with_suffix('.manifest.json'),dict(cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dataset_manifest_sha256=manifest_digest(),counts=manifest['paired_counts'],all_token_hashes_verified=True))
    references('development');references('test')
    return path


def qasper_inputs(student_path,output=None):
    from transformers import AutoTokenizer
    path=Path(output) if output else CACHE/'qasper_validation_inputs.jsonl'
    if path.exists():return path
    with preparation_lock(path):
        tokenizer=AutoTokenizer.from_pretrained(student_path,local_files_only=True)
        temporary=path.with_suffix(path.suffix+'.tmp');count=0
        with temporary.open('w',encoding='utf8') as stream:
            for raw in read_rows('qasper/validation'):
                ids=tokenizer.encode(raw['prompt_text'],add_special_tokens=False)
                if token_digest(ids)!=raw['prompt_ids_sha256']:
                    raise ValueError('QASPER prompt IDs differ: '+raw['id'])
                row={key:value for key,value in raw.items() if key not in {'prompt_text','prompt_ids_sha256'}}
                row['prompt_ids']=ids;stream.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
        if count!=load_manifest()['full_qasper_questions']:raise ValueError('incomplete QASPER input coverage')
        temporary.replace(path)
    references('development');return path


def baseline_prompt_ids(row,student,teacher,system,maximum):
    """Keep complete paper paragraphs and the question/answer cue within the configured budget."""
    paragraphs=[(i,p) for i,section in enumerate(row['sections']) for p in section['paragraphs']]
    def render(count):
        document=copy.deepcopy(row)
        for section in document['sections']:section['paragraphs']=[]
        for index,paragraph in paragraphs[:count]:document['sections'][index]['paragraphs'].append(paragraph)
        if count<len(paragraphs):document['figures_and_tables']=[]
        result={'student_prompt_ids':student.encode(system+'\n\n'+user_prompt(document),add_special_tokens=False)}
        rendered=teacher.apply_chat_template([{'role':'system','content':system},
            {'role':'user','content':user_prompt(document)}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        result['teacher_normal_ids']=teacher.encode(rendered,add_special_tokens=False)
        return result
    low=0;high=len(paragraphs);best=None
    while low<=high:
        middle=(low+high)//2;candidate=render(middle)
        if max(map(len,candidate.values()))<=maximum:best=candidate;low=middle+1
        else:high=middle-1
    if best is None:raise ValueError('question exceeds configured prompt budget')
    return best


def baseline_cache(config):
    from transformers import AutoTokenizer
    from evidence_opd.methods import teacher_prompt_key
    from .privileged import load_privileged
    prompt_key=teacher_prompt_key(config)
    privileged_path=config['data'].get('privileged_path')
    if prompt_key!='teacher_normal_ids' and (not privileged_path or not Path(privileged_path).is_file()):
        raise ValueError('PI baselines require external data.privileged_path (PRIVILEGED_DATA_PATH); construction is not included')
    student=AutoTokenizer.from_pretrained(config['models']['student_tokenizer'],local_files_only=True)
    model_key='ema_tokenizer' if config['method']['family']=='ema_teacher' else 'teacher_tokenizer'
    teacher=AutoTokenizer.from_pretrained(config['models'][model_key],local_files_only=True)
    if student.get_vocab()!=teacher.get_vocab():raise ValueError('baseline teacher/student vocabularies differ')
    data=config['data'];training=config['training'];seed=int(config['experiment']['seed'])
    source=list(read_rows('baseline/train'))
    privileged=(load_privileged(privileged_path,prompt_key,{str(row['id']) for row in source},
        int(data['max_prompt_tokens']),set(teacher.get_vocab().values()))
        if prompt_key!='teacher_normal_ids' else {})
    identity=dict(dataset_manifest_sha256=manifest_digest(),system=config['prompt']['system'],
        vocabulary=token_digest(student.get_vocab()),teacher_chat_template=teacher.chat_template,
        teacher_prompt=config['method'].get('teacher_prompt','normal'),
        privileged_inputs_sha256=(hashlib.sha256(Path(privileged_path).read_bytes()).hexdigest() if privileged else None),
        preparation_version=2,
        max_prompt_tokens=data['max_prompt_tokens'],max_response_tokens=data['max_response_tokens'],
        student_budget=data['student_max_batch_tokens'],teacher_budget=data['teacher_max_batch_tokens'],
        buckets=data['length_buckets'],gradient_accumulation=training['gradient_accumulation_steps'],seed=seed,
        packing='leading complete paragraphs; answer cue retained; no student gold evidence')
    tag=token_digest(identity)[:16]
    path=CACHE/f'baseline_train_{tag}.pt';batches_path=CACHE/f'baseline_batches_{tag}.jsonl'
    if not path.exists() or not batches_path.exists():
        with preparation_lock(path):
            records=[]
            for row in source:
                values=baseline_prompt_ids(row,student,teacher,config['prompt']['system'],int(data['max_prompt_tokens']))
                if privileged:values[prompt_key]=privileged[str(row['id'])]
                values.update(id=row['id'],paper_id=row['paper_id'],input_length=len(values['student_prompt_ids']),
                    max_teacher_input_length=max(len(ids) for key,ids in values.items() if key.startswith('teacher_')))
                records.append(values)
            grouped={int(boundary):[] for boundary in data['length_buckets']};rng=random.Random(seed)
            for row in records:
                boundary=next((b for b in grouped if row['input_length']<=b),max(grouped));grouped[boundary].append(row)
            for rows in grouped.values():rng.shuffle(rows)
            order=list(grouped);rng.shuffle(order);ordered=[row for boundary in order for row in grouped[boundary]]
            batches=[];batch=[];student_cost=teacher_cost=0;response=int(data['max_response_tokens'])
            for row in ordered:
                a=row['input_length']+response;b=row['max_teacher_input_length']+response
                if batch and (student_cost+a>data['student_max_batch_tokens'] or teacher_cost+b>data['teacher_max_batch_tokens']):
                    batches.append(batch);batch=[];student_cost=teacher_cost=0
                batch.append(row);student_cost+=a;teacher_cost+=b
            if batch:batches.append(batch)
            accumulation=int(training['gradient_accumulation_steps'])
            atomic_torch(path,records)
            with batches_path.open('w',encoding='utf8') as stream:
                for index,batch in enumerate(batches):
                    stream.write(json.dumps(dict(optimizer_step=index//accumulation,micro_batch=index%accumulation,
                        ids=[row['id'] for row in batch],estimated_student_tokens=sum(row['input_length']+response for row in batch),
                        estimated_teacher_tokens=sum(row['max_teacher_input_length']+response for row in batch)))+'\n')
            atomic_json(path.with_suffix('.manifest.json'),dict(identity=identity,records=len(records),
                microbatches=len(batches),optimizer_steps=math.ceil(len(batches)/accumulation)))
    return path,batches_path


def prepare_training(config,entry):
    """Called once by the VERL controller before workers are allocated."""
    if entry=='baseline':
        explicit=os.environ.get('BASELINE_CACHE_PATH');batch_env=os.environ.get('BASELINE_BATCH_MANIFEST')
        if bool(explicit)!=bool(batch_env):
            raise ValueError('BASELINE_CACHE_PATH and BASELINE_BATCH_MANIFEST must be set together')
        if explicit and batch_env:
            config['data'].update(tokenized_path=explicit,batch_manifest=batch_env)
        else:
            cache,batches=baseline_cache(config)
            config['data'].update(tokenized_path=str(cache),batch_manifest=str(batches))
    else:
        config['data']['cache']=str(paired_cache(config['models']['student_tokenizer'],
            config['models']['teacher_tokenizer'],config['data']['cache']))


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--student-tokenizer',default=os.environ.get('STUDENT_MODEL_PATH','models/Qwen3-4B-Base'))
    parser.add_argument('--teacher-tokenizer',default=os.environ.get('TEACHER_MODEL_PATH','models/Qwen3-30B-A3B-Instruct-2507'))
    args=parser.parse_args()
    print(paired_cache(args.student_tokenizer,args.teacher_tokenizer))
    print(qasper_inputs(args.student_tokenizer))


if __name__=='__main__':main()
