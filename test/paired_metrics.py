"""Deterministic atomic-answer scoring for the paper's paired evaluations."""
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from test.qa_metrics import token_f1,annotation_f1

SCORER_VERSION = 'paired-answer-v3-f1-primary'


def final_answer(text):
    value=text.rsplit('</think>',1)[-1]
    matches=list(re.finditer(r'(?im)^\s*ANSWER\s*:\s*',value))
    value=value[matches[-1].end():].strip() if matches else value.strip()
    # The answer field ends at a new paragraph or another structured field.
    # Keep continuation lines together: "A\nor B" must not become just "A".
    return re.split(r'\n[ \t]*\n|\n[ \t]*(?:CLAIM|EVIDENCE|RELATION|QUESTION)\s*:',
                    value,maxsplit=1,flags=re.I)[0].strip()


def normalize(text):
    value=unicodedata.normalize('NFKC',str(text)).casefold()
    # Paired dollar signs delimit mathematical typography, not answer content.
    # A lone currency sign is retained.
    value=re.sub(r'\$([^$\n]+)\$',r'\1',value)
    value=re.sub(r'\[(?:s\d+[-\w]*|\d+(?:\s*[,;-]\s*\d+)*)\]','',value)
    value=value.strip().strip('`"\' .;:')
    return re.sub(r'\s+',' ',value)


def numeric(text):
    text=normalize(text).replace('−','-')
    match=re.fullmatch(r'([+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z%µμ/²³0-9 ^-]*)',text)
    if not match:
        return None
    try:
        number=Decimal(match[1].replace(',',''))
    except InvalidOperation:
        return None
    unit=match[2].strip()
    conversions={'percent':('%',Decimal(1)),'percentage points':('pp',Decimal(1)),
                 'milliseconds':('s',Decimal('.001')),'ms':('s',Decimal('.001')),
                 'seconds':('s',Decimal(1)),'second':('s',Decimal(1)),
                 'minutes':('s',Decimal(60)),'min':('s',Decimal(60)),
                 'kilometers':('m',Decimal(1000)),'km':('m',Decimal(1000)),
                 'centimeters':('m',Decimal('.01')),'cm':('m',Decimal('.01'))}
    target,factor=conversions.get(unit,(unit,Decimal(1)))
    return number*factor,target


def correct(text, answer):
    value=normalize(final_answer(text))
    aliases=[answer['canonical']]+list(answer.get('aliases',[]))
    if value in {normalize(a) for a in aliases}:
        return True
    kind=answer.get('type')
    if kind=='numeric':
        number=numeric(value)
        return number is not None and any(number==numeric(a) for a in aliases)
    if kind in {'boolean','direction','comparison'}:
        ontology={'true':'yes','false':'no','improved':'increase','improves':'increase',
                  'increases':'increase','increased':'increase','higher':'increase',
                  'decreases':'decrease','decreased':'decrease','reduces':'decrease',
                  'reduced':'decrease','lower':'decrease','unchanged':'no change'}
        value=ontology.get(value,value)
        return value in {ontology.get(normalize(a),normalize(a)) for a in aliases}
    if kind=='bounded_set':
        def members(value):
            value=normalize(value).strip('{}[]()')
            # A thousands separator is part of one member, not a set boundary.
            value=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',value)
            return {normalize(v) for v in re.split(r'\s*[,;]\s*|\s+and\s+',value) if v}
        return any(members(value)==members(a) for a in aliases)
    return False


def summarize(pairs, predictions, original_references=None):
    rows=[]
    for pair in pairs:
        pid=pair['pair_id']
        scores={v:correct(predictions[(pid,v)]['text'],pair['answers'][v]) for v in pair['answers']}
        f1s={v+'_f1':max(token_f1(final_answer(predictions[(pid,v)]['text']),a)
                        for a in [pair['answers'][v]['canonical']]+list(pair['answers'][v].get('aliases',[])))
             for v in pair['answers']}
        if original_references is not None:
            references=original_references[pair['source']['question_id']]
            for variant in ['Dorig','Dpara','Dirr']:
                f1s[variant+'_f1']=annotation_f1(final_answer(predictions[(pid,variant)]['text']),references)
        persistence=correct(predictions[(pid,'Dcf')]['text'],pair['answers']['Dorig'])
        rows.append({'pair_id':pid,'paper_id':pair['source']['paper_id'],**scores,**f1s,'persistence':persistence,
                     'question_origin':pair['source'].get('question_origin','original_dataset_question')})
    summary=aggregate_rows(rows)
    summary['scorer_version']=SCORER_VERSION
    summary['f1_reference_source']='original QASPER annotations for answer-preserving variants; counterfactual aliases' if original_references is not None else 'validated canonical answers and aliases'
    summary['by_question_origin']={origin:aggregate_rows([r for r in rows if r['question_origin']==origin])
                                   for origin in sorted({r['question_origin'] for r in rows})}
    return summary,rows


def aggregate_rows(rows):
    total=len(rows)
    orig=sum(x['Dorig'] for x in rows)
    summary={'pairs':total,'papers':len({x['paper_id'] for x in rows}),
             'original_accuracy':orig/total,'counterfactual_accuracy':sum(x['Dcf'] for x in rows)/total,
             'bypass_numerator':sum(x['Dorig'] and x['persistence'] for x in rows),
             'bypass_denominator':orig,
             'bypass_rate':sum(x['Dorig'] and x['persistence'] for x in rows)/orig if orig else None,
             'abstention_accuracy':sum(x['Ddrop'] for x in rows)/total,
             'paraphrase_accuracy':sum(x['Dpara'] for x in rows)/total,
             'irrelevant_accuracy':sum(x['Dirr'] for x in rows)/total,
             'claim_support_precision':None,'claim_evaluation_status':'requires_blinded_claim_labels',
             'manual_pair_review_status':'not_performed_by_automated_pipeline'}
    summary.update(primary_metric='answer_f1',acc_definition='Acc(F1): mean max-reference token F1, not binary accuracy',
                   acc_f1=sum(x['Dorig_f1'] for x in rows)/total,
                   original_answer_f1=sum(x['Dorig_f1'] for x in rows)/total,
                   counterfactual_answer_f1=sum(x['Dcf_f1'] for x in rows)/total,
                   paraphrase_answer_f1=sum(x['Dpara_f1'] for x in rows)/total,
                   irrelevant_answer_f1=sum(x['Dirr_f1'] for x in rows)/total,
                   bypass_correctness_definition='binary canonical answer correctness; F1 overlap does not define correctness')
    return summary
